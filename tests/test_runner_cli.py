import dataclasses
import json
from pathlib import Path

import pytest

from conftest import night, src_run
from myopenfund_agents import cli
from myopenfund_agents.bench import traps
from myopenfund_agents.bench.fixtures import Expected, Meta, Night, save
from myopenfund_agents.bench.runner import (
    DEFAULT_CORPUS_REPOS,
    DEFAULT_REPO,
    BenchConfigError,
    repo_of,
    run_bench,
    run_night,
    write_report,
)
from myopenfund_agents.bench.score import load_thresholds
from myopenfund_agents.core.detector import detect
from myopenfund_agents.core.explanations import Explanation, Explanations, Quote
from myopenfund_agents.core.findings import Category, Severity
from myopenfund_agents.core.summary import build_summary
from myopenfund_agents.engines.base import EngineError, EngineResult
from myopenfund_agents.engines.fake import FakeEngine
from myopenfund_agents.sanitise import DenyList
from myopenfund_agents.settings import A1Settings

THRESHOLDS = Path(__file__).resolve().parents[1] / "bench" / "thresholds.toml"


def _rec(profile: str, repeat: int = 1, settings: A1Settings | None = None) -> dict:
    """How a report was produced: repeat count, A1 settings and threshold profile values."""
    return {"repeat": repeat, "settings": settings or A1Settings(), "thresholds": load_thresholds(THRESHOLDS, profile)}


def _known_problem_night(green_history):
    base = night(source_runs=[src_run("us", seen=150), src_run("fr"), src_run("se")], history=green_history)
    return traps.known_problem(base, "se", DEFAULT_CORPUS_REPOS["central-bank"])


def _quiet_night(**kw):
    inputs = night([src_run("ok1")], night_id="2026-11-20", **kw)
    return Night(inputs=inputs, expected=Expected(), meta=Meta(origin="green", description="quiet"))


QUOTE = "bad_content=2"


def _exp(fid="f1", diagnosis="Text layer empty.", quote=QUOTE, cause=Category.UNREADABLE_PDF) -> Explanation:
    return Explanation(
        finding_id=fid, diagnosis=diagnosis, likely_cause=cause, severity=Severity.WARN,
        quotes=[Quote(source="summary", text=quote)],
    )  # fmt: skip


def _answer(*explanations, observations=()) -> Explanations:
    return Explanations(explanations=list(explanations or [_exp()]), observations=list(observations))


def _fake(**kw) -> FakeEngine:
    return FakeEngine(_answer(), **kw)


class _BrokenEngine:
    name = "broken"
    model = "none"

    def explain(self, prompt, ids):
        raise EngineError("engine unreachable")


def test_fake_engine_scores_known_problem_as_comment(tmp_path, green_history):
    n, green = _known_problem_night(green_history), _quiet_night()
    engine = _fake()
    t = load_thresholds(THRESHOLDS, "local")
    scores, verdicts = run_bench([n, green], engine, A1Settings(), repeat=3, thresholds=t)
    assert [s.detector_exact for s in scores] == [True, True]
    assert scores[0].attempts[0].actions == [("comment", "se", "unreadable_pdf")]
    assert len(engine.prompts) == 3  # every attempt of the night with a finding, none of the quiet one
    assert all(v.passed for v in verdicts), verdicts
    md = write_report(tmp_path, engine, "local", scores, verdicts, **_rec("local", repeat=3))
    assert md.suffix == ".md" and md.with_suffix(".json").is_file()
    text = md.read_text(encoding="utf-8")
    assert "| detector exact | 100% (2/2) | 100% | yes |" in text
    assert "| findings explained | 100% (3/3) | 90% | yes |" in text
    assert "| likely_cause agreement (worst) | 100% |" in text
    assert "| fallback nights | 0 |" in text
    assert (
        "| Night | Origin | Findings | Detector | Explained | Valid | Quotes | Fallback | Unknown numbers "
        "| Cause agreement | Max s |" in text
    )
    assert f"| {n.inputs.night_id} | trap | `se/unreadable_pdf` | ok | 3/3 | 3/3 | 3/3 | no | 0 | 100% |" in text
    assert "| 2026-11-20 | green | - | ok | - | - | - | - | - | - | - |" in text
    payload = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["engine"] == "fake" and payload["model"] == "scripted" and payload["profile"] == "local"
    assert payload["nights"][0]["night_id"] == n.inputs.night_id
    assert payload["reported"]["fallback nights"] == 0 and payload["reported"]["unknown numbers"] == 0


def test_report_name_replaces_model_separators(tmp_path, green_history):
    n = _known_problem_night(green_history)
    engine = _fake(model="ollama/qwen3:8b")
    scores, verdicts = run_bench([n], engine, A1Settings(), 1, load_thresholds(THRESHOLDS, "prod"))
    md = write_report(tmp_path, engine, "prod", scores, verdicts, **_rec("prod"))
    assert md.name.endswith("-fake-ollama_qwen3_8b.md")


def test_report_refused_when_payload_has_infra_detail(tmp_path, green_history):
    n = _known_problem_night(green_history)
    engine = _fake(model="100.64.1.2")
    scores, verdicts = run_bench([n], engine, A1Settings(), 1, load_thresholds(THRESHOLDS, "local"))
    with pytest.raises(ValueError, match="report refused") as err:
        write_report(tmp_path / "out", engine, "local", scores, verdicts, **_rec("local"))
    assert "100.64.1.2" not in str(err.value)
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_engine_error_attempt_counted_invalid(green_history):
    n = _known_problem_night(green_history)
    score = run_night(n, _BrokenEngine(), A1Settings(), 2, DEFAULT_CORPUS_REPOS)
    assert score.valid == 0 and score.explained == 0 and score.explainable == 2 and score.fallback
    assert [a.error for a in score.attempts] == ["engine unreachable", "engine unreachable"]
    assert all(a.valid is False and a.explanations == [] for a in score.attempts)
    # the alert does not need the model
    assert score.detector_exact and score.attempts[0].actions == [("comment", "se", "unreadable_pdf")]


def test_repo_of_maps_known_corpora_only(green_history):
    n = _known_problem_night(green_history)
    n.inputs.source_runs.append(src_run("xx", corpus="elsewhere"))
    n.inputs.history.append(src_run("co", night=2, corpus="company"))
    got = repo_of(n, DEFAULT_CORPUS_REPOS)
    assert got["se"] == "central-bank-corpus" and got["co"] == "company-corpus"
    assert "xx" not in got
    assert DEFAULT_REPO == "data-orchestrator"


def test_cli_run_bad_engine_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MOF_AGENTS_ENGINE", "nope")
    code = cli.main(["bench-run", "--profile", "local", "--nights", str(tmp_path), "--thresholds", str(THRESHOLDS)])
    assert code == 2
    assert "MOF_AGENTS_ENGINE" in capsys.readouterr().err


def test_cli_run_with_scripted_engine(tmp_path, monkeypatch, capsys, green_history):
    n = _known_problem_night(green_history)
    save(n, tmp_path / "nights")
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: _fake())
    args = ["bench-run", "--profile", "local", "--nights", str(tmp_path / "nights"),
            "--thresholds", str(THRESHOLDS), "--results", str(tmp_path / "results")]  # fmt: skip
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert str(tmp_path / "results") in out
    assert "PASS findings explained: 100% (3/3) (threshold 90%)" in out
    assert "INFO unknown numbers: 0" in out and "MISMATCH" not in out
    assert cli.main([*args, "--only", "no-such-night"]) == 2
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: FakeEngine(None))
    assert cli.main(args) == 1


def test_cli_capture_missing_env_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MOF_AGENTS_MCP_URL", raising=False)
    monkeypatch.setenv("MOF_AGENTS_MCP_TOKEN", "sekret-value-123")
    args = ["bench-capture", "--night", "2026-11-22", "--origin", "green", "--description", "d",
            "--nights", str(tmp_path)]  # fmt: skip
    assert cli.main(args) == 2
    err = capsys.readouterr().err
    assert "MOF_AGENTS_MCP_URL" in err and "sekret-value-123" not in err


def test_cli_capture_writes_pending_night_and_refuses_existing(tmp_path, monkeypatch, green_history):
    monkeypatch.setenv("MOF_AGENTS_MCP_URL", "https://mcp.example.com/mcp")
    monkeypatch.setenv("MOF_AGENTS_MCP_TOKEN", "t")
    monkeypatch.delenv("MOF_AGENTS_DENYLIST", raising=False)
    seen = {}

    def fake_capture(call, **kw):
        seen.update(kw)
        return night(source_runs=[src_run("se")], history=green_history)

    monkeypatch.setattr(cli, "mcp_caller", lambda url, token: None)
    monkeypatch.setattr(cli, "capture_night", fake_capture)
    args = ["bench-capture", "--night", "2026-11-22", "--origin", "green", "--description", "quiet",
            "--sources", "se,fr", "--nights", str(tmp_path), "--no-denylist"]  # fmt: skip
    assert cli.main(args) == 0
    assert seen["night_end"].isoformat() == "2026-11-22T06:00:00+00:00"
    assert seen["sources"] == ["se", "fr"] and seen["window_nights"] == 14 and seen["corpus"] == "central-bank"
    meta = json.loads((tmp_path / "2026-11-22" / "meta.json").read_text())
    assert meta["origin"] == "green" and meta["validated_by"] is None
    assert "expected answers pending validation" in meta["notes"]
    assert cli.main(args) == 1


# --- Fix round 1 ---

PRIVATE_ERROR = "connect to http://100.64.1.2:11434 refused"


class _PrivateErrorEngine:
    name = "broken"
    model = "none"

    def explain(self, prompt, ids):
        raise EngineError(PRIVATE_ERROR)


def _run_args(tmp_path):
    return ["bench-run", "--profile", "local", "--repeat", "1", "--nights", str(tmp_path / "nights"),
            "--thresholds", str(THRESHOLDS), "--results", str(tmp_path / "results")]  # fmt: skip


def test_run_night_withholds_infra_error_text(green_history):
    score = run_night(_known_problem_night(green_history), _PrivateErrorEngine(), A1Settings(), 1, None)
    assert score.attempts[0].error == "EngineError: withheld (private_ip)"


def test_cli_private_engine_error_keeps_verdicts_and_report(tmp_path, monkeypatch, capsys, green_history):
    save(_known_problem_night(green_history), tmp_path / "nights")
    monkeypatch.delenv("MOF_AGENTS_DENYLIST", raising=False)
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: _PrivateErrorEngine())
    assert cli.main(_run_args(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "FAIL valid output: 0% (0/1)" in out and "PASS detector exact" in out
    (report,) = (tmp_path / "results").glob("*.json")
    text = report.read_text(encoding="utf-8")
    assert "withheld (private_ip)" in text and "100.64.1.2" not in text


def test_cli_run_bad_denylist_exits_2(tmp_path, monkeypatch, capsys, green_history):
    save(_known_problem_night(green_history), tmp_path / "nights")
    monkeypatch.setenv("MOF_AGENTS_DENYLIST", str(tmp_path / "missing.toml"))
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: _fake())
    assert cli.main(_run_args(tmp_path)) == 2
    assert "MOF_AGENTS_DENYLIST" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_write_report_uses_given_deny_list(tmp_path, green_history):
    engine = _fake(model="mirror.example.org")
    scores, verdicts = run_bench([_known_problem_night(green_history)], engine, A1Settings(), 1,
                                 load_thresholds(THRESHOLDS, "local"))  # fmt: skip
    with pytest.raises(ValueError, match="report refused"):
        write_report(
            tmp_path, engine, "local", scores, verdicts, deny=DenyList(hosts=["mirror.example.org"]), **_rec("local")
        )


def test_report_wraps_keys_in_backticks(tmp_path, green_history):
    engine = _fake()
    scores, verdicts = run_bench([_known_problem_night(green_history)], engine, A1Settings(), 1,
                                 load_thresholds(THRESHOLDS, "local"))  # fmt: skip
    text = write_report(tmp_path, engine, "local", scores, verdicts, **_rec("local")).read_text(encoding="utf-8")
    assert "| `se/unreadable_pdf` |" in text


def test_criterion_constants_exported():
    from myopenfund_agents.bench import score

    assert score.DETECTOR_EXACT == "detector exact"
    assert score.SLOWEST_NIGHT == "slowest night (s)"


def test_cli_run_corrupt_night_exits_2(tmp_path, monkeypatch, capsys, green_history):
    d = save(_known_problem_night(green_history), tmp_path / "nights")
    (d / "expected.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: _fake())
    assert cli.main(_run_args(tmp_path)) == 2
    assert str(d) in capsys.readouterr().err


def test_cli_run_unknown_only_id_exits_2(tmp_path, monkeypatch, capsys, green_history):
    n = _known_problem_night(green_history)
    save(n, tmp_path / "nights")
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: _fake())
    assert cli.main([*_run_args(tmp_path), "--only", f"{n.inputs.night_id},ghost-night"]) == 2
    assert "ghost-night" in capsys.readouterr().err


@pytest.mark.parametrize("content", ["{oops", "[1, 2]", '{"central-bank": 3}'])
def test_cli_run_bad_corpus_repos_exits_2(tmp_path, monkeypatch, green_history, content):
    save(_known_problem_night(green_history), tmp_path / "nights")
    f = tmp_path / "repos.json"
    f.write_text(content, encoding="utf-8")
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: _fake())
    assert cli.main([*_run_args(tmp_path), "--corpus-repos", str(f)]) == 2


def test_cli_run_corpus_repos_mapping_applies(tmp_path, monkeypatch, green_history):
    # the open issue is on central-bank-corpus; mapping the corpus elsewhere turns the comment into a new issue
    save(_known_problem_night(green_history), tmp_path / "nights")
    f = tmp_path / "repos.json"
    f.write_text('{"central-bank": "other-repo"}', encoding="utf-8")
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: _fake())
    assert cli.main(_run_args(tmp_path)) == 0
    assert cli.main([*_run_args(tmp_path), "--corpus-repos", str(f)]) == 1


def _capture_env(monkeypatch):
    monkeypatch.setenv("MOF_AGENTS_MCP_URL", "https://mcp.example.com/mcp")
    monkeypatch.setenv("MOF_AGENTS_MCP_TOKEN", "t")
    monkeypatch.delenv("MOF_AGENTS_DENYLIST", raising=False)
    monkeypatch.setattr(cli, "mcp_caller", lambda url, token: None)


def _capture_args(tmp_path, *extra):
    return ["bench-capture", "--night", "2026-11-22", "--origin", "green", "--description", "d",
            "--nights", str(tmp_path), *extra]  # fmt: skip


def test_cli_capture_cut_with_timezone_exits_2(tmp_path, monkeypatch):
    _capture_env(monkeypatch)
    assert cli.main(_capture_args(tmp_path, "--cut", "06:00+02:00")) == 2


def test_cli_capture_error_exits_1(tmp_path, monkeypatch, capsys):
    from myopenfund_agents.bench.capture import CaptureError

    _capture_env(monkeypatch)

    def failing(call, **kw):
        raise CaptureError("history incomplete for source 'se'")

    monkeypatch.setattr(cli, "capture_night", failing)
    assert cli.main(_capture_args(tmp_path, "--no-denylist")) == 1
    assert "history incomplete" in capsys.readouterr().err
    assert not (tmp_path / "2026-11-22").exists()


# --- Final review fix wave ---


class _TokenEngine:
    """Returns a fixed answer with token counts, like a real adapter."""

    name = "tokens"
    model = "scripted"

    def __init__(self, answer: Explanations) -> None:
        self.answer = answer

    def explain(self, prompt, ids):
        return EngineResult(self.name, self.model, 0.5, 1, 1200, 80, explanations=self.answer)


def test_attempt_keeps_explanations_and_tokens(green_history):
    score = run_night(_known_problem_night(green_history), _TokenEngine(_answer()), A1Settings(), 1, None)
    (attempt,) = score.attempts
    assert (attempt.input_tokens, attempt.output_tokens, attempt.seconds, attempt.retries) == (1200, 80, 0.5, 1)
    assert attempt.valid is True and attempt.error is None
    assert attempt.explanations == [
        {
            "finding_id": "f1", "diagnosis": "Text layer empty.", "likely_cause": "unreadable_pdf", "severity": "warn",
            "quotes": [{"source": "summary", "text": QUOTE}],
        }
    ]  # fmt: skip
    assert (attempt.quotes_total, attempt.quotes_exact) == (1, 1)


def test_failed_attempt_has_no_explanations_or_tokens(green_history):
    score = run_night(_known_problem_night(green_history), _BrokenEngine(), A1Settings(), 1, None)
    (attempt,) = score.attempts
    assert attempt.explanations == [] and attempt.input_tokens is None and attempt.output_tokens is None
    assert attempt.seconds >= 0 and attempt.model is None


def test_an_unusable_answer_is_invalid_but_keeps_its_cost(green_history):
    engine = _TokenEngine(_answer(_exp(fid="f99")))
    score = run_night(_known_problem_night(green_history), engine, A1Settings(), 1, None)
    (attempt,) = score.attempts
    assert attempt.valid is False and "not one of: f1" in attempt.error
    assert attempt.explanations == [] and attempt.input_tokens == 1200 and attempt.model == "scripted"
    assert "f99" not in json.dumps(dataclasses.asdict(attempt))


def test_a_quiet_night_asked_about_nothing_has_no_validity():
    score = run_night(_quiet_night(), _fake(), A1Settings(), 1, None)
    assert score.attempts[0].valid is None and score.valid == 0 and score.detector_exact


def test_dropped_explanation_is_counted_with_its_reason(green_history):
    engine = _TokenEngine(_answer(_exp(quote="never in the summary"), observations=["odd run 987654"]))
    score = run_night(_known_problem_night(green_history), engine, A1Settings(), 1, None)
    (attempt,) = score.attempts
    assert attempt.valid is True and attempt.explanations == [] and attempt.dropped == {"f1": "false quote"}
    assert (score.quotes_total, score.quotes_exact, score.explained, score.fallback) == (1, 0, 0, True)
    assert attempt.observations == ["odd run 987654"]


def test_unknown_numbers_are_counted(green_history):
    engine = _TokenEngine(_answer(_exp(diagnosis="987654 files are gone.")))
    score = run_night(_known_problem_night(green_history), engine, A1Settings(), 2, None)
    assert score.unknown_numbers == 2


def test_detector_mismatch_is_seen_by_the_runner(green_history):
    n = _known_problem_night(green_history)
    n = n.model_copy(update={"expected": Expected()})
    score = run_night(n, _fake(), A1Settings(), 1, None)
    assert score.detector_exact is False and score.detected == [("se", "unreadable_pdf")]
    scores, verdicts = run_bench([n], _fake(), A1Settings(), 1, load_thresholds(THRESHOLDS, "local"))
    assert [v.criterion for v in verdicts if not v.passed] == ["detector exact"]


def test_run_bench_without_any_detected_finding_never_calls_the_engine():
    engine = _fake()
    settings = A1Settings(explain_green_nights=True)
    with pytest.raises(BenchConfigError, match="no detected finding"):
        run_bench([_quiet_night()], engine, settings, 1, load_thresholds(THRESHOLDS, "local"))
    assert engine.prompts == []


def test_cli_run_without_any_detected_finding_exits_2(tmp_path, monkeypatch, capsys, green_history):
    save(_quiet_night(), tmp_path / "nights")
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: _fake())
    assert cli.main(_run_args(tmp_path)) == 2
    captured = capsys.readouterr()
    assert "no detected finding" in captured.err and "PASS" not in captured.out
    assert not (tmp_path / "results").exists()
    # the same through --only, next to a night that has one
    save(_known_problem_night(green_history), tmp_path / "nights")
    assert cli.main(_run_args(tmp_path)) == 0
    assert cli.main([*_run_args(tmp_path), "--only", "2026-11-20"]) == 2


def test_report_json_carries_the_detector_findings_and_the_answer(tmp_path, green_history):
    n = _known_problem_night(green_history)
    engine = _TokenEngine(_answer(observations=["odd"]))
    scores, verdicts = run_bench([n], engine, A1Settings(), 1, load_thresholds(THRESHOLDS, "local"))
    md = write_report(tmp_path, engine, "local", scores, verdicts, **_rec("local"))
    attempt = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))["nights"][0]["attempts"][0]
    assert attempt["input_tokens"] == 1200 and attempt["output_tokens"] == 80
    (d,) = detect(build_summary(n.inputs, A1Settings()), A1Settings())
    assert attempt["findings"] == [
        {
            "finding_id": "f1", "source_code": "se", "category": "unreadable_pdf", "severity": "warn",
            "title": d.finding.title, "evidence": d.finding.evidence, "other_flags": list(d.other_flags),
        }
    ]  # fmt: skip
    assert attempt["actions"] == [["comment", "se", "unreadable_pdf"]]
    assert attempt["explanations"][0]["diagnosis"] == "Text layer empty."
    assert attempt["explanations"][0]["quotes"] == [{"source": "summary", "text": QUOTE}]
    assert attempt["observations"] == ["odd"] and attempt["dropped"] == {} and attempt["error"] is None
    assert "result" not in attempt


def test_report_keeps_drop_reasons_and_engine_errors(tmp_path, green_history):
    n = _known_problem_night(green_history)
    t = load_thresholds(THRESHOLDS, "local")

    def attempt_of(engine, out):
        scores, verdicts = run_bench([n], engine, A1Settings(), 1, t)
        md = write_report(out, engine, "local", scores, verdicts, **_rec("local"))
        return json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))["nights"][0]["attempts"][0]

    got = attempt_of(_TokenEngine(_answer(_exp(quote="never in the summary"))), tmp_path / "a")
    assert got["dropped"] == {"f1": "false quote"} and got["explanations"] == []
    assert attempt_of(_BrokenEngine(), tmp_path / "b")["error"] == "engine unreachable"


def test_infra_detail_in_the_answer_never_reaches_the_report(tmp_path, green_history):
    """The raw answer is not in the report: not the dropped text, not a withheld observation."""
    answer = _answer(_exp(diagnosis="Fetch from 100.64.1.2 failed."), observations=["see 10.1.2.3", "fine"])
    engine = _TokenEngine(answer)
    n = _known_problem_night(green_history)
    scores, verdicts = run_bench([n], engine, A1Settings(), 1, load_thresholds(THRESHOLDS, "local"))
    md = write_report(tmp_path, engine, "local", scores, verdicts, **_rec("local"))
    text = md.with_suffix(".json").read_text(encoding="utf-8")
    assert "100.64.1.2" not in text and "10.1.2.3" not in text
    attempt = json.loads(text)["nights"][0]["attempts"][0]
    assert attempt["dropped"] == {"f1": "withheld"} and attempt["observations"] == ["fine"]


def test_report_withholds_texts_with_infra_detail(tmp_path, green_history):
    """Texts let through by the run are checked again with the report's deny-list."""
    answer = _answer(_exp(diagnosis="mirror.example.org is down"), observations=["see mirror.example.org", "fine"])
    engine = _TokenEngine(answer)
    n = _known_problem_night(green_history)
    scores, verdicts = run_bench([n], engine, A1Settings(), 1, load_thresholds(THRESHOLDS, "local"))
    assert scores[0].explained == 1
    attempt = scores[0].attempts[0]
    attempt.findings[0]["title"] = "se: mirror.example.org unreachable"
    attempt.findings[0]["evidence"] = ["bad_content=2", "GET http://100.64.1.2/x"]
    attempt.explanations[0]["quotes"].append({"source": "summary", "text": "password=abcdef123456"})
    attempt.error = "connect to mirror.example.org refused"
    deny = DenyList(hosts=["mirror.example.org"])
    md = write_report(tmp_path, engine, "local", scores, verdicts, deny=deny, **_rec("local"))
    text = md.with_suffix(".json").read_text(encoding="utf-8")
    assert "100.64.1.2" not in text and "mirror.example.org" not in text and "abcdef123456" not in text
    got = json.loads(text)["nights"][0]["attempts"][0]
    assert got["findings"][0]["title"] == "withheld (deny_host)"
    assert got["findings"][0]["evidence"] == ["bad_content=2", "withheld (private_ip)"]
    assert got["findings"][0]["source_code"] == "se"
    assert got["explanations"][0]["diagnosis"] == "withheld (deny_host)"
    assert [q["text"] for q in got["explanations"][0]["quotes"]] == [QUOTE, "withheld (credential)"]
    assert got["observations"] == ["withheld (deny_host)", "fine"]
    assert got["error"] == "withheld (deny_host)"


def test_report_refused_when_any_other_field_has_infra_detail(tmp_path, green_history):
    engine = _fake()
    n = _known_problem_night(green_history)
    scores, verdicts = run_bench([n], engine, A1Settings(), 1, load_thresholds(THRESHOLDS, "local"))
    scores[0].attempts[0].findings[0]["source_code"] = "100.64.1.2"
    with pytest.raises(ValueError, match="report refused"):
        write_report(tmp_path / "out", engine, "local", scores, verdicts, **_rec("local"))
    assert not (tmp_path / "out").exists()


def test_run_night_applies_the_given_deny_list(green_history):
    n = _known_problem_night(green_history)
    engine = _TokenEngine(_answer(_exp(diagnosis="mirror.example.org is down")))
    deny = DenyList(hosts=["mirror.example.org"])
    assert run_night(n, engine, A1Settings(), 1, None, deny).attempts[0].dropped == {"f1": "withheld"}


def _fake_capture(green_history):
    def capture(call, **kw):
        return night(source_runs=[src_run("se")], history=green_history)

    return capture


def test_cli_capture_requires_denylist(tmp_path, monkeypatch, capsys, green_history):
    _capture_env(monkeypatch)
    monkeypatch.setattr(cli, "capture_night", _fake_capture(green_history))
    assert cli.main(_capture_args(tmp_path)) == 2
    assert "MOF_AGENTS_DENYLIST" in capsys.readouterr().err
    assert not (tmp_path / "2026-11-22").exists()
    assert cli.main(_capture_args(tmp_path, "--no-denylist")) == 0
    assert (tmp_path / "2026-11-22" / "inputs.json").is_file()


def test_cli_capture_refuses_empty_denylist(tmp_path, monkeypatch, capsys, green_history):
    _capture_env(monkeypatch)
    monkeypatch.setattr(cli, "capture_night", _fake_capture(green_history))
    deny = tmp_path / "deny.toml"
    deny.write_text('hosts = [""]\npath_roots = []\n', encoding="utf-8")
    monkeypatch.setenv("MOF_AGENTS_DENYLIST", str(deny))
    nights = tmp_path / "nights"
    assert cli.main(_capture_args(nights)) == 2
    assert "no host and no path root" in capsys.readouterr().err
    assert cli.main(_capture_args(nights, "--no-denylist")) == 0


def test_cli_capture_passes_denylist_entries(tmp_path, monkeypatch, green_history):
    _capture_env(monkeypatch)
    seen = {}

    def capture(call, **kw):
        seen.update(kw)
        return night(source_runs=[src_run("se")], history=green_history)

    monkeypatch.setattr(cli, "capture_night", capture)
    deny = tmp_path / "deny.toml"
    deny.write_text('hosts = ["nas.example.lan"]\npath_roots = ["/srv/private/"]\n', encoding="utf-8")
    monkeypatch.setenv("MOF_AGENTS_DENYLIST", str(deny))
    assert cli.main(_capture_args(tmp_path / "nights")) == 0
    assert seen["deny"].hosts == ["nas.example.lan"] and seen["deny"].path_roots == ["/srv/private/"]


def test_report_records_how_it_was_produced(tmp_path, green_history):
    engine = _fake()
    settings = A1Settings(volume_drop_ratio=0.3, engine_timeout_s=120.0)
    t = load_thresholds(THRESHOLDS, "local")
    scores, verdicts = run_bench([_known_problem_night(green_history)], engine, settings, 2, t)
    md = write_report(tmp_path, engine, "local", scores, verdicts, repeat=2, settings=settings, thresholds=t)
    payload = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["repeat"] == 2
    assert payload["settings"]["volume_drop_ratio"] == 0.3 and payload["settings"]["engine_timeout_s"] == 120.0
    assert set(payload["settings"]) == set(dataclasses.asdict(A1Settings()))
    assert payload["thresholds"] == t
    text = md.read_text(encoding="utf-8")
    assert "Profile `local`, repeat 2." in text


def test_write_report_empty_denylist_env_reads_as_unset(tmp_path, monkeypatch, green_history):
    monkeypatch.setenv("MOF_AGENTS_DENYLIST", "")
    engine = _fake()
    scores, verdicts = run_bench([_known_problem_night(green_history)], engine, A1Settings(), 1,
                                 load_thresholds(THRESHOLDS, "local"))  # fmt: skip
    md = write_report(tmp_path, engine, "local", scores, verdicts, **_rec("local"))
    assert md.exists()


def test_run_night_no_run_does_not_comment_on_chronic_issue(green_history):
    """A no-run night with an open chronic ('*', unknown) issue opens its own no-run issue."""
    from myopenfund_agents.core.inputs import OpenIssue

    base = night(source_runs=[src_run("us", seen=150), src_run("fr"), src_run("se")], history=green_history)
    base.open_issues.append(OpenIssue(fingerprint=f"a1:{DEFAULT_REPO}:*:unknown", number=7, repo=DEFAULT_REPO))
    n = traps.no_run(base)
    score = run_night(n, _BrokenEngine(), A1Settings(), 1, None)
    assert score.attempts[0].actions == [("new_issue", "*", "unknown")]
    assert score.detector_exact


def test_report_names_the_models_that_answered(tmp_path, green_history):
    class Resolving:
        name, model = "claude-code", "default"

        def explain(self, prompt, ids):
            return EngineResult(self.name, "real-model", 0.1, explanations=_answer())

    score = run_night(_known_problem_night(green_history), Resolving(), A1Settings(), repeat=1, corpus_repos=None)
    md = write_report(
        tmp_path, Resolving(), "prod", [score], [], DenyList(), repeat=1, settings=A1Settings(), thresholds={}
    )
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert payload["models_used"] == ["real-model"]
    assert "Model that answered: real-model" in md.read_text()


# --- review round 1 ---


class _ScriptedEngine:
    """Plays one step per call: an answer, or an exception to raise; `clock` advances by `cost` each call."""

    name = "steps"
    model = "scripted"

    def __init__(self, *steps, clock=None, cost=0.0, tokens=(None, None)):
        self.steps, self.clock, self.cost, self.tokens = list(steps), clock, cost, tokens
        self.calls = 0

    def explain(self, prompt, ids):
        step = self.steps[min(self.calls, len(self.steps) - 1)]
        self.calls += 1
        if self.clock is not None:
            self.clock[0] += self.cost
        if isinstance(step, BaseException):
            raise step
        return EngineResult(self.name, self.model, 2.0, 0, *self.tokens, explanations=step)


def _clock(monkeypatch):
    from types import SimpleNamespace

    from myopenfund_agents.bench import runner

    clock = [100.0]
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    return clock


def _md(tmp_path, engine, scores, verdicts, profile="local"):
    md = write_report(tmp_path, engine, profile, scores, verdicts, **_rec(profile))
    return md.read_text(encoding="utf-8")


def test_a_timed_out_engine_can_fail_the_time_limit(monkeypatch, green_history):
    clock = _clock(monkeypatch)
    engine = _ScriptedEngine(_answer(), EngineError("timed out"), clock=clock, cost=700.0)
    n = _known_problem_night(green_history)
    scores, verdicts = run_bench([n], engine, A1Settings(), 2, load_thresholds(THRESHOLDS, "local"))
    assert [a.seconds for a in scores[0].attempts] == [2.0, 700.0]
    (slow,) = [v for v in verdicts if v.criterion == "slowest night (s)"]
    assert slow.value == 700.0 and not slow.passed


def test_an_unusable_answer_takes_the_elapsed_time(monkeypatch, green_history):
    engine = _ScriptedEngine(_answer(_exp(fid="f99")), clock=_clock(monkeypatch), cost=50.0)
    score = run_night(_known_problem_night(green_history), engine, A1Settings(), 1, None)
    assert score.attempts[0].valid is False and score.attempts[0].seconds == 50.0


def test_failing_report(tmp_path, monkeypatch, green_history):
    n, quiet = _known_problem_night(green_history), _quiet_night()
    other = _exp(diagnosis="987654 files are gone.", cause=Category.UNKNOWN)
    engine = _ScriptedEngine(
        _answer(), EngineError("down"), _answer(other), _answer(_exp(quote="never in the summary")),
        clock=_clock(monkeypatch), cost=10.0, tokens=(100, None),
    )  # fmt: skip
    t = load_thresholds(THRESHOLDS, "local") | {"max_seconds_per_night": 5}
    scores, verdicts = run_bench([n, quiet], engine, A1Settings(), 4, t)
    text = _md(tmp_path, engine, scores, verdicts)
    for line in (
        "| detector exact | 100% (2/2) | 100% | yes |",
        "| findings explained | 50% (2/4) | 90% | NO |",
        "| valid output | 75% (3/4) | 90% | NO |",
        "| quotes exact (of quotes given) | 67% (2/3) | 95% | NO |",
        "| slowest night (s) | 10.0 | 5.0 | NO |",
        "| fallback nights | 1 |",
        "| unknown numbers | 1 |",
        "| likely_cause agreement (mean) | 25% |",
        "| likely_cause agreement (worst) | 25% |",
        "| seconds (mean) | 4.0 |",
        "| seconds (worst) | 10.0 |",
        "| input tokens | 300, partial (3 of 4 attempts) |",
        "| output tokens | - |",
        f"| {n.inputs.night_id} | trap | `se/unreadable_pdf` | ok | 2/4 | 3/4 | 2/3 | YES | 1 | 25% | 10.0 |",
        "| 2026-11-20 | green | - | ok | - | - | - | - | - | - | - |",
    ):
        assert line in text.splitlines(), line
    assert "Detector mismatches" not in text


def test_report_row_shows_the_worst_finding_and_the_unknown_agreement(tmp_path, green_history):
    inputs = night([src_run("aa", fetch_errors=3), src_run("bb", fetch_errors=3)])
    n = Night(inputs=inputs, expected=Expected(), meta=Meta(origin="incident", description="d"))
    ids = [d.finding_id for d in detect(build_summary(inputs, A1Settings()), A1Settings())]

    def answer(second):
        return _answer(*(_exp(fid=i, quote="fetch_errors=3", cause=c) for i, c in zip(ids, second, strict=True)))

    u, v = Category.UNKNOWN, Category.PARSE_ERROR
    engine = _ScriptedEngine(answer((u, u)), answer((u, v)))
    t = load_thresholds(THRESHOLDS, "prod")
    scores, verdicts = run_bench([n], engine, A1Settings(), 2, t)
    text = _md(tmp_path / "a", engine, scores, verdicts, "prod")
    assert "| likely_cause agreement (mean) | 75% |" in text and "| likely_cause agreement (worst) | 50% |" in text
    assert "| 4/4 | 2/2 | 4/4 | no | 0 | 50% | 2.0 |" in text
    # judged in prod, so listed once
    assert text.count("| fallback nights |") == 1 and "| fallback nights | 0 | 0 | yes |" in text
    engine = _ScriptedEngine(answer((u, u)))
    scores, verdicts = run_bench([n], engine, A1Settings(), 1, t)
    text = _md(tmp_path / "b", engine, scores, verdicts, "prod")
    assert "| likely_cause agreement (mean) | - |" in text and "| likely_cause agreement (worst) | - |" in text
    assert "| 2/2 | 1/1 | 2/2 | no | 0 | - | 2.0 |" in text
    assert "| input tokens | - |" in text


def test_report_lists_detector_mismatches(tmp_path, green_history):
    n = _known_problem_night(green_history)
    wrong = n.model_copy(update={"expected": Expected()})
    engine = _fake()
    t = load_thresholds(THRESHOLDS, "local")
    scores, verdicts = run_bench([wrong, _quiet_night()], engine, A1Settings(), 2, t)
    text = _md(tmp_path, engine, scores, verdicts)
    assert "| detector exact | 50% (1/2) | 100% | NO |" in text
    assert "| trap | `se/unreadable_pdf` | NO |" in text
    section = text.split("## Detector mismatches", 1)[1].split("##", 1)[0]
    assert (
        f"| {n.inputs.night_id} | 1, 2 | - | `se/unreadable_pdf` | - | `comment se/unreadable_pdf` |" in section
    )
    assert "2026-11-20" not in section
    (payload,) = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert payload["nights"][0]["expected_actions"] == []


def test_an_early_mismatch_is_reported_even_if_the_last_attempt_matches(tmp_path, green_history):
    from myopenfund_agents.bench.score import score_night, verdicts

    n = _known_problem_night(green_history)
    good = run_night(n, _fake(), A1Settings(), 2, None).attempts
    early = dataclasses.replace(good[0], findings=[], actions=[])
    scores = [score_night(n, [early, good[1]])]
    engine = _fake()
    text = _md(tmp_path, engine, scores, verdicts(scores, load_thresholds(THRESHOLDS, "local")))
    row = "| 1 | `se/unreadable_pdf` | - | `comment se/unreadable_pdf` | - |"
    assert row in text.split("## Detector mismatches", 1)[1]


def test_cli_prints_one_line_per_mismatching_night(tmp_path, monkeypatch, capsys, green_history):
    n = _known_problem_night(green_history)
    save(n.model_copy(update={"expected": Expected()}), tmp_path / "nights")
    save(_quiet_night(), tmp_path / "nights")
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: _fake())
    assert cli.main([*_run_args(tmp_path), "--repeat", "2"]) == 1
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("MISMATCH")]
    assert lines == [
        f"MISMATCH {n.inputs.night_id} (attempts 1, 2): findings expected - got se/unreadable_pdf; "
        "actions expected - got comment se/unreadable_pdf"
    ]


def test_a_night_with_two_wrong_results_has_one_line(green_history):
    from myopenfund_agents.bench.runner import mismatch_lines
    from myopenfund_agents.bench.score import score_night

    n = _known_problem_night(green_history).model_copy(update={"expected": Expected()})
    (good,) = run_night(n, _fake(), A1Settings(), 1, None).attempts
    score = score_night(n, [good, dataclasses.replace(good, actions=[])])
    assert len(score.mismatches()) == 2 and len(mismatch_lines([score], DenyList())) == 1


def test_cli_mismatch_line_withholds_an_infra_detail(tmp_path, monkeypatch, capsys, green_history):
    n = _known_problem_night(green_history)
    save(n.model_copy(update={"expected": Expected()}), tmp_path / "nights")
    deny = tmp_path / "deny.toml"
    deny.write_text(f'hosts = ["{n.inputs.night_id}"]\n', encoding="utf-8")
    monkeypatch.setenv("MOF_AGENTS_DENYLIST", str(deny))
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: _fake())
    assert cli.main(_run_args(tmp_path)) == 1
    assert "MISMATCH withheld (deny_host)" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("criterion", "value", "shown"),
    [
        ("findings explained", 299 / 300, "99.7%"),
        ("findings explained", 9999 / 10000, "99.9%"),
        ("findings explained", 0.994, "99%"),
        ("findings explained", 0.995, "99.5%"),
        ("findings explained", 1.0, "100%"),
        ("likely_cause agreement (worst)", 299 / 300, "99.7%"),
        ("slowest night (s)", 12.34, "12.3"),
        ("seconds (mean)", 600, "600.0"),
        ("fallback nights", 0, "0"),
        ("unknown numbers", None, "-"),
        ("likely_cause agreement (mean)", None, "-"),
        ("input tokens", {"total": 1234567, "known": 3, "attempts": 3}, "1234567"),
        ("input tokens", {"total": 0, "known": 3, "attempts": 3}, "0"),
        ("input tokens", {"total": 12, "known": 2, "attempts": 3}, "12, partial (2 of 3 attempts)"),
        ("input tokens", {"total": None, "known": 0, "attempts": 3}, "-"),
    ],
)
def test_fmt(criterion, value, shown):
    from myopenfund_agents.bench.runner import fmt

    assert fmt(criterion, value) == shown


def test_green_nights_asked_about_nothing_are_in_no_model_figure(monkeypatch, green_history):
    n, quiet = _known_problem_night(green_history), _quiet_night()
    settings = A1Settings(explain_green_nights=True)
    t = load_thresholds(THRESHOLDS, "local")
    empty = Explanations(explanations=[], observations=[])
    for green_step in (empty, EngineError("down")):
        engine = _ScriptedEngine(_answer(), green_step, clock=_clock(monkeypatch), cost=700.0, tokens=(5, 5))
        scores, verdicts = run_bench([n, quiet], engine, settings, 1, t)
        assert engine.calls == 2
        assert scores[1].attempts[0].valid is (green_step is empty)
        assert scores[1].valid == 0 and not scores[1].fallback
        assert {v.criterion: v.detail for v in verdicts}["valid output"] == "1/1"
        assert all(v.passed for v in verdicts), verdicts


def test_repeat_below_one_is_refused(tmp_path, monkeypatch, capsys, green_history):
    with pytest.raises(ValueError, match="repeat must be >= 1"):
        run_night(_known_problem_night(green_history), _fake(), A1Settings(), 0, None)
    save(_known_problem_night(green_history), tmp_path / "nights")
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: _fake())
    assert cli.main([*_run_args(tmp_path), "--repeat", "0"]) == 2
    assert "--repeat must be >= 1" in capsys.readouterr().err


def test_cli_forwards_repeat(tmp_path, monkeypatch, green_history):
    save(_known_problem_night(green_history), tmp_path / "nights")
    engine = _fake()
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: engine)
    assert cli.main([*_run_args(tmp_path), "--repeat", "4"]) == 0
    assert len(engine.prompts) == 4
    (report,) = (tmp_path / "results").glob("*.json")
    payload = json.loads(report.read_text())
    assert payload["repeat"] == 4 and len(payload["nights"][0]["attempts"]) == 4


def test_run_bench_itself_withholds_with_the_given_deny_list(green_history):
    n, t = _known_problem_night(green_history), load_thresholds(THRESHOLDS, "local")
    deny = DenyList(hosts=["mirror.example.org"])
    answer = _ScriptedEngine(_answer(_exp(diagnosis="mirror.example.org is down")))
    scores, _ = run_bench([n], answer, A1Settings(), 1, t, None, deny)
    assert scores[0].attempts[0].dropped == {"f1": "withheld"}
    failing = _ScriptedEngine(EngineError("mirror.example.org refused"))
    scores, _ = run_bench([n], failing, A1Settings(), 1, t, None, deny)
    assert scores[0].attempts[0].error == "EngineError: withheld (deny_host)"


def test_cli_forwards_the_deny_list_to_the_run(tmp_path, monkeypatch, green_history):
    save(_known_problem_night(green_history), tmp_path / "nights")
    deny = tmp_path / "deny.toml"
    deny.write_text('hosts = ["mirror.example.org"]\n', encoding="utf-8")
    monkeypatch.setenv("MOF_AGENTS_DENYLIST", str(deny))
    engine = _ScriptedEngine(EngineError("mirror.example.org refused"))
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: engine)
    assert cli.main(_run_args(tmp_path)) == 1
    (report,) = (tmp_path / "results").glob("*.json")
    # "EngineError: " is written by the run; a text only withheld by the report has no type name
    assert json.loads(report.read_text())["nights"][0]["attempts"][0]["error"] == "EngineError: withheld (deny_host)"


# --- review round 2: a value is never shown on the wrong side of its threshold ----------------------------


def _verdict(criterion, value, threshold, passed, detail=""):
    from myopenfund_agents.bench.score import Verdict

    return Verdict(criterion, value, threshold, passed, detail)


@pytest.mark.parametrize(
    ("criterion", "value", "threshold", "passed", "shown"),
    [
        ("findings explained", 44 / 49, 0.9, False, ("89.8%", "90.0%")),
        ("quotes exact (of quotes given)", 52 / 55, 0.95, False, ("94.5%", "95.0%")),
        ("slowest night (s)", 600.04, 600, False, ("600.04", "600.00")),
        ("slowest night (s)", 599.96, 600, True, ("599.96", "600.00")),
        ("slowest night (s)", 600.0, 600, True, ("600.0", "600.0")),
        ("findings explained", 299 / 300, 1.0, False, ("99.7%", "100%")),
        ("findings explained", 299 / 300, 0.9, True, ("99.7%", "90%")),
        ("findings explained", 99999 / 100000, 1.0, False, ("99.9%", "100%")),
        ("findings explained", 0.99995, 0.9999, True, ("99.995%", "99.990%")),
        ("findings explained", 0.9, 0.9, True, ("90%", "90%")),
        ("findings explained", 19 / 20, 0.95, True, ("95%", "95%")),
        ("findings explained", 1.0, 1.0, True, ("100%", "100%")),
        ("findings explained", 0.904, 0.9, True, ("90.4%", "90.0%")),
        ("findings explained", 0.9249, 0.925, False, ("92.49%", "92.50%")),
        ("fallback nights", 1, 0, False, ("1", "0")),
        ("fallback nights", 3, 2.9999999, False, ("3.0000000", "2.9999999")),
    ],
)
def test_verdict_texts(criterion, value, threshold, passed, shown):
    from myopenfund_agents.bench.runner import verdict_texts

    assert verdict_texts(_verdict(criterion, value, threshold, passed)) == shown


def test_verdict_texts_sweep():
    from myopenfund_agents.bench.runner import verdict_texts

    for threshold in (0.9, 0.95, 1.0):
        for n in range(1, 121):
            for k in range(n + 1):
                value = k / n
                texts = verdict_texts(_verdict("findings explained", value, threshold, value >= threshold))
                a, b = (float(t.removesuffix("%")) for t in texts)
                assert (a >= b) == (value >= threshold) and (a == b) == (value == threshold), (k, n, texts)
                assert (a == 100) == (k == n), (k, n, texts)


def test_report_and_cli_never_show_a_failing_value_as_its_threshold(tmp_path, monkeypatch, capsys, green_history):
    failing = _verdict("findings explained", 44 / 49, 0.9, False, "44/49")
    engine = _fake()
    n = _known_problem_night(green_history)
    scores, _ = run_bench([n], engine, A1Settings(), 1, load_thresholds(THRESHOLDS, "local"))
    assert "| findings explained | 89.8% (44/49) | 90.0% | NO |" in _md(tmp_path / "r", engine, scores, [failing])
    save(n, tmp_path / "nights")
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: engine)
    monkeypatch.setattr(cli, "run_bench", lambda *a: (scores, [failing]))
    assert cli.main(_run_args(tmp_path)) == 1
    assert "FAIL findings explained: 89.8% (44/49) (threshold 90.0%)" in capsys.readouterr().out
