import dataclasses
import json
from pathlib import Path

import pytest

from conftest import night, src_run
from myopenfund_agents import cli
from myopenfund_agents.bench import traps
from myopenfund_agents.bench.fixtures import save
from myopenfund_agents.bench.runner import (
    DEFAULT_CORPUS_REPOS,
    DEFAULT_REPO,
    repo_of,
    run_bench,
    run_night,
    write_report,
)
from myopenfund_agents.bench.score import load_thresholds
from myopenfund_agents.core.findings import Category, Finding, Findings, Severity
from myopenfund_agents.engines.base import EngineError
from myopenfund_agents.engines.fake import FakeEngine
from myopenfund_agents.settings import A1Settings

THRESHOLDS = Path(__file__).resolve().parents[1] / "bench" / "thresholds.toml"


def _rec(profile: str, repeat: int = 1, settings: A1Settings | None = None) -> dict:
    """How a report was produced: repeat count, A1 settings and threshold profile values."""
    return {"repeat": repeat, "settings": settings or A1Settings(), "thresholds": load_thresholds(THRESHOLDS, profile)}


def _known_problem_night(green_history):
    base = night(source_runs=[src_run("us", seen=150), src_run("fr"), src_run("se")], history=green_history)
    return traps.known_problem(base, "se", DEFAULT_CORPUS_REPOS["central-bank"])


def _se_unreadable() -> Findings:
    finding = Finding(
        source_code="se", category=Category.UNREADABLE_PDF, severity=Severity.WARN,
        title="se: unreadable PDFs", diagnosis="Text layer empty.", evidence=["bad_content=2"],
    )  # fmt: skip
    return Findings(findings=[finding])


class _BrokenEngine:
    name = "broken"
    model = "none"

    def diagnose(self, summary_text: str):
        raise EngineError("engine unreachable")


def test_fake_engine_scores_known_problem_as_comment(tmp_path, green_history):
    n = _known_problem_night(green_history)
    engine = FakeEngine(_se_unreadable())
    t = load_thresholds(THRESHOLDS, "local")
    scores, verdicts = run_bench([n], engine, A1Settings(), repeat=3, thresholds=t)
    assert [s.actions_ok for s in scores] == [True]
    assert scores[0].attempts[0].actions == [("comment", "se", "unreadable_pdf")]
    assert all(v.passed for v in verdicts), verdicts
    md = write_report(tmp_path, engine, "local", scores, verdicts, **_rec("local", repeat=3))
    assert md.suffix == ".md" and md.with_suffix(".json").is_file()
    text = md.read_text(encoding="utf-8")
    assert "| problems found | 100% (1/1)" in text
    assert "| Night | Origin | Found | Extra | False alarms | Actions | Valid | Agreement | Max s |" in text
    assert "false alarms on green nights" in text
    payload = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["engine"] == "fake" and payload["model"] == "scripted" and payload["profile"] == "local"
    assert payload["nights"][0]["night_id"] == n.inputs.night_id


def test_report_name_replaces_model_separators(tmp_path, green_history):
    n = _known_problem_night(green_history)
    engine = FakeEngine(_se_unreadable(), model="ollama/qwen3:8b")
    scores, verdicts = run_bench([n], engine, A1Settings(), 1, load_thresholds(THRESHOLDS, "prod"))
    md = write_report(tmp_path, engine, "prod", scores, verdicts, **_rec("prod"))
    assert md.name.endswith("-fake-ollama_qwen3_8b.md")


def test_report_refused_when_payload_has_infra_detail(tmp_path, green_history):
    n = _known_problem_night(green_history)
    engine = FakeEngine(_se_unreadable(), model="100.64.1.2")
    scores, verdicts = run_bench([n], engine, A1Settings(), 1, load_thresholds(THRESHOLDS, "local"))
    with pytest.raises(ValueError, match="report refused") as err:
        write_report(tmp_path / "out", engine, "local", scores, verdicts, **_rec("local"))
    assert "100.64.1.2" not in str(err.value)
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_engine_error_attempt_counted_invalid(green_history):
    n = _known_problem_night(green_history)
    score = run_night(n, _BrokenEngine(), A1Settings(), 2, DEFAULT_CORPUS_REPOS)
    assert score.valid is False
    assert [a.error for a in score.attempts] == ["engine unreachable", "engine unreachable"]
    assert all(a.keys is None and a.actions is None for a in score.attempts)
    assert score.found == [] and score.actions_ok is None


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
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: FakeEngine(_se_unreadable()))
    args = ["bench-run", "--profile", "local", "--nights", str(tmp_path / "nights"),
            "--thresholds", str(THRESHOLDS), "--results", str(tmp_path / "results")]  # fmt: skip
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert str(tmp_path / "results") in out and "problems found" in out
    assert cli.main([*args, "--only", "no-such-night"]) == 2
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: FakeEngine(Findings(findings=[])))
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

    def diagnose(self, summary_text: str):
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
    assert "FAIL valid output" in out and "PASS false alarms on green nights" in out
    (report,) = (tmp_path / "results").glob("*.json")
    text = report.read_text(encoding="utf-8")
    assert "withheld (private_ip)" in text and "100.64.1.2" not in text


def test_cli_run_bad_denylist_exits_2(tmp_path, monkeypatch, capsys, green_history):
    save(_known_problem_night(green_history), tmp_path / "nights")
    monkeypatch.setenv("MOF_AGENTS_DENYLIST", str(tmp_path / "missing.toml"))
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: FakeEngine(_se_unreadable()))
    assert cli.main(_run_args(tmp_path)) == 2
    assert "MOF_AGENTS_DENYLIST" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_write_report_uses_given_deny_list(tmp_path, green_history):
    from myopenfund_agents.sanitise import DenyList

    engine = FakeEngine(_se_unreadable(), model="mirror.example.org")
    scores, verdicts = run_bench([_known_problem_night(green_history)], engine, A1Settings(), 1,
                                 load_thresholds(THRESHOLDS, "local"))  # fmt: skip
    with pytest.raises(ValueError, match="report refused"):
        write_report(
            tmp_path, engine, "local", scores, verdicts, deny=DenyList(hosts=["mirror.example.org"]), **_rec("local")
        )


def test_report_wraps_keys_in_backticks(tmp_path, green_history):
    engine = FakeEngine(_se_unreadable())
    scores, verdicts = run_bench([_known_problem_night(green_history)], engine, A1Settings(), 1,
                                 load_thresholds(THRESHOLDS, "local"))  # fmt: skip
    text = write_report(tmp_path, engine, "local", scores, verdicts, **_rec("local")).read_text(encoding="utf-8")
    assert "| `se/unreadable_pdf` |" in text


def test_criterion_constants_exported():
    from myopenfund_agents.bench import score

    assert score.PROBLEMS_FOUND == "problems found"
    assert score.SLOWEST_NIGHT == "slowest night (s)"


def test_cli_run_corrupt_night_exits_2(tmp_path, monkeypatch, capsys, green_history):
    d = save(_known_problem_night(green_history), tmp_path / "nights")
    (d / "expected.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: FakeEngine(_se_unreadable()))
    assert cli.main(_run_args(tmp_path)) == 2
    assert str(d) in capsys.readouterr().err


def test_cli_run_unknown_only_id_exits_2(tmp_path, monkeypatch, capsys, green_history):
    n = _known_problem_night(green_history)
    save(n, tmp_path / "nights")
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: FakeEngine(_se_unreadable()))
    assert cli.main([*_run_args(tmp_path), "--only", f"{n.inputs.night_id},ghost-night"]) == 2
    assert "ghost-night" in capsys.readouterr().err


@pytest.mark.parametrize("content", ["{oops", "[1, 2]", '{"central-bank": 3}'])
def test_cli_run_bad_corpus_repos_exits_2(tmp_path, monkeypatch, green_history, content):
    save(_known_problem_night(green_history), tmp_path / "nights")
    f = tmp_path / "repos.json"
    f.write_text(content, encoding="utf-8")
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: FakeEngine(_se_unreadable()))
    assert cli.main([*_run_args(tmp_path), "--corpus-repos", str(f)]) == 2


def test_cli_run_corpus_repos_mapping_applies(tmp_path, monkeypatch, green_history):
    # the open issue is on central-bank-corpus; mapping the corpus elsewhere turns the comment into a new issue
    save(_known_problem_night(green_history), tmp_path / "nights")
    f = tmp_path / "repos.json"
    f.write_text('{"central-bank": "other-repo"}', encoding="utf-8")
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: FakeEngine(_se_unreadable()))
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
    """Returns fixed findings with token counts, like a real adapter."""

    name = "tokens"
    model = "scripted"

    def __init__(self, findings: Findings) -> None:
        self.findings = findings

    def diagnose(self, summary_text: str):
        from myopenfund_agents.engines.base import EngineResult

        return EngineResult(self.findings, self.name, self.model, 0.5, 1, input_tokens=1200, output_tokens=80)


def test_attempt_keeps_findings_and_tokens(green_history):
    score = run_night(_known_problem_night(green_history), _TokenEngine(_se_unreadable()), A1Settings(), 1, None)
    (attempt,) = score.attempts
    assert attempt.input_tokens == 1200 and attempt.output_tokens == 80
    assert attempt.findings == [_se_unreadable().model_dump(mode="json")["findings"][0]]
    assert attempt.findings[0]["diagnosis"] == "Text layer empty."


def test_failed_attempt_has_no_findings_or_tokens(green_history):
    score = run_night(_known_problem_night(green_history), _BrokenEngine(), A1Settings(), 1, None)
    (attempt,) = score.attempts
    assert attempt.findings is None and attempt.input_tokens is None and attempt.output_tokens is None


def test_report_json_carries_findings_and_tokens(tmp_path, green_history):
    engine = _TokenEngine(_se_unreadable())
    scores, verdicts = run_bench([_known_problem_night(green_history)], engine, A1Settings(), 1,
                                 load_thresholds(THRESHOLDS, "local"))  # fmt: skip
    md = write_report(tmp_path, engine, "local", scores, verdicts, **_rec("local"))
    attempt = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))["nights"][0]["attempts"][0]
    assert attempt["input_tokens"] == 1200 and attempt["output_tokens"] == 80
    assert attempt["findings"][0]["title"] == "se: unreadable PDFs"
    assert attempt["findings"][0]["evidence"] == ["bad_content=2"]


def test_report_withholds_finding_texts_with_infra_detail(tmp_path, green_history):
    from myopenfund_agents.sanitise import DenyList

    finding = Finding(
        source_code="se", category=Category.UNREADABLE_PDF, severity=Severity.WARN,
        title="se: mirror.example.org unreachable", diagnosis="Fetch from 100.64.1.2 failed. password=abcdef123456",
        evidence=["bad_content=2", "GET http://100.64.1.2/x"],
    )  # fmt: skip
    engine = _TokenEngine(Findings(findings=[finding]))
    scores, verdicts = run_bench([_known_problem_night(green_history)], engine, A1Settings(), 1,
                                 load_thresholds(THRESHOLDS, "local"))  # fmt: skip
    deny = DenyList(hosts=["mirror.example.org"])
    md = write_report(tmp_path, engine, "local", scores, verdicts, deny=deny, **_rec("local"))
    text = md.with_suffix(".json").read_text(encoding="utf-8")
    assert "100.64.1.2" not in text and "mirror.example.org" not in text and "abcdef123456" not in text
    got = json.loads(text)["nights"][0]["attempts"][0]["findings"][0]
    assert got["title"] == "withheld (deny_host)"
    assert got["diagnosis"] == "withheld (credential, private_ip)"
    assert got["evidence"] == ["bad_content=2", "withheld (private_ip)"]
    assert got["source_code"] == "se"


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
    engine = FakeEngine(_se_unreadable())
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
    engine = FakeEngine(_se_unreadable())
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
    finding = Finding(
        source_code="*", category=Category.UNKNOWN, severity=Severity.ERROR,
        title="No run finished", diagnosis="No run finished tonight.", evidence=["No run finished during this night."],
    )  # fmt: skip
    score = run_night(n, FakeEngine(Findings(findings=[finding])), A1Settings(), 1, None)
    assert score.attempts[0].actions == [("new_issue", "*", "unknown")]
    assert score.actions_ok


def test_report_names_the_models_that_answered(tmp_path):
    from conftest import night, src_run
    from myopenfund_agents.bench.fixtures import Expected, Meta, Night
    from myopenfund_agents.bench.runner import run_night, write_report
    from myopenfund_agents.engines.base import EngineResult
    from myopenfund_agents.sanitise import DenyList

    class Resolving:
        name, model = "claude-code", "default"

        def diagnose(self, text):
            return EngineResult(Findings(findings=[]), self.name, "real-model", 0.1)

    n = Night(inputs=night([src_run("se")]), expected=Expected(), meta=Meta(origin="green", description="d"))
    score = run_night(n, Resolving(), A1Settings(), repeat=1, corpus_repos=None)
    md = write_report(
        tmp_path, Resolving(), "prod", [score], [], DenyList(), repeat=1, settings=A1Settings(), thresholds={}
    )
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert payload["models_used"] == ["real-model"]
    assert "Model that answered: real-model" in md.read_text()
