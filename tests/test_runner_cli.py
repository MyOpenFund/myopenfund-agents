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
    md = write_report(tmp_path, engine, "local", scores, verdicts)
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
    md = write_report(tmp_path, engine, "prod", scores, verdicts)
    assert md.name.endswith("-fake-ollama_qwen3_8b.md")


def test_report_refused_when_payload_has_infra_detail(tmp_path, green_history):
    n = _known_problem_night(green_history)
    engine = FakeEngine(_se_unreadable(), model="100.64.1.2")
    scores, verdicts = run_bench([n], engine, A1Settings(), 1, load_thresholds(THRESHOLDS, "local"))
    with pytest.raises(ValueError, match="report refused") as err:
        write_report(tmp_path / "out", engine, "local", scores, verdicts)
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
    monkeypatch.setattr(cli, "engine_from_env", lambda settings: FakeEngine(Findings()))
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
            "--sources", "se,fr", "--nights", str(tmp_path)]  # fmt: skip
    assert cli.main(args) == 0
    assert seen["night_end"].isoformat() == "2026-11-22T06:00:00+00:00"
    assert seen["sources"] == ["se", "fr"] and seen["window_nights"] == 14 and seen["corpus"] == "central-bank"
    meta = json.loads((tmp_path / "2026-11-22" / "meta.json").read_text())
    assert meta["origin"] == "green" and meta["validated_by"] is None
    assert "expected answers pending validation" in meta["notes"]
    assert cli.main(args) == 1
