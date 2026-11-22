"""Night capture from the MCP: fake `call`, no network."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent

from conftest import END, START
from myopenfund_agents.bench.capture import (
    PAGE,
    CaptureError,
    _unwrap_result,
    capture_night,
    source_codes,
)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _sr(code: str, finished: datetime | None, run_id: str | None = None, **kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_id or f"run-{code}-{finished.isoformat() if finished else 'open'}",
        "tool": "harvester",
        "command": f"harvest {code}",
        "corpus": "central-bank",
        "source_code": code,
        "finished_at": _iso(finished) if finished else None,
        "outcome": "ok",
        "docs_seen": 10,
        "error_samples": None,
    }
    row.update(kw)
    return row


def _run(run_id: str, finished: datetime | None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "tool": "harvester",
        "command": "harvest",
        "corpus": "central-bank",
        "finished_at": _iso(finished) if finished else None,
        "outcome": "ok",
    }


def _resp(source_runs=(), runs=(), errors=(), health=()) -> dict[str, Any]:
    return {
        "source_runs": list(source_runs),
        "runs": list(runs),
        "open_discovery_errors": list(errors),
        "source_health": list(health),
    }


class FakeCall:
    """Dict-driven fake MCP: responses keyed by (tool, source_code); records every call."""

    def __init__(self, responses: dict[tuple[str, str | None], Any]):
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, name: str, args: dict) -> Any:
        self.calls.append((name, dict(args)))
        return self.responses[(name, args.get("source_code"))]


TONIGHT = END - timedelta(hours=5)
LATE = END + timedelta(hours=1)
OLD = START - timedelta(days=3)


def _capture(call, **kw):
    kw.setdefault("now", END + timedelta(days=3))
    return capture_night(call, night_end=END, corpus="central-bank", window_nights=14, **kw)


def test_split_tonight_history_and_drop_after_night():
    call = FakeCall(
        {
            ("vault_runs_status", "aa"): _resp(
                source_runs=[
                    _sr("aa", TONIGHT),
                    _sr("aa", LATE),
                    _sr("aa", None),
                    _sr("aa", OLD),
                    _sr("aa", START),  # boundary: night_start belongs to history
                ],
                runs=[_run("r-tonight", TONIGHT), _run("r-late", LATE), _run("r-open", None), _run("r-old", OLD)],
            )
        }
    )
    inputs = _capture(call, sources=["aa"])

    assert call.calls == [
        (
            "vault_runs_status",
            {
                "corpus": "central-bank",
                "source_code": "aa",
                "since": (START - timedelta(days=14)).isoformat(),
                "limit": PAGE,
            },
        )
    ]
    assert [r.finished_at for r in inputs.source_runs] == [TONIGHT]
    assert sorted(r.finished_at for r in inputs.history) == [OLD, START]
    assert [r.run_id for r in inputs.runs] == ["r-tonight"]
    assert inputs.source_runs[0].error_samples == []
    assert inputs.night_id == "2026-11-22"
    assert inputs.night_start == START and inputs.night_end == END
    assert inputs.cadence is None
    assert inputs.open_issues == []


def test_fresh_night_keeps_cadence():
    health = [{"corpus": "central-bank", "source_code": "aa", "doc_type": "report", "days_late": 2.5,
               "cadence_status": "overdue"}]  # fmt: skip
    call = FakeCall({("vault_runs_status", "aa"): _resp(source_runs=[_sr("aa", TONIGHT)], health=health)})
    inputs = _capture(call, sources=["aa"], now=END + timedelta(hours=2))
    assert inputs.cadence is not None
    assert [(c.source_code, c.days_late, c.cadence_status) for c in inputs.cadence] == [("aa", 2.5, "overdue")]

    stale = _capture(call, sources=["aa"], now=END + timedelta(hours=13))
    assert stale.cadence is None


def test_incomplete_history_refused():
    rows = [_sr("aa", END - timedelta(minutes=10 * i), run_id=f"r{i}") for i in range(PAGE)]
    call = FakeCall({("vault_runs_status", "aa"): _resp(source_runs=rows)})
    with pytest.raises(CaptureError, match="history incomplete") as exc:
        _capture(call, sources=["aa"])
    assert "aa" in str(exc.value)


def test_infra_refused():
    row = _sr("aa", TONIGHT, error_samples=["timeout to 100.64.12.34:5432"])
    call = FakeCall({("vault_runs_status", "aa"): _resp(source_runs=[row])})
    with pytest.raises(CaptureError, match="infra details found") as exc:
        _capture(call, sources=["aa"])
    assert "private_ip" in str(exc.value)
    assert "100.64" not in str(exc.value)


def test_runs_deduplicated_across_sources():
    shared = _run("r-shared", TONIGHT)
    call = FakeCall(
        {
            ("vault_runs_status", "aa"): _resp(source_runs=[_sr("aa", TONIGHT)], runs=[shared]),
            ("vault_runs_status", "bb"): _resp(source_runs=[_sr("bb", TONIGHT)], runs=[shared, _run("r-b", TONIGHT)]),
        }
    )
    inputs = _capture(call, sources=["aa", "bb"])
    assert [r.run_id for r in inputs.runs] == ["r-shared", "r-b"]
    assert sorted(r.source_code for r in inputs.source_runs) == ["aa", "bb"]


def test_discovery_errors_after_night_dropped_and_deduplicated():
    errors = [
        {"fingerprint": "f-before", "source_code": "aa", "first_seen_at": _iso(START)},
        {"fingerprint": "f-after", "source_code": "aa", "first_seen_at": _iso(LATE)},
        {"fingerprint": "f-unknown", "source_code": "aa"},
    ]
    call = FakeCall(
        {
            ("vault_runs_status", "aa"): _resp(source_runs=[_sr("aa", TONIGHT)], errors=errors),
            ("vault_runs_status", "bb"): _resp(source_runs=[_sr("bb", TONIGHT)], errors=errors[:1]),
        }
    )
    inputs = _capture(call, sources=["aa", "bb"])
    assert [e.fingerprint for e in inputs.discovery_errors] == ["f-before", "f-unknown"]


def test_sources_listed_from_corpus_stats_when_not_given():
    call = FakeCall(
        {
            ("vault_corpus_stats", None): {"groups": [{"key": "bb", "count": 3}, {"key": "aa", "count": 1}]},
            ("vault_runs_status", "aa"): _resp(source_runs=[_sr("aa", TONIGHT)]),
            ("vault_runs_status", "bb"): _resp(),
        }
    )
    inputs = _capture(call)
    assert call.calls[0] == ("vault_corpus_stats", {"corpus": "central-bank", "group_by": "source_code"})
    assert [a["source_code"] for _, a in call.calls[1:]] == ["aa", "bb"]
    assert len(inputs.source_runs) == 1


def test_source_codes_key_and_value_shapes():
    by_key = FakeCall({("vault_corpus_stats", None): {"groups": [{"key": "bb"}, {"key": "aa"}, {"key": "aa"}]}})
    assert source_codes(by_key, "central-bank") == ["aa", "bb"]
    by_value = FakeCall({("vault_corpus_stats", None): [{"value": "cc"}, {"source_code": "dd"}, {"value": ""}]})
    assert source_codes(by_value, "central-bank") == ["cc", "dd"]
    empty = FakeCall({("vault_corpus_stats", None): {"groups": []}})
    with pytest.raises(CaptureError):
        source_codes(empty, "central-bank")


def test_unwrap_structured_result_key():
    res = CallToolResult(content=[], structured_content={"result": {"a": 1}})
    assert _unwrap_result(res) == {"a": 1}


def test_unwrap_structured_plain_dict():
    res = CallToolResult(content=[], structured_content={"a": 1, "result": 2})
    assert _unwrap_result(res) == {"a": 1, "result": 2}


def test_unwrap_text_json():
    res = CallToolResult(content=[TextContent(type="text", text='{"groups": []}')])
    assert _unwrap_result(res) == {"groups": []}


def test_unwrap_error_raises_short_message():
    res = CallToolResult(content=[TextContent(type="text", text="refused: " + "x" * 1000)], is_error=True)
    with pytest.raises(CaptureError, match="refused") as exc:
        _unwrap_result(res)
    assert len(str(exc.value)) < 400


def test_naive_night_end_treated_as_utc():
    call = FakeCall({("vault_runs_status", "aa"): _resp(source_runs=[_sr("aa", TONIGHT)])})
    inputs = capture_night(
        call, night_end=END.replace(tzinfo=None), corpus="central-bank", window_nights=14, sources=["aa"],
        now=datetime(2026, 11, 30, tzinfo=UTC),
    )  # fmt: skip
    assert inputs.night_end == END
    assert len(inputs.source_runs) == 1
