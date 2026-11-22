"""Shared test helpers: a fixed night and compact row builders."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from myopenfund_agents.core.inputs import NightInputs, SourceRunRow

END = datetime(2026, 11, 22, 6, 0, tzinfo=UTC)
START = END - timedelta(days=1)


def src_run(
    code: str,
    *,
    night: int = 0,
    seen: int | None = 10,
    outcome: str | None = "ok",
    exit_code: int | None = 0,
    run_id: str | None = None,
    **kw: Any,
) -> SourceRunRow:
    """A source run row finished `night` days before tonight (0 = tonight), at END - night days - 5 h."""
    finished = END - timedelta(days=night, hours=5)
    values: dict[str, Any] = {
        "run_id": run_id if run_id is not None else f"run-{code}-n{night}",
        "tool": "harvester",
        "command": f"harvest {code}",
        "corpus": "central-bank",
        "source_code": code,
        "started_at": finished - timedelta(minutes=10),
        "finished_at": finished,
        "outcome": outcome,
        "exit_code": exit_code,
        "docs_seen": seen,
        "docs_new": 0,
        "docs_failed": 0,
        "fetch_errors": 0,
        "truncated": False,
    }
    values.update(kw)
    return SourceRunRow(**values)


def night(source_runs=(), history=(), runs=(), **kw: Any) -> NightInputs:
    values: dict[str, Any] = {
        "night_id": END.date().isoformat(),
        "night_start": START,
        "night_end": END,
        "runs": list(runs),
        "source_runs": list(source_runs),
        "history": list(history),
        "discovery_errors": [],
        "cadence": None,
        "open_issues": [],
    }
    values.update(kw)
    return NightInputs(**values)


@pytest.fixture
def green_history() -> list[SourceRunRow]:
    """Nights 1-14: us at 150 docs, fr and se at 10, all ok."""
    rows = []
    for n in range(1, 15):
        rows.append(src_run("us", night=n, seen=150))
        rows.append(src_run("fr", night=n, seen=10))
        rows.append(src_run("se", night=n, seen=10))
    return rows
