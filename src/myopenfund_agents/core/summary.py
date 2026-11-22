"""Deterministic night summary: the only view of a night an LLM ever sees.

Every input-derived string placed into the text goes through `_clean`, which replaces `<` by `‹`
so no value can close the `<summary>` delimiter (Review Focus 2).
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from myopenfund_agents.core.inputs import CadenceRow, NightInputs, RunRow, SourceRunRow
from myopenfund_agents.settings import A1Settings

FLAG_ORDER = (
    "failed_run",
    "truncated",
    "fetch_errors",
    "bad_content",
    "docs_failed",
    "volume_drop",
    "cadence_overdue",
    "open_discovery_errors",
)
BAD_OUTCOMES = frozenset({"degraded", "failed"})
NO_DETAIL_SUFFIX = " (no per-source detail: failed before or outside sources)"
NO_RUN_LINE = "No run finished during this night."
CADENCE_UNKNOWN_LINE = "Cadence: not known for this night"
ONE_DAY = timedelta(days=1)


@dataclass
class SourceState:
    source_code: str
    corpus: str | None
    docs_seen: int
    docs_new: int
    docs_failed: int
    fetch_errors: int
    bad_content: int
    truncated: bool
    failed_commands: list[str] = field(default_factory=list)
    error_samples: list[str] = field(default_factory=list)
    median_docs_seen: float | None = None
    history_nights: int = 0
    consecutive_failed_nights: int = 0
    days_late: int | None = None
    open_discovery_errors: int = 0
    flags: list[str] = field(default_factory=list)


@dataclass
class NightSummary:
    text: str
    unusual: dict[str, SourceState]
    normal_count: int
    failing_runs: list[str]


def is_bad(row: RunRow | SourceRunRow) -> bool:
    return row.outcome in BAD_OUTCOMES or row.exit_code not in (0, None)


def _clean(value: Any) -> str:
    """One-line, delimiter-safe rendering of an input-derived value."""
    text = value if isinstance(value, str) else str(value)
    return " ".join(text.replace("<", "‹").split())


def _cut(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(max_chars - 1, 0)] + "…"


def _count(value: Any) -> int:
    """`bad_content` may be an int count, a list of items, or null."""
    if value is None or isinstance(value, bool):
        return int(bool(value))
    if isinstance(value, int):
        return value
    if isinstance(value, list | tuple | set | dict):
        return len(value)
    return 0


def _history_index(night_start: datetime, finished_at: datetime) -> int:
    return int((night_start - finished_at) / ONE_DAY) + 1


def _days_late(rows: list[CadenceRow]) -> int | None:
    values = [r.days_late for r in rows if r.days_late is not None]
    return int(max(values)) if values else None


def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _source_state(
    code: str,
    tonight: list[SourceRunRow],
    history: dict[int, list[SourceRunRow]],
    cadence_rows: list[CadenceRow] | None,
    discovery_count: int,
    corpus_hint: str | None,
    settings: A1Settings,
) -> SourceState:
    samples: list[str] = []
    for row in tonight:
        for sample in row.error_samples:
            if len(samples) < settings.error_samples_per_source:
                samples.append(_cut(_clean(sample), settings.error_sample_max_chars))
    corpus = next((r.corpus for r in tonight if r.corpus), corpus_hint)
    state = SourceState(
        source_code=code,
        corpus=corpus,
        docs_seen=sum(r.docs_seen or 0 for r in tonight),
        docs_new=sum(r.docs_new or 0 for r in tonight),
        docs_failed=sum(r.docs_failed or 0 for r in tonight),
        fetch_errors=sum(r.fetch_errors or 0 for r in tonight),
        bad_content=sum(_count(r.bad_content) for r in tonight),
        truncated=any(bool(r.truncated) for r in tonight),
        failed_commands=[r.command for r in tonight if is_bad(r)],
        error_samples=samples,
        history_nights=len(history),
        days_late=None if cadence_rows is None else _days_late(cadence_rows),
        open_discovery_errors=discovery_count,
    )
    if history:
        totals = [sum(r.docs_seen or 0 for r in rows) for rows in history.values()]
        state.median_docs_seen = float(statistics.median(totals))
    k = 1
    while k in history and any(is_bad(r) for r in history[k]):
        state.consecutive_failed_nights += 1
        k += 1

    checks = {
        "failed_run": bool(state.failed_commands),
        "truncated": state.truncated,
        "fetch_errors": state.fetch_errors > 0,
        "bad_content": state.bad_content > 0,
        "docs_failed": state.docs_failed > 0,
        # Volume is judged only for a source that ran tonight; a silent source is cadence's business.
        "volume_drop": bool(tonight)
        and state.history_nights >= settings.min_history_nights
        and state.median_docs_seen is not None
        and state.median_docs_seen > 0
        and state.docs_seen < settings.volume_drop_ratio * state.median_docs_seen,
        "cadence_overdue": state.days_late is not None and state.days_late > settings.cadence_grace_days,
        "open_discovery_errors": state.open_discovery_errors > 0,
    }
    state.flags = [flag for flag in FLAG_ORDER if checks[flag]]
    return state


def _block(state: SourceState) -> str:
    lines = [
        f"[{_clean(state.source_code)}] corpus={_clean(state.corpus or '-')} flags={','.join(state.flags)}",
        f"  tonight: seen={state.docs_seen} new={state.docs_new} failed={state.docs_failed} "
        f"fetch_errors={state.fetch_errors} bad_content={state.bad_content} "
        f"truncated={'yes' if state.truncated else 'no'}",
        f"  history: median_seen={_fmt(state.median_docs_seen)} over {state.history_nights} nights; "
        f"consecutive_failed_nights={state.consecutive_failed_nights}",
        f"  days_late={_fmt(state.days_late)} open_discovery_errors={state.open_discovery_errors}",
    ]
    if state.failed_commands:
        lines.append("  failed_commands: " + "; ".join(_clean(c) for c in state.failed_commands))
    lines += [f"  sample: {s}" for s in state.error_samples]
    return "\n".join(lines) + "\n"


def _failing_label(run: RunRow, has_detail: bool) -> str:
    return _clean(run.run_id) if has_detail else _clean(run.run_id) + NO_DETAIL_SUFFIX


def _run_line(run: RunRow, has_detail: bool) -> str:
    return (
        f"{_failing_label(run, has_detail)}: {_clean(run.tool)} {_clean(run.command)} "
        f"outcome={_clean(run.outcome or '-')} exit={_fmt(run.exit_code)}"
    )


def build_summary(inputs: NightInputs, settings: A1Settings) -> NightSummary:
    # Unfinished rows (in flight or crashed without a finish) never enter the night or the history.
    runs = [r for r in inputs.runs if r.finished_at is not None]
    tonight_rows = [r for r in inputs.source_runs if r.finished_at is not None]
    tonight: dict[str, list[SourceRunRow]] = defaultdict(list)
    for row in tonight_rows:
        tonight[row.source_code].append(row)

    history: dict[str, dict[int, list[SourceRunRow]]] = defaultdict(lambda: defaultdict(list))
    for row in inputs.history:
        if row.finished_at is None:
            continue
        index = _history_index(inputs.night_start, row.finished_at)
        if 1 <= index <= settings.volume_window_nights:
            history[row.source_code][index].append(row)

    cadence_by_code: dict[str, list[CadenceRow]] = defaultdict(list)
    for crow in inputs.cadence or []:
        cadence_by_code[crow.source_code].append(crow)
    discovery_by_code: dict[str, int] = defaultdict(int)
    corpus_hints: dict[str, str] = {}
    for err in inputs.discovery_errors:
        if err.source_code is not None:
            discovery_by_code[err.source_code] += 1
            if err.corpus:
                corpus_hints.setdefault(err.source_code, err.corpus)
    for code, crows in cadence_by_code.items():
        corpus = next((c.corpus for c in crows if c.corpus), None)
        if corpus:
            corpus_hints[code] = corpus

    overdue = {
        code
        for code, crows in cadence_by_code.items()
        if (late := _days_late(crows)) is not None and late > settings.cadence_grace_days
    }
    codes = sorted(set(tonight) | overdue | set(discovery_by_code))
    states = [
        _source_state(
            code,
            tonight.get(code, []),
            history.get(code, {}),
            None if inputs.cadence is None else cadence_by_code.get(code, []),
            discovery_by_code.get(code, 0),
            corpus_hints.get(code),
            settings,
        )
        for code in codes
    ]
    unusual_states = sorted((s for s in states if s.flags), key=lambda s: (-len(s.flags), s.source_code))
    unusual = {s.source_code: s for s in unusual_states}
    normal_count = len(states) - len(unusual)

    run_ids_with_detail = {r.run_id for r in tonight_rows}
    failing = [r for r in runs if is_bad(r)]
    failing_runs = [_failing_label(r, r.run_id in run_ids_with_detail) for r in failing]

    header = [
        f"Night {_clean(inputs.night_id)}: {inputs.night_start.isoformat()} to {inputs.night_end.isoformat()}",
    ]
    if not runs and not tonight_rows:
        header.append(NO_RUN_LINE)
    else:
        header.append(f"Runs: {len(runs)} finished, {len(failing)} failing")
    unfinished = (len(inputs.runs) - len(runs)) + (len(inputs.source_runs) - len(tonight_rows))
    if unfinished:
        header.append(f"Unfinished rows ignored: {unfinished}")
    if failing:
        header.append("Failing runs:")
        header += [f"- {_run_line(r, r.run_id in run_ids_with_detail)}" for r in failing]
    header.append(f"Sources: {normal_count} normal, {len(unusual)} unusual")
    if inputs.cadence is None:
        header.append(CADENCE_UNKNOWN_LINE)
    unattached = sum(1 for e in inputs.discovery_errors if e.source_code is None)
    if unattached:
        header.append(f"Open discovery errors without a source: {unattached}")

    text = "\n".join(header) + "\n"
    omitted: list[str] = []
    for state in unusual_states:
        block = "\n" + _block(state)
        # Most important blocks first; once one overflows, it and every later block are omitted.
        if omitted or len(text) + len(block) > settings.summary_char_budget:
            omitted.append(_clean(state.source_code))
        else:
            text += block
    if omitted:
        text += f"\nOmitted for size ({len(omitted)} unusual sources): {', '.join(omitted)}\n"
    return NightSummary(text=text, unusual=unusual, normal_count=normal_count, failing_runs=failing_runs)
