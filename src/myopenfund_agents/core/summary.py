"""Deterministic night summary: the only view of a night an LLM ever sees.

Every input-derived string placed into the text goes through `_clean`, which replaces `<` by `‹`
so no value can close the `<summary>` delimiter (Review Focus 2).
"""

import math
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
DEGRADED_EXIT = 3  # exit code of a run that completed with problems (crawler contract)
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
    no_run: bool = False  # True exactly when the text carries NO_RUN_LINE


def is_bad(row: RunRow | SourceRunRow) -> bool:
    return row.outcome in BAD_OUTCOMES or row.exit_code not in (0, None)


def _has_own_signal(row: SourceRunRow) -> bool:
    return bool(row.truncated or row.fetch_errors or row.docs_failed or _count(row.bad_content) or row.error_samples)


def _is_degraded(row: SourceRunRow) -> bool:
    """The run completed with problems (exit 3): only then does a silent row mean its source went fine."""
    return row.outcome == "degraded" and row.exit_code in (DEGRADED_EXIT, None)


def _implicated(rows: list[SourceRunRow]) -> list[SourceRunRow]:
    """Bad rows that a source answers for.

    A run over several sources stamps its one outcome on every source row. When some rows of a degraded run
    carry a signal of their own, they explain it and the silent rows are cleared. A crashed run clears nobody,
    nor does a run no row explains: an unexplained failure is reported, never dropped.
    """
    explained = {r.run_id for r in rows if _is_degraded(r) and _has_own_signal(r)}
    return [r for r in rows if is_bad(r) and not (r.run_id in explained and _is_degraded(r) and not _has_own_signal(r))]


def _cleared_lines(rows: list[SourceRunRow], implicated: set[int]) -> list[str]:
    cleared: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        if is_bad(r) and id(r) not in implicated:
            cleared[r.run_id, r.command].add(r.source_code)
    return [
        f"Degraded run {_clean(run_id)} ({_clean(command)}): no signal of their own, not flagged for it: "
        + ", ".join(_clean(c) for c in sorted(codes))
        for (run_id, command), codes in cleared.items()
    ]


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


def _long_overdue(row: CadenceRow, settings: A1Settings) -> bool:
    limit = settings.cadence_new_max_days
    return bool(limit) and row.days_late is not None and math.isfinite(row.days_late) and row.days_late > limit


def _days_late(rows: list[CadenceRow], settings: A1Settings) -> int | None:
    # Non-finite values (nan, inf) carry no usable lateness and are ignored; long overdue series are only counted.
    values = [
        r.days_late
        for r in rows
        if r.days_late is not None and math.isfinite(r.days_late) and not _long_overdue(r, settings)
    ]
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
    implicated: set[int],
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
        failed_commands=[r.command for r in tonight if id(r) in implicated],
        error_samples=samples,
        history_nights=len(history),
        days_late=None if cadence_rows is None else _days_late(cadence_rows, settings),
        open_discovery_errors=discovery_count,
    )
    if history:
        totals = [sum(r.docs_seen or 0 for r in rows) for rows in history.values()]
        state.median_docs_seen = float(statistics.median(totals))
    k = 1
    while k in history and any(id(r) in implicated for r in history[k]):
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


def _block(state: SourceState, max_chars: int) -> str:
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
        lines.append("  failed_commands: " + "; ".join(_cut(_clean(c), max_chars) for c in state.failed_commands))
    lines += [f"  sample: {s}" for s in state.error_samples]
    return "\n".join(lines) + "\n"


def _failing_label(run: RunRow, has_detail: bool) -> str:
    return _clean(run.run_id) if has_detail else _clean(run.run_id) + NO_DETAIL_SUFFIX


def _run_line(run: RunRow, has_detail: bool, max_chars: int) -> str:
    return (
        f"{_failing_label(run, has_detail)}: {_cut(_clean(run.tool), max_chars)} {_cut(_clean(run.command), max_chars)} "
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
        # Unfinished rows, and rows finished at or after night_start (tonight's own rows), are not history.
        if row.finished_at is None or row.finished_at >= inputs.night_start:
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
        if (late := _days_late(crows, settings)) is not None and late > settings.cadence_grace_days
    }
    codes = sorted(set(tonight) | overdue | set(discovery_by_code))
    history_rows = [r for by_night in history.values() for rows in by_night.values() for r in rows]
    implicated = {id(r) for r in [*_implicated(tonight_rows), *_implicated(history_rows)]}
    states = [
        _source_state(
            code,
            tonight.get(code, []),
            history.get(code, {}),
            None if inputs.cadence is None else cadence_by_code.get(code, []),
            discovery_by_code.get(code, 0),
            corpus_hints.get(code),
            implicated,
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
    no_run = not runs and not tonight_rows
    if no_run:
        header.append(NO_RUN_LINE)
    elif runs:
        header.append(f"Runs: {len(runs)} finished, {len(failing)} failing")
    else:
        # Source rows but no run row: a `Runs: 0 finished` line would read like a night where nothing ran.
        header.append(f"Source runs: {len(tonight_rows)} finished")
    unfinished = (len(inputs.runs) - len(runs)) + (len(inputs.source_runs) - len(tonight_rows))
    if unfinished:
        header.append(f"Unfinished rows ignored: {unfinished}")
    tail = [f"Sources: {normal_count} normal, {len(unusual)} unusual"]
    if inputs.cadence is None:
        tail.append(CADENCE_UNKNOWN_LINE)
    tail += _cleared_lines(tonight_rows, implicated)
    long_overdue = [c for c in inputs.cadence or [] if _long_overdue(c, settings)]
    if long_overdue:
        tail.append(
            f"Long overdue series (more than {settings.cadence_new_max_days} days late, not findings): "
            f"{len(long_overdue)} ({', '.join(_clean(c) for c in sorted({c.source_code for c in long_overdue}))})"
        )
    unattached = sum(1 for e in inputs.discovery_errors if e.source_code is None)
    if unattached:
        tail.append(f"Open discovery errors without a source: {unattached}")
    if failing:
        header.append("Failing runs:")
        # Failing runs are listed while the header stays within budget, keeping room for the tail lines.
        more_room = len(f"... and {len(failing)} more failing runs\n")
        used = sum(len(line) + 1 for line in header + tail) + more_room
        for listed, run in enumerate(failing):
            line = f"- {_run_line(run, run.run_id in run_ids_with_detail, settings.error_sample_max_chars)}"
            if used + len(line) + 1 > settings.summary_char_budget:
                header.append(f"... and {len(failing) - listed} more failing runs")
                break
            header.append(line)
            used += len(line) + 1
    header += tail

    text = "\n".join(header) + "\n"
    omitted: list[str] = []
    for state in unusual_states:
        block = "\n" + _block(state, settings.error_sample_max_chars)
        # Most important blocks first; once one overflows, it and every later block are omitted.
        if omitted or len(text) + len(block) > settings.summary_char_budget:
            omitted.append(_clean(state.source_code))
        else:
            text += block
    if omitted:
        text += f"\nOmitted for size ({len(omitted)} unusual sources): {', '.join(omitted)}\n"
    return NightSummary(text=text, unusual=unusual, normal_count=normal_count, failing_runs=failing_runs, no_run=no_run)
