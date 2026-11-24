"""Deterministic night summary: the only view of a night an LLM ever sees.

Every input-derived string placed into the text goes through `clean`, which replaces `<` by `‹`
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
# Line prefixes of a source block; the detector finds the lines of its evidence by them.
TONIGHT_PREFIX = "  tonight:"
HISTORY_PREFIX = "  history:"
DAYS_LATE_PREFIX = "  days_late="
FAILED_COMMANDS_PREFIX = "  failed_commands:"
SAMPLE_PREFIX = "  sample:"


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
    blocks: dict[str, str] = field(default_factory=dict)  # every unusual source's block, the omitted ones too
    # The rendered line of every failing run with no per-source detail, also those the budget cut from the text.
    unexplained_runs: list[str] = field(default_factory=list)


def is_bad(row: RunRow | SourceRunRow) -> bool:
    return row.outcome in BAD_OUTCOMES or row.exit_code not in (0, None)


def _has_own_signal(row: SourceRunRow) -> bool:
    return bool(row.truncated or row.fetch_errors or row.docs_failed or _count(row.bad_content) or row.error_samples)


def _raises_flag(row: SourceRunRow) -> bool:
    """The row alone flags its source; an error sample by itself does not."""
    counters = (row.fetch_errors or 0, row.docs_failed or 0, _count(row.bad_content))
    return bool(row.truncated) or any(n > 0 for n in counters)


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


def _cleared_lines(rows: list[SourceRunRow], implicated: set[int], max_chars: int, budget: int) -> list[str]:
    cleared: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        if is_bad(r) and id(r) not in implicated:
            cleared[r.run_id, r.command].add(r.source_code)
    lines: list[str] = []
    overflow = f"... and {len(cleared)} more degraded runs with sources not flagged"
    room = budget - len(overflow) - 1
    share = max(room // 2, 0)  # one huge run never takes the whole room from the next ones
    skipped = 0
    for (run_id, command), codes in cleared.items():
        head = (
            f"Degraded run {clean(run_id)} ({_cut(clean(command), max_chars)}): "
            "no signal of their own, not flagged for it: "
        )
        names = _names(sorted(codes), max(share - len(head), 0))
        line = head + names
        if len(line) + 1 > room or names.startswith("(+"):
            skipped += 1
            continue
        lines.append(line)
        room -= len(line) + 1
    if skipped:
        lines.append(f"... and {skipped} more degraded runs with sources not flagged")
    return lines


def clean(value: Any) -> str:
    """One-line, delimiter-safe rendering of an input-derived value."""
    text = value if isinstance(value, str) else str(value)
    return " ".join(text.replace("<", "‹").split())


def _cut(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(max_chars - 1, 0)] + "…"


def _names(codes: list[str], budget: int) -> str:
    """Whole names joined within ``budget`` characters; the rest is counted, never cut mid-name."""
    names = [clean(c) for c in codes]
    full = ", ".join(names)
    if len(full) <= budget:
        return full
    room = budget - len(f" (+{len(names)} more)")
    kept: list[str] = []
    used = 0
    for name in names:
        size = len(name) + (2 if kept else 0)
        if used + size <= room:
            kept.append(name)
            used += size
    return f"{', '.join(kept)} (+{len(names) - len(kept)} more)".lstrip()


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
                samples.append(_cut(clean(sample), settings.error_sample_max_chars))
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
        f"[{clean(state.source_code)}] corpus={clean(state.corpus or '-')} flags={','.join(state.flags)}",
        f"{TONIGHT_PREFIX} seen={state.docs_seen} new={state.docs_new} failed={state.docs_failed} "
        f"fetch_errors={state.fetch_errors} bad_content={state.bad_content} "
        f"truncated={'yes' if state.truncated else 'no'}",
        f"{HISTORY_PREFIX} median_seen={_fmt(state.median_docs_seen)} over {state.history_nights} nights; "
        f"consecutive_failed_nights={state.consecutive_failed_nights}",
        f"{DAYS_LATE_PREFIX}{_fmt(state.days_late)} open_discovery_errors={state.open_discovery_errors}",
    ]
    if state.failed_commands:
        commands = dict.fromkeys(_cut(clean(c), max_chars) for c in state.failed_commands)
        lines.append(f"{FAILED_COMMANDS_PREFIX} " + _names(list(commands), max_chars + 20))
    lines += [f"{SAMPLE_PREFIX} {s}" for s in state.error_samples]
    return "\n".join(lines) + "\n"


def _failing_label(run: RunRow, has_detail: bool) -> str:
    return clean(run.run_id) if has_detail else clean(run.run_id) + NO_DETAIL_SUFFIX


def _run_line(run: RunRow, has_detail: bool, max_chars: int) -> str:
    return (
        f"{_failing_label(run, has_detail)}: {_cut(clean(run.tool), max_chars)} {_cut(clean(run.command), max_chars)} "
        f"outcome={clean(run.outcome or '-')} exit={_fmt(run.exit_code)}"
    )


def _omitted_line(codes: list[str], budget: int) -> str:
    head = f"\nOmitted for size ({len(codes)} unusual sources): "
    return f"{head}{_names(codes, max(budget - len(head) - 1, 0))}\n"


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

    # A run has detail when a flagged source answers for it, not merely when some source row exists.
    run_ids_with_detail = {r.run_id for r in tonight_rows if id(r) in implicated or _raises_flag(r)}
    # Unexplained runs first: the budget may cut the list, and nothing else in the summary shows them.
    failing = sorted((r for r in runs if is_bad(r)), key=lambda r: r.run_id in run_ids_with_detail)
    failing_runs = [_failing_label(r, r.run_id in run_ids_with_detail) for r in failing]

    header = [
        # The date, not night_id: a trap's id names its answer.
        f"Night {inputs.night_end.date().isoformat()}: "
        f"{inputs.night_start.isoformat()} to {inputs.night_end.isoformat()}",
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
    note_budget = min(settings.note_char_budget, settings.summary_char_budget // 2) // 2
    tail += _cleared_lines(tonight_rows, implicated, settings.error_sample_max_chars, note_budget)
    long_overdue = [c for c in inputs.cadence or [] if _long_overdue(c, settings)]
    if long_overdue:
        names = _names(sorted({c.source_code for c in long_overdue}), max(note_budget - 100, 0))
        tail.append(
            f"Long overdue series (more than {settings.cadence_new_max_days} days late, not findings): "
            f"{len(long_overdue)} ({names})"
        )
    unattached = sum(1 for e in inputs.discovery_errors if e.source_code is None)
    if unattached:
        tail.append(f"Open discovery errors without a source: {unattached}")
    if failing:
        header.append("Failing runs:")
        # Failing runs keep to their own budget, so the sources that explain them always have room.
        room = min(settings.failing_runs_char_budget, settings.summary_char_budget // 4)
        used = len(f"... and {len(failing)} more failing runs, {len(failing)} with no per-source detail\n")
        for listed, run in enumerate(failing):
            line = f"- {_run_line(run, run.run_id in run_ids_with_detail, settings.error_sample_max_chars)}"
            if used + len(line) + 1 > room:
                bare = sum(1 for r in failing[listed:] if r.run_id not in run_ids_with_detail)
                note = f", {bare} with no per-source detail" if bare else ""
                header.append(f"... and {len(failing) - listed} more failing runs{note}")
                break
            header.append(line)
            used += len(line) + 1
    header += tail

    text = "\n".join(header) + "\n"
    budget = settings.summary_char_budget
    omitted: list[str] = []
    for i, state in enumerate(unusual_states):
        block = "\n" + _block(state, settings.error_sample_max_chars)
        # Room stays for naming every source after this one (at most a quarter of the budget).
        rest = [s.source_code for s in unusual_states[i + 1 :]]
        reserve = min(len(_omitted_line(rest, budget)), budget // 4) if rest else 0
        # Most important blocks first; once one overflows, it and every later block are omitted.
        if omitted or len(text) + len(block) + reserve > budget:
            omitted.append(state.source_code)
        else:
            text += block
    if omitted:
        text += _omitted_line(omitted, budget - len(text))
    return NightSummary(
        text=text,
        unusual=unusual,
        normal_count=normal_count,
        failing_runs=failing_runs,
        no_run=no_run,
        blocks={s.source_code: _block(s, settings.error_sample_max_chars) for s in unusual_states},
        # Sorted, so the evidence of the finding never depends on the order the rows came in.
        unexplained_runs=sorted(
            _run_line(r, False, settings.error_sample_max_chars) for r in failing if r.run_id not in run_ids_with_detail
        ),
    )
