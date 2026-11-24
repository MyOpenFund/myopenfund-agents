"""Deterministic detector: the code decides which findings exist, the model only explains them.

One finding per unusual source, its category the first rule whose flags match (spec section 3).
"""

from dataclasses import dataclass

from myopenfund_agents.core.actions import SEVERITY_RANK, group_outages, outage
from myopenfund_agents.core.findings import (
    EVIDENCE_MAX_ITEMS,
    TITLE_MAX_LENGTH,
    Category,
    Finding,
    Findings,
    Severity,
)
from myopenfund_agents.core.summary import (
    DAYS_LATE_PREFIX,
    FAILED_COMMANDS_PREFIX,
    FLAG_ORDER,
    HISTORY_PREFIX,
    NO_RUN_LINE,
    SAMPLE_PREFIX,
    TONIGHT_PREFIX,
    NightSummary,
)
from myopenfund_agents.settings import A1Settings

# (flags, category, severity). The first match gives the category, the highest severity of all matches is kept.
RULES: tuple[tuple[frozenset[str], Category, Severity], ...] = (
    (frozenset({"bad_content"}), Category.UNREADABLE_PDF, Severity.WARN),
    (frozenset({"fetch_errors", "truncated"}), Category.SOURCE_SITE_DOWN, Severity.ERROR),
    (frozenset({"volume_drop"}), Category.VOLUME_DROP, Severity.WARN),
    (frozenset({"docs_failed"}), Category.UNKNOWN, Severity.WARN),
    (frozenset({"failed_run"}), Category.UNKNOWN, Severity.ERROR),  # "only" = no row above matched
    (frozenset({"cadence_overdue"}), Category.CADENCE_OVERDUE, Severity.WARN),
    (frozenset({"open_discovery_errors"}), Category.UNKNOWN, Severity.WARN),
)
FAILED_RUN = "failed_run"
FAILED_RUN_RULE = frozenset({FAILED_RUN})  # the one `unknown` row that is grouped: many crashed sources are an outage
NO_EXPLANATION = "No explanation."
# The block lines that show each flag.
EVIDENCE_PREFIXES: dict[str, tuple[str, ...]] = {
    "bad_content": (TONIGHT_PREFIX,),
    "fetch_errors": (TONIGHT_PREFIX,),
    "truncated": (TONIGHT_PREFIX,),
    "docs_failed": (TONIGHT_PREFIX,),
    "volume_drop": (TONIGHT_PREFIX, HISTORY_PREFIX),
    "failed_run": (FAILED_COMMANDS_PREFIX,),
    "cadence_overdue": (DAYS_LATE_PREFIX,),
    "open_discovery_errors": (DAYS_LATE_PREFIX,),
}


@dataclass(frozen=True)
class Detected:
    finding_id: str
    finding: Finding
    other_flags: tuple[str, ...]


def _rule(flags: list[str]) -> tuple[frozenset[str], Category, Severity]:
    """The first matching row's flags and category, with the highest severity of every matching row."""
    matching = [rule for rule in RULES if rule[0] & set(flags)]
    if not matching:
        raise ValueError(f"no rule matches flags {flags}")
    wanted, category, _ = matching[0]
    return wanted, category, max((severity for _, _, severity in matching), key=SEVERITY_RANK.__getitem__)


def _evidence(block: str, flags: frozenset[str]) -> list[str]:
    """At most five lines of the block, verbatim and in block order.

    Kept first: the flags line, then the lines carrying the matched rule's flags, then error samples (they carry
    the cause), then the rest of the block.
    """
    lines = block.splitlines()
    if len(lines) <= EVIDENCE_MAX_ITEMS:
        return lines
    prefixes = tuple(p for flag in flags for p in EVIDENCE_PREFIXES.get(flag, ()))
    matched = [i for i, line in enumerate(lines) if i and line.startswith(prefixes)]
    samples = [i for i, line in enumerate(lines) if i and line.startswith(SAMPLE_PREFIX) and i not in matched]
    rest = [i for i in range(1, len(lines)) if i not in matched and i not in samples]
    return [lines[i] for i in sorted([0, *(matched + samples + rest)[: EVIDENCE_MAX_ITEMS - 1]])]


def _title(source: str, category: Category) -> str:
    suffix = f": {category.value}"
    if len(source) + len(suffix) <= TITLE_MAX_LENGTH:
        return source + suffix
    return source[: TITLE_MAX_LENGTH - len(suffix) - 1] + "…" + suffix


def _finding(
    source: str, category: Category, severity: Severity, evidence: list[str], title: str | None = None
) -> Finding:
    return Finding(
        source_code=source,
        category=category,
        severity=severity,
        title=title or _title(source, category),
        diagnosis=NO_EXPLANATION,
        evidence=evidence[:EVIDENCE_MAX_ITEMS],
    )


def _with_outage(
    found: list[Finding], failed_run: list[Finding], settings: A1Settings
) -> tuple[list[Finding], set[int]]:
    """The findings with at most one general outage, and the ids of the ones it absorbed.

    It absorbs what `group_outages` groups and, above general_outage_min_sources of them, the sources matched by the
    failed_run row."""
    kept = {id(f) for f in group_outages(Findings(findings=found), settings).findings}
    absorbed = [f for f in found if id(f) not in kept]
    if len(failed_run) > settings.general_outage_min_sources:
        absorbed += failed_run
    ids = {id(f) for f in absorbed}
    out: list[Finding] = []
    placed = False
    for f in found:
        if id(f) not in ids:
            out.append(f)
        elif not placed:
            out.append(outage(sorted(a.source_code for a in absorbed)))
            placed = True
    return out, ids


def detect(summary: NightSummary, settings: A1Settings) -> list[Detected]:
    found: list[Finding] = []
    others: dict[int, tuple[str, ...]] = {}
    crashing: list[Finding] = []  # findings of sources a failed run can be blamed on: site down, or failed_run
    failed_run: list[Finding] = []  # findings of sources matched by the failed_run row itself
    for code, state in summary.unusual.items():
        flags_wanted, category, severity = _rule(state.flags)
        finding = _finding(code, category, severity, _evidence(summary.blocks[code], flags_wanted & set(state.flags)))
        found.append(finding)
        if category == Category.SOURCE_SITE_DOWN or FAILED_RUN in state.flags:
            crashing.append(finding)
        if flags_wanted == FAILED_RUN_RULE:
            failed_run.append(finding)
        others[id(finding)] = tuple(f for f in FLAG_ORDER if f in state.flags and f not in flags_wanted)
    unexplained: Finding | None = None
    if summary.no_run:
        found.append(_finding("*", Category.UNKNOWN, Severity.ERROR, [NO_RUN_LINE]))
    elif summary.unexplained_runs:
        count = len(summary.unexplained_runs)
        title = f"*: unknown ({count} unexplained failing runs)" if count > EVIDENCE_MAX_ITEMS else None
        unexplained = _finding("*", Category.UNKNOWN, Severity.ERROR, summary.unexplained_runs, title)
        found.append(unexplained)
    grouped, absorbed = _with_outage(found, failed_run, settings)
    # A general outage covers the unexplained runs only when it absorbed more than general_outage_min_sources
    # sources that are site down or carry the failed_run flag: a source whose finding was grouped away.
    covers = sum(id(f) in absorbed for f in crashing) > settings.general_outage_min_sources
    if unexplained is not None and covers:
        grouped = [f for f in grouped if f is not unexplained]
    return [Detected(f"f{i}", f, others.get(id(f), ())) for i, f in enumerate(grouped, 1)]
