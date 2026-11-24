"""Deterministic detector: the code decides which findings exist, the model only explains them.

One finding per unusual source, its category the first rule whose flags match (spec section 3).
"""

from dataclasses import dataclass

from myopenfund_agents.core.actions import group_outages
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

# (flags, category, severity), in the order the first match wins.
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
    present = set(flags)
    for rule in RULES:
        if rule[0] & present:
            return rule
    raise ValueError(f"no rule matches flags {flags}")


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


def detect(summary: NightSummary, settings: A1Settings) -> list[Detected]:
    found: list[Finding] = []
    others: dict[int, tuple[str, ...]] = {}
    crashing: list[Finding] = []  # findings of sources a failed run can be blamed on: site down, or failed_run
    for code, state in summary.unusual.items():
        flags_wanted, category, severity = _rule(state.flags)
        finding = _finding(code, category, severity, _evidence(summary.blocks[code], flags_wanted & set(state.flags)))
        found.append(finding)
        if category == Category.SOURCE_SITE_DOWN or FAILED_RUN in state.flags:
            crashing.append(finding)
        others[id(finding)] = tuple(f for f in FLAG_ORDER if f in state.flags and f not in flags_wanted)
    unexplained: Finding | None = None
    if summary.no_run:
        found.append(_finding("*", Category.UNKNOWN, Severity.ERROR, [NO_RUN_LINE]))
    elif summary.unexplained_runs:
        count = len(summary.unexplained_runs)
        title = f"*: unknown ({count} unexplained failing runs)" if count > EVIDENCE_MAX_ITEMS else None
        unexplained = _finding("*", Category.UNKNOWN, Severity.ERROR, summary.unexplained_runs, title)
        found.append(unexplained)
    grouped = group_outages(Findings(findings=found), settings).findings
    # A general outage covers the unexplained runs only when it absorbed more than general_outage_min_sources
    # sources that are site down or carry the failed_run flag: a source whose finding was grouped away.
    kept = {id(f) for f in grouped}
    covers = sum(id(f) not in kept for f in crashing) > settings.general_outage_min_sources
    if unexplained is not None and covers and any(f.category == Category.GENERAL_OUTAGE for f in grouped):
        grouped = [f for f in grouped if f is not unexplained]
    return [Detected(f"f{i}", f, others.get(id(f), ())) for i, f in enumerate(grouped, 1)]
