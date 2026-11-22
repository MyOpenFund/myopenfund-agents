"""Deterministic layer: group outages, dedup by fingerprint, decide GitHub actions."""

from dataclasses import dataclass
from enum import StrEnum

from myopenfund_agents.core.findings import Category, Finding, Findings, Severity
from myopenfund_agents.core.inputs import OpenIssue
from myopenfund_agents.settings import A1Settings

AGENT = "a1"
_NOT_GROUPED = {Category.GENERAL_OUTAGE, Category.CADENCE_OVERDUE}
_RANK = {Severity.INFO: 0, Severity.WARN: 1, Severity.ERROR: 2}


class Kind(StrEnum):
    NEW_ISSUE = "new_issue"
    COMMENT = "comment"
    SUMMARY_ISSUE = "summary_issue"


@dataclass(frozen=True)
class Action:
    kind: Kind
    fingerprint: str
    repo: str
    finding: Finding
    issue_number: int | None = None


def fingerprint(repo: str, f: Finding) -> str:
    return f"{AGENT}:{repo}:{f.source_code}:{f.category.value}"


def summary_fingerprint(repo: str) -> str:
    """The summary issue's own fingerprint: never shared with a real ('*', unknown) finding."""
    return f"{AGENT}:{repo}:*:summary"


def group_outages(findings: Findings, settings: A1Settings) -> Findings:
    codes: dict[Category, set[str]] = {}
    for f in findings.findings:
        if f.source_code != "*" and f.category not in _NOT_GROUPED:
            codes.setdefault(f.category, set()).add(f.source_code)
    grouped = {c for c, s in codes.items() if len(s) > settings.general_outage_min_sources}
    all_codes = sorted(set().union(*(codes[c] for c in grouped))) if grouped else []

    out: list[Finding] = []
    placed = False
    for f in findings.findings:
        if f.source_code != "*" and f.category in grouped:
            if not placed:
                out.append(_outage(all_codes))
                placed = True
        elif f.category == Category.GENERAL_OUTAGE:
            if not placed:
                out.append(f if f.source_code == "*" else f.model_copy(update={"source_code": "*"}))
                placed = True
        else:
            out.append(f)
    return Findings(findings=out)


def _outage(codes: list[str]) -> Finding:
    evidence = f"sources: {', '.join(codes)}"
    return Finding(
        source_code="*",
        category=Category.GENERAL_OUTAGE,
        severity=Severity.ERROR,
        title=f"General outage: {len(codes)} sources failing"[:120],
        diagnosis="Many sources failed in the same way tonight, which points to a shared cause rather than "
        "independent source problems.",
        evidence=[evidence[:500]],
    )


def plan_actions(
    findings: Findings,
    open_issues: list[OpenIssue],
    repo_of: dict[str, str],
    default_repo: str,
    settings: A1Settings,
) -> list[Action]:
    open_by_fp = {i.fingerprint: i for i in open_issues}
    seen: set[str] = set()
    comments: list[Action] = []
    news: list[Action] = []
    for f in group_outages(findings, settings).findings:
        repo = repo_of.get(f.source_code, default_repo) if f.source_code != "*" else default_repo
        fp = fingerprint(repo, f)
        if fp in seen:
            continue
        seen.add(fp)
        issue = open_by_fp.get(fp)
        if issue is not None:
            comments.append(Action(Kind.COMMENT, fp, issue.repo, f, issue.number))
        else:
            news.append(Action(Kind.NEW_ISSUE, fp, repo, f))
    if len(news) <= settings.max_new_issues_per_run:
        return [*comments, *news]
    summary = _summary(news, default_repo)
    issue = open_by_fp.get(summary.fingerprint)
    if issue is not None:
        summary = Action(Kind.COMMENT, summary.fingerprint, issue.repo, summary.finding, issue.number)
    return [*comments, summary]


def _summary(news: list[Action], default_repo: str) -> Action:
    severity = max((a.finding.severity for a in news), key=_RANK.__getitem__)
    lines: list[str] = []
    size = 0
    for a in news:
        line = f"- {a.fingerprint}: {a.finding.title}"
        if size + len(line) + 1 > 1900:
            lines.append(f"- ... and {len(news) - len(lines)} more")
            break
        lines.append(line)
        size += len(line) + 1
    finding = Finding(
        source_code="*",
        category=Category.UNKNOWN,
        severity=severity,
        title=f"A1: {len(news)} new problems tonight",
        diagnosis="Too many new problems for individual issues:\n" + "\n".join(lines),
        evidence=[a.fingerprint for a in news[:5]],
    )
    return Action(Kind.SUMMARY_ISSUE, summary_fingerprint(default_repo), default_repo, finding)
