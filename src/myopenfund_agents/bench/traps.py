"""Traps: damage a green night in one known way and state exactly what A1 must then report."""

from myopenfund_agents.bench.fixtures import Expected, ExpectedAction, Key, Meta, Night
from myopenfund_agents.core.actions import fingerprint
from myopenfund_agents.core.findings import Category, Finding, Severity
from myopenfund_agents.core.inputs import NightInputs, OpenIssue, SourceRunRow

GENERAL_OUTAGE_ERROR = "getaddrinfo failed: Name or service not known"


def _tonight(night: NightInputs, code: str) -> list[SourceRunRow]:
    rows = [r for r in night.source_runs if r.source_code == code]
    if not rows:
        raise ValueError(f"source {code!r} has no run in night {night.night_id}")
    return rows


def _make(
    base: NightInputs,
    inputs: NightInputs,
    suffix: str,
    description: str,
    findings: list[tuple[str, Category]],
    kind: str | None,
    notes: list[str] | None = None,
) -> Night:
    inputs.night_id = f"{base.night_id}--{suffix}"
    expected = Expected(
        findings=[Key(source_code=s, category=c.value) for s, c in findings],
        actions=[ExpectedAction(kind=kind, source_code=s, category=c.value) for s, c in findings] if kind else [],
    )
    meta = Meta(
        origin="trap",
        description=description,
        base_night=base.night_id,
        validated_by=None,
        validated_on=None,
        notes=notes or [],
    )
    return Night(inputs=inputs, expected=expected, meta=meta)


def _damage_content(inputs: NightInputs, code: str, count: int) -> None:
    for row in _tonight(inputs, code):
        row.bad_content = count
        row.outcome = "degraded"
        row.exit_code = 3
        row.error_samples = [f"PDF text layer empty ({count} documents)"]


def volume_drop(base: NightInputs, code: str, seen: int = 3) -> Night:
    inputs = base.model_copy(deep=True)
    for row in _tonight(inputs, code):
        row.docs_seen = seen
    return _make(
        base, inputs, f"volume-drop-{code}", f"Source {code} saw only {seen} documents tonight",
        [(code, Category.VOLUME_DROP)], "new_issue",
    )  # fmt: skip


def bad_content(base: NightInputs, code: str, count: int = 2) -> Night:
    inputs = base.model_copy(deep=True)
    _damage_content(inputs, code, count)
    return _make(
        base, inputs, f"bad-content-{code}", f"Source {code} returned {count} unreadable documents tonight",
        [(code, Category.UNREADABLE_PDF)], "new_issue",
    )  # fmt: skip


def general_outage(base: NightInputs) -> Night:
    inputs = base.model_copy(deep=True)
    for row in inputs.source_runs:
        row.outcome = "failed"
        row.exit_code = 1
        row.docs_seen = 0
        row.error_samples = [GENERAL_OUTAGE_ERROR]
    for run in inputs.runs:
        run.outcome = "failed"
        run.exit_code = 1
    n = len({r.source_code for r in inputs.source_runs})
    notes = [
        "The deterministic outage grouping needs more than general_outage_min_sources (default 5) failing "
        f"sources; this base has {n}."
    ]
    return _make(
        base, inputs, "general-outage", "Every run and every source failed tonight with a DNS error",
        [("*", Category.GENERAL_OUTAGE)], "new_issue", notes,
    )  # fmt: skip


def new_source(base: NightInputs, code: str = "zz") -> Night:
    inputs = base.model_copy(deep=True)
    if not inputs.source_runs:
        raise ValueError(f"night {base.night_id} has no source run to copy")
    if any(r.source_code == code for r in inputs.source_runs):
        raise ValueError(f"source {code!r} already has a run in night {base.night_id}")
    row = inputs.source_runs[0]
    inputs.source_runs.append(
        SourceRunRow(
            run_id=row.run_id,
            tool=row.tool,
            command=f"harvest {code}",
            corpus=row.corpus,
            source_code=code,
            started_at=row.started_at,
            finished_at=row.finished_at,
            outcome="ok",
            exit_code=0,
            docs_seen=0,
            docs_new=0,
            docs_failed=0,
            fetch_errors=0,
            truncated=False,
        )
    )
    return _make(
        base, inputs, f"new-source-{code}", f"Source {code} appears for the first time with zero documents",
        [], None,
    )  # fmt: skip


def known_problem(base: NightInputs, code: str, repo: str, issue_number: int = 101) -> Night:
    inputs = base.model_copy(deep=True)
    _damage_content(inputs, code, 2)
    finding = Finding(
        source_code=code, category=Category.UNREADABLE_PDF, severity=Severity.WARN,
        title="known", diagnosis="known", evidence=[],
    )  # fmt: skip
    inputs.open_issues.append(OpenIssue(fingerprint=fingerprint(repo, finding), number=issue_number, repo=repo))
    return _make(
        base, inputs, f"known-problem-{code}", f"Source {code} has unreadable documents and an open issue already",
        [(code, Category.UNREADABLE_PDF)], "comment",
    )  # fmt: skip
