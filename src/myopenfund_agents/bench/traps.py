"""Traps: damage a green night in one known way and state exactly what A1 must then report."""

from myopenfund_agents.bench.fixtures import Expected, ExpectedAction, Key, Meta, Night
from myopenfund_agents.core.actions import fingerprint
from myopenfund_agents.core.findings import Category, Finding, Severity
from myopenfund_agents.core.inputs import NightInputs, OpenIssue, SourceRunRow
from myopenfund_agents.core.summary import NO_RUN_LINE, build_summary
from myopenfund_agents.settings import A1Settings

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
        findings=[Key(source_code=s, category=c) for s, c in findings],
        actions=[ExpectedAction(kind=kind, source_code=s, category=c) for s, c in findings] if kind else [],
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


def volume_drop(base: NightInputs, code: str, seen: int = 3, settings: A1Settings | None = None) -> Night:
    settings = settings or A1Settings()
    inputs = base.model_copy(deep=True)
    for i, row in enumerate(_tonight(inputs, code)):
        row.docs_seen = seen if i == 0 else 0
    state = build_summary(inputs, settings).unusual.get(code)
    if state is None or "volume_drop" not in state.flags:
        raise ValueError(f"seen={seen} does not raise a volume_drop flag for source {code!r} with these settings")
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


def general_outage(base: NightInputs, settings: A1Settings | None = None) -> Night:
    settings = settings or A1Settings()
    n = len({r.source_code for r in base.source_runs})
    if n <= settings.general_outage_min_sources:
        raise ValueError(
            f"base has {n} tonight sources; general_outage_min_sources={settings.general_outage_min_sources} "
            "requires more for the outage grouping to apply"
        )
    inputs = base.model_copy(deep=True)
    for row in inputs.source_runs:
        row.outcome = "failed"
        row.exit_code = 1
        row.docs_seen = 0
        row.docs_new = 0
        row.docs_failed = 0
        row.fetch_errors = 1
        row.error_samples = [GENERAL_OUTAGE_ERROR]
    for run in inputs.runs:
        run.outcome = "failed"
        run.exit_code = 1
    notes = [
        f"Needs more than general_outage_min_sources ({settings.general_outage_min_sources}) failing sources "
        f"for the outage grouping; this base has {n}."
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
    update = {
        "command": f"harvest {code}",
        "source_code": code,
        "outcome": "ok",
        "exit_code": 0,
        "docs_seen": 0,
        "docs_new": 0,
        "docs_failed": 0,
        "fetch_errors": 0,
        "truncated": False,
        "error_samples": [],
        "bad_content": None,
    }
    inputs.source_runs.append(row.model_copy(deep=True, update=update))
    return _make(
        base, inputs, f"new-source-{code}", f"Source {code} appears for the first time with zero documents",
        [], None,
    )  # fmt: skip


def known_problem(base: NightInputs, code: str, repo: str, issue_number: int = 101, count: int = 2) -> Night:
    inputs = base.model_copy(deep=True)
    _damage_content(inputs, code, count)
    finding = Finding(
        source_code=code, category=Category.UNREADABLE_PDF, severity=Severity.WARN,
        title="known", diagnosis="known", evidence=[],
    )  # fmt: skip
    fp = fingerprint(repo, finding)
    if not any(i.fingerprint == fp for i in inputs.open_issues):
        inputs.open_issues.append(OpenIssue(fingerprint=fp, number=issue_number, repo=repo))
    return _make(
        base, inputs, f"known-problem-{code}", f"Source {code} has unreadable documents and an open issue already",
        [(code, Category.UNREADABLE_PDF)], "comment",
    )  # fmt: skip


def no_run(base: NightInputs, settings: A1Settings | None = None) -> Night:
    """The scheduler did not fire: no run and no source run tonight, history unchanged."""
    settings = settings or A1Settings()
    if any(i.fingerprint.endswith(":*:unknown") for i in base.open_issues):
        raise ValueError("an open issue already has a ('*', unknown) fingerprint: the expected action would differ")
    inputs = base.model_copy(deep=True)
    inputs.runs = []
    inputs.source_runs = []
    summary = build_summary(inputs, settings)
    if NO_RUN_LINE not in summary.text or summary.unusual:
        raise ValueError(
            f"night {base.night_id} still shows unusual sources without any run (discovery errors or cadence)"
        )
    notes = ["An addition to the spec's trap list (Review Focus 4), pending Marc's approval."]
    return _make(
        base, inputs, "no-run", "No run and no source run finished tonight: the scheduler did not fire",
        [("*", Category.UNKNOWN)], "new_issue", notes,
    )  # fmt: skip
