"""Traps: damage a green night in one known way and state exactly what A1 must then report."""

import re
import tomllib
from datetime import timedelta
from pathlib import Path

from myopenfund_agents.bench.fixtures import Expected, ExpectedAction, Key, Meta, Night, load_all
from myopenfund_agents.core.actions import fingerprint
from myopenfund_agents.core.findings import Category, Finding, Severity
from myopenfund_agents.core.inputs import NightInputs, OpenIssue, SourceRunRow
from myopenfund_agents.core.summary import NO_RUN_LINE, build_summary, is_bad
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


def _degrade_runs(inputs: NightInputs, run_ids: set[str]) -> None:
    """Stamp a degraded outcome on the runs and on every source row they share, as the vault does."""
    for rows in (inputs.source_runs, inputs.runs):
        for row in rows:
            if row.run_id in run_ids:
                row.outcome, row.exit_code = "degraded", 3


def _damage_content(inputs: NightInputs, code: str, count: int) -> None:
    row = max(_tonight(inputs, code), key=lambda r: r.docs_seen or 0)
    row.bad_content = count
    row.error_samples = [f"PDF text layer empty ({count} documents)"]
    _degrade_runs(inputs, {row.run_id})
    state = build_summary(inputs, A1Settings()).unusual.get(code)
    if state is None or "bad_content" not in state.flags:
        raise ValueError(f"count={count} does not raise a bad_content flag for source {code!r}")


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
    if any(i.fingerprint.endswith(":*:no_run") for i in base.open_issues):
        raise ValueError("an open issue already has the no-run fingerprint: the expected action would be a comment")
    inputs = base.model_copy(deep=True)
    inputs.runs = []
    inputs.source_runs = []
    summary = build_summary(inputs, settings)
    if NO_RUN_LINE not in summary.text or summary.unusual:
        raise ValueError(
            f"night {base.night_id} still shows unusual sources without any run (discovery errors or cadence)"
        )
    notes = ["An addition to the spec's trap list (Review Focus 4), approved 2026-11-23."]
    return _make(
        base, inputs, "no-run", "No run and no source run finished tonight: the scheduler did not fire",
        [("*", Category.UNKNOWN)], "new_issue", notes,
    )  # fmt: skip


CHRONIC_ERROR = "page 3: GET bytes failed after retries"


def chronic(
    base: NightInputs, code: str, repo: str, issue_number: int = 102, settings: A1Settings | None = None
) -> list[Night]:
    """One source failing on three consecutive nights: new issue, then a comment on it twice.

    The three nights share the base's date; nights 2 and 3 carry one and two failed nights of ``code`` in
    their history, and the open issue. The run's degraded outcome is stamped on every row it shares.
    """
    settings = settings or A1Settings()
    _tonight(base, code)
    category = Category.SOURCE_SITE_DOWN
    finding = Finding(
        source_code=code, category=category, severity=Severity.WARN, title="known", diagnosis="known", evidence=[],
    )  # fmt: skip
    fp = fingerprint(repo, finding)
    if any(i.fingerprint == fp for i in base.open_issues):
        raise ValueError(f"source {code!r} already has an open issue for {category.value}")
    if any(is_bad(r) for r in base.runs):
        raise ValueError(f"night {base.night_id} already has a failing run")
    nights = []
    for past in range(3):
        inputs = base.model_copy(deep=True)
        target = max(_tonight(inputs, code), key=lambda r: r.docs_seen or 0)
        hit = [target]
        _degrade_runs(inputs, {r.run_id for r in hit})
        previous = [
            r
            for r in inputs.history
            if r.source_code == code
            and r.command == target.command
            and r.finished_at is not None
            and inputs.night_start - timedelta(days=past) < r.finished_at <= inputs.night_start
        ]
        for row in hit + previous:
            row.outcome, row.exit_code = "degraded", 3
            row.fetch_errors = 1
            row.error_samples = [CHRONIC_ERROR]
        if past:
            inputs.open_issues.append(OpenIssue(fingerprint=fp, number=issue_number, repo=repo))
        summary = build_summary(inputs, settings)
        if set(summary.unusual) != {code}:
            raise ValueError(f"other sources are unusual too: {sorted(set(summary.unusual) - {code})}")
        if summary.unusual[code].consecutive_failed_nights != past:
            raise ValueError(f"history of source {code!r} cannot carry {past} failed nights")
        nights.append(
            _make(
                base, inputs, f"chronic-{code}-{past + 1}", f"Source {code} fails, night {past + 1} of 3",
                [(code, category)], "comment" if past else "new_issue",
            )
        )  # fmt: skip
    return nights


RECIPES = "traps.toml"
_KINDS = {
    f.__name__: f for f in (volume_drop, bad_content, general_outage, new_source, known_problem, no_run, chronic)
}


def _valid_arg(name: str, value: object) -> bool:
    """A recipe may set only plain data: source code, repository and counts."""
    if name in ("code", "repo"):
        return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9._-]+", value) is not None
    minimum = {"seen": 0, "count": 1, "issue_number": 1}.get(name)
    return minimum is not None and type(value) is int and value >= minimum


def load_traps(root: Path, nights: list[Night]) -> list[Night]:
    """Trap nights built from ``root/traps.toml`` on one of ``nights``; none without that file."""
    path = root / RECIPES
    if not path.is_file():
        return []
    try:
        spec = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"{RECIPES}: {exc}") from exc
    unknown = sorted(set(spec) - {"base", "trap"})
    if unknown:
        raise ValueError(f"{RECIPES}: unknown key(s): {', '.join(unknown)}")
    base_night = next((n for n in nights if n.inputs.night_id == spec.get("base")), None)
    if base_night is None:
        raise ValueError(f"{RECIPES}: base night {spec.get('base')!r} not found")
    if base_night.meta.origin != "green":
        raise ValueError(f"{RECIPES}: base night {spec['base']!r} is not green")
    base = base_night.inputs
    recipes = spec.get("trap", [])
    if not isinstance(recipes, list) or not all(isinstance(r, dict) for r in recipes):
        raise ValueError(f"{RECIPES}: 'trap' must be a list of tables")
    made: dict[str, Night] = {}
    for recipe in recipes:
        args = dict(recipe)
        kind = args.pop("kind", None)
        build = _KINDS.get(kind) if isinstance(kind, str) else None
        if build is None:
            raise ValueError(f"{RECIPES}: unknown trap kind {kind!r}")
        for name, value in args.items():
            if not _valid_arg(name, value):
                raise ValueError(f"{RECIPES}: {kind}: invalid {name} = {value!r}")
        try:
            result = build(base, **args)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{RECIPES}: {kind}: {exc}") from exc
        for night in result if isinstance(result, list) else [result]:
            if night.inputs.night_id in made:
                raise ValueError(f"{RECIPES}: {night.inputs.night_id} made twice")
            made[night.inputs.night_id] = night
    return list(made.values())


def load_bench(root: Path) -> list[Night]:
    """Every stored night and every trap built from them, sorted by night id."""
    nights = load_all(root)
    return sorted([*nights, *load_traps(root, nights)], key=lambda n: n.inputs.night_id)
