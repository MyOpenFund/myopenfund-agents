"""Bench runner: run an engine over known nights, score it, and write a sanitised report."""

import dataclasses
import json
import os
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from myopenfund_agents.bench.fixtures import Night
from myopenfund_agents.bench.score import (
    FALSE_ALARMS,
    PROBLEMS_FOUND,
    RATIO_CRITERIA,
    SLOWEST_NIGHT,
    Attempt,
    NightScore,
    Verdict,
    score_night,
    verdicts,
)
from myopenfund_agents.core.actions import group_outages, plan_actions
from myopenfund_agents.core.summary import build_summary
from myopenfund_agents.engines.base import Engine, EngineError
from myopenfund_agents.sanitise import DenyList, check_tree, redact_paths, violations
from myopenfund_agents.settings import A1Settings

DEFAULT_CORPUS_REPOS = {"central-bank": "central-bank-corpus", "company": "company-corpus"}
DEFAULT_REPO = "data-orchestrator"


def repo_of(night: Night, corpus_repos: Mapping[str, str]) -> dict[str, str]:
    """source_code -> repo, from tonight's and history rows whose corpus is known (first row wins)."""
    out: dict[str, str] = {}
    for row in [*night.inputs.source_runs, *night.inputs.history]:
        if row.corpus in corpus_repos and row.source_code not in out:
            out[row.source_code] = corpus_repos[row.corpus]
    return out


def run_night(
    night: Night,
    engine: Engine,
    settings: A1Settings,
    repeat: int,
    corpus_repos: Mapping[str, str] | None,
    deny: DenyList | None = None,
) -> NightScore:
    if repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {repeat}")
    repos = repo_of(night, DEFAULT_CORPUS_REPOS if corpus_repos is None else corpus_repos)
    summary = build_summary(night.inputs, settings)
    attempts: list[Attempt] = []
    for _ in range(repeat):
        start = time.monotonic()
        try:
            result = engine.diagnose(summary.text)
        except EngineError as exc:
            attempts.append(Attempt(None, None, time.monotonic() - start, error=_error_text(exc, deny or DenyList())))
            continue
        # Scored after the deterministic grouping, as A1 would act on it.
        keys = group_outages(result.findings, settings).keys()
        planned = plan_actions(
            result.findings, night.inputs.open_issues, repos, DEFAULT_REPO, settings, no_run=summary.no_run
        )
        actions = [(a.kind.value, a.finding.source_code, a.finding.category.value) for a in planned]
        attempts.append(
            Attempt(
                keys,
                actions,
                result.seconds,
                result.retries,
                findings=result.findings.model_dump(mode="json")["findings"],
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
        )
    return score_night(night, attempts)


def _error_text(exc: Exception, deny: DenyList) -> str:
    """The engine error, or a withheld marker when it carries an infra detail (the report must stay publishable)."""
    reasons = violations(str(exc), deny)
    if reasons:
        return f"{type(exc).__name__}: withheld ({', '.join(reasons)})"
    return str(exc)


_FINDING_TEXTS = ("title", "diagnosis")


def _withheld(text: str, deny: DenyList) -> str:
    reasons = violations(text, deny)
    return f"withheld ({', '.join(reasons)})" if reasons else text


def _withhold_finding_texts(nights: list[dict[str, Any]], deny: DenyList) -> None:
    """Replace each finding text (title, diagnosis, evidence item) carrying an infra detail by a withheld marker.

    Diagnoses may quote an error sample; the rest of the payload stays behind the fail-closed gate.
    """
    for night in nights:
        for attempt in night.get("attempts", []):
            for finding in attempt.get("findings") or []:
                for name in _FINDING_TEXTS:
                    if isinstance(finding.get(name), str):
                        finding[name] = _withheld(finding[name], deny)
                if isinstance(finding.get("evidence"), list):
                    finding["evidence"] = [_withheld(e, deny) if isinstance(e, str) else e for e in finding["evidence"]]


def run_bench(
    nights: list[Night],
    engine: Engine,
    settings: A1Settings,
    repeat: int,
    thresholds: dict[str, Any],
    corpus_repos: Mapping[str, str] | None = None,
    deny: DenyList | None = None,
) -> tuple[list[NightScore], list[Verdict]]:
    scores = [run_night(n, engine, settings, repeat, corpus_repos, deny) for n in nights]
    return scores, verdicts(scores, thresholds)


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, set | frozenset):
        return sorted(_jsonable(v) for v in obj)
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    return obj


def _safe_name(text: str) -> str:
    return text.replace("/", "_").replace(":", "_")


def _fmt(criterion: str, value: float) -> str:
    if criterion in RATIO_CRITERIA:
        return f"{value:.0%}"
    if criterion == SLOWEST_NIGHT:
        return f"{value:.1f}"
    return f"{value:g}"


def _verdict_value(v: Verdict, scores: list[NightScore]) -> str:
    if v.criterion == PROBLEMS_FOUND:
        incidents = [s for s in scores if s.origin != "green"]
        found = sum(len(s.found) for s in incidents)
        expected = sum(len(s.expected) for s in incidents)
        return f"{_fmt(v.criterion, v.value)} ({found}/{expected})"
    return _fmt(v.criterion, v.value)


def _keys(keys: list[tuple[str, str]]) -> str:
    return ", ".join(f"`{s}/{c}`" for s, c in keys) or "-"


def _markdown(payload: dict[str, Any], scores: list[NightScore], verdict_list: list[Verdict]) -> str:
    lines = [
        f"# Bench: {payload['engine']} / {payload['model']} ({payload['profile']})",
        "",
        f"Run at {payload['stamp']}. Profile `{payload['profile']}`, repeat {payload['repeat']}.",
        "",
        "| Criterion | Value | Threshold | Pass |",
        "|---|---|---|---|",
    ]
    for v in verdict_list:
        passed = "yes" if v.passed else "NO"
        lines.append(f"| {v.criterion} | {_verdict_value(v, scores)} | {_fmt(v.criterion, v.threshold)} | {passed} |")
    lines += [
        "",
        f'"{FALSE_ALARMS}" counts extra findings on every night with no expected finding.',
        "",
        "| Night | Origin | Found | Extra | False alarms | Actions | Valid | Agreement | Max s |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in scores:
        actions = "-" if s.actions_ok is None else ("ok" if s.actions_ok else "NO")
        lines.append(
            f"| {s.night_id} | {s.origin} | {_keys(s.found)} | {_keys(s.extra)} | {s.false_alarms} | {actions} "
            f"| {'yes' if s.valid else 'NO'} | {s.agreement:.0%} | {s.seconds_max:.1f} |"
        )
    return "\n".join(lines) + "\n"


def write_report(
    out_dir: Path,
    engine: Engine,
    profile: str,
    scores: list[NightScore],
    verdict_list: list[Verdict],
    deny: DenyList | None = None,
    *,
    repeat: int,
    settings: A1Settings,
    thresholds: Mapping[str, Any],
) -> Path:
    """Write `<stamp>-<engine>-<model>.{json,md}` into out_dir; return the Markdown path.

    The payload records how it was produced: ``repeat``, every A1 setting and the threshold profile values.

    The payload must pass the sanitiser first: results are meant to be committed to a public repository.
    ``deny`` defaults to the deny-list named by ``MOF_AGENTS_DENYLIST`` (unset or empty: built-in checks only).
    """
    if deny is None:
        deny = DenyList.load(os.environ.get("MOF_AGENTS_DENYLIST") or None)  # empty reads as unset, as in the CLI
    now = datetime.now(UTC)
    nights = _jsonable(scores)
    _withhold_finding_texts(nights, deny)
    payload = {
        "engine": engine.name,
        "model": engine.model,
        "profile": profile,
        "stamp": now.isoformat(timespec="seconds"),
        "repeat": repeat,
        "settings": dataclasses.asdict(settings),
        "thresholds": dict(thresholds),
        "verdicts": _jsonable(verdict_list),
        "nights": nights,
    }
    found = check_tree(payload, deny)
    if found:
        raise ValueError("report refused: " + "; ".join(redact_paths(found, deny)))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{_safe_name(engine.name)}-{_safe_name(engine.model)}"
    (out_dir / f"{base}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = out_dir / f"{base}.md"
    md.write_text(_markdown(payload, scores, verdict_list), encoding="utf-8")
    return md
