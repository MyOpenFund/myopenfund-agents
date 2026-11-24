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
    AGREEMENT_WORST,
    NO_FINDING,
    RATIO_CRITERIA,
    SECONDS_CRITERIA,
    TOKEN_FIGURES,
    ActionTriple,
    Attempt,
    BenchConfigError,
    NightScore,
    Verdict,
    reported,
    score_night,
    verdicts,
)
from myopenfund_agents.core.detector import Detected, detect
from myopenfund_agents.core.summary import build_summary
from myopenfund_agents.core.triage import Triage, triage
from myopenfund_agents.engines.base import Engine
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


def _finding(d: Detected) -> dict[str, Any]:
    f = d.finding.model_dump(mode="json", exclude={"diagnosis"})  # a fixed text, the same for every finding
    return {"finding_id": d.finding_id, **f, "other_flags": list(d.other_flags)}


def _attempt(t: Triage, elapsed: float) -> Attempt:
    """What the report keeps of a triage. Only numbers and the model name are read from `t.result`: its texts
    are the model's, unchecked."""
    r, checked = t.result, t.checked
    out = Attempt(
        findings=[_finding(d) for d in t.detected],
        actions=[(a.kind.value, a.finding.source_code, a.finding.category.value) for a in t.actions],
        seconds=r.seconds if r and checked else elapsed,  # a failed attempt costs all the time it took
        valid=None if checked is None and t.engine_error is None else checked is not None,
        error=t.engine_error,
    )
    if r is not None:
        out.retries, out.model = r.retries, r.model
        out.input_tokens, out.output_tokens = r.input_tokens, r.output_tokens
    if checked is not None:
        out.explanations = [e.model_dump(mode="json") for e in checked.kept.values()]
        out.dropped = dict(checked.dropped)
        out.observations = list(checked.observations)
        out.quotes_total, out.quotes_exact = checked.quotes_total, checked.quotes_exact
        out.unknown_numbers = checked.unknown_numbers
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
    attempts: list[Attempt] = []
    for _ in range(repeat):
        start = time.monotonic()
        t = triage(night.inputs, engine, settings, repos, DEFAULT_REPO, deny or DenyList())
        attempts.append(_attempt(t, time.monotonic() - start))
    return score_night(night, attempts)


def _withheld(text: str, deny: DenyList) -> str:
    reasons = violations(text, deny)
    return f"withheld ({', '.join(reasons)})" if reasons else text


def _withhold_texts(nights: list[dict[str, Any]], deny: DenyList) -> None:
    """Replace each free text (finding title and evidence, diagnosis, quote, observation, error) carrying an infra
    detail by a withheld marker. The rest of the payload stays behind the fail-closed gate."""
    for attempt in (a for night in nights for a in night["attempts"]):
        for finding in attempt["findings"]:
            finding["title"] = _withheld(finding["title"], deny)
            finding["evidence"] = [_withheld(e, deny) for e in finding["evidence"]]
        for explanation in attempt["explanations"]:
            explanation["diagnosis"] = _withheld(explanation["diagnosis"], deny)
            for quote in explanation["quotes"]:
                quote["text"] = _withheld(quote["text"], deny)
        attempt["observations"] = [_withheld(o, deny) for o in attempt["observations"]]
        if attempt["error"] is not None:
            attempt["error"] = _withheld(attempt["error"], deny)


def run_bench(
    nights: list[Night],
    engine: Engine,
    settings: A1Settings,
    repeat: int,
    thresholds: dict[str, Any],
    corpus_repos: Mapping[str, str] | None = None,
    deny: DenyList | None = None,
) -> tuple[list[NightScore], list[Verdict]]:
    # Checked before any model call: a run that cannot be scored must cost nothing.
    if not any(detect(build_summary(n.inputs, settings), settings) for n in nights):
        raise BenchConfigError(NO_FINDING)
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


def _percent(value: float) -> str:
    """Never "100%" below 100%: 299/300 is 99.7%."""
    text = f"{value:.0%}"
    if value < 1 and text == "100%":
        text = f"{min(value, 0.999):.1%}"
    return text


def fmt(criterion: str, value: Any) -> str:
    """A figure as shown to a reader; "-" when unknown."""
    if criterion in TOKEN_FIGURES:
        total, known, attempts = value["total"], value["known"], value["attempts"]
        if total is None:
            return "-"
        return str(total) if known == attempts else f"{total}, partial ({known} of {attempts} attempts)"
    if value is None:
        return "-"
    if criterion in RATIO_CRITERIA:
        return _percent(value)
    if criterion in SECONDS_CRITERIA:
        return f"{value:.1f}"
    return f"{value:g}" if isinstance(value, float) else str(value)


def _ordered(criterion: str, value: float, threshold: float, shown: tuple[str, str]) -> bool:
    """The two texts compare as the two numbers do, and nothing below 100% reads 100%."""
    a, b = (float(t.removesuffix("%")) for t in shown)
    if criterion in RATIO_CRITERIA and value < 1 and a == 100:
        return False
    return (a > b) == (value > threshold) and (a < b) == (value < threshold)


def verdict_texts(v: Verdict) -> tuple[str, str]:
    """Value (with its counts) and threshold, with as many decimals as needed for the texts to compare as the
    numbers do: a failing value is never shown equal to its threshold, nor a passing one below it."""
    shown = fmt(v.criterion, v.value), fmt(v.criterion, v.threshold)
    kind = "%" if v.criterion in RATIO_CRITERIA else "f"
    decimals = 1
    # 17 decimals tell any two ratios or durations apart
    while not _ordered(v.criterion, v.value, v.threshold, shown) and decimals <= 17:
        shown = f"{v.value:.{decimals}{kind}}", f"{v.threshold:.{decimals}{kind}}"
        decimals += 1
    return shown[0] + (f" ({v.detail})" if v.detail else ""), shown[1]


def _plain_keys(keys: list[tuple[str, str]]) -> str:
    return ", ".join(f"{s}/{c}" for s, c in keys) or "-"


def _plain_actions(actions: list[ActionTriple]) -> str:
    return ", ".join(f"{k} {s}/{c}" for k, s, c in actions) or "-"


def mismatch_lines(scores: list[NightScore], deny: DenyList) -> list[str]:
    """One line per night the detector got wrong (its first wrong result; the report lists them all)."""
    out = []
    for s in scores:
        for numbers, keys, actions in s.mismatches()[:1]:
            attempts = ", ".join(map(str, numbers))
            line = (
                f"{s.night_id} (attempts {attempts}): "
                f"findings expected {_plain_keys(s.expected)} got {_plain_keys(keys)}; "
                f"actions expected {_plain_actions(s.expected_actions)} got {_plain_actions(actions)}"
            )
            out.append(f"MISMATCH {_withheld(line, deny)}")
    return out


def _keys(keys: list[tuple[str, str]]) -> str:
    return ", ".join(f"`{s}/{c}`" for s, c in keys) or "-"


def _actions(actions: list[ActionTriple]) -> str:
    return ", ".join(f"`{k} {s}/{c}`" for k, s, c in actions) or "-"


def _mismatch_section(scores: list[NightScore]) -> list[str]:
    rows = [
        f"| {s.night_id} | {', '.join(map(str, numbers))} | {_keys(s.expected)} | {_keys(keys)} "
        f"| {_actions(s.expected_actions)} | {_actions(actions)} |"
        for s in scores
        for numbers, keys, actions in s.mismatches()
    ]
    if not rows:
        return []
    return [
        "",
        "## Detector mismatches",
        "",
        "| Night | Attempts | Expected findings | Detected findings | Expected actions | Planned actions |",
        "|---|---|---|---|---|---|",
        *rows,
    ]


def _night_row(s: NightScore) -> str:
    cells = [s.night_id, s.origin, _keys(s.detected), "ok" if s.detector_exact else "NO"]
    if s.explainable:
        cells += [
            f"{s.explained}/{s.explainable}",
            f"{s.valid}/{len(s.attempts)}",
            f"{s.quotes_exact}/{s.quotes_total}",
            "YES" if s.fallback else "no",
            str(s.unknown_numbers),
            fmt(AGREEMENT_WORST, min(s.cause_agreement.values(), default=None)),
            f"{s.seconds_max:.1f}",
        ]
    else:
        cells += ["-"] * 7
    return "| " + " | ".join(cells) + " |"


def _markdown(payload: dict[str, Any], scores: list[NightScore], verdict_list: list[Verdict]) -> str:
    lines = [
        f"# Bench: {payload['engine']} / {payload['model']} ({payload['profile']})",
        "",
        f"Run at {payload['stamp']}. Profile `{payload['profile']}`, repeat {payload['repeat']}.",
        f"Model that answered: {', '.join(payload['models_used']) or '-'}.",
        "",
        "| Criterion | Value | Threshold | Pass |",
        "|---|---|---|---|",
    ]
    for v in verdict_list:
        passed = "yes" if v.passed else "NO"
        value, threshold = verdict_texts(v)
        lines.append(f"| {v.criterion} | {value} | {threshold} | {passed} |")
    lines += ["", "| Reported, no threshold | Value |", "|---|---|"]
    judged = {v.criterion for v in verdict_list}
    lines += [f"| {k} | {fmt(k, v)} |" for k, v in payload["reported"].items() if k not in judged]
    lines += [
        "",
        "Model figures run over every attempt of every night with a detected finding; the other nights show `-`.",
        "The cause agreement of a night is its worst finding's; an attempt without an explanation agrees with none.",
        "",
        "| Night | Origin | Findings | Detector | Explained | Valid | Quotes | Fallback | Unknown numbers "
        "| Cause agreement | Max s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [_night_row(s) for s in scores]
    lines += _mismatch_section(scores)
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
    _withhold_texts(nights, deny)
    payload = {
        "engine": engine.name,
        "model": engine.model,
        "models_used": sorted({a.model for s in scores for a in s.attempts if a.model}),
        "profile": profile,
        "stamp": now.isoformat(timespec="seconds"),
        "repeat": repeat,
        "settings": dataclasses.asdict(settings),
        "thresholds": dict(thresholds),
        "verdicts": _jsonable(verdict_list),
        "reported": reported(scores),
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
