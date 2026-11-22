"""Bench scoring: per-night quality from engine attempts, and pass/fail verdicts against thresholds."""

import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from myopenfund_agents.bench.fixtures import Night

Keys = set[tuple[str, str]]
ActionTriple = tuple[str, str, str]

OUTAGE = ("*", "general_outage")
_REQUIRED = ("min_recall", "max_false_alarms", "min_actions_ok", "min_valid", "min_agreement", "min_grouping")
_OPTIONAL = ("max_seconds_per_night",)


@dataclass
class Attempt:
    """One engine run on a night. `keys is None` means no valid output."""

    keys: Keys | None
    actions: list[ActionTriple] | None
    seconds: float
    retries: int = 0
    error: str | None = None


@dataclass
class NightScore:
    night_id: str
    origin: str
    expected: list[tuple[str, str]]
    found: list[tuple[str, str]]
    extra: list[tuple[str, str]]
    false_alarms: int
    actions_ok: bool | None
    valid: bool
    agreement: float
    seconds_max: float
    attempts: list[Attempt] = field(default_factory=list)


@dataclass
class Verdict:
    criterion: str
    value: float
    threshold: float
    passed: bool


def score_night(night: Night, attempts: list[Attempt]) -> NightScore:
    if not attempts:
        raise ValueError(f"night {night.inputs.night_id!r}: no attempts to score")
    expected = night.expected.keys()
    first = attempts[0]
    first_keys = first.keys
    if first_keys is None:
        found: list[tuple[str, str]] = []
        extra: list[tuple[str, str]] = []
    else:
        found = sorted(expected & first_keys)
        extra = sorted(first_keys - expected)
    origin = night.meta.origin
    actions_ok: bool | None = None
    if first_keys is not None and first_keys == expected:
        want = sorted((a.kind, a.source_code, a.category.value) for a in night.expected.actions)
        actions_ok = sorted(first.actions or []) == want
    outcomes = Counter(None if a.keys is None else frozenset(a.keys) for a in attempts)
    agreement = max(outcomes.values()) / len(attempts)
    return NightScore(
        night_id=night.inputs.night_id,
        origin=origin,
        expected=sorted(expected),
        found=found,
        extra=extra,
        false_alarms=len(extra) if not expected else 0,
        actions_ok=actions_ok,
        valid=all(a.keys is not None for a in attempts),
        agreement=agreement,
        seconds_max=max((a.seconds for a in attempts), default=0.0),
        attempts=list(attempts),
    )


def load_thresholds(path: Path, profile: str) -> dict[str, Any]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if profile not in data:
        raise ValueError(f"unknown threshold profile {profile!r} in {path}")
    t = data[profile]
    unknown = sorted(set(t) - set(_REQUIRED) - set(_OPTIONAL))
    missing = sorted(set(_REQUIRED) - set(t))
    if unknown or missing:
        raise ValueError(f"profile {profile!r}: unknown keys {unknown}, missing keys {missing}")
    for k, v in t.items():
        if isinstance(v, bool) or not isinstance(v, int | float):
            raise ValueError(f"profile {profile!r}: {k} must be a number, got {v!r}")
    return dict(t)


def verdicts(scores: list[NightScore], t: dict[str, Any]) -> list[Verdict]:
    incidents = [s for s in scores if s.origin != "green"]
    expected_n = sum(len(s.expected) for s in incidents)
    recall = sum(len(s.found) for s in incidents) / expected_n if expected_n else 1.0
    alarms = sum(s.false_alarms for s in scores)
    judged = [s for s in scores if s.actions_ok is not None]
    actions = sum(1 for s in judged if s.actions_ok) / len(judged) if judged else 1.0
    outage = [s for s in scores if OUTAGE in s.expected]
    # an invalid first attempt has found == [] and so never equals a non-empty expected set
    exact = sum(1 for s in outage if s.found == s.expected and not s.extra)
    grouped = exact / len(outage) if outage else 1.0
    valid = sum(1 for s in scores if s.valid) / len(scores) if scores else 1.0
    agreement = min((s.agreement for s in scores), default=1.0)
    out = [
        Verdict("problems found", recall, t["min_recall"], recall >= t["min_recall"]),
        Verdict("false alarms on green nights", alarms, t["max_false_alarms"], alarms <= t["max_false_alarms"]),
        Verdict("actions as expected", actions, t["min_actions_ok"], actions >= t["min_actions_ok"]),
        Verdict("general outage grouped", grouped, t["min_grouping"], grouped >= t["min_grouping"]),
        Verdict("valid output", valid, t["min_valid"], valid >= t["min_valid"]),
        Verdict("repeat agreement (worst night)", agreement, t["min_agreement"], agreement >= t["min_agreement"]),
    ]
    if "max_seconds_per_night" in t:
        slowest = max((s.seconds_max for s in scores), default=0.0)
        limit = t["max_seconds_per_night"]
        out.append(Verdict("slowest night (s)", slowest, limit, slowest <= limit))
    return out
