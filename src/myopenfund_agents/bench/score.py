"""Bench scoring: the detector against the expected answers, the model's explanations against thresholds."""

import math
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from myopenfund_agents.bench.fixtures import Night

ActionTriple = tuple[str, str, str]

DETECTOR_EXACT = "detector exact"
EXPLAINED = "findings explained"
VALID_OUTPUT = "valid output"
QUOTES_EXACT = "quotes exact (of quotes given)"
FALLBACK_NIGHTS = "fallback nights"
SLOWEST_NIGHT = "slowest night (s)"
UNKNOWN_NUMBERS = "unknown numbers"
AGREEMENT_MEAN = "likely_cause agreement (mean)"
AGREEMENT_WORST = "likely_cause agreement (worst)"
SECONDS_MEAN = "seconds (mean)"
SECONDS_WORST = "seconds (worst)"
INPUT_TOKENS = "input tokens"
OUTPUT_TOKENS = "output tokens"
TOKEN_FIGURES = {INPUT_TOKENS: "input_tokens", OUTPUT_TOKENS: "output_tokens"}
RATIO_CRITERIA = frozenset({DETECTOR_EXACT, EXPLAINED, VALID_OUTPUT, QUOTES_EXACT, AGREEMENT_MEAN, AGREEMENT_WORST})
SECONDS_CRITERIA = frozenset({SLOWEST_NIGHT, SECONDS_MEAN, SECONDS_WORST})

NO_FINDING = "no detected finding in the selected nights: nothing to explain, no model criterion can be scored"

_REQUIRED = ("min_explained", "min_valid", "min_quotes_exact")
_OPTIONAL = ("max_fallback_nights", "max_seconds_per_night")


class BenchConfigError(ValueError):
    """The selected nights cannot score a model."""


@dataclass
class Attempt:
    """One triage of a night.

    `findings` and `actions` are the detector's (no model). `valid` says whether the engine's answer was usable,
    None when the engine was not called. `explanations` are the ones kept, `dropped` the reasons of the others
    (finding id -> reason). Tokens are None when the engine does not report them.
    """

    findings: list[dict[str, Any]]
    actions: list[ActionTriple]
    seconds: float
    valid: bool | None = None
    explanations: list[dict[str, Any]] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    quotes_total: int = 0
    quotes_exact: int = 0
    unknown_numbers: int = 0
    retries: int = 0
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None  # the model that answered, as the engine reports it

    def keys(self) -> list[tuple[str, str]]:
        return sorted((f["source_code"], f["category"]) for f in self.findings)

    def causes(self) -> dict[str, str]:
        return {e["finding_id"]: e["likely_cause"] for e in self.explanations}


@dataclass
class NightScore:
    """Sums over every attempt of the night. `explainable` is 0 on a night without finding: it is in no ratio."""

    night_id: str
    origin: str
    expected: list[tuple[str, str]]
    expected_actions: list[ActionTriple]
    detected: list[tuple[str, str]]
    detector_exact: bool
    explained: int
    explainable: int
    valid: int
    quotes_total: int
    quotes_exact: int
    unknown_numbers: int
    fallback: bool  # some attempt sent a finding out unexplained
    cause_agreement: dict[str, float]  # finding id -> share of the attempts giving its top cause; empty below 2
    seconds_max: float
    attempts: list[Attempt] = field(default_factory=list)

    def mismatches(self) -> list[tuple[list[int], list[tuple[str, str]], list[ActionTriple]]]:
        """(attempt numbers, detected keys, actions) of each wrong detector result, over every attempt."""
        out: list[tuple[list[int], list[tuple[str, str]], list[ActionTriple]]] = []
        for number, attempt in enumerate(self.attempts, 1):
            keys, actions = attempt.keys(), sorted(attempt.actions)
            if keys == self.expected and actions == self.expected_actions:
                continue
            for numbers, seen_keys, seen_actions in out:
                if (seen_keys, seen_actions) == (keys, actions):
                    numbers.append(number)
                    break
            else:
                out.append(([number], keys, actions))
        return out


@dataclass
class Verdict:
    criterion: str
    value: float
    threshold: float
    passed: bool
    detail: str = ""  # the counts behind a ratio


def _agreement(attempts: list[Attempt]) -> dict[str, float]:
    """An attempt that did not explain a finding agrees with no other attempt, not even with another failed one."""
    out: dict[str, float] = {}
    if len(attempts) < 2:
        return out  # nothing to agree with
    for finding in attempts[0].findings:
        fid = finding["finding_id"]
        causes = Counter(a.causes()[fid] for a in attempts if fid in a.causes())
        out[fid] = max(causes.values(), default=0) / len(attempts)
    return out


def score_night(night: Night, attempts: list[Attempt]) -> NightScore:
    if not attempts:
        raise ValueError(f"night {night.inputs.night_id!r}: no attempts to score")
    expected = sorted(night.expected.keys())
    want = sorted((a.kind, a.source_code, a.category.value) for a in night.expected.actions)
    asked = [a for a in attempts if a.findings]
    score = NightScore(
        night_id=night.inputs.night_id,
        origin=night.meta.origin,
        expected=expected,
        expected_actions=want,
        detected=attempts[0].keys(),
        detector_exact=True,
        explained=sum(len(a.explanations) for a in asked),
        explainable=sum(len(a.findings) for a in asked),
        valid=sum(a.valid is True for a in asked),
        quotes_total=sum(a.quotes_total for a in asked),
        quotes_exact=sum(a.quotes_exact for a in asked),
        unknown_numbers=sum(a.unknown_numbers for a in asked),
        fallback=any(len(a.explanations) < len(a.findings) for a in asked),
        cause_agreement=_agreement(attempts),
        seconds_max=max(a.seconds for a in attempts),
        attempts=list(attempts),
    )
    score.detector_exact = not score.mismatches()
    return score


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
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise ValueError(f"profile {profile!r}: {k} must be a number, got {v!r}")
    return dict(t)


def _asked(scores: list[NightScore]) -> list[NightScore]:
    """The nights the model criteria run over: at least one detected finding."""
    return [s for s in scores if s.explainable]


def _ratio(criterion: str, part: int, whole: int, threshold: float) -> Verdict:
    value = part / whole if whole else 0.0
    return Verdict(criterion, value, threshold, value >= threshold, f"{part}/{whole}")


def verdicts(scores: list[NightScore], t: dict[str, Any]) -> list[Verdict]:
    asked = _asked(scores)
    if not asked:
        raise BenchConfigError(NO_FINDING)

    def total(name: str) -> int:
        return sum(getattr(s, name) for s in asked)

    out = [
        _ratio(DETECTOR_EXACT, sum(s.detector_exact for s in scores), len(scores), 1.0),
        _ratio(EXPLAINED, total("explained"), total("explainable"), t["min_explained"]),
        _ratio(VALID_OUTPUT, total("valid"), sum(len(s.attempts) for s in asked), t["min_valid"]),
        # no quote at all is 0, never 100%
        _ratio(QUOTES_EXACT, total("quotes_exact"), total("quotes_total"), t["min_quotes_exact"]),
    ]
    if "max_fallback_nights" in t:
        nights, limit = total("fallback"), t["max_fallback_nights"]
        out.append(Verdict(FALLBACK_NIGHTS, nights, limit, nights <= limit))
    if "max_seconds_per_night" in t:
        slowest, limit = max(s.seconds_max for s in asked), t["max_seconds_per_night"]
        out.append(Verdict(SLOWEST_NIGHT, slowest, limit, slowest <= limit))
    return out


def reported(scores: list[NightScore]) -> dict[str, Any]:
    """Figures without a threshold, over the same attempts as the model criteria; None when unknown."""
    asked = _asked(scores)
    attempts = [a for s in asked for a in s.attempts]
    agreement = [v for s in asked for v in s.cause_agreement.values()]
    seconds = [a.seconds for a in attempts]

    def tokens(name: str) -> dict[str, int | None]:
        known = [v for a in attempts if (v := getattr(a, name)) is not None]
        return {"total": sum(known) if known else None, "known": len(known), "attempts": len(attempts)}

    return {
        FALLBACK_NIGHTS: sum(s.fallback for s in asked),
        UNKNOWN_NUMBERS: sum(s.unknown_numbers for s in asked),
        AGREEMENT_MEAN: sum(agreement) / len(agreement) if agreement else None,
        AGREEMENT_WORST: min(agreement, default=None),
        SECONDS_MEAN: sum(seconds) / len(seconds) if seconds else None,
        SECONDS_WORST: max(seconds, default=None),
        **{figure: tokens(name) for figure, name in TOKEN_FIGURES.items()},
    }
