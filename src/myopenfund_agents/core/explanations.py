"""What the model returns: an explanation per finding the detector made. The code checks it, never trusts it."""

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from myopenfund_agents.core.actions import SEVERITY_RANK
from myopenfund_agents.core.detector import Detected
from myopenfund_agents.core.findings import Category, Finding, Severity
from myopenfund_agents.core.prompt import shown_evidence, shown_summary
from myopenfund_agents.sanitise import DenyList, violations, withheld
from myopenfund_agents.settings import A1Settings

DIAGNOSIS_MAX = 2000
FALSE_QUOTE = "false quote"
NO_QUOTE = "no quote"
WITHHELD = "withheld"

_SPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
EMPTY = "empty"
_PLAIN = re.compile(r"\d+(?:\.\d+)?")


class Quote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["summary"]
    text: str


class Explanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    diagnosis: str = Field(max_length=DIAGNOSIS_MAX)
    likely_cause: Category
    severity: Severity
    quotes: list[Quote]  # the caps are settings, see id_error


class Explanations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanations: list[Explanation]  # required: a bare {} is an invalid answer
    observations: list[str]


@dataclass(frozen=True)
class Checked:
    kept: dict[str, Explanation] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)
    # What the model said in the dropped ones, each text with an infra detail withheld. For reports, never an Action.
    dropped_copies: dict[str, Explanation] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    quotes_total: int = 0  # quotes given; an explanation returned without any counts as one inexact quote
    quotes_exact: int = 0
    unknown_numbers: int = 0


def id_error(e: Explanations, ids: frozenset[str], settings: A1Settings) -> str | None:
    """A reason to ask the model again, or None. A missing id is fine: that finding stays unexplained.

    Run it before `check`, which drops what is out of limits but never asks again."""
    seen: set[str] = set()
    for x in e.explanations:
        if x.finding_id not in ids:
            # Never echo the id: it is the model's own text, and this message goes back to the model.
            return f"an explanation uses an id that is not one of: {', '.join(sorted(ids)) or 'none'}"
        if x.finding_id in seen:
            return f"finding_id {x.finding_id!r} explained more than once"
        seen.add(x.finding_id)
        if not x.diagnosis.strip():
            return f"{x.finding_id}: empty diagnosis"
        if len(x.quotes) > settings.quotes_max:
            return f"{x.finding_id}: at most {settings.quotes_max} quotes per explanation"
    if len(e.observations) > settings.observations_max:
        return f"at most {settings.observations_max} observations"
    for o in e.observations:
        if len(o) > settings.observation_max_chars:
            return f"an observation is longer than {settings.observation_max_chars} characters"
    return None


def _collapse(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def _readings(text: str) -> dict[str, set[str]]:
    """Each number token read merged ("1,234" is 1234) -> its split readings (1 and 234), over all its spellings."""
    out: dict[str, set[str]] = {}
    for n in _NUMBER.findall(text):
        out.setdefault(n.replace(",", ""), set()).update(_PLAIN.findall(n))
    return out


def _known_numbers(text: str) -> set[str]:
    known: set[str] = set()
    for merged, parts in _readings(text).items():
        known.add(merged)
        known.update(parts)
    return known


def _unknown_numbers(text: str, known: set[str]) -> int:
    """Tokens that neither reading (merged or split) of the shown texts explains, each counted once."""
    return sum(
        1 for merged, parts in _readings(text).items() if merged not in known and not all(p in known for p in parts)
    )


def _safe_copy(e: Explanation, deny: DenyList) -> Explanation:
    quotes = [q.model_copy(update={"text": withheld(q.text, deny)}) for q in e.quotes]
    return e.model_copy(update={"diagnosis": withheld(e.diagnosis, deny), "quotes": quotes})


def check(e: Explanations, detected: list[Detected], summary_text: str, settings: A1Settings, deny: DenyList) -> Checked:
    """Keep the explanations whose quotes are really in the summary or in an evidence line, if the sanitiser agrees."""
    ids = {d.finding_id for d in detected}
    # What the model was shown. One text per evidence line: a quote never spans two of them.
    texts = [shown_summary(summary_text), *(line for d in detected for line in shown_evidence(d))]
    haystacks = [_collapse(t) for t in texts]
    known = set().union(*(_known_numbers(t) for t in texts))
    id_words = re.compile(r"\b(?:" + "|".join(map(re.escape, ids)) + r")\b") if ids else None
    kept: dict[str, Explanation] = {}
    dropped: dict[str, str] = {}
    copies: dict[str, Explanation] = {}
    total = exact = unknown = 0
    for x in e.explanations:
        if x.finding_id not in ids or x.finding_id in kept or x.finding_id in dropped:
            continue
        prose = id_words.sub(" ", x.diagnosis) if id_words else x.diagnosis
        unknown += _unknown_numbers(prose, known)
        good = 0
        total += max(1, len(x.quotes))
        for q in x.quotes:
            text = _collapse(q.text)
            if len(text) >= settings.quote_min_chars and any(text in h for h in haystacks):
                good += 1
        exact += good
        if not x.diagnosis.strip():
            dropped[x.finding_id] = EMPTY
        elif not x.quotes:
            dropped[x.finding_id] = NO_QUOTE
        elif good < len(x.quotes):
            dropped[x.finding_id] = FALSE_QUOTE
        elif any(violations(t, deny) for t in [x.diagnosis, *(q.text for q in x.quotes)]):
            dropped[x.finding_id] = WITHHELD
        else:
            kept[x.finding_id] = x
        if x.finding_id in dropped:
            copies[x.finding_id] = _safe_copy(x, deny)
    observations = [o for o in e.observations if len(o) <= settings.observation_max_chars and not violations(o, deny)][
        : settings.observations_max
    ]
    return Checked(kept, dropped, copies, observations, total, exact, unknown)


def apply(d: Detected, e: Explanation | None) -> Finding:
    """The detector's finding with the model's words on it. Source, category and evidence never change."""
    if e is None:
        return d.finding
    line = f"\n\nLikely cause: {e.likely_cause.value}"
    diagnosis = e.diagnosis[: DIAGNOSIS_MAX - len(line)] + line
    severity = max(d.finding.severity, e.severity, key=SEVERITY_RANK.__getitem__)
    return d.finding.model_copy(update={"diagnosis": diagnosis, "severity": severity})
