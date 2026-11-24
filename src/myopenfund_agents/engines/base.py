"""Engine interface: a summary text in, schema-validated findings out."""

from dataclasses import dataclass
from typing import Protocol

from myopenfund_agents.core.explanations import Explanations
from myopenfund_agents.core.findings import Findings


class EngineError(RuntimeError):
    """The engine could not produce valid findings (unreachable, invalid output, misconfigured)."""


@dataclass(frozen=True)
class EngineResult:
    findings: Findings
    engine: str
    model: str
    seconds: float  # all attempts, retries included
    retries: int = 0
    input_tokens: int | None = None  # all attempts (claude-code adds them up; pydantic-ai reports the run's usage)
    output_tokens: int | None = None
    explanations: Explanations | None = None  # set by `explain`; `findings` is then an empty placeholder


class Engine(Protocol):
    name: str
    model: str

    def diagnose(self, summary_text: str) -> EngineResult: ...

    def explain(self, prompt: str, ids: frozenset[str]) -> EngineResult: ...
