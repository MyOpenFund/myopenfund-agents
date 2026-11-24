"""Engine interface: the detected findings' prompt in, explanations out."""

from dataclasses import dataclass
from typing import Protocol

from myopenfund_agents.core.explanations import Explanations


class EngineError(RuntimeError):
    """The engine could not produce valid explanations (unreachable, invalid output, misconfigured)."""


@dataclass(frozen=True)
class EngineResult:
    engine: str
    model: str
    seconds: float  # all attempts, retries included
    retries: int = 0
    input_tokens: int | None = None  # all attempts (claude-code adds them up; pydantic-ai reports the run's usage)
    output_tokens: int | None = None
    explanations: Explanations | None = None


class Engine(Protocol):
    name: str
    model: str

    def explain(self, prompt: str, ids: frozenset[str]) -> EngineResult: ...
