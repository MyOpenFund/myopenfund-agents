"""Engine interface: a summary text in, schema-validated findings out."""

from dataclasses import dataclass
from typing import Protocol

from myopenfund_agents.core.findings import Findings


class EngineError(RuntimeError):
    """The engine could not produce valid findings (unreachable, invalid output, misconfigured)."""


@dataclass(frozen=True)
class EngineResult:
    findings: Findings
    engine: str
    model: str
    seconds: float
    retries: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None


class Engine(Protocol):
    name: str
    model: str

    def diagnose(self, summary_text: str) -> EngineResult: ...
