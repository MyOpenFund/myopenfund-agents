"""Scripted engine for tests: returns fixed findings and records the prompts it would send."""

import time

from myopenfund_agents.core.findings import Findings
from myopenfund_agents.core.prompt import user_prompt
from myopenfund_agents.engines.base import EngineResult


class FakeEngine:
    name = "fake"

    def __init__(self, findings: Findings, model: str = "scripted") -> None:
        self.findings = findings
        self.model = model
        self.prompts: list[str] = []

    def diagnose(self, summary_text: str) -> EngineResult:
        start = time.monotonic()
        self.prompts.append(user_prompt(summary_text))
        return EngineResult(self.findings, self.name, self.model, time.monotonic() - start)
