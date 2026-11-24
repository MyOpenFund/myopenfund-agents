"""Scripted engine for tests: returns fixed findings and records the prompts it would send."""

import time

from myopenfund_agents.core.explanations import Explanations
from myopenfund_agents.core.findings import Findings
from myopenfund_agents.core.prompt import user_prompt
from myopenfund_agents.engines.base import EngineResult


class FakeEngine:
    name = "fake"

    def __init__(
        self, findings: Findings, model: str = "scripted", explanations: Explanations | BaseException | None = None
    ) -> None:
        self.findings = findings
        self.explanations = explanations  # an exception instance is raised by `explain`
        self.model = model
        self.prompts: list[str] = []
        self.ids: list[frozenset[str]] = []  # the ids each `explain` call was given

    def diagnose(self, summary_text: str) -> EngineResult:
        start = time.monotonic()
        self.prompts.append(user_prompt(summary_text))
        return EngineResult(self.findings, self.name, self.model, time.monotonic() - start)

    def explain(self, prompt: str, ids: frozenset[str]) -> EngineResult:
        start = time.monotonic()
        self.prompts.append(prompt)
        self.ids.append(ids)
        if isinstance(self.explanations, BaseException):
            raise self.explanations
        return EngineResult(
            Findings(findings=[]), self.name, self.model, time.monotonic() - start, explanations=self.explanations
        )
