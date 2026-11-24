"""Scripted engine for tests: returns fixed explanations and records the prompts it was given."""

import time

from myopenfund_agents.core.explanations import Explanations
from myopenfund_agents.engines.base import EngineResult


class FakeEngine:
    name = "fake"

    def __init__(self, explanations: Explanations | BaseException | None, model: str = "scripted") -> None:
        self.explanations = explanations  # an exception instance is raised by `explain`
        self.model = model
        self.prompts: list[str] = []
        self.ids: list[frozenset[str]] = []  # the ids each `explain` call was given

    def explain(self, prompt: str, ids: frozenset[str]) -> EngineResult:
        start = time.monotonic()
        self.prompts.append(prompt)
        self.ids.append(ids)
        if isinstance(self.explanations, BaseException):
            raise self.explanations
        return EngineResult(self.name, self.model, time.monotonic() - start, explanations=self.explanations)
