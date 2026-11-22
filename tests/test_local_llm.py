"""Smoke test against a real local model: only schema-valid output is asserted, never its content.

Runs only with `-m local_llm` and `MOF_AGENTS_TEST_MODEL` set (CI sets it; no paid API is involved).
"""

import os

import pytest

from conftest import night, src_run
from myopenfund_agents.bench import traps
from myopenfund_agents.core.findings import Findings
from myopenfund_agents.core.summary import build_summary
from myopenfund_agents.engines.pydantic_ai_engine import PydanticAIEngine, build_model
from myopenfund_agents.settings import A1Settings

pytestmark = pytest.mark.local_llm

MODEL = os.environ.get("MOF_AGENTS_TEST_MODEL")
BASE_URL = os.environ.get("MOF_AGENTS_TEST_BASE_URL", "http://localhost:11434/v1")

if not MODEL:
    pytest.skip("MOF_AGENTS_TEST_MODEL is not set", allow_module_level=True)


def _green(history):
    rows = [src_run("us", seen=150), src_run("fr", seen=10), src_run("se", seen=10)]
    return night(source_runs=rows, history=history)


@pytest.mark.parametrize("case", ["green", "bad_content"])
def test_local_model_returns_valid_findings(case: str, green_history) -> None:
    base = _green(green_history)
    inputs = base if case == "green" else traps.bad_content(base, "se").inputs
    settings = A1Settings.from_env()
    engine = PydanticAIEngine(build_model(MODEL, BASE_URL, None), settings)
    result = engine.diagnose(build_summary(inputs, settings).text)
    assert isinstance(result.findings, Findings)
