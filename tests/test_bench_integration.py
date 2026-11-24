"""Every trap through `run_bench` with an oracle engine: the pipeline and the scorer agree in both profiles."""

from pathlib import Path

import pytest

from conftest import night, src_run
from myopenfund_agents.bench import traps
from myopenfund_agents.bench.fixtures import Expected, Meta, Night
from myopenfund_agents.bench.runner import DEFAULT_CORPUS_REPOS, run_bench
from myopenfund_agents.bench.score import load_thresholds
from myopenfund_agents.core.detector import detect
from myopenfund_agents.core.explanations import Explanation, Explanations, Quote
from myopenfund_agents.core.prompt import explain_prompt
from myopenfund_agents.core.summary import build_summary
from myopenfund_agents.engines.base import EngineResult
from myopenfund_agents.settings import A1Settings

THRESHOLDS = Path(__file__).resolve().parents[1] / "bench" / "thresholds.toml"
CODES = ("a1", "b2", "c3", "d4", "e5", "f6")
S = A1Settings()


class OracleEngine:
    """Explains each night's findings with a quote of the summary's first line."""

    name = "oracle"
    model = "expected"

    def __init__(self, nights: list[Night]) -> None:
        self.answers: dict[str, Explanations] = {}
        for n in nights:
            summary = build_summary(n.inputs, S)
            detected = detect(summary, S)
            self.answers[explain_prompt(summary.text, detected)] = Explanations(
                explanations=[
                    Explanation(
                        finding_id=d.finding_id, diagnosis="Oracle diagnosis.", likely_cause=d.finding.category,
                        severity=d.finding.severity, quotes=[Quote(source="summary", text=summary.text.splitlines()[0])],
                    )
                    for d in detected
                ],
                observations=[],
            )  # fmt: skip

    def explain(self, prompt: str, ids: frozenset[str]) -> EngineResult:
        answer = self.answers[prompt]
        assert ids == {e.finding_id for e in answer.explanations}
        return EngineResult(self.name, self.model, 0.01, explanations=answer)


def _nights() -> list[Night]:
    history = [src_run(c, night=n, seen=100) for n in range(1, 15) for c in CODES]
    base = night(source_runs=[src_run(c, seen=100) for c in CODES], history=history)
    green = Night(inputs=base, expected=Expected(), meta=Meta(origin="green", description="quiet night"))
    return [
        green,
        traps.volume_drop(base, "a1"),
        traps.bad_content(base, "b2"),
        traps.general_outage(base),
        traps.new_source(base),
        traps.known_problem(base, "c3", DEFAULT_CORPUS_REPOS["central-bank"]),
        traps.no_run(base),
    ]


@pytest.mark.parametrize("profile", ["prod", "local"])
def test_every_trap_passes_with_an_oracle(profile):
    nights = _nights()
    assert len({n.inputs.night_id for n in nights}) == len(nights)
    scores, verdicts = run_bench(nights, OracleEngine(nights), S, 3, load_thresholds(THRESHOLDS, profile))
    failed = [v for v in verdicts if not v.passed]
    assert not failed, failed
    by_id = {s.night_id: s for s in scores}
    for n in nights:
        s = by_id[n.inputs.night_id]
        assert s.detector_exact and not s.fallback, s
        assert s.detected == sorted(n.expected.keys()), s
        assert s.valid == (3 if n.expected.findings else 0), s  # a quiet night asks no model
        assert set(s.cause_agreement.values()) <= {1.0}, s
    outage = by_id[f"{nights[0].inputs.night_id}--general-outage"]
    assert [f["category"] for f in outage.attempts[0].findings] == ["general_outage"]
    no_run = by_id[f"{nights[0].inputs.night_id}--no-run"]
    assert no_run.attempts[0].actions == [("new_issue", "*", "unknown")]
