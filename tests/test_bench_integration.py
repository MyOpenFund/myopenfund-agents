"""Every trap through `run_bench` with an oracle engine: the pipeline and the scorer agree in both profiles."""

from pathlib import Path

import pytest

from conftest import night, src_run
from myopenfund_agents.bench import traps
from myopenfund_agents.bench.fixtures import Expected, Meta, Night
from myopenfund_agents.bench.runner import DEFAULT_CORPUS_REPOS, run_bench
from myopenfund_agents.bench.score import load_thresholds
from myopenfund_agents.core.findings import Category, Finding, Findings, Severity
from myopenfund_agents.core.summary import build_summary
from myopenfund_agents.engines.base import EngineResult
from myopenfund_agents.settings import A1Settings

THRESHOLDS = Path(__file__).resolve().parents[1] / "bench" / "thresholds.toml"
CODES = ("a1", "b2", "c3", "d4", "e5", "f6")
S = A1Settings()


def _finding(code: str, category: Category) -> Finding:
    return Finding(
        source_code=code, category=category, severity=Severity.ERROR, title=f"{code} {category.value}",
        diagnosis="Oracle diagnosis.", evidence=["oracle evidence"],
    )  # fmt: skip


class OracleEngine:
    """Answers each night's summary with exactly the expected findings (per-source for the outage trap)."""

    name = "oracle"
    model = "expected"

    def __init__(self, nights: list[Night]) -> None:
        self.answers: dict[str, Findings] = {}
        for n in nights:
            keys = n.expected.keys()
            if keys == {("*", "general_outage")}:
                # As a model would: one site-down finding per failing source, grouped by the deterministic layer.
                found = [_finding(r.source_code, Category.SOURCE_SITE_DOWN) for r in n.inputs.source_runs]
            else:
                found = [_finding(k.source_code, k.category) for k in n.expected.findings]
            self.answers[build_summary(n.inputs, S).text] = Findings(findings=found)

    def diagnose(self, summary_text: str) -> EngineResult:
        return EngineResult(self.answers[summary_text], self.name, self.model, 0.01)


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
        assert s.valid and s.agreement == 1.0 and s.extra == [], s
        assert s.found == sorted(n.expected.keys()), s
        if n.expected.findings:
            assert s.actions_ok is True, s
    outage = by_id[f"{nights[0].inputs.night_id}--general-outage"]
    assert len(outage.attempts[0].findings) == len(CODES)  # raw findings kept before grouping
    no_run = by_id[f"{nights[0].inputs.night_id}--no-run"]
    assert no_run.attempts[0].actions == [("new_issue", "*", "unknown")]
