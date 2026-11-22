"""Every committed bench night is clean and its expected answer is consistent with the pipeline."""

from pathlib import Path

import pytest

from myopenfund_agents.bench.fixtures import Night, load_all
from myopenfund_agents.bench.runner import DEFAULT_CORPUS_REPOS, run_night
from myopenfund_agents.core.findings import Finding, Findings, Severity
from myopenfund_agents.engines.fake import FakeEngine
from myopenfund_agents.sanitise import DenyList, check_tree
from myopenfund_agents.settings import A1Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
NIGHTS = load_all(REPO_ROOT / "bench" / "nights")
IDS = [n.inputs.night_id for n in NIGHTS]


@pytest.mark.parametrize("night", NIGHTS, ids=IDS)
def test_night_passes_sanitiser(night: Night) -> None:
    assert check_tree(night, DenyList()) == []


@pytest.mark.parametrize("night", NIGHTS, ids=IDS)
def test_expected_answer_yields_expected_actions(night: Night) -> None:
    findings = Findings(
        findings=[
            Finding(
                source_code=k.source_code,
                category=k.category,
                severity=Severity.WARN,
                title=f"{k.source_code} {k.category.value}",
                diagnosis="Scripted diagnosis from the expected answer.",
                evidence=["scripted evidence"],
            )
            for k in night.expected.findings
        ]
    )
    score = run_night(night, FakeEngine(findings), A1Settings(), repeat=1, corpus_repos=DEFAULT_CORPUS_REPOS)
    assert score.valid
    assert score.actions_ok is True
