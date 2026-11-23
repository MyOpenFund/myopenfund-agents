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


def _validated_or_trap(night: Night) -> bool:
    """A trap carries its own ground truth; any other night needs a human validation of its expected answer."""
    return night.meta.origin == "trap" or bool(night.meta.validated_by)


@pytest.mark.parametrize("night", NIGHTS, ids=IDS)
def test_night_is_trap_or_validated(night: Night) -> None:
    assert _validated_or_trap(night), f"{night.inputs.night_id}: expected answer not validated (meta.validated_by)"


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


def test_validation_rule_on_synthetic_nights():
    from conftest import night, src_run
    from myopenfund_agents.bench import traps
    from myopenfund_agents.bench.fixtures import Expected, Meta

    base = night(source_runs=[src_run("se")])
    pending = Night(inputs=base, expected=Expected(), meta=Meta(origin="incident", description="captured"))
    assert not _validated_or_trap(pending)
    validated = pending.model_copy(update={"meta": pending.meta.model_copy(update={"validated_by": "Marc"})})
    assert _validated_or_trap(validated)
    assert _validated_or_trap(traps.new_source(base))
