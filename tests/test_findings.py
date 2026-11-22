import pytest
from pydantic import ValidationError

from myopenfund_agents.core.findings import Finding, Findings
from myopenfund_agents.core.inputs import NightInputs


def _finding(category: str = "unreadable_pdf") -> dict:
    return {
        "source_code": "se",
        "category": category,
        "severity": "warn",
        "title": "PDF unreadable",
        "diagnosis": "Scanned PDFs.",
        "evidence": ["3 failures"],
    }


def test_findings_keys():
    findings = Findings(findings=[Finding(**_finding())])
    assert findings.keys() == {("se", "unreadable_pdf")}


def test_unknown_category_rejected():
    with pytest.raises(ValidationError):
        Finding(**_finding("weird"))


def test_empty_is_valid():
    assert Findings().findings == []
    assert Findings().keys() == set()


def test_inputs_keep_unknown_columns():
    inputs = NightInputs.model_validate(
        {
            "night_id": "2026-11-21",
            "night_start": "2026-11-21T00:00:00Z",
            "night_end": "2026-11-21T08:00:00Z",
            "runs": [],
            "source_runs": [{"run_id": "r1", "tool": "t", "command": "c", "source_code": "se", "new_col": 1}],
            "history": [],
            "discovery_errors": [],
            "open_issues": [],
        }
    )
    assert inputs.source_runs[0].model_extra == {"new_col": 1}
    assert inputs.cadence is None
