from myopenfund_agents.core.actions import Action, Kind, fingerprint, group_outages, plan_actions
from myopenfund_agents.core.findings import Category, Finding, Findings, Severity
from myopenfund_agents.core.inputs import OpenIssue
from myopenfund_agents.settings import A1Settings

S = A1Settings()


def _f(code: str, cat: Category = Category.SOURCE_SITE_DOWN, sev: Severity = Severity.WARN) -> Finding:
    return Finding(
        source_code=code, category=cat, severity=sev, title=f"{code} {cat.value}", diagnosis="d", evidence=["e"]
    )


def test_fingerprint_stable():
    assert fingerprint("r", _f("se")) == "a1:r:se:source_site_down"


def test_new_issue_then_comment():
    fs = Findings(findings=[_f("se")])
    repo_of = {"se": "central-bank-corpus"}
    acts = plan_actions(fs, [], repo_of, "default", S)
    assert [(a.kind, a.repo, a.issue_number) for a in acts] == [(Kind.NEW_ISSUE, "central-bank-corpus", None)]
    fp = acts[0].fingerprint
    assert fp == "a1:central-bank-corpus:se:source_site_down"
    acts = plan_actions(fs, [OpenIssue(fingerprint=fp, number=7, repo="central-bank-corpus")], repo_of, "default", S)
    assert [(a.kind, a.issue_number, a.repo) for a in acts] == [(Kind.COMMENT, 7, "central-bank-corpus")]
    assert isinstance(acts[0], Action)


def test_general_outage_grouping():
    fs = Findings(findings=[_f(f"s{i}") for i in range(6)])
    out = group_outages(fs, S)
    assert [(f.source_code, f.category) for f in out.findings] == [("*", Category.GENERAL_OUTAGE)]
    assert out.findings[0].severity == Severity.ERROR
    assert out.findings[0].evidence == ["sources: s0, s1, s2, s3, s4, s5"]
    assert group_outages(out, S) == out


def test_five_sources_stay_separate():
    fs = Findings(findings=[_f(f"s{i}") for i in range(5)])
    assert len(group_outages(fs, S).findings) == 5


def test_several_general_outages_collapse():
    fs = Findings(findings=[_f("*", Category.GENERAL_OUTAGE), _f("x"), _f("y", Category.GENERAL_OUTAGE)])
    out = group_outages(fs, S)
    assert [f.source_code for f in out.findings] == ["*", "x"]


def test_duplicate_fingerprints_collapse():
    acts = plan_actions(Findings(findings=[_f("se"), _f("se")]), [], {}, "d", S)
    assert len(acts) == 1 and acts[0].repo == "d"


def test_summary_issue_above_cap():
    cats = list(Category)[:4] + [Category.UNKNOWN]
    fs = Findings(
        findings=[_f(f"s{i}", cats[i % 5]) for i in range(5)]
        + [_f("t", Category.PARSE_ERROR, Severity.ERROR), _f("u", Category.UNREADABLE_PDF)]
    )
    existing = OpenIssue(fingerprint="a1:d:u:unreadable_pdf", number=3, repo="d")
    acts = plan_actions(fs, [existing], {}, "d", S)
    assert [a.kind for a in acts] == [Kind.COMMENT, Kind.SUMMARY_ISSUE]
    s = acts[1]
    assert s.repo == "d" and s.finding.severity == Severity.ERROR and s.finding.source_code == "*"
    assert "6 new problems" in s.finding.title and len(s.finding.evidence) == 5
    assert s.fingerprint == fingerprint("d", s.finding)
