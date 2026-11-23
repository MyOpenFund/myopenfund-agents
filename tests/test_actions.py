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
    assert s.fingerprint == "a1:d:*:summary"


def test_summary_issue_deduplicated_against_open_issue():
    cats = list(Category)[:4] + [Category.UNKNOWN]
    fs = Findings(findings=[_f(f"s{i}", cats[i % 5]) for i in range(5)] + [_f("t", Category.PARSE_ERROR)])
    open_summary = OpenIssue(fingerprint="a1:d:*:summary", number=9, repo="d")
    acts = plan_actions(fs, [open_summary], {}, "d", S)
    assert [(a.kind, a.issue_number, a.repo) for a in acts] == [(Kind.COMMENT, 9, "d")]
    assert acts[0].fingerprint == "a1:d:*:summary"


def test_summary_issue_does_not_collide_with_chronic_unknown():
    """The chronic ('*', unknown) issue gets its comment; six new problems still open one summary issue."""
    cats = list(Category)[:4] + [Category.UNKNOWN]
    news = [_f(f"s{i}", cats[i % 5]) for i in range(5)] + [_f("t", Category.PARSE_ERROR)]
    fs = Findings(findings=[*news, _f("*", Category.UNKNOWN, Severity.ERROR)])
    chronic = OpenIssue(fingerprint="a1:d:*:unknown", number=7, repo="d")
    acts = plan_actions(fs, [chronic], {}, "d", S)
    assert [(a.kind, a.fingerprint, a.issue_number) for a in acts] == [
        (Kind.COMMENT, "a1:d:*:unknown", 7),
        (Kind.SUMMARY_ISSUE, "a1:d:*:summary", None),
    ]
    assert "6 new problems" in acts[1].finding.title


def test_general_outage_with_real_source_normalised():
    out = group_outages(Findings(findings=[_f("zz", Category.GENERAL_OUTAGE)]), S)
    assert [f.source_code for f in out.findings] == ["*"]
    acts = plan_actions(Findings(findings=[_f("zz", Category.GENERAL_OUTAGE)]), [], {"zz": "other"}, "d", S)
    assert acts[0].fingerprint == "a1:d:*:general_outage"


def test_no_run_finding_does_not_collide_with_chronic_unknown():
    """Review probe: an open chronic ('*', unknown) issue must not swallow a no-run night."""
    from myopenfund_agents.core.actions import no_run_fingerprint

    chronic = OpenIssue(fingerprint="a1:data-orchestrator:*:unknown", number=7, repo="data-orchestrator")
    fs = Findings(findings=[_f("*", Category.UNKNOWN, Severity.ERROR)])
    acts = plan_actions(fs, [chronic], {}, "data-orchestrator", S, no_run=True)
    assert [(a.kind, a.fingerprint, a.issue_number) for a in acts] == [
        (Kind.NEW_ISSUE, "a1:data-orchestrator:*:no_run", None)
    ]
    assert no_run_fingerprint("data-orchestrator") == "a1:data-orchestrator:*:no_run"


def test_no_run_finding_comments_on_open_no_run_issue():
    open_no_run = OpenIssue(fingerprint="a1:d:*:no_run", number=11, repo="d")
    fs = Findings(findings=[_f("*", Category.UNKNOWN, Severity.ERROR), _f("se")])
    acts = plan_actions(fs, [open_no_run], {}, "d", S, no_run=True)
    assert [(a.kind, a.fingerprint, a.issue_number) for a in acts] == [
        (Kind.COMMENT, "a1:d:*:no_run", 11),
        (Kind.NEW_ISSUE, "a1:d:se:source_site_down", None),
    ]


def test_unknown_star_keeps_its_fingerprint_without_no_run():
    fs = Findings(findings=[_f("*", Category.UNKNOWN, Severity.ERROR)])
    assert plan_actions(fs, [], {}, "d", S)[0].fingerprint == "a1:d:*:unknown"
