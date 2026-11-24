import pytest

import myopenfund_agents.core.triage as triage_mod
from conftest import night, src_run
from myopenfund_agents.core.actions import Kind, plan_actions
from myopenfund_agents.core.detector import detect
from myopenfund_agents.core.explanations import Explanation, Explanations, Quote
from myopenfund_agents.core.findings import Category, Findings, Severity
from myopenfund_agents.core.inputs import OpenIssue
from myopenfund_agents.core.prompt import explain_prompt
from myopenfund_agents.core.summary import build_summary
from myopenfund_agents.core.triage import triage
from myopenfund_agents.engines.base import EngineError
from myopenfund_agents.engines.fake import FakeEngine
from myopenfund_agents.sanitise import DenyList, error_text, violations
from myopenfund_agents.settings import A1Settings

S = A1Settings()
DENY = DenyList()


def _inputs(codes=("aa", "bb"), **kw):
    return night([src_run(c, fetch_errors=3) for c in codes] + [src_run("ok1")], **kw)


def _detected(inputs, settings=S):
    return detect(build_summary(inputs, settings), settings)


def _exp(d, text="The site timed out.", severity=None, fid=None, quote=None):
    return Explanation(
        finding_id=fid or d.finding_id,
        diagnosis=text,
        likely_cause=Category.SOURCE_SITE_DOWN,
        severity=severity or d.finding.severity,
        quotes=[Quote(source="summary", text=quote or d.finding.evidence[0])],
    )


def _all(inputs, **kw):
    return Explanations(explanations=[_exp(d, **kw) for d in _detected(inputs)], observations=[])


def _run(inputs, engine, settings=S, deny=DENY):
    return triage(inputs, engine, settings, {}, "orch", deny)


def _keys(t):
    return [(a.kind, a.fingerprint, a.repo, a.issue_number) for a in t.actions]


def _engine(explanations):
    return FakeEngine(explanations)


def test_explained_run_puts_model_text_in_the_actions():
    inp = _inputs()
    t = _run(inp, _engine(_all(inp, text="Explained by the model.")))
    assert not t.degraded and t.unexplained == [] and t.engine_error is None
    assert all("Explained by the model." in a.finding.diagnosis for a in t.actions)
    assert [f.diagnosis for f in t.findings] == [a.finding.diagnosis for a in t.actions]


def test_engine_gets_the_explain_prompt():
    inp = _inputs()
    eng = _engine(_all(inp))
    _run(inp, eng)
    assert eng.prompts == [explain_prompt(build_summary(inp, S).text, _detected(inp))]


@pytest.mark.parametrize(
    "make",
    [
        lambda inp: None,
        lambda inp: _engine(EngineError("boom")),
        lambda inp: _engine(RuntimeError("boom")),
        lambda inp: _engine(ValueError("boom")),
        lambda inp: _engine(None),
        lambda inp: _engine(Explanations(explanations=[_exp(_detected(inp)[0], fid="f99")], observations=[])),
        lambda inp: _engine(
            Explanations(
                explanations=[_exp(d, quote="never in the summary at all") for d in _detected(inp)], observations=[]
            )
        ),
    ],
    ids=["no-engine", "engine-error", "runtime-error", "value-error", "no-explanations", "id-error", "all-dropped"],
)
def test_fallback_keeps_the_actions_of_the_explained_run(make):
    inp = _inputs(open_issues=[OpenIssue(fingerprint="a1:orch:aa:source_site_down", number=7, repo="orch")])
    explained = _run(inp, _engine(_all(inp)))
    t = _run(inp, make(inp))
    assert [(a.kind, a.fingerprint) for a in t.actions] == [(a.kind, a.fingerprint) for a in explained.actions]
    assert _keys(t) == _keys(explained)
    assert t.degraded
    assert t.unexplained == [d.finding_id for d in _detected(inp)]


def test_fallback_actions_are_the_detected_findings_exactly():
    inp = _inputs()
    t = _run(inp, None)
    detected = _detected(inp)
    expected = plan_actions(Findings(findings=[d.finding for d in detected]), [], {}, "orch", S)
    assert t.actions == expected
    assert t.findings == [d.finding for d in detected]


def test_keyboard_interrupt_is_not_swallowed():
    inp = _inputs()
    with pytest.raises(KeyboardInterrupt):
        _run(inp, _engine(KeyboardInterrupt()))
    with pytest.raises(SystemExit):
        _run(inp, _engine(SystemExit(3)))


def test_crash_inside_check_or_apply_falls_back(monkeypatch):
    inp = _inputs()
    base = _run(inp, None)

    def boom(*a, **k):
        raise RuntimeError("crash")

    for name in ("check", "apply"):
        monkeypatch.setattr(triage_mod, name, boom)
        t = _run(inp, _engine(_all(inp)))
        assert t.actions == base.actions and t.degraded and t.engine_error
        monkeypatch.undo()


def test_partial_explanation_leaves_only_the_missing_id_unexplained():
    inp = _inputs()
    first, second = _detected(inp)
    eng = _engine(Explanations(explanations=[_exp(first, text="Only the first.")], observations=[]))
    t = _run(inp, eng)
    assert t.unexplained == [second.finding_id]
    assert t.degraded
    assert "Only the first." in t.actions[0].finding.diagnosis
    assert t.actions[1].finding == second.finding


def test_dropped_explanation_is_unexplained_but_not_an_engine_error():
    inp = _inputs()
    first, second = _detected(inp)
    bad = _exp(second, quote="never in the summary at all")
    t = _run(inp, _engine(Explanations(explanations=[_exp(first), bad], observations=[])))
    assert t.unexplained == [second.finding_id] and t.engine_error is None and t.degraded
    assert t.checked is not None and second.finding_id in t.checked.dropped


def test_empty_night_does_not_call_the_engine_and_is_not_degraded():
    inp = night([src_run("ok1")])
    eng = _engine(Explanations(explanations=[], observations=[]))
    t = _run(inp, eng)
    assert eng.prompts == [] and t.actions == [] and not t.degraded and t.result is None
    assert not _run(inp, None).degraded


def test_green_night_asks_the_model_when_set():
    inp = night([src_run("ok1")])
    eng = _engine(Explanations(explanations=[], observations=["odd"]))
    t = _run(inp, eng, A1Settings(explain_green_nights=True))
    assert len(eng.prompts) == 1 and not t.degraded and t.checked.observations == ["odd"]
    failing = _run(inp, _engine(RuntimeError("x")), A1Settings(explain_green_nights=True))
    assert failing.degraded and failing.actions == []
    assert not _run(inp, None, A1Settings(explain_green_nights=True)).degraded


def test_infra_detail_in_engine_error_is_withheld():
    inp = _inputs()
    t = _run(inp, _engine(EngineError("cannot reach 10.1.2.3 now")))
    assert "10.1.2.3" not in t.engine_error and "withheld" in t.engine_error
    deny = DenyList(hosts=["internal.example"])
    t = _run(inp, _engine(RuntimeError("connect to gpu.internal.example failed")), deny=deny)
    assert "internal.example" not in t.engine_error and "withheld" in t.engine_error


def test_answer_for_an_undetected_source_changes_nothing():
    inp = _inputs()
    d = _detected(inp)[0]
    extra = _exp(d, fid="f77", text="Brand new problem.")
    t = _run(inp, _engine(Explanations(explanations=[*_all(inp).explanations, extra], observations=[])))
    base = _run(inp, None)
    assert _keys(t) == _keys(base) and t.degraded
    assert all("Brand new problem." not in a.finding.diagnosis for a in t.actions)


def test_model_cannot_change_what_the_action_is():
    inp = _inputs(open_issues=[OpenIssue(fingerprint="a1:orch:aa:source_site_down", number=7, repo="orch")])
    ex = Explanations(
        explanations=[
            _exp(d, severity=Severity.ERROR).model_copy(update={"likely_cause": Category.VOLUME_DROP})
            for d in _detected(inp)
        ],
        observations=[],
    )
    t = _run(inp, _engine(ex))
    base = _run(inp, None)
    assert _keys(t) == _keys(base)
    assert [(a.finding.source_code, a.finding.category) for a in t.actions] == [
        (a.finding.source_code, a.finding.category) for a in base.actions
    ]
    assert t.actions[0].kind == Kind.COMMENT and t.actions[0].issue_number == 7


def test_severity_is_never_lowered_and_can_be_raised():
    inp = night([src_run("aa", bad_content=1), src_run("bb", bad_content=1)])
    ds = _detected(inp)
    assert all(d.finding.severity == Severity.WARN for d in ds)
    ex = Explanations(
        explanations=[_exp(ds[0], severity=Severity.ERROR), _exp(ds[1], severity=Severity.INFO)], observations=[]
    )
    t = _run(inp, _engine(ex))
    assert [a.finding.severity for a in t.actions] == [Severity.ERROR, Severity.WARN]


def test_summary_issue_keeps_the_detectors_decision():
    st = A1Settings(max_new_issues_per_run=1)
    inp = night([src_run("aa", bad_content=1), src_run("bb", bad_content=1)])
    ds = _detected(inp, st)
    base = _run(inp, None, st)
    assert [a.kind for a in base.actions] == [Kind.SUMMARY_ISSUE]
    assert base.actions[0].finding.severity == Severity.WARN

    def explained(severity):
        ex = Explanations(explanations=[_exp(d, severity=severity) for d in ds], observations=[])
        return _run(inp, _engine(ex), st)

    low, high = explained(Severity.INFO), explained(Severity.ERROR)
    assert _keys(low) == _keys(high) == _keys(base)
    assert low.actions[0].finding.severity == Severity.WARN
    assert high.actions[0].finding.severity == Severity.ERROR


def test_no_run_night_uses_the_no_run_fingerprint():
    inp = night()
    t = _run(inp, None)
    assert [a.fingerprint for a in t.actions] == ["a1:orch:*:no_run"]
    assert t.degraded


# --- review round 1 ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["The log shows a token:", "x password:", "Authorization: basic"])
def test_a_secret_made_by_the_model_text_and_the_appended_line_is_withheld(text):
    inp = _inputs()
    first, second = _detected(inp)
    ex = Explanations(explanations=[_exp(first, text=text), _exp(second, text="Fine.")], observations=[])
    t = _run(inp, _engine(ex))
    assert t.actions[0].finding == first.finding and t.findings[0] == first.finding
    assert t.checked.dropped == {first.finding_id: "withheld"} and first.finding_id not in t.checked.kept
    assert t.unexplained == [first.finding_id] and t.degraded and t.engine_error is None
    assert "Fine." in t.actions[1].finding.diagnosis


def test_a_hit_already_in_the_detected_finding_is_not_the_models_doing():
    inp = night([src_run("aa.internal.example", fetch_errors=3)])
    deny = DenyList(hosts=["internal.example"])
    d = _detected(inp)[0]
    assert d.finding.title and "internal.example" in d.finding.title
    # the quote carries the host, so check() drops it on its own: the point is the title, a detected text
    ex = Explanations(explanations=[_exp(d, quote="fetch_errors")], observations=[])
    t = _run(inp, _engine(ex), deny=deny)
    assert t.checked.kept.keys() == {d.finding_id} and not t.degraded


def test_error_text_never_raises():
    class NoStr(Exception):
        def __str__(self):
            raise RuntimeError("no str")

        __repr__ = __str__

    assert error_text(NoStr(), DenyList()) == "NoStr"
    t = _run(_inputs(), _engine(NoStr()))
    assert t.engine_error == "NoStr" and t.degraded and t.actions == _run(_inputs(), None).actions


def test_error_text_is_capped_by_a_setting():
    t = _run(_inputs(), _engine(RuntimeError("x" * 900)))
    assert len(t.engine_error) == 500
    t = _run(_inputs(), _engine(RuntimeError("x" * 900)), A1Settings(engine_error_max_chars=20))
    assert t.engine_error == "x" * 20
    assert A1Settings.from_env({"MOF_AGENTS_A1_ENGINE_ERROR_MAX_CHARS": "7"}).engine_error_max_chars == 7
    with pytest.raises(ValueError):
        A1Settings.from_env({"MOF_AGENTS_A1_ENGINE_ERROR_MAX_CHARS": "0"})


def _other(kind):
    def fake_apply(d, e):
        f = d.finding
        if kind == "source":
            return f.model_copy(update={"source_code": "zz"})
        if kind == "category":
            return f.model_copy(update={"category": Category.VOLUME_DROP})
        return _detected(_inputs())[0].finding  # a copy of the first finding: the plan loses one action

    return fake_apply


@pytest.mark.parametrize("kind", ["source", "category", "duplicate"])
def test_an_explained_plan_that_differs_keeps_the_detected_actions(monkeypatch, kind):
    inp = _inputs()
    base = _run(inp, None)
    monkeypatch.setattr(triage_mod, "apply", _other(kind))
    t = _run(inp, _engine(_all(inp)))
    assert t.actions == base.actions and t.engine_error and t.degraded and t.findings == base.findings


def test_repo_of_routes_actions_in_both_plans():
    inp = _inputs()
    repos = {"aa": "repo-a"}
    for eng in (None, _engine(_all(inp))):
        t = triage(inp, eng, S, repos, "orch", DENY)
        assert [a.repo for a in t.actions] == ["repo-a", "orch"]
        assert t.actions[0].fingerprint == "a1:repo-a:aa:source_site_down"
    assert not t.degraded


def test_the_engine_gets_the_finding_ids():
    inp = _inputs()
    eng = _engine(_all(inp))
    _run(inp, eng)
    assert eng.ids == [frozenset(d.finding_id for d in _detected(inp))]


def test_result_and_summary_are_kept_on_success_and_failure():
    inp = _inputs()
    text = build_summary(inp, S).text
    ok = _run(inp, _engine(_all(inp)))
    assert ok.result is not None and ok.result.explanations is not None and ok.summary.text == text
    bad = _run(inp, _engine(None))
    assert bad.result is not None and bad.result.explanations is None and bad.summary.text == text
    assert bad.findings == [d.finding for d in _detected(inp)]
    assert _run(inp, None).summary.text == text


def test_quotes_are_checked_against_the_summary_not_the_prompt():
    inp = _inputs()
    first, second = _detected(inp)
    only_in_findings_block = f"{first.finding_id}: source=aa category=source_site_down"
    t = _run(inp, _engine(Explanations(explanations=[_exp(first, quote=only_in_findings_block)], observations=[])))
    assert t.checked.dropped == {first.finding_id: "false quote"}


def test_a_check_with_the_real_deny_list():
    inp = night([src_run("aa.internal.example", fetch_errors=3)])
    d = _detected(inp)[0]
    deny = DenyList(hosts=["internal.example"])
    quote = d.finding.evidence[0]
    assert "internal.example" in quote
    t = _run(inp, _engine(Explanations(explanations=[_exp(d, quote=quote)], observations=[])), deny=deny)
    assert t.checked.dropped == {d.finding_id: "withheld"}
    ex = Explanations(explanations=[_exp(d, text="gpu.internal.example is down")], observations=[])
    t = _run(inp, _engine(ex), deny=deny)
    assert t.checked.dropped == {d.finding_id: "withheld"}


def test_an_empty_engine_error_on_a_green_night_is_degraded():
    inp = night([src_run("ok1")])
    t = _run(inp, _engine(EngineError("")), A1Settings(explain_green_nights=True))
    assert t.engine_error == "" and t.degraded


def test_no_run_night_explained_keeps_the_no_run_fingerprint():
    inp = night()
    d = _detected(inp)[0]
    t = _run(inp, _engine(Explanations(explanations=[_exp(d, text="Nothing ran.")], observations=[])))
    assert [a.fingerprint for a in t.actions] == ["a1:orch:*:no_run"] and not t.degraded
    assert "Nothing ran." in t.actions[0].finding.diagnosis


@pytest.mark.parametrize("cap", [1, 5, 20, 500])
def test_an_engine_error_that_the_cut_makes_unclean_is_withheld(cap):
    deny = DenyList(path_roots=["/srv/data"])
    text = "x" * (cap - len("/srv/data")) + "/srv/database is full"
    if cap < len("/srv/data"):
        text = "/srv/database is full"
    assert not violations(text, deny)  # only the cut makes it a hit
    t = _run(_inputs(), _engine(RuntimeError(text)), A1Settings(engine_error_max_chars=cap), deny=deny)
    assert not violations(t.engine_error, deny) and len(t.engine_error) <= cap and t.degraded
    if cap >= 20:
        assert t.engine_error == "RuntimeError: withheld (deny_path)"[:cap]


def test_the_reviewers_engine_error_example():
    deny = DenyList(path_roots=["/srv/data"])
    s = A1Settings(engine_error_max_chars=20)
    t = _run(_inputs(), _engine(RuntimeError("failed on /srv/data-archive")), s, deny=deny)
    assert t.engine_error == "failed on /srv/data-"
    s = A1Settings(engine_error_max_chars=19)
    t = _run(_inputs(), _engine(RuntimeError("failed on /srv/data-archive")), s, deny=deny)
    assert t.engine_error == "RuntimeError: withhe"[:19]


def test_the_withheld_marker_is_within_the_cap_and_clean():
    s = A1Settings(engine_error_max_chars=15)
    t = _run(_inputs(), _engine(RuntimeError("at 10.1.2.3 " + "y" * 900)), s)
    assert t.engine_error == "RuntimeError: withheld (private_ip)"[:15]
    # the type name itself is denied: the marker cannot be used
    deny = DenyList(hosts=["runtimeerror", "10.1"])
    t = _run(_inputs(), _engine(RuntimeError("at 10.1.2.3")), deny=deny)
    assert t.engine_error == "withheld"
    t = _run(_inputs(), _engine(RuntimeError("at 10.1.2.3")), deny=DenyList(hosts=["runtimeerror", "withheld"]))
    assert t.engine_error == "" and t.degraded


def test_an_infra_detail_cut_in_two_or_away_is_still_withheld():
    s = A1Settings(engine_error_max_chars=40)
    for text in ("y" * 33 + "10.1.2.3", "y" * 60 + " 10.1.2.3"):
        t = _run(_inputs(), _engine(RuntimeError(text)), s)
        assert t.engine_error == "RuntimeError: withheld (private_ip)"


def test_the_type_name_used_for_an_unreadable_error_is_cut_and_scanned():
    class NasExample(Exception):
        def __str__(self):
            raise RuntimeError("no str")

    assert error_text(NasExample(), DenyList(), 3) == "Nas"
    assert error_text(NasExample(), DenyList(hosts=["nasexample"])) == "withheld"


def test_a_secret_in_the_applied_diagnosis_is_withheld_even_when_the_evidence_has_one():
    inp = night([src_run("aa", fetch_errors=3, error_samples=["password=hunter2222"])])
    d = _detected(inp)[0]
    assert any("hunter2222" in line for line in d.finding.evidence)
    ex = Explanations(explanations=[_exp(d, text="The log shows a token:", quote="fetch_errors")], observations=[])
    t = _run(inp, _engine(ex))
    assert t.checked.dropped == {d.finding_id: "withheld"} and t.findings == [d.finding]


def test_the_copy_of_a_dropped_explanation_never_reaches_an_action():
    inp = _inputs()
    first, second = _detected(inp)
    ex = [_exp(first, text="Zebra text.", quote="never in the summary"), _exp(second, text="A giraffe token:")]
    t = _run(inp, _engine(Explanations(explanations=ex, observations=[])))
    assert t.checked.dropped == {first.finding_id: "false quote", second.finding_id: "withheld"}
    copies = t.checked.dropped_copies
    assert copies[first.finding_id].diagnosis == "Zebra text."
    # the hit only exists once the cause line is appended: the diagnosis is withheld all the same
    assert copies[second.finding_id].diagnosis == "withheld (credential)"
    assert copies[second.finding_id].quotes == ex[1].quotes
    assert t.actions == _run(inp, None).actions and t.findings == [first.finding, second.finding]


def test_a_hit_only_in_the_joined_body_is_withheld():
    inp = _inputs(("aa",))
    deny = DenyList(hosts=["source_site_down\nthe site"])  # spans the title and the diagnosis
    t = _run(inp, _engine(_all(inp)), deny=deny)
    assert t.checked.dropped == {"f1": "withheld"} and t.actions == _run(inp, None).actions
