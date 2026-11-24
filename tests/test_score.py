from pathlib import Path

import pytest

from conftest import night as inputs
from myopenfund_agents.bench.fixtures import Expected, ExpectedAction, Key, Meta, Night
from myopenfund_agents.bench.score import Attempt, NightScore, load_thresholds, reported, score_night, verdicts
from myopenfund_agents.core.findings import Category

THRESHOLDS = Path(__file__).resolve().parents[1] / "bench" / "thresholds.toml"
KEY = ("us", "unreadable_pdf")
KEY2 = ("fr", "source_site_down")
ACT = ("new_issue", "us", "unreadable_pdf")
ACT2 = ("new_issue", "fr", "source_site_down")
QUOTES = "quotes exact (of quotes given)"
ZERO = {"min_explained": 0, "min_valid": 0, "min_quotes_exact": 0}


def make_night(origin="incident", keys=(KEY,), actions=(ACT,), night_id="n") -> Night:
    return Night(
        inputs=inputs(night_id=night_id),
        expected=Expected(
            findings=[Key(source_code=s, category=Category(c)) for s, c in keys],
            actions=[ExpectedAction(kind=k, source_code=s, category=Category(c)) for k, s, c in actions],
        ),
        meta=Meta(origin=origin, description="t"),
    )


GREEN = make_night("green", (), (), "green")
TWO = make_night(keys=(KEY, KEY2), actions=(ACT, ACT2), night_id="two")


def found(*keys):
    return [{"finding_id": f"f{i}", "source_code": s, "category": c} for i, (s, c) in enumerate(keys, 1)]


def ok(keys=(KEY,), actions=(ACT,), seconds=1.0, causes=None, **kw) -> Attempt:
    """Every finding explained, one exact quote each, unless `causes` (finding number -> cause) says otherwise."""
    causes = {i: "unknown" for i, _ in enumerate(keys, 1)} if causes is None else causes
    values = {
        "findings": found(*keys),
        "actions": list(actions),
        "seconds": seconds,
        "valid": True,
        "explanations": [{"finding_id": f"f{i}", "likely_cause": c} for i, c in causes.items()],
        "quotes_total": len(causes),
        "quotes_exact": len(causes),
    }
    return Attempt(**{**values, **kw})


def bad(keys=(KEY,), actions=(ACT,), seconds=1.0) -> Attempt:
    return Attempt(findings=found(*keys), actions=list(actions), seconds=seconds, valid=False, error="boom")


def quiet(**kw) -> Attempt:
    return Attempt(findings=[], actions=[], seconds=0.0, **kw)


def score(night: Night, *attempts: Attempt) -> NightScore:
    return score_night(night, list(attempts))


def by_name(vs):
    return {v.criterion: v for v in vs}


def check(scores, t=None):
    return by_name(verdicts(scores, t or load_thresholds(THRESHOLDS, "prod")))


def test_all_explained_passes_both_profiles():
    scores = [score(make_night(), ok(), ok(), ok()), score(GREEN, quiet(), quiet(), quiet())]
    for profile, names in (
        ("prod", ["detector exact", "findings explained", "valid output", QUOTES, "fallback nights"]),
        (
            "local",
            ["detector exact", "findings explained", "valid output", QUOTES, "slowest night (s)"],
        ),
    ):
        vs = verdicts(scores, load_thresholds(THRESHOLDS, profile))
        assert [v.criterion for v in vs] == names and all(v.passed for v in vs)


def test_explained_counts_every_finding_of_every_attempt():
    s = score(TWO, ok((KEY, KEY2), (ACT, ACT2), causes={1: "unknown"}), ok((KEY, KEY2), (ACT, ACT2)), bad(
        (KEY, KEY2), (ACT, ACT2)))  # fmt: skip
    assert (s.explained, s.explainable) == (3, 6)
    v = check([s, score(make_night(), ok())])
    assert v["findings explained"].value == pytest.approx(4 / 7) and not v["findings explained"].passed
    assert v["findings explained"].detail == "4/7"


def test_missed_night_is_not_skipped():
    """Regression: the old "actions as expected" skipped the nights a problem was missed on and showed 100%."""
    missed = score(make_night(night_id="a"), bad(), bad(), bad())
    good = score(make_night(night_id="b"), ok(), ok(), ok())
    v = check([good, missed])
    assert v["findings explained"].value == 0.5 and v["valid output"].value == 0.5
    assert not v["findings explained"].passed and not v["valid output"].passed
    # and a failed first, middle or last attempt weighs the same
    for attempts in ([bad(), ok(), ok()], [ok(), bad(), ok()], [ok(), ok(), bad()]):
        assert check([score(make_night(), *attempts)])["findings explained"].value == pytest.approx(2 / 3)


def test_valid_counts_attempts_not_nights():
    v = check([score(make_night(), ok(), bad(), ok()), score(make_night(night_id="b"), ok())])
    assert v["valid output"].value == 0.75 and v["valid output"].detail == "3/4"


def test_valid_answer_that_explains_nothing_is_valid_but_unexplained():
    v = check([score(make_night(), ok(causes={}), ok())])
    assert v["valid output"].value == 1.0 and v["findings explained"].value == 0.5


def test_green_nights_are_in_no_denominator():
    incident = score(make_night(), ok(), bad())
    # asked about nothing (explain_green_nights) and failing, slow, with quotes: none of it counts
    noise = quiet(valid=True, quotes_total=9, quotes_exact=0, unknown_numbers=4, input_tokens=5)
    noise.seconds = 9999.0
    green = score(GREEN, noise, noise)
    t = load_thresholds(THRESHOLDS, "local") | {"max_fallback_nights": 0}
    alone, both = check([incident], t), check([incident, green], t)
    for name in alone.keys() - {"detector exact"}:
        assert alone[name] == both[name]
    assert reported([incident]) == reported([incident, green])
    assert not green.fallback and (green.explainable, green.quotes_total, green.unknown_numbers) == (0, 0, 0)
    assert green.valid == 0
    assert both["detector exact"].detail == "2/2"


def test_quotes_exact_is_a_sum_over_the_run():
    a = score(make_night(), ok(quotes_total=3, quotes_exact=3), ok(quotes_total=1, quotes_exact=0))
    v = check([a])
    assert v[QUOTES].value == 0.75 and v[QUOTES].detail == "3/4"
    assert not v[QUOTES].passed


def test_no_quote_at_all_is_zero_not_100():
    for attempt in (ok(quotes_total=0, quotes_exact=0), bad()):
        v = check([score(make_night(), attempt)])
        assert v[QUOTES].value == 0.0 and not v[QUOTES].passed


def test_fallback_night_counts_once():
    worst = score(make_night(night_id="a"), bad(), bad(), ok(causes={}))
    good = score(make_night(night_id="b"), ok(), ok(), ok())
    last = score(TWO, ok((KEY, KEY2), (ACT, ACT2)), ok((KEY, KEY2), (ACT, ACT2), causes={2: "unknown"}))
    assert (worst.fallback, good.fallback, last.fallback) == (True, False, True)
    v = check([worst, good, last])
    assert v["fallback nights"].value == 2 and v["fallback nights"].threshold == 0 and not v["fallback nights"].passed
    assert reported([worst, good, last])["fallback nights"] == 2
    assert "fallback nights" not in check([worst], load_thresholds(THRESHOLDS, "local"))


def test_fallback_threshold_is_a_maximum():
    s = [score(make_night(night_id="a"), bad()), score(make_night(night_id="b"), ok())]
    assert check(s, {**ZERO, "max_fallback_nights": 1})["fallback nights"].passed
    assert not check(s, {**ZERO, "max_fallback_nights": 0})["fallback nights"].passed


@pytest.mark.parametrize(
    ("keys", "actions"),
    [
        ((), ()),  # missed
        ((KEY, KEY2), (ACT,)),  # extra finding
        ((("us", "unknown"),), (ACT,)),  # other category
        ((KEY,), (("comment", "us", "unreadable_pdf"),)),  # other action
        ((KEY,), (ACT, ACT)),  # multiplicity
        ((KEY, KEY), (ACT,)),  # a finding made twice
        ((KEY,), ()),
    ],
)
def test_detector_mismatch_fails(keys, actions):
    wrong = score(make_night(), ok(keys, actions))
    right = score(make_night(night_id="b"), ok())
    assert wrong.detector_exact is False and right.detector_exact is True
    v = check([right, wrong])["detector exact"]
    assert v.value == 0.5 and v.threshold == 1.0 and not v.passed and v.detail == "1/2"


def test_detector_mismatch_on_a_green_night_or_a_later_attempt_fails():
    extra = score(GREEN, ok())
    assert extra.detector_exact is False
    late = score(make_night(), ok(), ok((), ()))
    assert late.detector_exact is False
    assert not check([extra, late])["detector exact"].passed


def test_detector_order_does_not_matter():
    assert score(TWO, ok((KEY2, KEY), (ACT2, ACT))).detector_exact is True
    assert score(TWO, ok((KEY, KEY2), (ACT, ACT2))).detector_exact is True


def test_no_detected_finding_anywhere_is_an_error_not_100():
    for scores in ([], [score(GREEN, quiet())]):
        with pytest.raises(ValueError, match="no detected finding"):
            verdicts(scores, load_thresholds(THRESHOLDS, "prod"))


def test_slowest_night():
    slow = [score(make_night(), ok(seconds=100.0), ok(seconds=700.0)), score(make_night(night_id="b"), ok())]
    v = check(slow, load_thresholds(THRESHOLDS, "local"))["slowest night (s)"]
    assert v.value == 700.0 and v.threshold == 600 and not v.passed
    assert "slowest night (s)" not in check(slow)
    edge = [score(make_night(), ok(seconds=600.0))]
    assert check(edge, load_thresholds(THRESHOLDS, "local"))["slowest night (s)"].passed


def test_ratio_thresholds_are_inclusive():
    s = [score(make_night(), *[ok()] * 9, bad())]
    v = check(s, load_thresholds(THRESHOLDS, "local"))
    assert v["findings explained"].value == 0.9 and v["findings explained"].passed and v["valid output"].passed
    s = [score(make_night(), ok(quotes_total=20, quotes_exact=19))]
    assert check(s, load_thresholds(THRESHOLDS, "local"))[QUOTES].passed
    s = [score(make_night(), ok(quotes_total=20, quotes_exact=18))]
    assert not check(s, load_thresholds(THRESHOLDS, "local"))[QUOTES].passed


def test_cause_agreement_per_finding_over_repeats():
    def att(a, b):
        return ok((KEY, KEY2), (ACT, ACT2), causes={k: c for k, c in ((1, a), (2, b)) if c})

    s = score(TWO, att("unknown", "parse_error"), att("unknown", "source_site_down"), att("unknown", "parse_error"))
    assert s.cause_agreement == {"f1": 1.0, "f2": pytest.approx(2 / 3)}
    r = reported([s])
    assert r["likely_cause agreement (mean)"] == pytest.approx(5 / 6)
    assert r["likely_cause agreement (worst)"] == pytest.approx(2 / 3)
    # the mean is per finding, over every night
    r = reported([s, score(make_night(), ok(), ok())])
    assert r["likely_cause agreement (mean)"] == pytest.approx((1 + 2 / 3 + 1) / 3)


def test_failed_attempt_is_a_distinct_answer_in_the_agreement():
    assert score(make_night(), ok(), ok(), bad()).cause_agreement == {"f1": pytest.approx(2 / 3)}
    assert score(make_night(), ok(), bad(), ok(causes={})).cause_agreement == {"f1": pytest.approx(1 / 3)}
    # failures never agree with each other
    assert score(make_night(), bad(), bad(), bad()).cause_agreement == {"f1": 0.0}
    assert score(make_night(), bad(), bad(), ok()).cause_agreement == {"f1": pytest.approx(1 / 3)}


def test_reported_seconds_tokens_and_unknown_numbers():
    s = score(
        make_night(),
        ok(seconds=1.0, unknown_numbers=2, input_tokens=100, output_tokens=10),
        ok(seconds=3.0, unknown_numbers=1, input_tokens=50),
    )
    r = reported([s, score(make_night(night_id="b"), ok(seconds=8.0, unknown_numbers=4))])
    assert r["unknown numbers"] == 7
    assert r["seconds (mean)"] == 4.0 and r["seconds (worst)"] == 8.0
    assert r["input tokens"] == {"total": 150, "known": 2, "attempts": 3}
    assert r["output tokens"] == {"total": 10, "known": 1, "attempts": 3}
    assert reported([score(make_night(), bad())])["input tokens"] == {"total": None, "known": 0, "attempts": 1}
    assert reported([score(make_night(), ok(input_tokens=0))])["input tokens"]["total"] == 0


def test_agreement_needs_two_attempts():
    once = score(make_night(), ok())
    assert once.cause_agreement == {}
    r = reported([once])
    assert r["likely_cause agreement (mean)"] is None and r["likely_cause agreement (worst)"] is None
    assert reported([score(make_night(), bad())])["likely_cause agreement (worst)"] is None
    assert score(make_night(), ok(), ok()).cause_agreement == {"f1": 1.0}


def test_mismatches_name_the_attempts_and_both_sides():
    assert score(make_night(), ok(), ok()).mismatches() == []
    early = score(make_night(), ok((), ()), ok(), ok())
    assert early.detected == [] and early.expected_actions == [ACT]
    assert early.mismatches() == [([1], [], [])]
    late = score(make_night(), ok(), ok((KEY2,), (ACT2,)), ok((KEY2,), (ACT2,)))
    assert late.mismatches() == [([2, 3], [KEY2], [ACT2])]
    # right findings, wrong action; two variants
    acts = score(make_night(), ok(actions=()), ok(actions=(ACT2,)))
    assert acts.mismatches() == [([1], [KEY], []), ([2], [KEY], [ACT2])]
    both = score(make_night(), ok(actions=(("comment", *KEY),)), ok(), ok((), ()))
    assert both.mismatches() == [([1], [KEY], [("comment", *KEY)]), ([3], [], [])]


def test_night_score_summary():
    s = score(make_night(), ok(seconds=2.0), ok(seconds=3.0, unknown_numbers=2, quotes_total=2))
    assert (s.night_id, s.origin, s.expected, s.detected) == ("n", "incident", [KEY], [KEY])
    assert (s.valid, s.seconds_max, s.unknown_numbers, s.quotes_total, s.quotes_exact) == (2, 3.0, 2, 3, 2)
    assert len(s.attempts) == 2


def test_empty_attempts_rejected():
    with pytest.raises(ValueError, match="no attempts"):
        score(make_night())


def test_committed_thresholds():
    assert load_thresholds(THRESHOLDS, "prod") == {
        "min_explained": 1.0, "min_valid": 1.0, "min_quotes_exact": 1.0, "max_fallback_nights": 0,
    }  # fmt: skip
    assert load_thresholds(THRESHOLDS, "local") == {
        "min_explained": 0.9, "min_valid": 0.9, "min_quotes_exact": 0.95, "max_seconds_per_night": 600,
    }  # fmt: skip


BASE = "[x]\nmin_explained=1\nmin_valid=1\nmin_quotes_exact=1\n"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (BASE + "min_recal=1\n", "unknown keys"),
        (BASE + "min_recall=1\n", "unknown keys"),  # an old threshold file
        (BASE.replace("min_explained=1\n", ""), "missing keys"),
        (BASE.replace("min_valid=1\n", ""), "missing keys"),
        (BASE.replace("min_quotes_exact=1\n", ""), "missing keys"),
        (BASE.replace("min_valid=1", "min_valid=true"), "must be a number"),
        (BASE.replace("min_valid=1", 'min_valid="1"'), "must be a number"),
        (BASE.replace("min_valid=1", "min_valid=nan"), "must be a number"),
    ],
)
def test_load_thresholds_strict(tmp_path, text, message):
    p = tmp_path / "t.toml"
    p.write_text(BASE)
    assert load_thresholds(p, "x")["min_explained"] == 1
    p.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_thresholds(p, "x")
    with pytest.raises(ValueError, match="unknown threshold profile"):
        load_thresholds(p, "nope")
