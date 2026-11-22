from pathlib import Path

import pytest

from conftest import night as inputs
from myopenfund_agents.bench.fixtures import Expected, ExpectedAction, Key, Meta, Night
from myopenfund_agents.bench.score import Attempt, NightScore, load_thresholds, score_night, verdicts
from myopenfund_agents.core.findings import Category

THRESHOLDS = Path(__file__).resolve().parents[1] / "bench" / "thresholds.toml"
KEY = ("us", "parse_error")
ACT = ("new_issue", "us", "parse_error")


def make_night(origin="incident", keys=(KEY,), actions=(ACT,)) -> Night:
    return Night(
        inputs=inputs(),
        expected=Expected(
            findings=[Key(source_code=s, category=Category(c)) for s, c in keys],
            actions=[ExpectedAction(kind=k, source_code=s, category=Category(c)) for k, s, c in actions],
        ),
        meta=Meta(origin=origin, description="t"),
    )


def ok(keys=(KEY,), actions=(ACT,), seconds=1.0) -> Attempt:
    return Attempt(keys=set(keys), actions=list(actions), seconds=seconds)


def bad(seconds=1.0) -> Attempt:
    return Attempt(keys=None, actions=None, seconds=seconds, error="boom")


def score(night: Night, *attempts: Attempt) -> NightScore:
    return score_night(night, list(attempts))


def by_name(vs):
    return {v.criterion: v for v in vs}


def test_incident_found_and_stable():
    s = score(make_night(), ok(), ok(), ok(seconds=3.0))
    assert s.found == [KEY] and s.extra == [] and s.false_alarms == 0
    assert s.actions_ok is True and s.valid is True
    assert s.agreement == 1.0 and s.seconds_max == 3.0
    assert s.expected == [KEY] and s.origin == "incident" and len(s.attempts) == 3


def test_false_alarm_on_green_and_invalid():
    s = score(make_night("green", keys=(), actions=()), ok(keys=(KEY,), actions=(ACT,)), bad())
    assert s.extra == [KEY] and s.found == [] and s.false_alarms == 1
    assert s.valid is False and s.agreement == 0.5
    assert s.actions_ok is None


def test_extra_not_a_false_alarm_on_incident_night():
    s = score(make_night(), ok(keys=(KEY, ("fr", "unknown")), actions=(ACT,)))
    assert s.extra == [("fr", "unknown")] and s.false_alarms == 0 and s.actions_ok is None


def test_invalid_first_attempt():
    s = score(make_night(), bad(), ok())
    assert s.found == [] and s.extra == [] and s.actions_ok is None and s.valid is False


def test_actions_compare_order_insensitive_with_multiplicity():
    n = make_night(actions=(ACT, ("comment", "us", "parse_error")))
    assert score(n, ok(actions=(("comment", "us", "parse_error"), ACT))).actions_ok is True
    assert score(n, ok(actions=(ACT, ACT))).actions_ok is False


def test_actions_not_judged_when_findings_differ():
    s = score(make_night(), ok(keys=(), actions=()))
    assert s.found == [] and s.actions_ok is None


def test_verdicts_prod_and_local():
    slow = score(make_night(), ok(seconds=700.0))
    prod = by_name(verdicts([slow], load_thresholds(THRESHOLDS, "prod")))
    local = by_name(verdicts([slow], load_thresholds(THRESHOLDS, "local")))
    assert "slowest night (s)" not in prod and all(v.passed for v in prod.values())
    assert local["slowest night (s)"].passed is False and local["slowest night (s)"].value == 700.0
    assert [v.criterion for v in verdicts([slow], load_thresholds(THRESHOLDS, "local"))] == [
        "problems found",
        "false alarms on green nights",
        "actions as expected",
        "valid output",
        "repeat agreement (worst night)",
        "slowest night (s)",
    ]


def test_two_of_three_agreement():
    s = score(make_night(), ok(), ok(), ok(keys=(), actions=()))
    assert s.agreement == pytest.approx(2 / 3)
    t = by_name(verdicts([s], load_thresholds(THRESHOLDS, "local")))
    assert t["repeat agreement (worst night)"].passed is True
    t = by_name(verdicts([s], load_thresholds(THRESHOLDS, "prod")))
    assert t["repeat agreement (worst night)"].passed is False


def test_recall_green_alarms_and_empty():
    t = load_thresholds(THRESHOLDS, "prod")
    miss = score(make_night(), ok(keys=(), actions=()))
    green = score(make_night("green", (), ()), ok())
    v = by_name(verdicts([miss, green], t))
    assert v["problems found"].value == 0.0 and not v["problems found"].passed
    assert v["false alarms on green nights"].value == 1 and not v["false alarms on green nights"].passed
    assert v["actions as expected"].value == 1.0  # nothing judged
    v = by_name(verdicts([], t))
    assert v["problems found"].value == 1.0 and all(x.passed for x in v.values())


def test_load_thresholds_strict(tmp_path):
    with pytest.raises(ValueError):
        load_thresholds(THRESHOLDS, "nope")
    p = tmp_path / "t.toml"
    p.write_text("[x]\nmin_recall=1\nmin_recal=1\nmax_false_alarms=0\nmin_actions_ok=1\nmin_valid=1\nmin_agreement=1\n")
    with pytest.raises(ValueError):
        load_thresholds(p, "x")
    p.write_text("[x]\nmin_recall=1\n")
    with pytest.raises(ValueError):
        load_thresholds(p, "x")
    assert load_thresholds(THRESHOLDS, "local")["max_seconds_per_night"] == 600
    assert "max_seconds_per_night" not in load_thresholds(THRESHOLDS, "prod")
