import random

import pytest

from conftest import night, src_run
from myopenfund_agents.core.actions import fingerprint, plan_actions
from myopenfund_agents.core.detector import RULES, Detected, detect
from myopenfund_agents.core.findings import Category, Findings, Severity
from myopenfund_agents.core.inputs import DiscoveryError, RunRow
from myopenfund_agents.core.summary import NO_RUN_LINE, build_summary
from myopenfund_agents.settings import A1Settings

SETTINGS = A1Settings()


def _detect(inputs, settings=SETTINGS):
    return detect(build_summary(inputs, settings), settings)


def _keys(detected):
    return [(d.finding.source_code, d.finding.category.value) for d in detected]


def _bad_run(run_id="run-x", outcome="failed", exit_code=1):
    return RunRow(
        run_id=run_id,
        tool="harvester",
        command="harvest all",
        started_at=night_start(),
        finished_at=night_start(),
        outcome=outcome,
        exit_code=exit_code,
    )


def night_start():
    from conftest import END

    return END.replace(hour=3)


def _discovery(code="se"):
    return DiscoveryError(fingerprint=f"d-{code}", source_code=code, corpus="central-bank", error="unreachable")


def _history(code="se", seen=10):
    return [src_run(code, night=n, seen=seen) for n in range(1, 15)]


CASES = {
    "bad_content": (dict(bad_content=2), "unreadable_pdf", "warn"),
    "fetch_errors": (dict(fetch_errors=3), "source_site_down", "error"),
    "truncated": (dict(truncated=True), "source_site_down", "error"),
    "docs_failed": (dict(docs_failed=1), "unknown", "warn"),
    "failed_run": (dict(outcome="failed", exit_code=1), "unknown", "error"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_row_rules(name):
    kw, category, severity = CASES[name]
    found = _detect(night([src_run("se", **kw)], _history()))
    assert [(d.finding.source_code, d.finding.category.value, d.finding.severity.value) for d in found] == [
        ("se", category, severity)
    ]
    assert found[0].finding_id == "f1"
    assert found[0].finding.title == f"se: {category}"
    assert found[0].finding.diagnosis == "No explanation."
    assert 1 <= len(found[0].finding.evidence) <= 5
    assert found[0].finding.evidence[0].startswith("[se] ")


def test_volume_drop_rule():
    found = _detect(night([src_run("se", seen=1)], _history()))
    assert [(d.finding.category.value, d.finding.severity.value) for d in found] == [("volume_drop", "warn")]


def test_cadence_rule():
    from myopenfund_agents.core.inputs import CadenceRow

    cadence = [CadenceRow(source_code="se", corpus="central-bank", days_late=2)]
    found = _detect(night([src_run("us")], _history("us"), cadence=cadence))
    assert [(d.finding.source_code, d.finding.category.value, d.finding.severity.value) for d in found] == [
        ("se", "cadence_overdue", "warn")
    ]


def test_discovery_rule():
    found = _detect(night([src_run("se")], _history(), discovery_errors=[_discovery()]))
    assert [(d.finding.category.value, d.finding.severity.value) for d in found] == [("unknown", "warn")]


def _cadence():
    from myopenfund_agents.core.inputs import CadenceRow

    return [CadenceRow(source_code="se", corpus="central-bank", days_late=2)]


FAILED = dict(outcome="failed", exit_code=1)
PAIRS = {
    "bad_content+fetch_errors": (dict(bad_content=1, fetch_errors=1), "unreadable_pdf", "error", ("fetch_errors",)),
    "fetch_errors+volume_drop": (dict(fetch_errors=1, seen=1), "source_site_down", "error", ("volume_drop",)),
    "truncated+failed_run": (dict(truncated=True, **FAILED), "source_site_down", "error", ("failed_run",)),
    "fetch_errors+failed_run": (dict(fetch_errors=2, **FAILED), "source_site_down", "error", ("failed_run",)),
    "volume_drop+docs_failed": (dict(seen=1, docs_failed=1), "volume_drop", "warn", ("docs_failed",)),
    "docs_failed+failed_run": (dict(docs_failed=1, **FAILED), "unknown", "error", ("failed_run",)),
    "volume_drop+failed_run": (dict(seen=1, **FAILED), "volume_drop", "error", ("failed_run",)),
}


@pytest.mark.parametrize("name", list(PAIRS))
def test_adjacent_rows_first_wins(name):
    kw, category, severity, other = PAIRS[name]
    kw = dict(kw)
    seen = kw.pop("seen", 10)
    found = _detect(night([src_run("se", seen=seen, **kw)], _history()))
    assert [(d.finding.category.value, d.finding.severity.value) for d in found] == [(category, severity)]
    assert found[0].other_flags == other


def test_failed_run_beats_cadence_and_discovery():
    found = _detect(night([src_run("se", **FAILED)], _history(), cadence=_cadence()))
    assert [(d.finding.category.value, d.finding.severity.value) for d in found] == [("unknown", "error")]
    assert found[0].other_flags == ("cadence_overdue",)
    found = _detect(night([src_run("se", **FAILED)], _history(), discovery_errors=[_discovery()]))
    assert [(d.finding.category.value, d.finding.severity.value) for d in found] == [("unknown", "error")]
    assert found[0].other_flags == ("open_discovery_errors",)


def test_cadence_beats_discovery():
    found = _detect(night([src_run("us")], _history("us"), cadence=_cadence(), discovery_errors=[_discovery()]))
    assert [(d.finding.category.value, d.finding.severity.value) for d in found] == [("cadence_overdue", "warn")]
    assert found[0].other_flags == ("open_discovery_errors",)


SAMPLES = ["boom one", "boom two", "boom three"]


def _evidence_of(rows, history=None, **kw):
    return _detect(night(rows, _history() if history is None else history, **kw))[0].finding.evidence


def test_evidence_keeps_the_line_of_the_matched_rule():
    lines = _evidence_of([src_run("se", fetch_errors=1, error_samples=SAMPLES)])
    assert lines[0].startswith("[se] ") and lines[1].startswith("  tonight:")
    assert sum(line.startswith("  sample: ") for line in lines) >= 1 and len(lines) == 5


def test_evidence_cadence_keeps_days_late_line():
    rows = [src_run("se", error_samples=SAMPLES, fetch_errors=0)]
    lines = _evidence_of(rows, cadence=_cadence())
    assert any(line.startswith("  days_late=2") for line in lines)
    assert any(line.startswith("  sample: ") for line in lines)


def test_evidence_discovery_keeps_days_late_line():
    lines = _evidence_of([src_run("se", error_samples=SAMPLES)], discovery_errors=[_discovery()])
    assert any("open_discovery_errors=1" in line for line in lines)
    assert any(line.startswith("  sample: ") for line in lines)


def test_evidence_failed_run_keeps_failed_commands():
    lines = _evidence_of([src_run("se", error_samples=SAMPLES, **FAILED)])
    assert any(line.startswith("  failed_commands:") for line in lines)
    assert any(line.startswith("  sample: ") for line in lines)
    assert lines[0].startswith("[se] ") and len(lines) == 5


def test_evidence_volume_drop_keeps_history_and_tonight():
    lines = _evidence_of([src_run("se", seen=1, error_samples=SAMPLES)])
    assert any(line.startswith("  history:") for line in lines)
    assert any(line.startswith("  tonight:") for line in lines)


def test_outage_does_not_cover_unexplained_runs_for_other_categories():
    rows = [src_run(f"s{i}", seen=1) for i in range(6)]
    history = [src_run(f"s{i}", night=n) for i in range(6) for n in range(1, 8)]
    found = _detect(night(rows, history, runs=[_bad_run()]))
    assert sorted(_keys(found)) == [("*", "general_outage"), ("*", "unknown")]


def test_outage_counts_only_the_sources_it_absorbed():
    rows = [src_run(f"v{i}", seen=1) for i in range(6)]  # volume drop only: grouped
    rows += [src_run(f"s{i}", fetch_errors=1) for i in range(3)]  # site down, too few to group
    rows += [src_run(f"c{i}", **FAILED) for i in range(3)]  # failed_run, too few to group
    history = [src_run(f"v{i}", night=n) for i in range(6) for n in range(1, 8)]
    found = _detect(night(rows, history, runs=[_bad_run()]))
    assert ("*", "general_outage") in _keys(found)
    assert ("*", "unknown") in _keys(found)


def test_exactly_threshold_site_down_keeps_everything():
    rows = [src_run(f"s{i}", fetch_errors=1) for i in range(5)]
    found = _detect(night(rows, [], runs=[_bad_run()]))
    assert len(found) == 6 and ("*", "unknown") in _keys(found)


def test_many_unexplained_runs_are_counted_in_the_title():
    runs = [_bad_run(f"run-{i:03d}") for i in range(40)]
    found = _detect(night([src_run("us")], _history("us"), runs=runs))
    assert found[0].finding.title == "*: unknown (40 unexplained failing runs)"
    assert len(found[0].finding.evidence) == 5
    one = _detect(night([src_run("us")], _history("us"), runs=runs[:2]))
    assert one[0].finding.title == "*: unknown"


def test_unexplained_runs_do_not_depend_on_run_order():
    runs = [_bad_run(f"run-{i:03d}") for i in range(8)]
    a = build_summary(night([src_run("us")], _history("us"), runs=runs), SETTINGS)
    b = build_summary(night([src_run("us")], _history("us"), runs=runs[::-1]), SETTINGS)
    assert a.unexplained_runs == b.unexplained_runs


def test_rules_table_is_the_spec_table():
    assert [(c, s) for _, c, s in RULES] == [
        (Category.UNREADABLE_PDF, Severity.WARN),
        (Category.SOURCE_SITE_DOWN, Severity.ERROR),
        (Category.VOLUME_DROP, Severity.WARN),
        (Category.UNKNOWN, Severity.WARN),
        (Category.UNKNOWN, Severity.ERROR),
        (Category.CADENCE_OVERDUE, Severity.WARN),
        (Category.UNKNOWN, Severity.WARN),
    ]
    assert RULES[1][0] == frozenset({"fetch_errors", "truncated"})


def test_first_rule_wins_and_lists_the_others():
    rows = [src_run("se", seen=1, bad_content=1)]
    found = _detect(night(rows, _history()))
    assert len(found) == 1
    assert isinstance(found[0], Detected)
    assert found[0].finding.category == Category.UNREADABLE_PDF
    assert found[0].other_flags == ("volume_drop",)


def test_no_run_adds_unknown():
    found = _detect(night([], [], discovery_errors=[_discovery()]))
    assert _keys(found) == [("se", "unknown"), ("*", "unknown")]
    assert [d.finding_id for d in found] == ["f1", "f2"]
    assert found[1].finding.severity == Severity.ERROR
    assert found[1].finding.evidence == [NO_RUN_LINE]


def test_unexplained_run():
    found = _detect(night([src_run("us")], _history("us"), runs=[_bad_run()]))
    assert _keys(found) == [("*", "unknown")]
    assert found[0].finding.severity == Severity.ERROR
    assert found[0].finding.evidence[0].startswith("run-x (no per-source detail")


def test_unexplained_runs_keep_runs_cut_from_the_text():
    settings = A1Settings(failing_runs_char_budget=200)
    runs = [_bad_run(f"run-{i:03d}") for i in range(40)]
    summary = build_summary(night([src_run("us")], _history("us"), runs=runs), settings)
    assert len(summary.unexplained_runs) == 40
    assert "more failing runs" in summary.text


def test_outage_covers_unexplained_runs():
    rows = [src_run(f"s{i}", fetch_errors=1) for i in range(6)]
    found = _detect(night(rows, [], runs=[_bad_run()]))
    assert _keys(found) == [("*", "general_outage")]


def test_outage_keeps_another_category():
    from myopenfund_agents.core.inputs import CadenceRow

    rows = [src_run(f"s{i}", fetch_errors=1) for i in range(6)]
    cadence = [CadenceRow(source_code="late", corpus="central-bank", days_late=3)]
    found = _detect(night(rows, [], cadence=cadence))
    assert sorted(_keys(found)) == [("*", "general_outage"), ("late", "cadence_overdue")]
    assert [d.finding_id for d in found] == ["f1", "f2"]


def test_omitted_source_is_detected_with_evidence():
    rows = [src_run(f"s{i:03d}", docs_failed=1) for i in range(400)]
    settings = A1Settings(summary_char_budget=1000)
    summary = build_summary(night(rows, []), settings)
    assert len(summary.unusual) == 400
    assert "Omitted for size" in summary.text
    assert "[s399]" not in summary.text
    # 400 sources of one category would group as an outage: raise the threshold to see each finding
    settings = A1Settings(summary_char_budget=1000, general_outage_min_sources=1000)
    found = detect(build_summary(night(rows, []), settings), settings)
    assert len(found) == 400
    last = next(d for d in found if d.finding.source_code == "s399")
    assert last.finding.category == Category.UNKNOWN
    assert last.finding.evidence[0].startswith("[s399] ")
    assert len(last.finding.evidence) == 4


def test_hostile_source_code_respects_schema_limits():
    code = "x" * 500
    found = _detect(night([src_run(code, bad_content=1, error_samples=["a", "b", "c"])], _history(code)))
    assert len(found[0].finding.title) <= 120
    assert found[0].finding.title.endswith(": unreadable_pdf")
    assert len(found[0].finding.evidence) == 5


def test_category_change_changes_the_fingerprint():
    first = _detect(night([src_run("se", seen=1)], _history()))[0].finding
    second = _detect(night([src_run("se", seen=1, bad_content=1)], _history()))[0].finding
    assert first.source_code == second.source_code
    assert fingerprint("repo", first) != fingerprint("repo", second)


def test_green_night_is_empty():
    assert _detect(night([src_run("se")], _history())) == []


def test_detection_is_deterministic():
    rows = [src_run(f"s{i}", fetch_errors=i % 2, docs_failed=1) for i in range(8)]
    rows += [src_run(f"s{i}", run_id=f"extra-{i}", seen=3, docs_failed=1) for i in range(8)]  # several rows a source
    rows += [src_run("f1", outcome="failed", exit_code=1, run_id="rf1"), src_run("f1", seen=2, run_id="rf2")]
    history = [src_run(f"s{i}", night=n) for i in range(8) for n in range(1, 8)]
    runs = [_bad_run(f"run-{i}") for i in range(4)]
    expected = _detect(night(rows, history, runs=runs))
    rng = random.Random(7)
    for _ in range(5):
        shuffled, shuffled_runs = list(rows), list(runs)
        rng.shuffle(shuffled)
        rng.shuffle(shuffled_runs)
        assert _detect(night(shuffled, history, runs=shuffled_runs)) == expected


@pytest.mark.parametrize("crashed, covered", [(5, False), (6, True)])
def test_covering_boundary(crashed, covered):
    rows = [src_run(f"v{i}", seen=1, **(FAILED if i < crashed else {})) for i in range(6)]
    history = [src_run(f"v{i}", night=n) for i in range(6) for n in range(1, 8)]
    found = _detect(night(rows, history, runs=[_bad_run()]))
    assert _keys(found) == [("*", "general_outage")] + ([] if covered else [("*", "unknown")])


def test_small_unrelated_unknowns_are_never_an_outage():
    rows = [src_run(f"s{i}", docs_failed=1) for i in range(6)]
    found = _detect(night(rows))
    assert _keys(found) == [(f"s{i}", "unknown") for i in range(6)]
    assert all(d.finding.evidence[0].startswith(f"[{d.finding.source_code}]") for d in found)
    rows = [src_run(f"d{i}", docs_failed=1) for i in range(2)] + [src_run(f"c{i}", **FAILED) for i in range(2)]
    rows += [src_run(f"o{i}") for i in range(2)]
    found = _detect(night(rows, discovery_errors=[_discovery("o0"), _discovery("o1")]))
    assert sorted(_keys(found)) == [(c, "unknown") for c in ("c0", "c1", "d0", "d1", "o0", "o1")]


def test_docs_failed_with_failed_run_is_not_matched_by_the_failed_run_rule():
    rows = [src_run(f"c{i}", **FAILED) for i in range(5)] + [src_run("d0", docs_failed=1, **FAILED)]
    assert len(_detect(night(rows))) == 6


@pytest.mark.parametrize("n", [5, 6])
def test_failed_run_sources_are_grouped_above_the_threshold(n):
    rows = [src_run(f"c{i}", **FAILED) for i in range(n)]
    found = _detect(night(rows, runs=[_bad_run()]))
    if n == 5:
        assert sorted(_keys(found)) == [("*", "unknown")] + [(f"c{i}", "unknown") for i in range(5)]
        return
    assert _keys(found) == [("*", "general_outage")]  # six absorbed failed_run sources cover the unexplained run
    f = found[0].finding
    assert f.severity == Severity.ERROR and f.evidence == ["sources: c0, c1, c2, c3, c4, c5"]
    assert found[0].finding_id == "f1" and "6 sources" in f.title


def test_failed_run_threshold_is_the_setting():
    rows = [src_run(f"c{i}", **FAILED) for i in range(3)]
    assert len(_detect(night(rows), A1Settings(general_outage_min_sources=3))) == 3
    assert _keys(_detect(night(rows), A1Settings(general_outage_min_sources=2))) == [("*", "general_outage")]


def test_failed_run_with_lower_rows_is_still_grouped():
    rows = [src_run(f"c{i}", **FAILED) for i in range(6)]
    found = _detect(night(rows, discovery_errors=[_discovery("c5")]))
    assert _keys(found) == [("*", "general_outage")]
    assert found[0].finding.evidence == ["sources: c0, c1, c2, c3, c4, c5"]  # c5 has more flags, so it comes first


def test_both_outages_make_one_finding_that_plan_actions_keeps():
    rows = [src_run(f"a{i}", fetch_errors=1) for i in range(6)] + [src_run(f"c{i}", **FAILED) for i in range(6)]
    rows.append(src_run("pdf", bad_content=1))
    found = _detect(night(rows))
    assert sorted(_keys(found)) == [("*", "general_outage"), ("pdf", "unreadable_pdf")]
    outage = next(d.finding for d in found if d.finding.source_code == "*")
    codes = [f"a{i}" for i in range(6)] + [f"c{i}" for i in range(6)]
    assert outage.evidence == [f"sources: {', '.join(codes)}"] and "12 sources" in outage.title
    acts = plan_actions(Findings(findings=[d.finding for d in found]), [], {}, "d", SETTINGS)
    assert [a.finding for a in acts] == [d.finding for d in found]


def test_an_outage_leaves_the_other_kind_below_the_threshold_alone():
    rows = [src_run(f"a{i}", fetch_errors=1) for i in range(6)] + [src_run(f"c{i}", **FAILED) for i in range(2)]
    found = _detect(night(rows))
    assert sorted(_keys(found)) == [("*", "general_outage"), ("c0", "unknown"), ("c1", "unknown")]
    rows = [src_run(f"a{i}", fetch_errors=1) for i in range(2)] + [src_run(f"c{i}", **FAILED) for i in range(6)]
    found = _detect(night(rows))
    assert sorted(_keys(found)) == [("*", "general_outage"), ("a0", "source_site_down"), ("a1", "source_site_down")]


def test_severity_is_the_highest_of_every_matching_row():
    found = _detect(night([src_run("se", docs_failed=1)], _history(), cadence=_cadence()))
    assert [(d.finding.category.value, d.finding.severity.value) for d in found] == [("unknown", "warn")]
    found = _detect(night([src_run("se", seen=1, **FAILED)], _history()))
    assert [(d.finding.category.value, d.finding.severity.value) for d in found] == [("volume_drop", "error")]
