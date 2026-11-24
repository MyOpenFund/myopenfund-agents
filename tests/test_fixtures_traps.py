import pytest
from pydantic import ValidationError

from conftest import night, src_run
from myopenfund_agents.bench import traps
from myopenfund_agents.bench.fixtures import Expected, ExpectedAction, Key, Meta, Night, load, load_all, save
from myopenfund_agents.core.inputs import DiscoveryError, OpenIssue, RunRow
from myopenfund_agents.core.summary import NO_RUN_LINE, build_summary
from myopenfund_agents.settings import A1Settings


def _base(green_history):
    rows = [src_run("us", seen=150), src_run("fr"), src_run("se")]
    return night(source_runs=rows, history=green_history)


def _wide_base():
    rows = [src_run(c) for c in ("a1", "b2", "c3", "d4", "e5", "f6")]
    return night(
        source_runs=rows,
        runs=[RunRow(run_id="run-a1-n0", tool="harvester", command="harvest", outcome="ok", exit_code=0)],
    )


def _summary(inputs):
    return build_summary(inputs, A1Settings())


def test_roundtrip(tmp_path, green_history):
    n = Night(
        inputs=_base(green_history),
        expected=Expected(findings=[Key(source_code="us", category="volume_drop")]),
        meta=Meta(origin="green", description="d", base_night=None, validated_by=None, validated_on=None, notes=[]),
    )
    d = save(n, tmp_path)
    assert d == tmp_path / n.inputs.night_id
    assert (d / "inputs.json").read_text().endswith("\n")
    assert load(d) == n
    assert load_all(tmp_path) == [n]
    assert n.expected.keys() == {("us", "volume_drop")}


def test_traps_show_in_summary(green_history):
    base = _base(green_history)
    before = base.model_copy(deep=True)

    t = traps.volume_drop(base, "us")
    assert t.inputs.night_id == f"{base.night_id}--volume-drop-us"
    assert t.meta.origin == "trap" and t.meta.base_night == base.night_id
    assert _summary(t.inputs).unusual["us"].flags == ["volume_drop"]
    assert t.expected.keys() == {("us", "volume_drop")}
    assert [a.kind for a in t.expected.actions] == ["new_issue"]

    t = traps.bad_content(base, "fr")
    assert _summary(t.inputs).unusual["fr"].flags == ["failed_run", "bad_content"]
    assert t.expected.keys() == {("fr", "unreadable_pdf")}
    assert [(a.kind, a.source_code, a.category) for a in t.expected.actions] == [("new_issue", "fr", "unreadable_pdf")]

    t = traps.general_outage(_wide_base())
    unusual = _summary(t.inputs).unusual
    assert len(unusual) == 6
    assert all("failed_run" in s.flags for s in unusual.values())
    assert t.expected.keys() == {("*", "general_outage")}
    assert [(a.kind, a.source_code, a.category) for a in t.expected.actions] == [("new_issue", "*", "general_outage")]
    assert all(r.outcome == "failed" and r.docs_new == 0 and r.fetch_errors >= 1 for r in t.inputs.source_runs)
    assert all(r.outcome == "failed" and r.exit_code == 1 for r in t.inputs.runs)
    assert t.meta.notes

    t = traps.new_source(base)
    assert "zz" not in _summary(t.inputs).unusual
    assert t.expected.findings == [] and t.expected.actions == []

    t = traps.known_problem(base, "se", "org/repo")
    assert _summary(t.inputs).unusual["se"].flags == ["failed_run", "bad_content"]
    assert len(t.inputs.open_issues) == 1
    assert t.inputs.open_issues[0].fingerprint == "a1:org/repo:se:unreadable_pdf"
    assert [a.kind for a in t.expected.actions] == ["comment"]

    assert base == before


def test_trap_errors(green_history):
    base = _base(green_history)
    with pytest.raises(ValueError, match="xx"):
        traps.volume_drop(base, "xx")
    with pytest.raises(ValueError, match="already"):
        traps.new_source(base, "us")
    with pytest.raises(ValueError, match="no source run"):
        traps.new_source(night())
    with pytest.raises(ValueError, match="general_outage_min_sources"):
        traps.general_outage(base)
    with pytest.raises(ValueError, match="xx"):
        traps.bad_content(base, "xx")


def test_volume_drop_guarantee(green_history):
    base = _base(green_history)
    with pytest.raises(ValueError, match="volume_drop"):
        traps.volume_drop(base, "fr")  # median 10 * 0.2 = 2, seen 3 is not a drop
    with pytest.raises(ValueError, match="volume_drop"):
        traps.volume_drop(night(source_runs=[src_run("us", seen=150)], history=green_history[:6]), "us")
    t = traps.volume_drop(base, "fr", seen=1)
    assert _summary(t.inputs).unusual["fr"].flags == ["volume_drop"]


def test_volume_drop_multiple_rows_sums_to_seen(green_history):
    rows = [src_run("us", seen=100, run_id="r1"), src_run("us", seen=50, run_id="r2")]
    t = traps.volume_drop(night(source_runs=rows, history=green_history), "us", seen=3)
    assert [r.docs_seen for r in t.inputs.source_runs] == [3, 0]


def test_closed_categories():
    with pytest.raises(ValidationError):
        Key(source_code="us", category="unreadable-pdf")
    with pytest.raises(ValidationError):
        ExpectedAction(kind="comment", source_code="us", category="nope")


def test_known_problem_details(green_history):
    base = _base(green_history)
    t = traps.known_problem(base, "se", "org/repo", count=4)
    assert t.inputs.source_runs[2].bad_content == 4
    again = base.model_copy(deep=True)
    again.open_issues.append(OpenIssue(fingerprint="a1:org/repo:se:unreadable_pdf", number=7, repo="org/repo"))
    t = traps.known_problem(again, "se", "org/repo")
    assert len(t.inputs.open_issues) == 1 and t.inputs.open_issues[0].number == 7


def test_new_source_keeps_extra_columns(green_history):
    row = src_run("us", seen=150, extra_col="keep")
    t = traps.new_source(night(source_runs=[row], history=green_history))
    assert t.inputs.source_runs[1].model_extra["extra_col"] == "keep"


def test_save_rejects_unsafe_night_id(tmp_path, green_history):
    n = traps.new_source(_base(green_history))
    for bad in ("a/b", "a\\b", ".", ".."):
        n.inputs.night_id = bad
        with pytest.raises(ValueError, match="night_id"):
            save(n, tmp_path)


def test_trap_roundtrip_and_load_all(tmp_path, green_history):
    base = _base(green_history)
    t1 = traps.volume_drop(base, "us")
    t2 = traps.bad_content(base, "fr")
    save(t2, tmp_path)
    save(t1, tmp_path)
    (tmp_path / "empty-dir").mkdir()
    assert load(tmp_path / t1.inputs.night_id) == t1
    assert [n.inputs.night_id for n in load_all(tmp_path)] == sorted([t1.inputs.night_id, t2.inputs.night_id])


def test_no_run_trap(green_history):
    base = _base(green_history)
    base.runs.append(RunRow(run_id="run-1", tool="harvester", command="harvest", outcome="ok", exit_code=0,
                            finished_at=base.night_end))  # fmt: skip
    before = base.model_copy(deep=True)
    t = traps.no_run(base)
    assert t.inputs.night_id == f"{base.night_id}--no-run"
    assert t.inputs.runs == [] and t.inputs.source_runs == []
    assert t.inputs.history == base.history
    summary = _summary(t.inputs)
    assert NO_RUN_LINE in summary.text
    assert summary.unusual == {}
    assert t.expected.keys() == {("*", "unknown")}
    assert [(a.kind, a.source_code, a.category) for a in t.expected.actions] == [("new_issue", "*", "unknown")]
    assert any("addition to the spec's trap list" in note for note in t.meta.notes)
    assert base == before


def test_no_run_trap_guards_its_ground_truth(green_history):
    base = _base(green_history)
    with_no_run_issue = base.model_copy(deep=True)
    with_no_run_issue.open_issues.append(OpenIssue(fingerprint="a1:data-orchestrator:*:no_run", number=4, repo="r"))
    with pytest.raises(ValueError, match="no-run fingerprint"):
        traps.no_run(with_no_run_issue)
    with_errors = base.model_copy(deep=True)
    with_errors.discovery_errors.append(DiscoveryError(fingerprint="f1", source_code="us"))
    with pytest.raises(ValueError, match="unusual"):
        traps.no_run(with_errors)


def test_no_run_trap_on_base_with_open_chronic_issue(green_history):
    """The no-run fingerprint differs from the chronic ('*', unknown) one: the expected action stays new_issue."""
    base = _base(green_history)
    base.open_issues.append(OpenIssue(fingerprint="a1:data-orchestrator:*:unknown", number=3, repo="r"))
    t = traps.no_run(base)
    assert [(a.kind, a.source_code, a.category) for a in t.expected.actions] == [("new_issue", "*", "unknown")]


def _shared_base(green_history):
    """us, fr and se share one run, as the real corpus commands do."""
    rows = [src_run(c, seen=s, run_id="shared") for c, s in (("us", 150), ("fr", 10), ("se", 10))]
    done = rows[0].finished_at
    run = RunRow(run_id="shared", tool="harvester", command="repec", outcome="ok", exit_code=0, finished_at=done)
    return night(source_runs=rows, runs=[run], history=green_history)


def test_chronic_sequence(green_history):
    base = _shared_base(green_history)
    before = base.model_copy(deep=True)
    seq = traps.chronic(base, "se", "central-bank-corpus")
    assert base == before
    assert [n.inputs.night_id for n in seq] == [f"{base.night_id}--chronic-se-{k}" for k in (1, 2, 3)]
    assert [[a.kind for a in n.expected.actions] for n in seq] == [["new_issue"], ["comment"], ["comment"]]
    assert [len(n.inputs.open_issues) for n in seq] == [0, 1, 1]
    for k, n in enumerate(seq):
        assert n.expected.keys() == {("se", "source_site_down")}
        assert n.meta.origin == "trap" and n.meta.base_night == base.night_id
        s = _summary(n.inputs)
        assert set(s.unusual) == {"se"}  # the siblings of the shared degraded run stay unflagged
        assert s.unusual["se"].consecutive_failed_nights == k
        assert {"failed_run", "fetch_errors"} <= set(s.unusual["se"].flags)
        assert "volume_drop" not in s.unusual["se"].flags
        assert s.failing_runs


def test_chronic_refuses_what_it_cannot_carry(green_history):
    with pytest.raises(ValueError, match="no run in night"):
        traps.chronic(_shared_base(green_history), "zz", "r")
    with pytest.raises(ValueError, match="history"):
        traps.chronic(_shared_base([r for r in green_history if r.finished_at < green_history[0].finished_at]), "se", "r")
    noisy = _shared_base(green_history)
    noisy.source_runs[0].fetch_errors = 2
    with pytest.raises(ValueError, match="other"):
        traps.chronic(noisy, "se", "r")


def test_chronic_refuses_an_issue_already_open(green_history):
    base = _shared_base(green_history)
    base.open_issues.append(OpenIssue(fingerprint="a1:r:se:source_site_down", number=1, repo="r"))
    seq_fp = traps.chronic(_shared_base(green_history), "se", "r")[1].inputs.open_issues[0].fingerprint
    base.open_issues[0].fingerprint = seq_fp
    with pytest.raises(ValueError, match="already"):
        traps.chronic(base, "se", "r")


def test_summary_never_shows_the_trap_name(green_history):
    """The night id of a trap names its answer: the model must not read it."""
    base = _shared_base(green_history)
    made = [traps.volume_drop(base, "us"), traps.bad_content(base, "fr"), traps.no_run(base)]
    made += [traps.new_source(base), traps.known_problem(base, "se", "r"), traps.general_outage(_wide_base())]
    made += traps.chronic(base, "se", "r")
    for n in made:
        text = _summary(n.inputs).text
        assert text.startswith(f"Night {base.night_id}: ")
        assert "--" not in text


@pytest.mark.parametrize("make", [traps.bad_content, lambda b, c: traps.known_problem(b, c, "r")])
def test_content_traps_look_like_a_real_degraded_run(green_history, make):
    """A source with two commands is damaged once, and its run is degraded for every source it shares."""
    base = _shared_base(green_history)
    base.source_runs.append(src_run("se", seen=0, run_id="other", command="discover"))
    s = _summary(make(base, "se").inputs)
    assert s.unusual["se"].bad_content == 2
    assert set(s.unusual) == {"se"}
    assert len(s.failing_runs) == 1


def test_content_trap_refuses_a_count_that_raises_no_signal(green_history):
    with pytest.raises(ValueError, match="bad_content"):
        traps.bad_content(_shared_base(green_history), "se", count=0)
    with pytest.raises(ValueError, match="bad_content"):
        traps.known_problem(_shared_base(green_history), "se", "r", count=0)


def test_chronic_refuses_a_base_with_a_failing_run(green_history):
    base = _shared_base(green_history)
    done = base.runs[0].finished_at
    base.runs.append(RunRow(run_id="c", tool="t", command="c", outcome="failed", exit_code=1, finished_at=done))
    with pytest.raises(ValueError, match="failing run"):
        traps.chronic(base, "se", "r")


def test_chronic_damages_one_command_only(green_history):
    base = _shared_base(green_history)
    base.source_runs.append(src_run("se", seen=0, run_id="other", command="discover"))
    for n in traps.chronic(base, "se", "r"):
        assert _summary(n.inputs).unusual["se"].fetch_errors == 1
