import pytest

from conftest import night, src_run
from myopenfund_agents.bench import traps
from myopenfund_agents.bench.fixtures import Expected, Key, Meta, Night, load, load_all, save
from myopenfund_agents.core.summary import build_summary
from myopenfund_agents.settings import A1Settings


def _base(green_history):
    rows = [src_run("us", seen=150), src_run("fr"), src_run("se")]
    return night(source_runs=rows, history=green_history)


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

    t = traps.general_outage(base)
    unusual = _summary(t.inputs).unusual
    assert set(unusual) == {"us", "fr", "se"}
    assert all("failed_run" in s.flags for s in unusual.values())
    assert t.expected.keys() == {("*", "general_outage")}
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
    with pytest.raises(ValueError):
        traps.volume_drop(base, "xx")
    with pytest.raises(ValueError):
        traps.new_source(base, "us")
    with pytest.raises(ValueError):
        traps.new_source(night())
