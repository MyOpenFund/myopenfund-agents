"""Traps are rebuilt from their base night by `traps.toml`, never stored; nights are stored one row per line."""

import json

import pytest

from conftest import night, src_run
from myopenfund_agents.bench.fixtures import Expected, Meta, Night, load, save
from myopenfund_agents.bench.traps import load_bench
from myopenfund_agents.cli import main
from myopenfund_agents.core.inputs import RunRow
from myopenfund_agents.engines.fake import FakeEngine


def _save_base(root, green_history):
    rows = [src_run(c, seen=s, run_id="shared") for c, s in (("us", 150), ("fr", 10), ("se", 10))]
    done = rows[0].finished_at
    run = RunRow(run_id="shared", tool="t", command="repec", outcome="ok", exit_code=0, finished_at=done)
    base = Night(
        inputs=night(source_runs=rows, runs=[run], history=green_history),
        expected=Expected(),
        meta=Meta(origin="green", description="d", validated_by="Marc"),
    )
    save(base, root)
    return base


def _recipe(root, base, *lines):
    (root / "traps.toml").write_text("\n".join([f'base = "{base.inputs.night_id}"', *lines]) + "\n")


def test_night_file_has_one_row_per_line(tmp_path, green_history):
    base = _save_base(tmp_path, green_history)
    d = tmp_path / base.inputs.night_id
    lines = (d / "inputs.json").read_text().splitlines()
    assert len([x for x in lines if '"run_id"' in x]) == len(green_history) + 3 + 1
    assert all(json.loads(x.rstrip(",")) for x in lines if '"run_id"' in x)
    assert load(d) == base


def test_load_bench_builds_the_traps(tmp_path, green_history):
    base = _save_base(tmp_path, green_history)
    _recipe(tmp_path, base, '[[trap]]', 'kind = "volume_drop"', 'code = "us"',
            '[[trap]]', 'kind = "chronic"', 'code = "se"', 'repo = "central-bank-corpus"')  # fmt: skip
    i = base.inputs.night_id
    expected = [i, f"{i}--chronic-se-1", f"{i}--chronic-se-2", f"{i}--chronic-se-3", f"{i}--volume-drop-us"]
    assert [n.inputs.night_id for n in load_bench(tmp_path)] == expected
    assert len(list(tmp_path.iterdir())) == 2  # nothing written


def test_no_recipe_file_means_no_trap(tmp_path, green_history):
    base = _save_base(tmp_path, green_history)
    assert load_bench(tmp_path) == [base]
    assert load_bench(tmp_path / "missing") == []


@pytest.mark.parametrize(
    ("lines", "match"),
    [
        (['[[trap]]', 'kind = "load"'], "unknown trap kind"),
        (['[[trap]]', 'code = "us"'], "unknown trap kind"),
        (['[[trap]]', 'kind = "volume_drop"', 'code = "us"', 'nope = 1'], "volume_drop"),
        (['[[trap]]', 'kind = "volume_drop"', 'code = "zz"'], "no run in night"),
        (['[[trap]]', 'kind = "no_run"'] * 2, "twice"),
        (['[[trap]'], "traps.toml"),
        (['trap = 3'], "traps.toml"),
    ],
)  # fmt: skip
def test_bad_recipes_are_refused(tmp_path, green_history, lines, match):
    _recipe(tmp_path, _save_base(tmp_path, green_history), *lines)
    with pytest.raises(ValueError, match=match):
        load_bench(tmp_path)


def test_unknown_base_is_refused(tmp_path, green_history):
    _save_base(tmp_path, green_history)
    (tmp_path / "traps.toml").write_text('base = "1999-01-01"\n')
    with pytest.raises(ValueError, match="1999-01-01"):
        load_bench(tmp_path)


def test_cli_reports_a_bad_recipe_as_configuration_error(tmp_path, green_history, monkeypatch, capsys):
    _recipe(tmp_path, _save_base(tmp_path, green_history), '[[trap]]', 'kind = "nope"')
    monkeypatch.setattr("myopenfund_agents.cli.engine_from_env", lambda settings: FakeEngine(None))
    assert main(["bench-run", "--profile", "local", "--nights", str(tmp_path)]) == 2
    assert "unknown trap kind" in capsys.readouterr().err


def test_base_must_be_green(tmp_path, green_history):
    base = _save_base(tmp_path, green_history)
    d = tmp_path / base.inputs.night_id
    save(base.model_copy(update={"meta": base.meta.model_copy(update={"origin": "incident"})}), tmp_path)
    assert d.is_dir()
    _recipe(tmp_path, base, "[[trap]]", 'kind = "new_source"')
    with pytest.raises(ValueError, match="green"):
        load_bench(tmp_path)


@pytest.mark.parametrize(
    "lines",
    [
        ["[[traps]]", 'kind = "no_run"'],
        ["[[trap]]", 'kind = "no_run"', 'settings = "x"'],
        ["[[trap]]", 'kind = "new_source"', "code = 5"],
        ["[[trap]]", 'kind = "new_source"', 'code = ""'],
        ["[[trap]]", 'kind = "new_source"', 'code = "a/b"'],
        ["[[trap]]", 'kind = "volume_drop"', 'code = "us"', "seen = -5"],
        ["[[trap]]", 'kind = "bad_content"', 'code = "fr"', "count = [1]"],
        ["[[trap]]", 'kind = "bad_content"', 'code = "fr"', "count = true"],
        ["[[trap]]", 'kind = "chronic"', 'code = "se"', "repo = 3"],
        ["[[trap]]", 'kind = "bad_content"', 'code = "zz"'],
        ["[[trap]]", 'kind = "bad_content"'],
    ],
)
def test_bad_recipe_errors_name_the_file(tmp_path, green_history, lines):
    _recipe(tmp_path, _save_base(tmp_path, green_history), *lines)
    with pytest.raises(ValueError, match=r"^traps\.toml: "):
        load_bench(tmp_path)


@pytest.mark.parametrize("kind", ["known_problem", "chronic"])
def test_recipe_repo_must_be_the_one_the_planner_uses(tmp_path, green_history, kind):
    _recipe(tmp_path, _save_base(tmp_path, green_history), "[[trap]]", f'kind = "{kind}"', 'code = "se"', 'repo = "typo"')
    with pytest.raises(ValueError, match=r"^traps\.toml: .*typo.*central-bank-corpus"):
        load_bench(tmp_path)


def test_base_must_show_no_unusual_source(tmp_path, green_history):
    base = _save_base(tmp_path, green_history)
    base.inputs.source_runs[0].fetch_errors = 3
    save(base, tmp_path)
    _recipe(tmp_path, base, "[[trap]]", 'kind = "bad_content"', 'code = "fr"')
    with pytest.raises(ValueError, match=r"^traps\.toml: .*unusual"):
        load_bench(tmp_path)


def test_new_source_must_be_new_in_history_too(tmp_path, green_history):
    base = _save_base(tmp_path, green_history)
    base.inputs.source_runs = [r for r in base.inputs.source_runs if r.source_code != "us"]
    save(base, tmp_path)
    _recipe(tmp_path, base, "[[trap]]", 'kind = "new_source"', 'code = "us"')
    with pytest.raises(ValueError, match=r"^traps\.toml: .*history"):
        load_bench(tmp_path)


def test_recipe_file_without_trap_is_refused(tmp_path, green_history):
    _recipe(tmp_path, _save_base(tmp_path, green_history))
    with pytest.raises(ValueError, match=r"^traps\.toml: no trap"):
        load_bench(tmp_path)
