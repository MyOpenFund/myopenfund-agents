import pytest

from myopenfund_agents.settings import A1Settings


def test_defaults():
    s = A1Settings()
    assert s.volume_window_nights == 14
    assert s.volume_drop_ratio == 0.2


def test_env_override():
    s = A1Settings.from_env({"MOF_AGENTS_A1_VOLUME_DROP_RATIO": "0.5", "MOF_AGENTS_A1_MIN_HISTORY_NIGHTS": "3"})
    assert s.volume_drop_ratio == 0.5
    assert s.min_history_nights == 3


def test_env_bad_value_raises():
    with pytest.raises(ValueError, match="MOF_AGENTS_A1_MIN_HISTORY_NIGHTS"):
        A1Settings.from_env({"MOF_AGENTS_A1_MIN_HISTORY_NIGHTS": "three"})


def test_engine_timeout_setting():
    assert A1Settings().engine_timeout_s == 600.0
    assert A1Settings.from_env({"MOF_AGENTS_A1_ENGINE_TIMEOUT_S": "30"}).engine_timeout_s == 30.0


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("ENGINE_TIMEOUT_S", "inf"),
        ("ENGINE_TIMEOUT_S", "nan"),
        ("ENGINE_TIMEOUT_S", "-inf"),
        ("ENGINE_TIMEOUT_S", "0"),
        ("ENGINE_TIMEOUT_S", "-5"),
        ("VOLUME_DROP_RATIO", "0"),
        ("VOLUME_DROP_RATIO", "nan"),
        ("TEMPERATURE", "-0.1"),
        ("TEMPERATURE", "inf"),
        ("MIN_HISTORY_NIGHTS", "-1"),
        ("MAX_SCHEMA_RETRIES", "-1"),
        ("CADENCE_GRACE_DAYS", "-1"),
        ("VOLUME_WINDOW_NIGHTS", "0"),
        ("SUMMARY_CHAR_BUDGET", "0"),
        ("ERROR_SAMPLE_MAX_CHARS", "0"),
    ],
)
def test_env_out_of_range_raises(name, value):
    with pytest.raises(ValueError, match=f"MOF_AGENTS_A1_{name}"):
        A1Settings.from_env({f"MOF_AGENTS_A1_{name}": value})


def test_env_zero_allowed_where_meaningful():
    s = A1Settings.from_env(
        {
            "MOF_AGENTS_A1_TEMPERATURE": "0",
            "MOF_AGENTS_A1_CADENCE_GRACE_DAYS": "0",
            "MOF_AGENTS_A1_MAX_SCHEMA_RETRIES": "0",
            "MOF_AGENTS_A1_MIN_HISTORY_NIGHTS": "0",
        }
    )
    assert s.temperature == 0.0 and s.cadence_grace_days == 0 and s.max_schema_retries == 0
