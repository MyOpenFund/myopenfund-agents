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
