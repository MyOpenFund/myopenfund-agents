"""Settings for agent A1. Every threshold is configurable via MOF_AGENTS_A1_<NAME>."""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields

ENV_PREFIX = "MOF_AGENTS_A1_"

# Lowest accepted value per setting read from the environment. Floats must also be finite.
# Floats default to strictly positive; ints default to non-negative (0 is meaningful, e.g. no grace day).
_FLOAT_MIN_EXCLUSIVE = 0.0
_FLOAT_MIN_INCLUSIVE = {"temperature": 0.0}
_INT_MIN = {"volume_window_nights": 1, "summary_char_budget": 1, "error_sample_max_chars": 1}


@dataclass(frozen=True)
class A1Settings:
    volume_window_nights: int = 14
    volume_drop_ratio: float = 0.2
    min_history_nights: int = 5
    cadence_grace_days: int = 0
    cadence_new_max_days: int = 14  # later than this = long overdue, one count line only; 0 = no upper bound
    summary_char_budget: int = 12000
    error_sample_max_chars: int = 300
    error_samples_per_source: int = 3
    general_outage_min_sources: int = 5
    max_new_issues_per_run: int = 5
    max_schema_retries: int = 2
    temperature: float = 0.0
    engine_timeout_s: float = 600.0  # one model request (pydantic-ai) or one `claude -p` call (claude-code)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "A1Settings":
        env = os.environ if env is None else env
        values: dict[str, int | float] = {}
        for f in fields(cls):
            name = ENV_PREFIX + f.name.upper()
            if name not in env:
                continue
            caster = float if f.type in (float, "float") else int
            try:
                value = caster(env[name])
            except ValueError as exc:
                raise ValueError(f"{name}: invalid {caster.__name__} value {env[name]!r}") from exc
            _check_range(name, f.name, value)
            values[f.name] = value
        return cls(**values)


def _check_range(env_name: str, field: str, value: int | float) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{env_name}: must be finite, got {value!r}")
        if field in _FLOAT_MIN_INCLUSIVE:
            if value < _FLOAT_MIN_INCLUSIVE[field]:
                raise ValueError(f"{env_name}: must be >= {_FLOAT_MIN_INCLUSIVE[field]}, got {value!r}")
        elif value <= _FLOAT_MIN_EXCLUSIVE:
            raise ValueError(f"{env_name}: must be > {_FLOAT_MIN_EXCLUSIVE}, got {value!r}")
    elif value < (floor := _INT_MIN.get(field, 0)):
        raise ValueError(f"{env_name}: must be >= {floor}, got {value!r}")
