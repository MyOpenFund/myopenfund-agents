"""Settings for agent A1. Every threshold is configurable via MOF_AGENTS_A1_<NAME>."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields

ENV_PREFIX = "MOF_AGENTS_A1_"


@dataclass(frozen=True)
class A1Settings:
    volume_window_nights: int = 14
    volume_drop_ratio: float = 0.2
    min_history_nights: int = 5
    cadence_grace_days: int = 0
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
                values[f.name] = caster(env[name])
            except ValueError as exc:
                raise ValueError(f"{name}: invalid {caster.__name__} value {env[name]!r}") from exc
        return cls(**values)
