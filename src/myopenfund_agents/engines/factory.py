"""Choose the engine from the environment only (MOF_AGENTS_ENGINE, _MODEL, _BASE_URL, _API_KEY)."""

import os
from collections.abc import Mapping

from myopenfund_agents.engines.base import Engine, EngineError
from myopenfund_agents.settings import A1Settings

DEFAULT_ENGINE = "pydantic-ai"


def engine_from_env(settings: A1Settings, env: Mapping[str, str] | None = None) -> Engine:
    env = os.environ if env is None else env
    name = env.get("MOF_AGENTS_ENGINE") or DEFAULT_ENGINE
    if name == "pydantic-ai":
        from myopenfund_agents.engines.pydantic_ai_engine import PydanticAIEngine, build_model

        model = env.get("MOF_AGENTS_MODEL")
        if not model:
            raise EngineError("pydantic-ai: MOF_AGENTS_MODEL is required")
        built = build_model(model, env.get("MOF_AGENTS_BASE_URL") or None, env.get("MOF_AGENTS_API_KEY") or None)
        return PydanticAIEngine(built, settings, model_label=model)
    if name == "claude-code":
        from myopenfund_agents.engines.claude_code_engine import DEFAULT_MODEL, ClaudeCodeEngine

        return ClaudeCodeEngine(
            settings,
            model=env.get("MOF_AGENTS_MODEL") or DEFAULT_MODEL,
            timeout_s=settings.engine_timeout_s,
            env=env,
        )
    raise EngineError(f"unknown MOF_AGENTS_ENGINE {name!r} (expected 'pydantic-ai' or 'claude-code')")
