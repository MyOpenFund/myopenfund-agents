"""pydantic-ai engine: any OpenAI-compatible endpoint (Ollama, llama.cpp, vLLM) or a pydantic-ai model string."""

import os
import time

import openai
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import AgentRunError, ModelRetry, UserError
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from myopenfund_agents.core.explanations import Explanations, id_error
from myopenfund_agents.core.prompt import explain_instructions
from myopenfund_agents.engines.base import EngineError, EngineResult
from myopenfund_agents.settings import A1Settings

# AgentRunError covers UnexpectedModelBehavior and every ModelAPIError (HTTP, connection, timeout);
# the others are transport errors a model implementation may let through unwrapped.
ENGINE_FAILURES: tuple[type[BaseException], ...] = (
    AgentRunError,
    UserError,
    OSError,  # ConnectionError, ConnectionRefusedError, TimeoutError
    openai.APIError,  # pydantic-ai already maps these to ModelAPIError; kept as a safety net
)
# A provider string whose package is not installed (`anthropic:...`, `groq:...`) fails while the agent is built.
CONSTRUCTION_FAILURES: tuple[type[BaseException], ...] = (*ENGINE_FAILURES, ImportError)


def build_model(model: str, base_url: str | None, api_key: str | None) -> Model | str:
    """An OpenAI-compatible chat model on `base_url`, else the pydantic-ai model string (key from its env var)."""
    if base_url:
        return OpenAIChatModel(model, provider=OpenAIProvider(base_url=base_url, api_key=api_key or "unused"))
    return model


class PydanticAIEngine:
    name = "pydantic-ai"

    def __init__(self, model: Model | str, settings: A1Settings, model_label: str | None = None) -> None:
        self.model = model_label or (model if isinstance(model, str) else model.model_name)
        os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # default only; the user may override it
        model_settings = {
            "temperature": settings.temperature,
            # `timeout`: each HTTP attempt; `request_timeout`: the whole request, SDK retries included.
            "timeout": settings.engine_timeout_s,
            "request_timeout": settings.engine_timeout_s,
        }
        try:
            # The ids differ per call, so they travel as run dependencies: one agent serves every night.
            self._agent = Agent(
                model,
                output_type=Explanations,
                deps_type=frozenset[str],
                instructions=explain_instructions(settings),
                retries=settings.max_schema_retries,
                model_settings=model_settings,
            )
        except CONSTRUCTION_FAILURES as exc:
            raise EngineError(f"{self.name} ({self.model}): cannot build the agent: {exc}") from exc

        @self._agent.output_validator
        def _ids_ok(ctx: RunContext[frozenset[str]], output: Explanations) -> Explanations:
            reason = id_error(output, ctx.deps, settings)
            if reason:
                raise ModelRetry(reason)
            return output

    def explain(self, prompt: str, ids: frozenset[str]) -> EngineResult:
        start = time.monotonic()
        try:
            result = self._agent.run_sync(prompt, deps=ids)
        except ENGINE_FAILURES as exc:
            raise EngineError(f"{self.name} ({self.model}): {type(exc).__name__}: {exc}") from exc
        seconds = time.monotonic() - start
        # A property in pydantic-ai 2.x, a method in some 1.x releases. The 2.x RunUsage is itself callable
        # (deprecated shim), so test for the data rather than `callable`.
        usage = result.usage
        if not hasattr(usage, "requests"):
            usage = usage()
        return EngineResult(
            self.name,
            self.model,
            seconds,
            max(usage.requests - 1, 0),
            usage.input_tokens or None,
            usage.output_tokens or None,
            explanations=result.output,
        )
