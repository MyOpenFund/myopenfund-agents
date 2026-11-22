"""pydantic-ai engine: any OpenAI-compatible endpoint (Ollama, llama.cpp, vLLM) or a pydantic-ai model string."""

import os

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

import time  # noqa: E402

import openai  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.exceptions import AgentRunError, UserError  # noqa: E402
from pydantic_ai.models import Model  # noqa: E402
from pydantic_ai.models.openai import OpenAIChatModel  # noqa: E402
from pydantic_ai.providers.openai import OpenAIProvider  # noqa: E402

from myopenfund_agents.core.findings import Findings  # noqa: E402
from myopenfund_agents.core.prompt import A1_INSTRUCTIONS, user_prompt  # noqa: E402
from myopenfund_agents.engines.base import EngineError, EngineResult  # noqa: E402
from myopenfund_agents.settings import A1Settings  # noqa: E402

# AgentRunError covers UnexpectedModelBehavior and every ModelAPIError (HTTP, connection, timeout);
# the others are transport errors a model implementation may let through unwrapped.
ENGINE_FAILURES: tuple[type[BaseException], ...] = (
    AgentRunError,
    UserError,
    OSError,  # ConnectionError, ConnectionRefusedError, TimeoutError
    openai.APIError,  # pydantic-ai already maps these to ModelAPIError; kept as a safety net
)


def build_model(model: str, base_url: str | None, api_key: str | None) -> Model | str:
    """An OpenAI-compatible chat model on `base_url`, else the pydantic-ai model string (key from its env var)."""
    if base_url:
        return OpenAIChatModel(model, provider=OpenAIProvider(base_url=base_url, api_key=api_key or "unused"))
    return model


class PydanticAIEngine:
    name = "pydantic-ai"

    def __init__(self, model: Model | str, settings: A1Settings, model_label: str | None = None) -> None:
        self.model = model_label or (model if isinstance(model, str) else model.model_name)
        try:
            self._agent = Agent(
                model,
                output_type=Findings,
                instructions=A1_INSTRUCTIONS,
                retries=settings.max_schema_retries,
                model_settings={"temperature": settings.temperature},
            )
        except ENGINE_FAILURES as exc:
            raise EngineError(f"{self.name} ({self.model}): cannot build the agent: {exc}") from exc

    def diagnose(self, summary_text: str) -> EngineResult:
        start = time.monotonic()
        try:
            result = self._agent.run_sync(user_prompt(summary_text))
        except ENGINE_FAILURES as exc:
            raise EngineError(f"{self.name} ({self.model}): {type(exc).__name__}: {exc}") from exc
        seconds = time.monotonic() - start
        usage = result.usage  # a property in pydantic-ai 2.x
        return EngineResult(
            findings=result.output,
            engine=self.name,
            model=self.model,
            seconds=seconds,
            retries=max(usage.requests - 1, 0),
            input_tokens=usage.input_tokens or None,
            output_tokens=usage.output_tokens or None,
        )
