"""claude-code engine: `claude -p` on the subscription (CLAUDE_CODE_OAUTH_TOKEN), no tools, JSON-schema output.

Guards: refuses to start when any of ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, CLAUDE_CODE_USE_BEDROCK or
CLAUDE_CODE_USE_VERTEX is set non-empty (each would bill an API account instead of the subscription), requires
CLAUDE_CODE_OAUTH_TOKEN, never passes `--bare`, disables every tool (`--tools ""`) and every MCP server
(`--strict-mcp-config`). The call is bounded by `A1Settings.engine_timeout_s` unless `timeout_s` is given.
"""

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from myopenfund_agents.core.explanations import Explanations, id_error
from myopenfund_agents.core.findings import Findings
from myopenfund_agents.core.prompt import A1_INSTRUCTIONS, explain_instructions, user_prompt
from myopenfund_agents.engines.base import EngineError, EngineResult
from myopenfund_agents.settings import A1Settings

Runner = Callable[[list[str], str, Mapping[str, str], float], subprocess.CompletedProcess]

TOKEN_VAR = "CLAUDE_CODE_OAUTH_TOKEN"
API_KEY_VAR = "ANTHROPIC_API_KEY"
# Each of these can divert the CLI off the subscription to billed usage (API key auth, Bedrock, Vertex).
BILLING_VARS = (API_KEY_VAR, "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX")
SECRET_VARS = (TOKEN_VAR, API_KEY_VAR, "ANTHROPIC_AUTH_TOKEN")
DEFAULT_MODEL = "default"
REASON_CHARS = 300  # longest slice of CLI output quoted in an error message


def _run(cmd: list[str], stdin_text: str, env: Mapping[str, str], timeout_s: float) -> subprocess.CompletedProcess:
    """Run in a neutral directory so no project CLAUDE.md is picked up; `env` is the full child environment."""
    return subprocess.run(
        cmd,
        input=stdin_text,
        capture_output=True,
        text=True,
        errors="replace",  # undecodable bytes must not raise UnicodeDecodeError
        timeout=timeout_s,
        cwd=tempfile.gettempdir(),
        env=dict(env),
        check=False,
    )


def check_env(env: Mapping[str, str]) -> None:
    """Refuse an environment that would bill the API or has no subscription token. Never quotes the values."""
    for var in BILLING_VARS:
        if env.get(var):
            raise EngineError(f"claude-code: {var} is set; unset it (claude-code must run on the subscription)")
    if not env.get(TOKEN_VAR):
        raise EngineError(f"claude-code: {TOKEN_VAR} is required (run `claude setup-token` to create one)")


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= REASON_CHARS else text[: REASON_CHARS - 1] + "…"


class CLIError(EngineError):
    """The CLI itself reported `is_error` (rate limit, auth, execution error): not retried."""


class AnswerError(EngineError):
    """The model's answer is unusable. `str()` is the detailed reason (logs, EngineError); `feedback` is what the
    model is told on the next attempt: fixed wording and our own field names only, never its text or the CLI's."""

    def __init__(self, detail: str, feedback: str) -> None:
        super().__init__(detail)
        self.feedback = feedback


NOT_JSON = "not valid JSON"


def _field_names(model: type[BaseModel]) -> frozenset[str]:
    """Every property name of the schema: the only strings of a validation error that are ours."""
    schema = model.model_json_schema()
    names = set(schema.get("properties", {}))
    for definition in schema.get("$defs", {}).values():
        names |= set(definition.get("properties", {}))
    return frozenset(names)


def _schema_feedback(exc: ValidationError, model: type[BaseModel], limit: int) -> str:
    known = _field_names(model)
    places: list[str] = []
    kinds: list[str] = []
    for error in exc.errors()[:limit]:
        loc = error["loc"][:-1] if error["type"] == "extra_forbidden" else error["loc"]  # the last one is the model's
        place = ""
        for part in loc:
            place += f"[{part}]" if isinstance(part, int) else (f".{part}" if part in known else "")
        places.append(place.lstrip(".") or "the top level")
        kinds.append(error["type"])
    return f"does not match the schema at: {', '.join(places)} ({', '.join(sorted(set(kinds)))})"


def _validate[M: BaseModel](data: Any, where: str, model: type[M], required: tuple[str, ...], limit: int) -> M:
    # A missing key would validate as an empty list: a silent clean night (Review Focus 3).
    missing = [key for key in required if not isinstance(data, dict) or key not in data]
    if missing:
        names = " and ".join(repr(key) for key in required)
        raise AnswerError(f"{where} has no {names} key", f"missing key(s): {', '.join(missing)}")
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise AnswerError(
            f"{where} does not match the schema: {_short(str(exc))}", _schema_feedback(exc, model, limit)
        ) from exc


def _loads(text: str, detail: str) -> Any:
    """json.loads that fails only with AnswerError: a huge integer (ValueError) or deep nesting (RecursionError)
    is an invalid answer like any other."""
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise AnswerError(detail, NOT_JSON) from exc


def _envelope(stdout: str) -> dict[str, Any]:
    data = _loads(stdout, f"output is not JSON: {_short(stdout) or '(empty)'}")
    if not isinstance(data, dict):
        raise AnswerError(f"output is not a JSON object: {_short(stdout)}", NOT_JSON)
    return data


def _identity(value: Any) -> Any:
    return value


def _parse_data[M: BaseModel](
    data: dict[str, Any],
    model: type[M],
    required: tuple[str, ...],
    limit: int,
    clean: Callable[[Any], Any] = _identity,
) -> M:
    """`model` from the envelope: `structured_output`, else the outermost {…} in `result` (decoded, then `clean`ed)."""
    if data.get("is_error"):
        detail = data.get("result")
        detail = _short(detail) if isinstance(detail, str) else ""
        raise CLIError(f"CLI error ({data.get('subtype', 'unknown')}): {detail}".rstrip(": "))
    structured = data.get("structured_output")
    if isinstance(structured, dict):
        return _validate(structured, "structured_output", model, required, limit)
    result = data.get("result")
    if isinstance(result, str):
        start, end = result.find("{"), result.rfind("}")
        if start != -1 and end > start:
            obj = _loads(result[start : end + 1], f"result holds no valid JSON object: {_short(result)}")
            return _validate(clean(obj), "result", model, required, limit)
    raise AnswerError("output holds neither structured_output nor a JSON object in result", NOT_JSON)


def parse_output(stdout: str) -> Findings:
    return _parse_data(_envelope(stdout), Findings, ("findings",), A1Settings.feedback_errors_max)


def parse_explanations(stdout: str) -> Explanations:
    """Both keys are required: a bare {} or a missing key is never "nothing to explain"."""
    return _parse_data(
        _envelope(stdout), Explanations, ("explanations", "observations"), A1Settings.feedback_errors_max
    )


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _input_tokens(usage: dict[str, Any]) -> int | None:
    """Every input token; the CLI reports cached prompt tokens apart from `input_tokens`."""
    base = _int_or_none(usage.get("input_tokens"))
    if base is None:
        return None
    cached = ("cache_creation_input_tokens", "cache_read_input_tokens")
    return base + sum(_int_or_none(usage.get(name)) or 0 for name in cached)


def _usage_of(data: Any) -> tuple[int | None, int | None]:
    """Input and output tokens of one attempt, when its output is a JSON envelope that reports them."""
    usage = data.get("usage") if isinstance(data, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    return _input_tokens(usage), _int_or_none(usage.get("output_tokens"))


def _add(total: int | None, value: int | None) -> int | None:
    return total if value is None else (value if total is None else total + value)


def _model_used(data: dict[str, Any], configured: str) -> str:
    """The model(s) the CLI says answered, so a report never says only `default`."""
    used = data.get("modelUsage")
    return "+".join(sorted(used)) if isinstance(used, dict) and used else configured


class ClaudeCodeEngine:
    name = "claude-code"

    def __init__(
        self,
        settings: A1Settings,
        model: str = DEFAULT_MODEL,
        binary: str = "claude",
        timeout_s: float | None = None,
        env: Mapping[str, str] | None = None,
        runner: Runner = _run,
    ) -> None:
        self.model = model
        self._settings = settings
        self._binary = binary
        self._timeout_s = settings.engine_timeout_s if timeout_s is None else timeout_s
        self._env: Mapping[str, str] = os.environ if env is None else env
        self._runner = runner
        check_env(self._env)

    def _command(self, schema: type[BaseModel], instructions: str) -> list[str]:
        cmd = [
            self._binary,
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema.model_json_schema(), separators=(",", ":")),
            "--tools",
            "",
            "--strict-mcp-config",
            "--system-prompt",
            instructions,
        ]
        if self.model != DEFAULT_MODEL:
            cmd += ["--model", self.model]
        return cmd

    def command(self) -> list[str]:
        return self._command(Findings, A1_INSTRUCTIONS)

    def explain_command(self) -> list[str]:
        return self._command(Explanations, explain_instructions(self._settings))

    def _redact(self, text: str) -> str:
        """Remove secret values from CLI output before anything is parsed, shortened or quoted."""
        for var in SECRET_VARS:
            secret = self._env.get(var)
            if secret:
                text = text.replace(secret, "<redacted>")
        return text

    def _redact_decoded(self, value: Any) -> Any:
        """Redact the decoded strings (keys included) of a parsed JSON value, in place and without recursion:
        `\\u` escapes hide a token from the text replacement."""
        if isinstance(value, str):
            return self._redact(value)
        stack = [value]
        while stack:
            node = stack.pop()
            if isinstance(node, list):
                pairs = list(enumerate(node))
            elif isinstance(node, dict):
                pairs = list(node.items())
                node.clear()
            else:
                continue
            for key, item in pairs:
                if isinstance(item, str):
                    item = self._redact(item)
                elif isinstance(item, list | dict):
                    stack.append(item)
                if isinstance(node, list):
                    node[key] = item
                else:
                    node[self._redact(key)] = item
        return value

    def _fail(self, message: str) -> EngineError:
        return EngineError(f"{self.name} ({self.model}): {self._redact(message)}")

    def _ask[M: BaseModel](
        self,
        cmd: list[str],
        prompt: str,
        model: type[M],
        required: tuple[str, ...],
        check: Callable[[M], str | None],
        what: str,
        tell_model: bool,
    ) -> tuple[M, dict[str, Any], int, float, int | None, int | None]:
        """Run the CLI until `parse` and `check` accept its answer; `tell_model` appends our feedback to the prompt.

        The feedback is fixed wording about the last attempt only (see AnswerError); a failed process is retried
        with the original prompt. Seconds and tokens add up over every attempt."""
        attempts = self._settings.max_schema_retries + 1
        limit = self._settings.feedback_errors_max
        reason, feedback = "", None
        tokens_in: int | None = None
        tokens_out: int | None = None
        start = time.monotonic()
        for attempt in range(attempts):
            sent = prompt
            if tell_model and feedback:
                sent += f"\nYour previous answer was rejected: {feedback}.\nAnswer again, fixing this.\n"
            try:
                completed = self._runner(cmd, sent, self._env, self._timeout_s)
            except subprocess.TimeoutExpired as exc:
                raise self._fail(f"timed out after {self._timeout_s:g} s") from exc
            except (OSError, ValueError) as exc:  # binary missing; ValueError covers UnicodeDecodeError
                raise self._fail(f"cannot run {self._binary!r}: {type(exc).__name__}: {exc}") from exc
            stdout = self._redact(completed.stdout or "")
            stderr = _short(self._redact(completed.stderr or ""))
            feedback = None
            data: dict[str, Any] = {}
            try:
                data = self._redact_decoded(_envelope(stdout))
                used_in, used_out = _usage_of(data)
                tokens_in = _add(tokens_in, used_in)
                tokens_out = _add(tokens_out, used_out)
                parsed = _parse_data(data, model, required, limit, self._redact_decoded)
            except CLIError as exc:  # rate limit, auth, execution error: fail fast, no retry
                raise self._fail(f"{exc} (exit code {completed.returncode})") from exc
            except EngineError as exc:
                reason = str(exc)
                if completed.returncode != 0:
                    reason = f"exit code {completed.returncode}: {reason}" + (f"; stderr: {stderr}" if stderr else "")
                else:
                    feedback = exc.feedback if isinstance(exc, AnswerError) else NOT_JSON
                continue
            if completed.returncode != 0:
                reason = f"exit code {completed.returncode}" + (f"; stderr: {stderr}" if stderr else "")
                continue
            problem = check(parsed)
            if problem:
                reason = feedback = problem
                continue
            return parsed, data, attempt, time.monotonic() - start, tokens_in, tokens_out
        raise self._fail(f"no valid {what} after {attempts} attempts; last failure: {reason}")

    def _result(
        self, answer: tuple[Any, dict[str, Any], int, float, int | None, int | None], **payload: Any
    ) -> EngineResult:
        _, data, attempt, seconds, tokens_in, tokens_out = answer
        return EngineResult(
            engine=self.name,
            model=_model_used(data, self.model),
            seconds=seconds,
            retries=attempt,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            **payload,
        )

    def diagnose(self, summary_text: str) -> EngineResult:
        answer = self._ask(
            self.command(),
            user_prompt(summary_text),
            Findings,
            ("findings",),
            lambda _: None,
            "findings",
            tell_model=False,
        )
        return self._result(answer, findings=answer[0])

    def explain(self, prompt: str, ids: frozenset[str]) -> EngineResult:
        answer = self._ask(
            self.explain_command(),
            prompt,
            Explanations,
            ("explanations", "observations"),
            lambda e: id_error(e, ids, self._settings),
            "explanations",
            tell_model=True,
        )
        return self._result(answer, findings=Findings(findings=[]), explanations=answer[0])
