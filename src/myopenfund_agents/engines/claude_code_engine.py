"""claude-code engine: `claude -p` on the subscription (CLAUDE_CODE_OAUTH_TOKEN), no tools, JSON-schema output.

Guards: refuses to start when ANTHROPIC_API_KEY is set (it would bill the API), requires CLAUDE_CODE_OAUTH_TOKEN,
never passes `--bare`, disables every tool (`--tools ""`) and every MCP server (`--strict-mcp-config`).
"""

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError

from myopenfund_agents.core.findings import Findings
from myopenfund_agents.core.prompt import A1_INSTRUCTIONS, user_prompt
from myopenfund_agents.engines.base import EngineError, EngineResult
from myopenfund_agents.settings import A1Settings

Runner = Callable[[list[str], str, Mapping[str, str], float], subprocess.CompletedProcess]

TOKEN_VAR = "CLAUDE_CODE_OAUTH_TOKEN"
API_KEY_VAR = "ANTHROPIC_API_KEY"
DEFAULT_MODEL = "default"
REASON_CHARS = 300  # longest slice of CLI output quoted in an error message


def _run(cmd: list[str], stdin_text: str, env: Mapping[str, str], timeout_s: float) -> subprocess.CompletedProcess:
    """Run in a neutral directory so no project CLAUDE.md is picked up; `env` is the full child environment."""
    return subprocess.run(
        cmd,
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=tempfile.gettempdir(),
        env=dict(env),
        check=False,
    )


def check_env(env: Mapping[str, str]) -> None:
    """Refuse an environment that would bill the API or has no subscription token. Never quotes the values."""
    if env.get(API_KEY_VAR):
        raise EngineError(f"claude-code: {API_KEY_VAR} is set; unset it (claude-code must run on the subscription)")
    if not env.get(TOKEN_VAR):
        raise EngineError(f"claude-code: {TOKEN_VAR} is required (run `claude setup-token` to create one)")


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= REASON_CHARS else text[: REASON_CHARS - 1] + "…"


def _validate(data: Any, where: str) -> Findings:
    try:
        return Findings.model_validate(data)
    except ValidationError as exc:
        raise EngineError(f"{where} does not match the schema: {_short(str(exc))}") from exc


def _envelope(stdout: str) -> dict[str, Any]:
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise EngineError(f"output is not JSON: {_short(stdout) or '(empty)'}") from exc
    if not isinstance(data, dict):
        raise EngineError(f"output is not a JSON object: {_short(stdout)}")
    return data


def parse_output(stdout: str) -> Findings:
    """Findings from the `--output-format json` envelope: `structured_output`, else the outermost {…} in `result`."""
    data = _envelope(stdout)
    if data.get("is_error"):
        detail = data.get("result")
        detail = _short(detail) if isinstance(detail, str) else ""
        raise EngineError(f"CLI error ({data.get('subtype', 'unknown')}): {detail}".rstrip(": "))
    structured = data.get("structured_output")
    if isinstance(structured, dict):
        return _validate(structured, "structured_output")
    result = data.get("result")
    if isinstance(result, str):
        start, end = result.find("{"), result.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(result[start : end + 1])
            except json.JSONDecodeError as exc:
                raise EngineError(f"result holds no valid JSON object: {_short(result)}") from exc
            return _validate(obj, "result")
    raise EngineError("output holds neither structured_output nor a JSON object in result")


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class ClaudeCodeEngine:
    name = "claude-code"

    def __init__(
        self,
        settings: A1Settings,
        model: str = DEFAULT_MODEL,
        binary: str = "claude",
        timeout_s: float = 600,
        env: Mapping[str, str] | None = None,
        runner: Runner = _run,
    ) -> None:
        self.model = model
        self._settings = settings
        self._binary = binary
        self._timeout_s = timeout_s
        self._env: Mapping[str, str] = os.environ if env is None else env
        self._runner = runner
        check_env(self._env)

    def command(self) -> list[str]:
        schema = json.dumps(Findings.model_json_schema(), separators=(",", ":"))
        cmd = [
            self._binary,
            "-p",
            "--output-format",
            "json",
            "--json-schema",
            schema,
            "--tools",
            "",
            "--strict-mcp-config",
            "--system-prompt",
            A1_INSTRUCTIONS,
        ]
        if self.model != DEFAULT_MODEL:
            cmd += ["--model", self.model]
        return cmd

    def _fail(self, message: str) -> EngineError:
        for var in (TOKEN_VAR, API_KEY_VAR):  # defence in depth: CLI output must never leak a secret
            secret = self._env.get(var)
            if secret:
                message = message.replace(secret, "<redacted>")
        return EngineError(f"{self.name} ({self.model}): {message}")

    def diagnose(self, summary_text: str) -> EngineResult:
        cmd, prompt = self.command(), user_prompt(summary_text)
        attempts = self._settings.max_schema_retries + 1
        reason = ""
        start = time.monotonic()
        for attempt in range(attempts):
            try:
                completed = self._runner(cmd, prompt, self._env, self._timeout_s)
            except subprocess.TimeoutExpired as exc:
                raise self._fail(f"timed out after {self._timeout_s:g} s") from exc
            except OSError as exc:
                raise self._fail(f"cannot run {self._binary!r}: {type(exc).__name__}: {exc}") from exc
            stdout = completed.stdout or ""
            if completed.returncode != 0:
                try:
                    parse_output(stdout)
                    detail = _short(completed.stderr or "")
                except EngineError as exc:
                    detail = str(exc) if stdout.strip() else _short(completed.stderr or "")
                reason = f"exit code {completed.returncode}" + (f": {detail}" if detail else "")
                continue
            try:
                findings = parse_output(stdout)
            except EngineError as exc:
                reason = str(exc)
                continue
            usage = json.loads(stdout).get("usage")
            usage = usage if isinstance(usage, dict) else {}
            return EngineResult(
                findings=findings,
                engine=self.name,
                model=self.model,
                seconds=time.monotonic() - start,
                retries=attempt,
                input_tokens=_int_or_none(usage.get("input_tokens")),
                output_tokens=_int_or_none(usage.get("output_tokens")),
            )
        raise self._fail(f"no valid findings after {attempts} attempts; last failure: {reason}")
