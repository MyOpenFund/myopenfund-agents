"""claude-code adapter: fake runners only, the real `claude -p` is never called."""

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from myopenfund_agents.core.findings import Findings
from myopenfund_agents.core.prompt import A1_INSTRUCTIONS, user_prompt
from myopenfund_agents.engines.base import EngineError
from myopenfund_agents.engines.claude_code_engine import (
    REASON_CHARS,
    ClaudeCodeEngine,
    _run,
    check_env,
    parse_output,
)
from myopenfund_agents.engines.factory import engine_from_env
from myopenfund_agents.settings import A1Settings

TOKEN = "sk-ant-oat-fake-token-value"
ENV = {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN, "PATH": "/usr/bin"}
GOOD = {
    "findings": [
        {
            "source_code": "se",
            "category": "unreadable_pdf",
            "severity": "warn",
            "title": "se: PDFs unreadable",
            "diagnosis": "Three PDFs failed text extraction. The site still answers.",
            "evidence": ["docs_failed=3"],
        }
    ]
}
BAD = {"findings": [{**GOOD["findings"][0], "category": "not_a_category"}]}


def envelope(**kw: Any) -> str:
    values: dict[str, Any] = {"type": "result", "subtype": "success", "is_error": False, "result": ""}
    values.update(kw)
    return json.dumps(values)


class FakeRunner:
    """Answers each call with the next (returncode, stdout) pair, or raises it if it is an exception."""

    def __init__(self, *answers: tuple[int, str] | BaseException) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[list[str], str, Mapping[str, str], float]] = []

    def __call__(self, cmd, stdin_text, env, timeout_s):
        self.calls.append((cmd, stdin_text, env, timeout_s))
        answer = self.answers[min(len(self.calls), len(self.answers)) - 1]
        if isinstance(answer, BaseException):
            raise answer
        code, stdout = answer
        return subprocess.CompletedProcess(cmd, code, stdout=stdout, stderr="")


def engine(runner: FakeRunner, **kw: Any) -> ClaudeCodeEngine:
    return ClaudeCodeEngine(A1Settings(), env=ENV, runner=runner, **kw)


# --- guards -----------------------------------------------------------------------------------------------


def test_claude_env_guards():
    with pytest.raises(EngineError, match="ANTHROPIC_API_KEY") as info:
        check_env({**ENV, "ANTHROPIC_API_KEY": "sk-ant-api-secret"})
    assert "sk-ant-api-secret" not in str(info.value) and TOKEN not in str(info.value)
    with pytest.raises(EngineError, match="CLAUDE_CODE_OAUTH_TOKEN"):
        check_env({"PATH": "/usr/bin"})
    with pytest.raises(EngineError, match="CLAUDE_CODE_OAUTH_TOKEN"):
        check_env({"CLAUDE_CODE_OAUTH_TOKEN": ""})
    check_env({**ENV, "ANTHROPIC_API_KEY": ""})  # empty key: nothing would be billed
    with pytest.raises(EngineError, match="ANTHROPIC_API_KEY"):
        ClaudeCodeEngine(A1Settings(), env={**ENV, "ANTHROPIC_API_KEY": "k"}, runner=FakeRunner())


def test_claude_command_shape():
    cmd = engine(FakeRunner()).command()
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--output-format") + 1] == "json"
    schema = cmd[cmd.index("--json-schema") + 1]
    assert json.loads(schema) == Findings.model_json_schema()
    assert schema == json.dumps(Findings.model_json_schema(), separators=(",", ":"))  # compact
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == A1_INSTRUCTIONS
    assert "--bare" not in cmd and "--mcp-config" not in cmd and "--model" not in cmd
    cmd = engine(FakeRunner(), model="sonnet", binary="/opt/claude").command()
    assert cmd[0] == "/opt/claude"
    assert cmd[-2:] == ["--model", "sonnet"]


# --- parse_output -----------------------------------------------------------------------------------------


def test_parse_structured_and_text():
    assert parse_output(envelope(structured_output=GOOD)).keys() == {("se", "unreadable_pdf")}
    text = "Here are the findings:\n```json\n" + json.dumps(GOOD) + "\n```\nDone."
    assert parse_output(envelope(result=text)).keys() == {("se", "unreadable_pdf")}
    assert parse_output(envelope(result='{"findings": []}')) == Findings(findings=[])
    with pytest.raises(EngineError):
        parse_output(envelope(result="no json at all"))
    with pytest.raises(EngineError):
        parse_output(envelope(structured_output=BAD))
    with pytest.raises(EngineError):
        parse_output("not json")
    with pytest.raises(EngineError):
        parse_output("[1, 2]")


def test_parse_is_error():
    with pytest.raises(EngineError, match="error_during_execution") as info:
        parse_output(envelope(subtype="error_during_execution", is_error=True, result="rate limit reached"))
    assert "rate limit" in str(info.value)
    long = "x" * 5000
    with pytest.raises(EngineError) as info:
        parse_output(envelope(is_error=True, result=long))
    assert len(str(info.value)) < 600


# --- diagnose ---------------------------------------------------------------------------------------------


def test_claude_retry_then_ok():
    runner = FakeRunner(
        (0, "this is not json"),
        (0, envelope(structured_output=GOOD, usage={"input_tokens": 1200, "output_tokens": 80})),
    )
    result = engine(runner).diagnose("night summary")
    assert result.findings.keys() == {("se", "unreadable_pdf")}
    assert result.retries == 1
    assert (result.engine, result.model) == ("claude-code", "default")
    assert (result.input_tokens, result.output_tokens) == (1200, 80)
    assert result.seconds >= 0
    cmd, stdin_text, env, timeout_s = runner.calls[0]
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--bare" not in cmd
    assert stdin_text == user_prompt("night summary")
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == TOKEN and "ANTHROPIC_API_KEY" not in env
    assert timeout_s == 600


def test_claude_retries_nonzero_exit_and_schema_failure():
    runner = FakeRunner((1, ""), (0, envelope(structured_output=BAD)), (0, envelope(structured_output=GOOD)))
    result = engine(runner).diagnose("S")
    assert result.retries == 2
    assert result.input_tokens is None and result.output_tokens is None


def test_claude_gives_up_after_cap():
    runner = FakeRunner((0, envelope(structured_output=BAD)))
    with pytest.raises(EngineError, match=r"claude-code \(sonnet\).*3 attempts") as info:
        ClaudeCodeEngine(A1Settings(max_schema_retries=2), model="sonnet", env=ENV, runner=runner).diagnose("S")
    assert len(runner.calls) == 3
    assert TOKEN not in str(info.value)


def test_claude_error_never_quotes_the_token():
    def leaky(cmd, stdin_text, env, timeout_s):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr=f"invalid token {TOKEN}")

    with pytest.raises(EngineError, match="exit code 1: .*stderr: invalid token <redacted>") as info:
        ClaudeCodeEngine(A1Settings(max_schema_retries=0), env=ENV, runner=leaky).diagnose("S")
    assert TOKEN not in str(info.value)


@pytest.mark.parametrize("code", [0, 1])
def test_claude_rate_limit_fails_fast(code):
    runner = FakeRunner((code, envelope(subtype="error_during_execution", is_error=True, result="rate limit")))
    with pytest.raises(EngineError, match=r"claude-code \(default\).*error_during_execution.*rate limit"):
        ClaudeCodeEngine(A1Settings(max_schema_retries=2), env=ENV, runner=runner).diagnose("S")
    assert len(runner.calls) == 1  # is_error is not retried


def test_parse_requires_findings_key():
    with pytest.raises(EngineError, match="findings"):
        parse_output(envelope(result="Sorry, I cannot {}"))
    with pytest.raises(EngineError, match="findings"):
        parse_output(envelope(structured_output={}))
    with pytest.raises(EngineError, match="findings"):
        parse_output(envelope(result='{"answer": 1}'))


def test_claude_no_silent_empty_findings():
    runner = FakeRunner((0, envelope(result="Sorry, I cannot {}")))
    with pytest.raises(EngineError, match=r"3 attempts.*findings"):
        engine(runner).diagnose("S")
    assert len(runner.calls) == 3


@pytest.mark.parametrize("cut", [-5, -1, 0, 3, 10])
def test_claude_redacts_token_straddling_the_cut(cut):
    # Place the token so that it straddles the REASON_CHARS truncation of stdout (quoted after a prefix) or stderr.
    pad = "x" * (REASON_CHARS - len("output is not JSON: ") + cut - len(TOKEN) // 2)
    leaky_out = pad + TOKEN + " tail" * 80
    leaky_err = "y" * (REASON_CHARS + cut - len(TOKEN) // 2) + TOKEN + " tail" * 80

    def leaky(cmd, stdin_text, env, timeout_s):
        return subprocess.CompletedProcess(cmd, 1, stdout=leaky_out, stderr=leaky_err)

    with pytest.raises(EngineError) as info:
        ClaudeCodeEngine(A1Settings(max_schema_retries=0), env=ENV, runner=leaky).diagnose("S")
    message = str(info.value)
    for prefix in (TOKEN[:6], TOKEN[-6:]):
        assert prefix not in message


def test_claude_nonzero_exit_non_json_keeps_stderr():
    def failing(cmd, stdin_text, env, timeout_s):
        return subprocess.CompletedProcess(cmd, 2, stdout="Error: something", stderr="network unreachable")

    with pytest.raises(EngineError, match=r"exit code 2: output is not JSON.*stderr: network unreachable"):
        ClaudeCodeEngine(A1Settings(max_schema_retries=0), env=ENV, runner=failing).diagnose("S")


def test_default_runner_replaces_undecodable_bytes():
    cmd = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'ok\\xff\\xfe')"]
    completed = _run(cmd, "", dict(os.environ), 30)
    assert completed.returncode == 0
    assert completed.stdout.startswith("ok") and "\ufffd" in completed.stdout


def test_claude_decode_error_is_engine_error():
    runner = FakeRunner(UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"))
    with pytest.raises(EngineError, match=r"claude-code \(default\).*UnicodeDecodeError") as info:
        engine(runner).diagnose("S")
    assert isinstance(info.value.__cause__, UnicodeDecodeError)


@pytest.mark.parametrize(
    "var", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX"]
)
def test_claude_billing_vars_refused(var):
    with pytest.raises(EngineError, match=var) as info:
        check_env({**ENV, var: "secret-value-1"})
    assert "secret-value-1" not in str(info.value)
    check_env({**ENV, var: ""})


def test_claude_timeout_is_engine_error():
    runner = FakeRunner(subprocess.TimeoutExpired(["claude", "-p"], 5))
    with pytest.raises(EngineError, match=r"claude-code \(default\).*timed out") as info:
        engine(runner, timeout_s=5).diagnose("S")
    assert isinstance(info.value.__cause__, subprocess.TimeoutExpired)
    assert len(runner.calls) == 1  # no retry


def test_claude_binary_missing_is_engine_error():
    runner = FakeRunner(FileNotFoundError(2, "No such file or directory", "claude"))
    with pytest.raises(EngineError, match=r"claude-code \(default\)") as info:
        engine(runner).diagnose("S")
    assert isinstance(info.value.__cause__, FileNotFoundError)
    assert len(runner.calls) == 1


# --- factory ----------------------------------------------------------------------------------------------


def test_factory_claude_code():
    built = engine_from_env(A1Settings(), {"MOF_AGENTS_ENGINE": "claude-code", **ENV})
    assert isinstance(built, ClaudeCodeEngine)
    assert (built.name, built.model) == ("claude-code", "default")
    built = engine_from_env(A1Settings(), {"MOF_AGENTS_ENGINE": "claude-code", "MOF_AGENTS_MODEL": "opus", **ENV})
    assert built.model == "opus" and built.command()[-2:] == ["--model", "opus"]
    with pytest.raises(EngineError, match="CLAUDE_CODE_OAUTH_TOKEN"):
        engine_from_env(A1Settings(), {"MOF_AGENTS_ENGINE": "claude-code"})
    with pytest.raises(EngineError, match="ANTHROPIC_API_KEY"):
        engine_from_env(A1Settings(), {"MOF_AGENTS_ENGINE": "claude-code", "ANTHROPIC_API_KEY": "k", **ENV})


def test_claude_timeout_from_settings_unless_explicit():
    runner = FakeRunner((0, envelope(structured_output=GOOD)))
    ClaudeCodeEngine(A1Settings(engine_timeout_s=42.0), env=ENV, runner=runner).diagnose("S")
    assert runner.calls[-1][3] == 42.0
    ClaudeCodeEngine(A1Settings(engine_timeout_s=42.0), env=ENV, runner=runner, timeout_s=7).diagnose("S")
    assert runner.calls[-1][3] == 7


def test_factory_claude_code_uses_timeout_setting():
    built = engine_from_env(A1Settings(engine_timeout_s=42.0), {**ENV, "MOF_AGENTS_ENGINE": "claude-code"})
    assert built._timeout_s == 42.0


def test_real_envelope_names_the_model_and_counts_cached_input():
    """A real `claude -p` output (ids replaced): cached prompt tokens are input too, and `default` is resolved."""
    raw = (Path(__file__).parent / "data" / "claude_output.json").read_text(encoding="utf-8")
    result = engine(FakeRunner((0, raw))).diagnose("s")
    assert result.findings.keys() == {("se", "source_site_down")}
    assert result.model == "claude-sonnet-5-5"
    assert (result.input_tokens, result.output_tokens) == (2570, 378)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"cache_read_input_tokens": 5}, None),
        ({"input_tokens": 1, "cache_read_input_tokens": 5}, 6),
        ({"input_tokens": 1, "cache_creation_input_tokens": "x"}, 1),
        ({"input_tokens": 1, "cache_creation_input_tokens": True}, 1),
    ],
)
def test_input_tokens_with_partial_usage(usage, expected):
    runner = FakeRunner((0, envelope(structured_output=GOOD, usage=usage)))
    assert engine(runner).diagnose("s").input_tokens == expected


@pytest.mark.parametrize("used", [None, {}, [], "x"])
def test_model_falls_back_to_the_configured_one(used):
    runner = FakeRunner((0, envelope(structured_output=GOOD, modelUsage=used)))
    assert engine(runner).diagnose("s").model == "default"


def test_several_models_are_all_named():
    runner = FakeRunner((0, envelope(structured_output=GOOD, modelUsage={"b": {}, "a": {}})))
    assert engine(runner).diagnose("s").model == "a+b"
