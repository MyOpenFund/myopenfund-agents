"""Rebuild one past night's inputs from the read-only vault MCP, failing closed.

The captured night is meant to be committed as a bench fixture in a public repository: any
infrastructure detail found by the sanitiser refuses the whole capture, and an incomplete history
page refuses it too, rather than silently producing a wrong baseline.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import TypeAdapter, ValidationError

from myopenfund_agents.core.inputs import CadenceRow, DiscoveryError, NightInputs, RunRow, SourceRunRow
from myopenfund_agents.sanitise import DenyList, check_tree, redact_paths

CallTool = Callable[[str, dict], dict]

PAGE = 200  # vault_runs_status refuses limit > 200
FRESH_CADENCE = timedelta(hours=12)  # source_health is "now" only: trusted only right after the night
_ERROR_TEXT_MAX = 200

_DT = TypeAdapter(datetime)


class CaptureError(RuntimeError):
    """The night cannot be captured faithfully or safely; nothing is written."""


def _aware(dt: datetime) -> datetime:
    """Aware UTC; a naive timestamp is read as UTC."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


_TIMESTAMP_FIELDS = ("started_at", "finished_at")


def _utc_row(row: Any) -> Any:
    """Store the row's compared timestamps as aware UTC, so a night never mixes naive and aware values."""
    for name in _TIMESTAMP_FIELDS:
        value = getattr(row, name, None)
        if isinstance(value, datetime):
            setattr(row, name, _aware(value))
    return row


def _parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        return _aware(_DT.validate_python(value))
    except ValidationError as exc:
        raise CaptureError("unreadable timestamp in MCP response") from exc


def _rows(resp: Any, key: str, *, required: bool = False) -> list[dict[str, Any]]:
    if not isinstance(resp, dict):
        raise CaptureError(f"unexpected MCP response type: {type(resp).__name__}")
    if required and key not in resp:
        raise CaptureError(f"unexpected vault_runs_status shape: no {key!r} key")
    rows = resp.get(key)
    if rows is None:
        rows = []
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise CaptureError(f"unexpected shape for {key!r} in MCP response")
    return rows


def _model(cls: type, row: dict[str, Any]) -> Any:
    try:
        return _utc_row(cls(**row))
    except ValidationError as exc:
        raise CaptureError(f"invalid {cls.__name__} row in MCP response") from exc


def source_codes(call: CallTool, corpus: str) -> list[str]:
    """Source codes of a corpus, from the vault's per-source document statistics."""
    resp = call("vault_corpus_stats", {"corpus": corpus, "group_by": "source_code"})
    # unverified offline: confirm in Task 14 (a top-level list, or the first list value such as "groups")
    groups: Any = resp
    if isinstance(resp, dict):
        groups = next((v for v in resp.values() if isinstance(v, list)), [])
    if not isinstance(groups, list):
        groups = []
    codes: set[str] = set()
    for g in groups:
        if not isinstance(g, dict):
            continue
        # unverified offline: confirm in Task 14 (group code under "key", else "value", else "source_code")
        for field in ("key", "value", "source_code"):
            if field in g:
                code = g[field]
                if isinstance(code, str) and code:
                    codes.add(code)
                break
    if not codes:
        raise CaptureError(f"no source code found for corpus {corpus!r}")
    return sorted(codes)


def capture_night(
    call: CallTool,
    *,
    night_end: datetime,
    corpus: str,
    window_nights: int,
    sources: list[str] | None = None,
    now: datetime | None = None,
    deny: DenyList | None = None,
) -> NightInputs:
    """Rebuild the night ending at ``night_end`` (tonight = (night_end - 1 day, night_end])."""
    night_end = _aware(night_end)
    night_start = night_end - timedelta(days=1)
    since = night_start - timedelta(days=window_nights)
    now = _aware(now) if now is not None else datetime.now(UTC)
    codes = list(sources) if sources is not None else source_codes(call, corpus)

    tonight: list[SourceRunRow] = []
    history: list[SourceRunRow] = []
    seen_source_runs: set[str] = set()  # exact duplicate rows only: no assumption on the row grain
    runs: dict[str, RunRow] = {}
    errors: dict[str, DiscoveryError] = {}
    health: dict[tuple[Any, ...], CadenceRow] = {}

    def when(finished: datetime | None) -> str | None:
        # Review Focus 5: unfinished rows and rows after the night never enter it nor its history.
        if finished is None or finished > night_end:
            return None
        if finished > night_start:
            return "tonight"
        if finished >= since:
            return "history"
        return None

    for code in codes:
        resp = call(
            "vault_runs_status",
            {"corpus": corpus, "source_code": code, "since": since.isoformat(), "limit": PAGE},
        )
        raw_source_runs = _rows(resp, "source_runs", required=True)
        parsed = [_model(SourceRunRow, r) for r in raw_source_runs]
        if len(raw_source_runs) >= PAGE and not _reaches(parsed, since):
            raise CaptureError(f"history incomplete for source {code!r}: a full page of {PAGE} rows")
        for raw, row in zip(raw_source_runs, parsed, strict=True):
            slot = when(row.finished_at)
            key = json.dumps(raw, sort_keys=True, default=str)
            if slot is None or key in seen_source_runs:
                continue
            seen_source_runs.add(key)
            (tonight if slot == "tonight" else history).append(row)

        raw_runs = _rows(resp, "runs")  # run grain, not filtered by source: union across calls
        parsed_runs = [_model(RunRow, r) for r in raw_runs]
        # A full page must provably cover tonight, else tonight's runs may be cut (a false "no run tonight").
        if len(raw_runs) >= PAGE and not _reaches(parsed_runs, night_start):
            raise CaptureError(f"runs page full for source {code!r}: {PAGE} rows do not cover the whole night")
        for run in parsed_runs:
            if run.run_id not in runs and when(run.finished_at) == "tonight":
                runs[run.run_id] = run

        for raw in _rows(resp, "open_discovery_errors"):
            err = _model(DiscoveryError, raw)
            if err.fingerprint in errors:
                continue
            # Approximation: errors open "now" and first seen by the end of the night.
            first_seen = _parse_dt(raw.get("first_seen_at"))
            if first_seen is None or first_seen <= night_end:
                errors[err.fingerprint] = err

        for raw in _rows(resp, "source_health"):
            row = _model(CadenceRow, raw)
            health.setdefault((row.corpus, row.source_code, row.doc_type), row)

    inputs = NightInputs(
        night_id=night_end.date().isoformat(),
        night_start=night_start,
        night_end=night_end,
        runs=list(runs.values()),
        source_runs=tonight,
        history=history,
        discovery_errors=list(errors.values()),
        cadence=list(health.values()) if timedelta(0) <= now - night_end < FRESH_CADENCE else None,
        open_issues=[],
    )
    deny = deny or DenyList()
    found = check_tree(inputs, deny)
    if found:
        raise CaptureError("infra details found: " + "; ".join(redact_paths(found, deny)))
    return inputs


def _reaches(rows: list[Any], bound: datetime) -> bool:
    """True when finished rows exist and the oldest finished_at is at or before ``bound``."""
    finished = [r.finished_at for r in rows if r.finished_at is not None]
    return bool(finished) and min(finished) <= bound


def _unwrap_result(result: Any) -> Any:
    """Payload of an MCP CallToolResult: structured content (lone 'result' key unwrapped), else JSON text."""
    if result.is_error:
        text = _first_text(result) or "tool error"
        raise CaptureError(f"MCP tool error: {text[:_ERROR_TEXT_MAX]}")
    structured = result.structured_content
    if structured is not None:
        if isinstance(structured, dict) and set(structured) == {"result"}:
            return structured["result"]
        return structured
    text = _first_text(result)
    if text is None:
        raise CaptureError("MCP result has neither structured nor text content")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CaptureError("MCP result text is not JSON") from exc


def _first_text(result: Any) -> str | None:
    for item in result.content or []:
        if getattr(item, "type", None) == "text":
            return item.text
    return None


def _guarded(run: Callable[[], Any]) -> Any:
    """Run one MCP call; any failure other than CaptureError becomes a CaptureError naming its type only.

    Never the message: it may quote the request, its headers or the token. The SDK's transport, protocol
    and task-group errors all land here, so no transport package is imported directly.
    """
    try:
        return run()
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"MCP call failed: {type(exc).__name__}") from None


def mcp_caller(url: str, token: str) -> CallTool:  # pragma: no cover - network; exercised in Task 14
    """A synchronous read-only tool caller over MCP streamable HTTP with a bearer token."""
    from mcp import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
    from mcp.types import CallToolResult

    async def _call(name: str, args: dict) -> Any:
        async with create_mcp_http_client(headers={"Authorization": f"Bearer {token}"}) as http:
            async with streamable_http_client(url, http_client=http) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, args)
        if not isinstance(result, CallToolResult):
            raise CaptureError(f"unexpected MCP result kind: {type(result).__name__}")
        return _unwrap_result(result)

    def call(name: str, args: dict) -> Any:
        return _guarded(lambda: asyncio.run(_call(name, args)))

    return call
