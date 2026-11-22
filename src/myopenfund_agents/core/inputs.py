"""Night inputs: typed rows read from the vault MCP. New vault columns never break A1."""

from datetime import UTC, datetime
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, field_validator


def _naive_as_utc(dt: datetime) -> datetime:
    """A timestamp without offset (e.g. in a hand-edited fixture) is read as UTC; aware values are kept."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt


UtcDatetime = Annotated[datetime, AfterValidator(_naive_as_utc)]


class _Row(BaseModel):
    model_config = ConfigDict(extra="allow")


class RunRow(_Row):
    run_id: str
    tool: str
    command: str
    corpus: str | None = None
    started_at: UtcDatetime | None = None
    finished_at: UtcDatetime | None = None
    outcome: str | None = None
    exit_code: int | None = None
    totals: dict[str, Any] | None = None


class SourceRunRow(_Row):
    run_id: str
    tool: str
    command: str
    corpus: str | None = None
    source_code: str
    started_at: UtcDatetime | None = None
    finished_at: UtcDatetime | None = None
    outcome: str | None = None
    exit_code: int | None = None
    docs_seen: int | None = None
    docs_new: int | None = None
    docs_failed: int | None = None
    fetch_errors: int | None = None
    truncated: bool | None = None
    error_samples: list[Any] = []
    bad_content: Any = None

    @field_validator("error_samples", mode="before")
    @classmethod
    def _null_samples_to_empty(cls, v: Any) -> Any:
        return [] if v is None else v


class DiscoveryError(_Row):
    fingerprint: str
    corpus: str | None = None
    source_code: str | None = None
    doc_type: str | None = None
    error_class: str | None = None
    error: str | None = None
    http_status: int | None = None
    occurrences: int | None = None


class CadenceRow(_Row):
    corpus: str | None = None
    source_code: str
    doc_type: str | None = None
    days_late: float | None = None
    cadence_status: str | None = None


class OpenIssue(BaseModel):
    fingerprint: str
    number: int
    repo: str


class NightInputs(BaseModel):
    night_id: str
    night_start: UtcDatetime
    night_end: UtcDatetime
    runs: list[RunRow]
    source_runs: list[SourceRunRow]
    history: list[SourceRunRow]
    discovery_errors: list[DiscoveryError]
    cadence: list[CadenceRow] | None = None  # None: not known for this night
    open_issues: list[OpenIssue]
