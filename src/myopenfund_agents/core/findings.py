"""Findings produced by an engine; its JSON schema is sent to the model."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

TITLE_MAX_LENGTH = 120
EVIDENCE_MAX_ITEMS = 5


class Category(StrEnum):
    SOURCE_SITE_DOWN = "source_site_down"
    PARSE_ERROR = "parse_error"
    UNREADABLE_PDF = "unreadable_pdf"
    VOLUME_DROP = "volume_drop"
    CADENCE_OVERDUE = "cadence_overdue"
    GENERAL_OUTAGE = "general_outage"
    UNKNOWN = "unknown"


class Severity(StrEnum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_code: str
    category: Category
    severity: Severity
    title: str = Field(max_length=TITLE_MAX_LENGTH)
    diagnosis: str = Field(max_length=2000)
    evidence: list[str] = Field(max_length=EVIDENCE_MAX_ITEMS)


class Findings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[Finding]  # required: a bare {} must never read as a clean night

    def keys(self) -> set[tuple[str, str]]:
        return {(f.source_code, f.category.value) for f in self.findings}
