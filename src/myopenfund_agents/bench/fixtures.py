"""Bench night fixtures: a night's inputs, what A1 must find, and where it comes from."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from myopenfund_agents.core.inputs import NightInputs


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Key(_Model):
    source_code: str
    category: str


class ExpectedAction(_Model):
    kind: Literal["new_issue", "comment", "summary_issue"]
    source_code: str
    category: str


class Expected(_Model):
    findings: list[Key] = []
    actions: list[ExpectedAction] = []

    def keys(self) -> set[tuple[str, str]]:
        return {(k.source_code, k.category) for k in self.findings}


class Meta(_Model):
    origin: Literal["green", "incident", "trap"]
    description: str
    base_night: str | None = None
    validated_by: str | None = None
    validated_on: str | None = None
    notes: list[str] = []


class Night(_Model):
    inputs: NightInputs
    expected: Expected
    meta: Meta


_PARTS = ("inputs", "expected", "meta")


def save(night: Night, root: Path) -> Path:
    d = root / night.inputs.night_id
    d.mkdir(parents=True, exist_ok=True)
    dumped = night.model_dump(mode="json")
    for part in _PARTS:
        text = json.dumps(dumped[part], indent=2, ensure_ascii=False) + "\n"
        (d / f"{part}.json").write_text(text, encoding="utf-8")
    return d


def load(d: Path) -> Night:
    parts = {p: json.loads((d / f"{p}.json").read_text(encoding="utf-8")) for p in _PARTS}
    return Night.model_validate(parts)


def load_all(root: Path) -> list[Night]:
    if not root.is_dir():
        return []
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "inputs.json").is_file())
    return [load(d) for d in dirs]
