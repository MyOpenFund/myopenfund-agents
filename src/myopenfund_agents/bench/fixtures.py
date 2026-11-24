"""Bench night fixtures: a night's inputs, what A1 must find, and where it comes from."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from myopenfund_agents.core.findings import Category
from myopenfund_agents.core.inputs import NightInputs


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Key(_Model):
    source_code: str
    category: Category


class ExpectedAction(_Model):
    kind: Literal["new_issue", "comment", "summary_issue"]
    source_code: str
    category: Category


class Expected(_Model):
    findings: list[Key] = []
    actions: list[ExpectedAction] = []

    def keys(self) -> set[tuple[str, str]]:
        return {(k.source_code, k.category.value) for k in self.findings}


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


def _by_row(data: dict) -> str:
    """JSON with one line per list item: a night holds hundreds of rows, and a diff should show rows."""

    def value(v: object) -> str:
        if not isinstance(v, list) or not v:
            return json.dumps(v, ensure_ascii=False)
        return "[\n" + ",\n".join("    " + json.dumps(x, ensure_ascii=False) for x in v) + "\n  ]"

    return "{\n" + ",\n".join(f"  {json.dumps(k)}: {value(v)}" for k, v in data.items()) + "\n}"


def save(night: Night, root: Path) -> Path:
    night_id = night.inputs.night_id
    if not night_id or "/" in night_id or "\\" in night_id or night_id in (".", ".."):
        raise ValueError(f"unsafe night_id {night_id!r}")
    d = root / night_id
    d.mkdir(parents=True, exist_ok=True)
    dumped = night.model_dump(mode="json")
    for part in _PARTS:
        text = _by_row(dumped[part]) if part == "inputs" else json.dumps(dumped[part], indent=2, ensure_ascii=False)
        text += "\n"
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
