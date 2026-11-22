"""Fail-closed sanitiser: detects infrastructure details; never edits text, never echoes matches."""

from __future__ import annotations

import dataclasses
import ipaddress
import re
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_V4_NETS = [
    ipaddress.ip_network(n)
    for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10", "127.0.0.0/8", "169.254.0.0/16")
]
_V6_NETS = [ipaddress.ip_network(n) for n in ("::1/128", "fc00::/7", "fe80::/10")]

_V4_RUN_RE = re.compile(r"[0-9]+(?:\.[0-9]+)+")
_V6_RUN_RE = re.compile(r"[0-9A-Fa-f:.]+")
_V6_MAX_LEN = 45
_CRED_WORDS = r"(?:api[_-]?key|apikey|token|secret|password|passwd|pwd)"
_CREDENTIAL_RE = re.compile(_CRED_WORDS + r"""["']?\s*[:=]\s*["']?[^\s"']{6,}""", re.I)
_CRED_KEY_RE = re.compile(_CRED_WORDS, re.I)
_USERINFO_RE = re.compile(r"[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@", re.I)
_BEARER_RE = re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]{8,}", re.I)
_GITHUB_RE = re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")
_SIGNED_RE = re.compile(r"(?:x-amz-signature|signature|sig|token)=[^\s&\"']{8,}", re.I)


class DenyList(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    hosts: list[str] = Field(default_factory=list)
    path_roots: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path | None) -> DenyList:
        if path is None:
            return cls()
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        unknown = sorted(set(data) - {"hosts", "path_roots"})
        if unknown:
            raise ValueError(f"unknown deny-list keys: {', '.join(unknown)}")
        return cls(**data)


def _v4_private(ip: ipaddress.IPv4Address) -> bool:
    return any(ip in n for n in _V4_NETS)


def _v6_private(ip: ipaddress.IPv6Address) -> bool:
    if ip.ipv4_mapped is not None:
        return _v4_private(ip.ipv4_mapped)
    return any(ip in n for n in _V6_NETS)


def _has_private_ip(text: str) -> bool:
    for m in _V4_RUN_RE.finditer(text):
        parts = m.group().split(".")
        octets = [int(x) for x in parts]
        for i in range(len(octets) - 3):
            win = octets[i : i + 4]
            if all(o <= 255 for o in win) and _v4_private(ipaddress.IPv4Address(bytes(win))):
                return True
    for m in _V6_RUN_RE.finditer(text):
        run = m.group()
        if run.count(":") < 2:
            continue
        starts = [0] + [i + 1 for i, c in enumerate(run) if c in ":."]
        ends = [len(run)] + [i for i, c in enumerate(run) if c in ":."]
        for s in starts:
            for e in ends:
                if e <= s or e - s > _V6_MAX_LEN or run.count(":", s, e) < 2:
                    continue
                try:
                    ip = ipaddress.IPv6Address(run[s:e])
                except ValueError:
                    continue
                if _v6_private(ip):
                    return True
    return False


def _path_hit(root: str, text: str) -> bool:
    if not root:
        return False
    if root in text:
        return True
    bare = root.rstrip("/")
    return bool(bare) and re.search(re.escape(bare) + r"(?:/|\Z)", text) is not None


def violations(text: str, deny: DenyList) -> list[str]:
    """Sorted unique reason names; never the matched value."""
    found: set[str] = set()
    if _has_private_ip(text):
        found.add("private_ip")
    if _CREDENTIAL_RE.search(text) or _USERINFO_RE.search(text):
        found.add("credential")
    if _BEARER_RE.search(text):
        found.add("bearer")
    if _GITHUB_RE.search(text):
        found.add("github_token")
    if _SIGNED_RE.search(text):
        found.add("signed_url")
    low = text.lower()
    if any(h and h.lower() in low for h in deny.hosts):
        found.add("deny_host")
    if any(_path_hit(p, text) for p in deny.path_roots):
        found.add("deny_path")
    return sorted(found)


def _as_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray)):
        return bytes(x).decode("utf-8", errors="replace")
    return str(x)


def check_tree(obj: Any, deny: DenyList, where: str = "$", _cred: bool = False) -> list[str]:
    """Walk an object; return '<json path>: <reason>' per violation, in walk order.

    Fails closed: every non-trivial type is stringified and checked; only None/bool/int/float pass.
    """
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    out: list[str] = []
    if obj is None or isinstance(obj, (bool, int, float)):
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = _as_text(k)
            child = f"{where}.{ks}"
            out += [f"{child}: {r}" for r in violations(ks, deny)]
            out += check_tree(v, deny, child, _cred=_CRED_KEY_RE.search(ks) is not None)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            out += check_tree(v, deny, f"{where}[{i}]")
    else:
        text = _as_text(obj)
        reasons = set(violations(text, deny))
        if _cred and isinstance(obj, str) and len(text) >= 6:
            reasons.add("credential")
        out += [f"{where}: {r}" for r in sorted(reasons)]
    return out
