"""Fail-closed sanitiser: detects infrastructure details; never edits text, never echoes matches."""

from __future__ import annotations

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

_V4_RE = re.compile(r"(?<!\d)(?<!\d\.)\d{1,3}(?:\.\d{1,3}){3}(?!\d)(?!\.\d)")
_V6_RE = re.compile(r"(?<![\w:.])[0-9A-Fa-f:.]*:[0-9A-Fa-f:.]*(?:%[\w.-]+)?(?![\w])")
_CREDENTIAL_RE = re.compile(r"(?:api_key|token|secret|password|passwd)\s*[:=]\s*\S{6,}", re.I)
_BEARER_RE = re.compile(r"\bbearer\s+[A-Za-z0-9._~+/=-]{8,}", re.I)
_GITHUB_RE = re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")
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


def _parse(candidate: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    base = candidate.rstrip(".")
    tries = [base]
    if base.endswith(":") and not base.endswith("::"):
        tries.append(base[:-1])  # sentence/log punctuation after an address
    for c in tries:
        try:
            return ipaddress.ip_address(c)
        except ValueError:
            continue
    return None


def _has_private_ip(text: str) -> bool:
    for m in _V4_RE.finditer(text):
        try:
            if _v4_private(ipaddress.IPv4Address(m.group())):
                return True
        except ValueError:
            continue
    for m in _V6_RE.finditer(text):
        ip = _parse(m.group())
        if isinstance(ip, ipaddress.IPv6Address) and _v6_private(ip):
            return True
    return False


def violations(text: str, deny: DenyList) -> list[str]:
    """Sorted unique reason names; never the matched value."""
    found: set[str] = set()
    if _has_private_ip(text):
        found.add("private_ip")
    if _CREDENTIAL_RE.search(text):
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
    if any(p and p in text for p in deny.path_roots):
        found.add("deny_path")
    return sorted(found)


def check_tree(obj: Any, deny: DenyList, where: str = "$") -> list[str]:
    """Walk an object; return '<json path>: <reason>' per violation, in walk order."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    out: list[str] = []
    if isinstance(obj, str):
        out += [f"{where}: {r}" for r in violations(obj, deny)]
    elif isinstance(obj, dict):
        for k, v in obj.items():
            child = f"{where}.{k}"
            if isinstance(k, str):
                out += [f"{child}: {r}" for r in violations(k, deny)]
            out += check_tree(v, deny, child)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += check_tree(v, deny, f"{where}[{i}]")
    return out
