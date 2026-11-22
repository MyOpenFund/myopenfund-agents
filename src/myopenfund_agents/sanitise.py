"""Fail-closed sanitiser: detects infrastructure details; never edits text, never echoes matches."""

from __future__ import annotations

import dataclasses
import ipaddress
import re
import tomllib
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path, PurePath
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
_CRED_WORDS = r"(?:api[_-]?key|apikey|token|secret|password|passwd)"
_CREDENTIAL_RE = re.compile(
    r"""(?:""" + _CRED_WORDS + r"""\\?["']?\s*[:=]|pwd\\?["']?\s*=)\s*\\?["']?[^\s"']{6,}""", re.I
)
_CRED_KEYS = (
    "password", "passwd", "pwd", "secret", "token", "api_key", "apikey", "api-key",
    "access_token", "auth_token", "secret_key", "client_secret",
)  # fmt: skip
_USERINFO_RE = re.compile(
    r"(?<![a-z0-9+.-])[a-z][a-z0-9+.-]{0,30}://(?:[^/\s@:]{0,256}:[^\s@]{1,512}|[A-Za-z0-9_-]{8,128})@", re.I
)
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


def _octet(x: str) -> int:
    z = x.lstrip("0")
    if not z:
        return 0
    return int(z) if len(z) <= 3 else 999  # over-long components are never an address octet


@lru_cache(maxsize=8192)
def _v6_private_text(cand: str) -> bool:
    try:
        return _v6_private(ipaddress.IPv6Address(cand))
    except ValueError:
        return False


def _v6_run_private(run: str) -> bool:
    colons = [i for i, c in enumerate(run) if c == ":"]
    if len(colons) < 2:
        return False
    ks = range(2, 8) if "::" in run else ((7, 6) if "." in run else (7,))
    starts = [0] + [i + 1 for i, c in enumerate(run) if c in ":."]
    for s in starts:
        j = bisect_left(colons, s)
        for k in ks:
            if j + k - 1 >= len(colons):
                continue
            ck = colons[j + k - 1]
            if ck - s >= _V6_MAX_LEN:
                break
            nxt = colons[j + k] if j + k < len(colons) else len(run)
            for e in range(ck + 1, min(nxt, s + _V6_MAX_LEN) + 1):
                if (e == nxt or run[e] == ".") and _v6_private_text(run[s:e]):
                    return True
    return False


def _has_private_ip(text: str) -> bool:
    for m in _V4_RUN_RE.finditer(text):
        octets = [_octet(x) for x in m.group().split(".")]
        for i in range(len(octets) - 3):
            win = octets[i : i + 4]
            if max(win) <= 255 and _v4_private(ipaddress.IPv4Address(bytes(win))):
                return True
    return any(_v6_run_private(m.group()) for m in _V6_RUN_RE.finditer(text))


def _path_hit(root: str, text: str) -> bool:
    if not root:
        return False
    bare = root.rstrip("/")
    if not bare:
        return root in text
    return re.search(re.escape(bare) + r"(?:(?![\w.-])|(?=\.(?:\s|\Z)))", text) is not None


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


def _is_cred_key(key: str) -> bool:
    k = key.lower()
    return any(k == w or k.endswith(("_" + w, "-" + w)) for w in _CRED_KEYS)


def check_tree(obj: Any, deny: DenyList, where: str = "$", _cred: bool = False) -> list[str]:
    """Walk an object; return '<json path>: <reason>' per violation, in walk order.

    Fails closed: only None/bool/int/float/str/bytes/containers/pydantic models/dataclasses/paths are
    understood; any other type is reported as ``unsupported_type``.
    """
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    elif isinstance(obj, PurePath):
        obj = str(obj)
    out: list[str] = []
    if obj is None or isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if _cred and len(str(obj)) >= 6:
            out.append(f"{where}: credential")
        return out
    if isinstance(obj, (str, bytes, bytearray)):
        text = _as_text(obj)
        reasons = set(violations(text, deny))
        if _cred and len(text) >= 6:
            reasons.add("credential")
        return [f"{where}: {r}" for r in sorted(reasons)]
    if isinstance(obj, (dict, list, tuple, set, frozenset)):
        if _cred and obj:
            out.append(f"{where}: credential")
        if isinstance(obj, dict):
            for k, v in obj.items():
                ks = _as_text(k)
                child = f"{where}.{ks}"
                out += [f"{child}: {r}" for r in violations(ks, deny)]
                out += check_tree(v, deny, child, _cred=_cred or _is_cred_key(ks))
        else:
            for i, v in enumerate(obj):
                out += check_tree(v, deny, f"{where}[{i}]", _cred=_cred)
        return out
    return [f"{where}: unsupported_type"]
