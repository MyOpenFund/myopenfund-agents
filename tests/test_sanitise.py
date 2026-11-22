import pytest
from pydantic import BaseModel

from myopenfund_agents.sanitise import DenyList, check_tree, violations


def test_seeded_tailnet_ip_and_nas_path_refused():
    deny = DenyList(path_roots=["/srv/private/"])
    text = "fetch failed from 100.64.1.2 writing /srv/private/data/x.pdf"
    assert violations(text, deny) == ["deny_path", "private_ip"]


@pytest.mark.parametrize("addr", ["fd7a:115c:a1e0::1", "fe80::1", "::1", "fc00::5", "fe80::1%eth0"])
def test_ipv6_private_refused(addr):
    assert violations(f"connect to {addr} failed", DenyList()) == ["private_ip"]


@pytest.mark.parametrize("addr", ["10.0.0.1", "172.16.5.4", "172.31.255.1", "192.168.1.1", "127.0.0.1", "169.254.1.1"])
def test_ipv4_private_refused(addr):
    assert violations(f"host {addr}.", DenyList()) == ["private_ip"]


def test_public_text_passes():
    url = "https://www.riksbank.se/en-gb/press-and-published/notices-and-press-releases/"
    assert violations(url, DenyList()) == []
    assert violations("resolver 8.8.8.8 timed out", DenyList()) == []
    assert violations("172.32.0.1 and 1.2.3.4.5", DenyList()) == []


def test_adversarial_ip_forms():
    assert violations("peer ::ffff:10.0.0.1 reset", DenyList()) == ["private_ip"]
    assert violations("peer ::ffff:8.8.8.8 reset", DenyList()) == []
    assert violations("http://[fd7a:115c:a1e0::5]:8080/", DenyList()) == ["private_ip"]
    assert violations("dns 2001:4860:4860::8888 ok", DenyList()) == []
    for text in ["started 06:00:00", "at 12:34", "a:b", "x::y", "1:2:3:4:5:6:7:8", "12:34:56.789", ":::"]:
        assert violations(text, DenyList()) == [], text


def test_credentials_and_signed_urls():
    d = DenyList()
    assert violations("api_key=abcdef123", d) == ["credential"]
    assert violations("Password: hunter22", d) == ["credential"]
    assert violations("password=abc", d) == []
    assert violations("Authorization: Bearer abcdefgh12345", d) == ["bearer"]
    assert violations("bearer abc", d) == []
    assert violations("tok ghp_" + "a" * 20, d) == ["github_token"]
    assert violations("tok ghp_short", d) == []
    assert violations("https://x.example/f?X-Amz-Signature=0123456789abcdef", d) == ["signed_url"]
    assert violations("https://x.example/f?sig=0123456789", d) == ["signed_url"]
    assert violations("https://x.example/f?token=0123456789", d) == ["credential", "signed_url"]
    assert violations("sig=short", d) == []


def test_reasons_never_contain_value():
    out = violations("password=SuperSecretValue1 at 10.1.2.3", DenyList())
    assert out == ["credential", "private_ip"]
    assert not any("Secret" in r or "10.1" in r for r in out)


def test_host_case_insensitive_and_tree_paths():
    deny = DenyList(hosts=["nas.example.lan"])
    assert check_tree({"a": [{"b": "see NAS.example.lan"}]}, deny) == ["$.a[0].b: deny_host"]


def test_path_case_sensitive():
    deny = DenyList(path_roots=["/srv/private/"])
    assert violations("/SRV/PRIVATE/x", deny) == []


def test_tree_keys_scalars_and_models():
    deny = DenyList(hosts=["nas.example.lan"])
    tree = {"nas.example.lan": 1, "n": 5, "ok": None, "l": [True, "10.0.0.1"]}
    assert check_tree(tree, deny) == ["$.nas.example.lan: deny_host", "$.l[1]: private_ip"]

    class M(BaseModel):
        note: str

    assert check_tree(M(note="from 100.64.1.2"), DenyList()) == ["$.note: private_ip"]
    assert check_tree("clean", DenyList()) == []


def test_load(tmp_path):
    p = tmp_path / "deny.toml"
    p.write_text('hosts = ["nas.example.lan"]\npath_roots = ["/srv/private/"]\n')
    d = DenyList.load(p)
    assert d.hosts == ["nas.example.lan"] and d.path_roots == ["/srv/private/"]
    assert DenyList.load(None) == DenyList()
    p.write_text('hosts = ["a"]\n')
    assert DenyList.load(str(p)).path_roots == []
    p.write_text('host = ["a"]\n')
    with pytest.raises(ValueError, match="host"):
        DenyList.load(p)


@pytest.mark.parametrize(
    "text",
    [
        "host:fd7a:115c:a1e0::1",
        "Host:fe80::1 down",
        "[IPv6:fe80::1]",
        "ip6:fd7a:115c:a1e0::1",
        "addr_fe80::1",
        "error:::1 refused",
        "at 12:00:01:fe80::1",
        "range fe80::1..fe80::5",
        "fe80::1.a",
        "::1:5432",
        "localhost ::1:5432 refused",
        "IP 10.0.0.1.443 > 10.0.0.2.5555",
        "10.0.0.1.5",
        "3.10.0.0.1",
        "010.0.0.1",
        "10.000.0.1",
        "user:pw@x https://bob:hunter22@nas.example.lan/x",
        '{"password": "hunter22"}',
        "'password': 'hunter22'",
        "pwd=hunter22",
        "apikey=abcdef123",
        "api-key: abcdef123",
        "github_pat_" + "A1_" * 10,
    ],
)
def test_review_round1_bypasses_refused(text):
    assert violations(text, DenyList()) != []


def test_path_root_without_trailing_slash():
    deny = DenyList(path_roots=["/srv/private/"])
    assert violations("mount /srv/private", deny) == ["deny_path"]
    assert violations("mount /srv/private/", deny) == ["deny_path"]
    assert violations("/srv/privateer", deny) == []


def test_check_tree_unknown_types_fail_closed():
    import dataclasses
    from pathlib import PurePosixPath

    deny = DenyList(path_roots=["/srv/private/"])

    @dataclasses.dataclass
    class D:
        x: str

    assert check_tree(PurePosixPath("/srv/private/a"), deny) == ["$: deny_path"]
    assert check_tree(b"from 100.64.1.2", deny) == ["$: private_ip"]
    assert check_tree({"100.64.1.2"}, deny) == ["$[0]: private_ip"]
    assert check_tree(frozenset({"100.64.1.2"}), deny) == ["$[0]: private_ip"]
    assert check_tree(("a", "100.64.1.2"), deny) == ["$[1]: private_ip"]
    assert check_tree(D("100.64.1.2"), deny) == ["$.x: private_ip"]
    assert check_tree(RuntimeError("connect 100.64.1.2"), deny) == ["$: unsupported_type"]
    assert check_tree({("100.64.1.2", 1): 1}, deny) == ["$.('100.64.1.2', 1): private_ip"]
    assert check_tree([None, True, 3, 1.5], deny) == []


def test_check_tree_credential_keys():
    assert check_tree({"password": "hunter22abc"}, DenyList()) == ["$.password: credential"]
    assert check_tree({"api_key": "abcdef123", "n": "ok"}, DenyList()) == ["$.api_key: credential"]
    assert check_tree({"password": "abc"}, DenyList()) == []
    assert check_tree({"password": "password=hunter22"}, DenyList()) == ["$.password: credential"]


def test_linear_time_on_adversarial_input():
    import time

    t = time.monotonic()
    assert violations("ab" * 100_000, DenyList()) == []
    assert violations("1:" * 100_000, DenyList()) == []
    assert violations("1::" * 66_000, DenyList()) == []
    assert violations("1." * 100_000, DenyList()) == []
    assert violations("http://" + "a:" * 100_000, DenyList()) == []
    assert violations("9" * 5000 + ".1.1.1", DenyList()) == []
    assert time.monotonic() - t < 2


def test_unsupported_types_fail_closed():
    import enum

    class Color(enum.Enum):
        RED = 1

    def gen():
        yield "100.64.1.2"

    exc_group = ExceptionGroup("g", [ValueError("x")])
    for bad in (gen(), iter(["a"]), Color.RED, object(), memoryview(b"abc"), exc_group):
        assert check_tree({"k": bad}, DenyList()) == ["$.k: unsupported_type"]


@pytest.mark.parametrize(
    "text",
    [
        r"{\"password\": \"hunter22\"}",
        "redis://:hunter22@cache:6379",
        "postgres://u:pa/ss@db/x",
        "https://tokenvalue123@github.com/x",
        "pwd=hunter22",
    ],
)
def test_round2_credentials_refused(text):
    assert violations(text, DenyList()) == ["credential"]


def test_round2_credential_false_positives_pass():
    assert violations("pwd: /home/x/project", DenyList()) == []
    assert violations("https://user@example.org/x", DenyList()) == []
    assert check_tree({"tokenizer": "cl100k_base", "token_count": "123456"}, DenyList()) == []


def test_round2_credential_keys_structured():
    d = DenyList()
    assert check_tree({"db_password": "hunter22"}, d) == ["$.db_password: credential"]
    assert check_tree({"auth-token": b"hunter22"}, d) == ["$.auth-token: credential"]
    assert check_tree({"secret": ["x"]}, d) == ["$.secret: credential"]
    assert check_tree({"secret": {"a": "abcdefgh"}}, d) == ["$.secret: credential", "$.secret.a: credential"]
    assert check_tree({"secret": []}, d) == []


@pytest.mark.parametrize(
    "text",
    [
        "mount /srv/private now",
        "ls /srv/private\n",
        "'/srv/private'",
        "/srv/private, then",
        "/srv/private:ro",
        "see /srv/private.",
        "/srv/private",
    ],
)
def test_round2_path_root_boundaries(text):
    assert violations(text, DenyList(path_roots=["/srv/private/"])) == ["deny_path"]
    assert violations(text, DenyList(path_roots=["/srv/private"])) == ["deny_path"]


def test_round2_path_root_lookalikes_pass():
    deny = DenyList(path_roots=["/srv/private/"])
    assert violations("/srv/privateer /srv/private.bak /srv/private-x", deny) == []


@pytest.mark.parametrize(
    "text",
    [
        "fd00:1:2:3:4:5:6::",
        "fe80:1:2:3:4:5:6::",
        "fd00:0:0:0:0:0:1.2.3.4.0.0.0.0.0.0.0.0.0.0.0.0.0.0.0:1",
        "a" * 40 + ".https://u:pw@h",
        "1https://u:pw@h",
    ],
)
def test_round3_bypasses_refused(text):
    assert violations(text, DenyList()) != []


@pytest.mark.parametrize("text", ["(see /srv/private.)", '"/srv/private."', "/srv/private.,", "/srv/private..."])
def test_round3_path_trailing_dot(text):
    assert violations(text, DenyList(path_roots=["/srv/private/"])) == ["deny_path"]
    assert violations("/srv/private.bak", DenyList(path_roots=["/srv/private/"])) == []


def test_round3_huge_int_under_credential_key():
    assert check_tree({"password": 10**5000}, DenyList()) == ["$.password: credential"]
    assert check_tree({"token": 12}, DenyList()) == []


def test_round3_unsupported_key_types():
    import enum

    class Color(enum.Enum):
        RED = 1

    def gen():
        yield 1

    for key in (Color.RED, object(), (Color.RED, 1), gen()):
        out = check_tree({key: 1}, DenyList())
        assert len(out) == 1 and out[0].endswith(": unsupported_type")
    assert check_tree({1: "a", None: "b", 1.5: "d", b"k": "e"}, DenyList()) == []
    assert check_tree({(b"100.64.1.2", 1): 1}, DenyList()) == ["$.(b'100.64.1.2', 1): private_ip"]
