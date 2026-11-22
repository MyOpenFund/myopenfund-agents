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
