"""Tests for the URL-derived local-vs-remote distinction.

Mirrors the host-as-SSOT pattern adopted by amplifier-module-provider-ollama.
The base_url is the single source of truth: localhost URLs are treated as
local; anything else is treated as remote (capability-tagged accordingly).
This avoids the divergence cases the ollama provider's reviewer flagged
(see microsoft/amplifier-module-provider-ollama#13).
"""

from amplifier_module_provider_vllm import _is_remote_host


# ---------- _is_remote_host() pure function ----------


def test_localhost_is_not_remote():
    assert _is_remote_host("http://localhost:8000/v1") is False


def test_127_0_0_1_is_not_remote():
    assert _is_remote_host("http://127.0.0.1:8000/v1") is False


def test_ipv6_loopback_is_not_remote():
    assert _is_remote_host("http://[::1]:8000/v1") is False


def test_0_0_0_0_is_not_remote():
    """0.0.0.0 means "all interfaces" but in practice users use it for local binds."""
    assert _is_remote_host("http://0.0.0.0:8000/v1") is False


def test_lan_ip_is_remote():
    assert _is_remote_host("http://192.168.1.10:8000/v1") is True


def test_public_hostname_is_remote():
    assert _is_remote_host("https://my-vllm.example.com/v1") is True


def test_runpod_style_url_is_remote():
    assert _is_remote_host("https://abc123.proxy.runpod.net/v1") is True


def test_anyscale_style_url_is_remote():
    assert _is_remote_host("https://api.endpoints.anyscale.com/v1") is True


def test_empty_url_is_not_remote():
    assert _is_remote_host("") is False


def test_none_url_is_not_remote():
    assert _is_remote_host(None) is False


def test_localhost_with_no_port_is_not_remote():
    assert _is_remote_host("http://localhost/v1") is False


def test_bare_hostname_is_remote():
    """A non-loopback bare hostname (no scheme) parses as netloc=''.

    urlparse("foo.example.com") returns ParseResult with path='foo.example.com'
    and netloc=''. We treat empty netloc as not-remote (defensive default —
    this is malformed input).
    """
    # Confirms the defensive default: malformed URLs don't get tagged remote.
    assert _is_remote_host("foo.example.com") is False
