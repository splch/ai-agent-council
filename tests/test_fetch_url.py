"""fetch_url tool tests — network is monkeypatched; no real HTTP hits the wire."""

from typing import Any, ClassVar

import pytest

from ai_agent_council import tools


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch):
    """Deny all real outbound HTTP by default in this module. Tests that want to
    simulate a response patch the opener explicitly."""

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("network calls are blocked in this test file")

    import urllib.request

    monkeypatch.setattr(urllib.request, "build_opener", _boom)


def test_rejects_non_http_schemes() -> None:
    out = tools.get("fetch_url").fn(url="file:///etc/passwd")
    assert out.startswith("error:")
    assert "http" in out.lower()


def test_rejects_no_hostname() -> None:
    out = tools.get("fetch_url").fn(url="http:///")
    assert out.startswith("error:")


def test_rejects_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    out = tools.get("fetch_url").fn(url="http://127.0.0.1/foo")
    assert out.startswith("error:")
    assert "private" in out.lower() or "loopback" in out.lower()


def test_rejects_private_10_range() -> None:
    out = tools.get("fetch_url").fn(url="http://10.0.0.1/x")
    assert out.startswith("error:")


def test_rejects_link_local(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate DNS returning a link-local (169.254/16) address.
    import socket as real_socket

    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host == "evil.example.com":
            return [(real_socket.AF_INET, 0, 0, "", ("169.254.169.254", 80))]
        raise real_socket.gaierror
    monkeypatch.setattr(tools.socket, "getaddrinfo", fake_getaddrinfo)
    out = tools.get("fetch_url").fn(url="http://evil.example.com/")
    assert out.startswith("error:")


def test_successful_fetch_returns_truncated_body(monkeypatch: pytest.MonkeyPatch) -> None:
    # Let the SSRF check pass.
    monkeypatch.setattr(tools, "_is_public_address", lambda host: True)

    # Unblock build_opener and give it a mock.
    import urllib.request as urllib_request

    class _FakeResp:
        status = 200
        headers: ClassVar[dict[str, str]] = {}

        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self, n: int = -1) -> bytes:
            return self._body[:n] if n > 0 else self._body

        def __enter__(self):
            return self

        def __exit__(self, *a: Any) -> bool:
            return False

    class _FakeOpener:
        def open(self, req: Any, timeout: float) -> Any:
            return _FakeResp(b"hello world" * 500)  # ~5500 bytes

    monkeypatch.setattr(urllib_request, "build_opener", lambda *a, **k: _FakeOpener())

    out = tools.get("fetch_url").fn(url="https://example.com/")
    assert out.startswith("hello world")
    assert len(out) <= 8192


def test_fetch_cap_truncates_at_8kb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "_is_public_address", lambda host: True)
    import urllib.request as urllib_request

    class _FakeResp:
        status = 200
        headers: ClassVar[dict[str, str]] = {}

        def read(self, n: int = -1) -> bytes:
            return (b"x" * 100_000)[:n]

        def __enter__(self):
            return self

        def __exit__(self, *a: Any) -> bool:
            return False

    class _FakeOpener:
        def open(self, req: Any, timeout: float) -> Any:
            return _FakeResp()

    monkeypatch.setattr(urllib_request, "build_opener", lambda *a, **k: _FakeOpener())

    out = tools.get("fetch_url").fn(url="https://example.com/")
    assert len(out) <= 8192


def test_redirect_to_private_host_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Initial host is public; the redirect targets a private host.
    public_then_private = {"initial.example.com": True, "internal.example.com": False}
    monkeypatch.setattr(
        tools, "_is_public_address", lambda host: public_then_private.get(host, False)
    )

    import urllib.request as urllib_request

    class _RedirectResp:
        status = 302
        headers: ClassVar[dict[str, str]] = {"Location": "http://internal.example.com/secret"}

        def read(self, n: int = -1) -> bytes:
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a: Any) -> bool:
            return False

    class _FakeOpener:
        def open(self, req: Any, timeout: float) -> Any:
            return _RedirectResp()

    monkeypatch.setattr(urllib_request, "build_opener", lambda *a, **k: _FakeOpener())

    out = tools.get("fetch_url").fn(url="http://initial.example.com/")
    assert out.startswith("error:")
    non_public_hit = "non-public" in out.lower() or "private" in out.lower()
    assert non_public_hit


def test_max_redirects_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "_is_public_address", lambda host: True)

    import urllib.request as urllib_request

    class _RedirectResp:
        status = 302
        headers: ClassVar[dict[str, str]] = {"Location": "http://example.com/next"}

        def read(self, n: int = -1) -> bytes:
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a: Any) -> bool:
            return False

    class _FakeOpener:
        def open(self, req: Any, timeout: float) -> Any:
            return _RedirectResp()

    monkeypatch.setattr(urllib_request, "build_opener", lambda *a, **k: _FakeOpener())

    out = tools.get("fetch_url").fn(url="http://example.com/")
    assert out.startswith("error:")
    assert "redirect" in out.lower()


def test_http_error_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "_is_public_address", lambda host: True)
    import urllib.request as urllib_request
    from email.message import Message as EmailMessage
    from urllib.error import HTTPError

    class _FakeOpener:
        def open(self, req: Any, timeout: float) -> Any:
            raise HTTPError(req.full_url, 404, "Not Found", EmailMessage(), None)

    monkeypatch.setattr(urllib_request, "build_opener", lambda *a, **k: _FakeOpener())

    out = tools.get("fetch_url").fn(url="https://example.com/missing")
    assert out.startswith("error: HTTP 404")


def test_registered_in_default_tools() -> None:
    assert "fetch_url" in tools.available()


def test_schema_shape() -> None:
    schema = tools.get("fetch_url").to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "fetch_url"
    assert "url" in schema["function"]["parameters"]["properties"]
