import asyncio
import ipaddress

import pytest

from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    ensure_path_within_root,
    sanitize_upload_filename,
    validate_openai_media_url,
)


class DummyResponse:
    def __init__(self, status_code=200, headers=None, chunks=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks or []

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class DummyStream:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class DummyAsyncClient:
    def __init__(self, responses, *args, **kwargs):
        self._responses = iter(responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, timeout=None):
        return DummyStream(next(self._responses))


def test_sanitize_upload_filename_basename():
    assert sanitize_upload_filename("../../evil.txt", "fallback") == "evil.txt"


def test_sanitize_upload_filename_fallback_for_dotdot():
    assert sanitize_upload_filename("..", "fallback") == "fallback"


def test_ensure_path_within_root_rejects_escape(tmp_path):
    root = tmp_path / "uploads"
    root.mkdir()
    with pytest.raises(ValueError):
        ensure_path_within_root(root / ".." / "evil.txt", root)


def test_validate_openai_media_url_blocks_private(monkeypatch):
    monkeypatch.setenv("SGLANG_OPENAI_MEDIA_URL_ALLOWED_SCHEMES", "http,https")
    monkeypatch.delenv("SGLANG_OPENAI_MEDIA_URL_ALLOWLIST", raising=False)
    with pytest.raises(ValueError):
        validate_openai_media_url("http://127.0.0.1/test")


def test_validate_openai_media_url_allows_allowlist(monkeypatch):
    monkeypatch.setenv("SGLANG_OPENAI_MEDIA_URL_ALLOWED_SCHEMES", "http,https")
    monkeypatch.setenv("SGLANG_OPENAI_MEDIA_URL_ALLOWLIST", "127.0.0.1")
    validate_openai_media_url("http://127.0.0.1/test")


def test_validate_openai_media_url_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_OPENAI_MEDIA_URL_FETCH_ENABLED", "false")
    with pytest.raises(ValueError):
        validate_openai_media_url("https://example.com/image.png")


def test_validate_openai_media_url_blocks_resolved_private(monkeypatch):
    monkeypatch.setenv("SGLANG_OPENAI_MEDIA_URL_ALLOWED_SCHEMES", "http,https")
    monkeypatch.delenv("SGLANG_OPENAI_MEDIA_URL_ALLOWLIST", raising=False)
    monkeypatch.setattr(
        openai_utils,
        "_resolve_host_ips",
        lambda hostname: [ipaddress.ip_address("10.0.0.1")],
    )
    with pytest.raises(ValueError):
        validate_openai_media_url("http://example.com/test")


def test_openai_media_url_redirect_limit(monkeypatch, tmp_path):
    policy = {
        "enabled": True,
        "schemes": {"http", "https"},
        "allow_hosts": set(),
        "allow_nets": [],
        "max_bytes": 1024,
        "max_redirects": 1,
        "timeout": 1.0,
    }
    responses = [
        DummyResponse(302, headers={"location": "http://example.com/next"}),
        DummyResponse(302, headers={"location": "http://example.com/final"}),
    ]
    monkeypatch.setattr(
        openai_utils.httpx,
        "AsyncClient",
        lambda *args, **kwargs: DummyAsyncClient(responses),
    )
    monkeypatch.setattr(
        openai_utils, "_validate_media_url_with_policy", lambda url, policy: None
    )
    target_path = tmp_path / "image"
    with pytest.raises(ValueError, match="Too many redirects"):
        asyncio.run(
            openai_utils._save_url_image_to_path(
                "http://example.com/start", str(target_path), policy=policy
            )
        )


def test_openai_media_url_size_limit(monkeypatch, tmp_path):
    policy = {
        "enabled": True,
        "schemes": {"http", "https"},
        "allow_hosts": set(),
        "allow_nets": [],
        "max_bytes": 4,
        "max_redirects": 0,
        "timeout": 1.0,
    }
    responses = [
        DummyResponse(
            200,
            headers={"content-length": "10", "content-type": "image/png"},
            chunks=[b"1234"],
        )
    ]
    monkeypatch.setattr(
        openai_utils.httpx,
        "AsyncClient",
        lambda *args, **kwargs: DummyAsyncClient(responses),
    )
    monkeypatch.setattr(
        openai_utils, "_validate_media_url_with_policy", lambda url, policy: None
    )
    target_path = tmp_path / "image"
    with pytest.raises(ValueError, match="Remote content exceeds max size limit"):
        asyncio.run(
            openai_utils._save_url_image_to_path(
                "http://example.com/image.png", str(target_path), policy=policy
            )
        )


def test_openai_media_url_stream_limit(monkeypatch, tmp_path):
    policy = {
        "enabled": True,
        "schemes": {"http", "https"},
        "allow_hosts": set(),
        "allow_nets": [],
        "max_bytes": 4,
        "max_redirects": 0,
        "timeout": 1.0,
    }
    responses = [
        DummyResponse(
            200,
            headers={"content-type": "image/png"},
            chunks=[b"123", b"456"],
        )
    ]
    monkeypatch.setattr(
        openai_utils.httpx,
        "AsyncClient",
        lambda *args, **kwargs: DummyAsyncClient(responses),
    )
    monkeypatch.setattr(
        openai_utils, "_validate_media_url_with_policy", lambda url, policy: None
    )
    target_path = tmp_path / "image"
    with pytest.raises(ValueError, match="Remote content exceeds max size limit"):
        asyncio.run(
            openai_utils._save_url_image_to_path(
                "http://example.com/image.png", str(target_path), policy=policy
            )
        )
