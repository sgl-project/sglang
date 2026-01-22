import pytest

from sglang.srt.utils.common import validate_vlm_media_url


def test_validate_vlm_media_url_blocks_private(monkeypatch):
    monkeypatch.setenv("SGLANG_VLM_MEDIA_URL_ALLOWED_SCHEMES", "http,https")
    monkeypatch.delenv("SGLANG_VLM_MEDIA_URL_ALLOWLIST", raising=False)
    with pytest.raises(ValueError):
        validate_vlm_media_url("http://127.0.0.1/test", default_timeout=3.0)


def test_validate_vlm_media_url_allows_allowlist(monkeypatch):
    monkeypatch.setenv("SGLANG_VLM_MEDIA_URL_ALLOWED_SCHEMES", "http,https")
    monkeypatch.setenv("SGLANG_VLM_MEDIA_URL_ALLOWLIST", "127.0.0.1")
    validate_vlm_media_url("http://127.0.0.1/test", default_timeout=3.0)


def test_validate_vlm_media_url_disabled(monkeypatch):
    monkeypatch.setenv("SGLANG_VLM_MEDIA_URL_FETCH_ENABLED", "false")
    with pytest.raises(ValueError):
        validate_vlm_media_url("https://example.com/image.png", default_timeout=3.0)
