import pytest

from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    ensure_path_within_root,
    sanitize_upload_filename,
    validate_openai_media_url,
)


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
