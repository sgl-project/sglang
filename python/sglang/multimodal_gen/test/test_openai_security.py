import pytest

from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    ensure_path_within_root,
    sanitize_upload_filename,
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
