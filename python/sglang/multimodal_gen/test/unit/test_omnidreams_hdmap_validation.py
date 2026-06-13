# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the OmniDreams online HD-map guard (no server, no GPU).

``hdmap_path`` is fed to ``load_video`` / ``load_image`` which open local files
directly, so a raw filesystem path from an untrusted HTTP body is an
arbitrary-file-read vector. ``_validate_http_hdmap_path`` enforces that, over the
HTTP API, ``hdmap_path`` may only be an ``http(s)://`` or ``data:`` URL. CLI
callers build sampling params directly and bypass this guard.

These tests pin that contract (the only OmniDreams-specific online code path).
"""

import pytest
from fastapi import HTTPException

from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _validate_http_hdmap_path,
)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "http://host/clip_hdmap.mp4",
        "https://host/clip_hdmap.mp4",
        "data:video/mp4;base64,AAAA",
        ["http://host/a.mp4", "https://host/b.mp4"],
        # Non-string entries are tolerated (skipped) rather than rejected.
        [123, "https://host/b.mp4"],
        # Scheme matching is case-insensitive and ignores surrounding space.
        "  HTTPS://host/clip.mp4  ",
    ],
)
def test_allows_urls_and_none(value):
    # Must not raise.
    _validate_http_hdmap_path(value)


@pytest.mark.parametrize(
    "value",
    [
        "/root/blockdata/omni-dreams/clip_hdmap.mp4",
        "relative/clip_hdmap.mp4",
        "file:///root/blockdata/clip_hdmap.mp4",
        "ftp://host/clip_hdmap.mp4",
        # A single bad entry in an otherwise-valid list still fails.
        ["https://host/ok.mp4", "/root/blockdata/local.mp4"],
    ],
)
def test_rejects_local_and_non_http_schemes(value):
    with pytest.raises(HTTPException) as exc_info:
        _validate_http_hdmap_path(value)
    assert exc_info.value.status_code == 400
    assert "hdmap_path" in str(exc_info.value.detail)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
