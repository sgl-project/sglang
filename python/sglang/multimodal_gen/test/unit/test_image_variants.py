"""Regression tests for url-format image responses with n > 1.

Covers the fix for the bug where `response_format="url"` with `n > 1` returned
only one `data` item and `/v1/images/{id}/content?variant=N` ignored `variant`.
"""

import tempfile

import pytest

from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    download_image_content,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import IMAGE_STORE


@pytest.mark.asyncio
async def test_download_content_resolves_variant():
    p0 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    p1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    await IMAGE_STORE.upsert(
        "rid",
        {"file_path": p0, "url": None, "file_paths": [p0, p1], "urls": [None, None]},
    )

    # variant 0 (default) and variant 1 map to the right file
    assert (await download_image_content(image_id="rid", variant=None)).path == p0
    assert (await download_image_content(image_id="rid", variant="1")).path == p1

    # out-of-range variant is a 404, not a silent wrong file
    with pytest.raises(Exception) as exc:
        await download_image_content(image_id="rid", variant="5")
    assert getattr(exc.value, "status_code", None) == 404


@pytest.mark.asyncio
async def test_download_content_backward_compatible():
    # items stored before per-variant lists existed still serve variant 0
    p0 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    await IMAGE_STORE.upsert("old", {"file_path": p0, "url": None})
    assert (await download_image_content(image_id="old", variant=None)).path == p0
