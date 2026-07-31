from pathlib import Path

import pytest
from PIL import Image

from sglang.multimodal_gen.runtime.server_warmup import (
    MINIMUM_PICTURE_BASE64_FOR_WARMUP,
)
from sglang.multimodal_gen.test.server import test_server_utils
from sglang.multimodal_gen.test.test_utils import is_image_url


@pytest.mark.parametrize(
    ("image_source", "expected"),
    [
        ("https://example.com/input.png", True),
        ("HTTP://example.com/input.png", True),
        (MINIMUM_PICTURE_BASE64_FOR_WARMUP, True),
        ("data:text/plain;base64,SGVsbG8=", False),
        ("input.png", False),
        (Path("input.png"), False),
        (None, False),
    ],
)
def test_is_image_url_supports_embedded_images(
    image_source: str | Path | None,
    expected: bool,
) -> None:
    assert is_image_url(image_source) is expected


def test_download_image_from_data_url_preserves_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(test_server_utils.tempfile, "gettempdir", lambda: str(tmp_path))

    image_path = test_server_utils.download_image_from_url(
        MINIMUM_PICTURE_BASE64_FOR_WARMUP
    )

    assert image_path.parent == tmp_path
    assert image_path.suffix == ".png"
    with Image.open(image_path) as image:
        assert image.format == "PNG"
        assert image.size == (64, 64)
