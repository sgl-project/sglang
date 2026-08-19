import sys
from io import BytesIO

import pytest
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor
from sglang.srt.multimodal.processors.kimi_k3 import KimiK3ImageProcessor
from sglang.srt.utils.common import load_image, smart_to_rgb
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=10,
    suite="base-a-test-cpu",
    nightly=False,
    disabled=None,
)

_RGBA_PIXEL = (10, 20, 30, 200)


def _rgba_png_bytes() -> bytes:
    img = Image.new("RGBA", (4, 4), _RGBA_PIXEL)
    buf = BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_load_image_preserves_rgba_fidelity():
    img, _ = load_image(_rgba_png_bytes())
    assert img.mode == "RGBA"
    assert img.getpixel((0, 0)) == _RGBA_PIXEL


def test_kimi_k3_no_discard_keeps_rgba():
    out = KimiK3ImageProcessor._load_single_item(
        _rgba_png_bytes(),
        modality=Modality.IMAGE,
        discard_alpha_channel=False,
    )
    assert out.mode == "RGBA"
    assert out.getpixel((0, 0)) == _RGBA_PIXEL


def test_default_processor_plain_convert_drops_alpha_only():
    out = BaseMultimodalProcessor._load_single_item(
        _rgba_png_bytes(),
        modality=Modality.IMAGE,
        discard_alpha_channel=True,
    )
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == _RGBA_PIXEL[:3]


def test_glm4v_optin_smart_composite_matches_reference():
    rgba = Image.open(BytesIO(_rgba_png_bytes()))
    reference = smart_to_rgb(rgba.copy())
    out = Glm4vImageProcessor._load_single_item(
        _rgba_png_bytes(),
        modality=Modality.IMAGE,
        discard_alpha_channel=True,
    )
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == reference.getpixel((0, 0))
    plain = rgba.convert("RGB")
    assert out.getpixel((0, 0)) != plain.getpixel((0, 0))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
