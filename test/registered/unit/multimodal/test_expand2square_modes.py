from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import pytest
from PIL import Image

from sglang.srt.multimodal.mm_utils import expand2square

MODES = ["RGB", "L", "LA", "RGBA", "P", "CMYK", "I;16", "1"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("size", [(64, 32), (64, 64)])
def test_expand2square_returns_rgb(mode, size):
    """Every PIL mode must come back as RGB, including square inputs.

    Square inputs used to return early, so even the old "L"-only conversion was
    skipped; LA/I;16/1 raised TypeError from Image.new because the background is
    a 3-tuple, and P stayed paletted.
    """
    image = Image.new(mode, size, 128)
    out = expand2square(image, (127, 127, 127))
    assert out.mode == "RGB"
    side = max(size)
    assert out.size == (side, side)


def test_expand2square_pads_with_background():
    image = Image.new("L", (4, 2), 255)
    out = expand2square(image, (10, 20, 30))
    assert out.getpixel((0, 0)) == (10, 20, 30)
    assert out.getpixel((2, 1)) == (255, 255, 255)
