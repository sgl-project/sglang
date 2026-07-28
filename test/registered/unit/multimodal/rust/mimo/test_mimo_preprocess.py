"""MiMo native image preprocessing parity against the Python ``MiMoProcessor``.

Covers ``MimoV2Processor::process_item`` and MiMo's ``smart_resize`` variant
in ``rust/sglang-mm/src/mimo_v2/mod.rs`` (via the ``_core.mimo_v2.preprocess``
and ``smart_resize_py`` bindings), against ``MiMoProcessor.get_visual_transform``
(torch bilinear ``F.interpolate`` + 0..255-scale standardization) and
``_flatten_visual_inputs``. Unlike the Qwen path there is no u8 quantization
after resize, so the envelope is float accumulation noise only.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.multimodal.processors.mimo_v2 import MiMoProcessor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import image_bytes, load_core, make_image  # noqa: E402

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

MIMO_CORE = getattr(load_core(), "mimo_v2", None)

FACTOR = 32  # patch 16 * merge 2 (MiMo-V2.5)
MIN_PIXELS, MAX_PIXELS = 8192, 8388608
SPEC = (
    '{"family":"mimo_v2","image_token_id":2,"patch_size":16,"merge_size":2,'
    '"temporal_patch_size":2,"min_pixels":8192,"max_pixels":8388608,'
    '"image_mean":[123.675,116.28,103.53],"image_std":[58.395,57.12,57.375]}'
)

# (width, height): plain resize, min_pixels upscale, min-side-below-factor
# upscale, extreme-thin upscale, and a max_pixels downscale.
SIZES = ((640, 480), (96, 80), (23, 301), (4000, 20), (3600, 2700))


@unittest.skipUnless(MIMO_CORE, "sglang-mm MiMo binding not built")
class TestMimoImagePreprocess(CustomTestCase):
    def python_reference(self, image):
        tensor, _, _ = MiMoProcessor.get_visual_transform(
            image, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        processor = MiMoProcessor.__new__(MiMoProcessor)  # pixel path only
        processor.patch_size, processor.merge_size = 16, 2
        processor.temporal_patch_size = 2
        return processor._flatten_visual_inputs(tensor, "image")

    def test_preprocess_matches_python(self):
        for index, (width, height) in enumerate(SIZES):
            with self.subTest(size=(width, height)):
                image = make_image(width, height, seed=index)
                patches, grid = self.python_reference(image)
                native, native_grid = MIMO_CORE.preprocess(
                    image_bytes(width, height, seed=index), SPEC
                )
                self.assertEqual(tuple(grid.tolist()), native_grid)
                diff = np.abs(native - patches.numpy().ravel())
                self.assertLess(diff.max(), 5e-3)
                self.assertLess(diff.mean(), 1e-4)

    def test_smart_resize_matches_python(self):
        # Sweep all branches: identity, round-down (no max(factor) clamp),
        # min-side upscale, min_pixels/max_pixels rescale, and the branch
        # order that lets a sub-factor min side skip the ratio guard.
        for height, width in (
            (480, 640),
            (100, 100),
            (80, 96),
            (301, 23),
            (20, 4000),
            (2700, 3600),
            (1, 300),
            (31, 31),
        ):
            with self.subTest(size=(height, width)):
                self.assertEqual(
                    MIMO_CORE.smart_resize_py(
                        height, width, FACTOR, MIN_PIXELS, MAX_PIXELS
                    ),
                    tuple(
                        MiMoProcessor.smart_resize(
                            height, width, FACTOR, MIN_PIXELS, MAX_PIXELS
                        )
                    ),
                )

    def test_smart_resize_rejects_what_python_rejects(self):
        # Ratio guard fires only once the min side reaches `factor`.
        with self.assertRaises(ValueError):
            MiMoProcessor.smart_resize(33, 33 * 201, FACTOR, MIN_PIXELS, MAX_PIXELS)
        with self.assertRaises(ValueError):
            MIMO_CORE.smart_resize_py(33, 33 * 201, FACTOR, MIN_PIXELS, MAX_PIXELS)


if __name__ == "__main__":
    unittest.main()
