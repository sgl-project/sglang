"""Qwen native image preprocessing parity against Transformers.

Covers ``QwenVlProcessor::process_item`` and ``smart_resize`` in
``rust/sglang-mm/src/qwen_vl/mod.rs`` (via the ``_core.qwen_vl.preprocess``
and ``smart_resize_py`` bindings), against the HF Qwen2-VL image processors
and the Python ``smart_resize``.

Both HF processors are pinned, because they resample differently and only one
of them is what a server actually runs. On transformers 5.x
``Qwen2VLImageProcessor`` is the torchvision path (the ``Fast`` suffix was
dropped) and is what ``AutoImageProcessor`` hands SGLang by default;
``Qwen2VLImageProcessorPil`` is the PIL path, reachable via
``--disable-fast-image-processor``. The Rust resize is a bit-exact clone of
PIL's fixed-point kernel, so the PIL processor is asserted exactly and the
torchvision one carries the cross-implementation envelope.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import (  # noqa: E402
    PROCESSOR_CONFIGS,
    image_bytes,
    load_core,
    make_image,
    spec_json,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

QWEN_CORE = getattr(load_core(), "qwen_vl", None)


SIZES = ((640, 480), (1024, 683), (50, 40), (300, 301))


@unittest.skipUnless(QWEN_CORE, "sglang-mm Qwen binding not built")
class TestQwenImagePreprocess(CustomTestCase):
    def _assert_matches(self, processor, max_diff, mean_diff):
        for family, config in PROCESSOR_CONFIGS.items():
            hf = processor(**config)
            for index, size in enumerate(SIZES):
                with self.subTest(family=family, size=size):
                    image = make_image(*size, seed=index)
                    actual, grid = QWEN_CORE.preprocess(
                        image_bytes(*size, seed=index), spec_json(config)
                    )
                    expected = hf(images=[image], return_tensors="pt")
                    self.assertEqual(grid, tuple(expected.image_grid_thw[0].tolist()))
                    diff = np.abs(
                        np.asarray(actual).reshape(expected.pixel_values.shape)
                        - expected.pixel_values.numpy()
                    )
                    # LessEqual, not Less: max_diff=0.0 is a real bound here.
                    self.assertLessEqual(diff.max(), max_diff)
                    self.assertLessEqual(diff.mean(), mean_diff)

    def test_features_match_pil_processor_exactly(self):
        """Against the PIL processor the native path is bit-exact, so this is
        asserted with zero tolerance: every stage (smart_resize geometry,
        the fixed-point bicubic kernel, rescale/normalize, HF patch order) is
        pinned, and any drift in any of them shows up here rather than being
        absorbed by a tolerance."""
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
            Qwen2VLImageProcessorPil,
        )

        self._assert_matches(Qwen2VLImageProcessorPil, max_diff=0.0, mean_diff=0.0)

    def test_features_match_torchvision_processor_within_envelope(self):
        """The torchvision processor is what a default server runs, so its
        divergence is bounded separately — it is a different antialiased-bicubic
        implementation, which the bit-exact PIL assertion above says nothing
        about. Measured worst case is max 0.030 / mean 6.7e-5 (≈2 u8 levels
        after normalize with the qwen2_vl std)."""
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor,
        )

        self._assert_matches(Qwen2VLImageProcessor, max_diff=0.035, mean_diff=1e-3)

    def test_smart_resize_matches_python(self):
        from sglang.srt.multimodal.processors.qwen_vl import smart_resize

        cases = (
            (1365, 2048, 28, 3136, 12845056),
            (3000, 4000, 28, 3136, 1003520),
            (20, 20, 28, 3136, 12845056),
            (1365, 2048, 32, 65536, 16777216),
            (4000, 48, 32, 4, 1 << 30),
        )
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(QWEN_CORE.smart_resize_py(*case), smart_resize(*case))


if __name__ == "__main__":
    unittest.main()
