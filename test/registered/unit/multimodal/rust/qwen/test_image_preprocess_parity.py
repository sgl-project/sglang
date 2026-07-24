"""Qwen native image preprocessing parity against Transformers."""

import sys
import unittest
from pathlib import Path

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _utils import (  # noqa: E402
    PROCESSOR_CONFIGS,
    image_bytes,
    load_core,
    make_image,
    spec_json,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

QWEN_CORE = getattr(load_core(), "qwen_vl", None)


@unittest.skipUnless(QWEN_CORE, "sglang-mm Qwen binding not built")
class TestQwenImagePreprocess(CustomTestCase):
    def test_features_and_grids_match_hf(self):
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
            Qwen2VLImageProcessor,
        )

        for family, config in PROCESSOR_CONFIGS.items():
            processor = Qwen2VLImageProcessor(**config)
            sizes = ((640, 480), (1024, 683), (50, 40), (300, 301))
            for index, size in enumerate(sizes):
                with self.subTest(family=family, size=size):
                    image = make_image(*size, seed=index)
                    actual, grid = QWEN_CORE.preprocess(
                        image_bytes(*size, seed=index), spec_json(config)
                    )
                    expected = processor(images=[image], return_tensors="pt")
                    self.assertEqual(grid, tuple(expected.image_grid_thw[0].tolist()))
                    diff = np.abs(
                        np.asarray(actual).reshape(expected.pixel_values.shape)
                        - expected.pixel_values.numpy()
                    )
                    self.assertLess(diff.max(), 0.06)
                    self.assertLess(diff.mean(), 1e-3)

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
