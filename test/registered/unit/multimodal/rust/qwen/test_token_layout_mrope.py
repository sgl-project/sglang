"""Qwen M-RoPE parity.

Covers ``mrope_image_only`` in ``rust/sglang-mm/src/qwen_vl/mod.rs`` (via the
``_core.qwen_vl.mrope_image_only_py`` binding), against
``MRotaryEmbedding.get_rope_index``.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import (  # noqa: E402
    IMAGE_TOKEN_ID,
    VIDEO_TOKEN_ID,
    VISION_END_ID,
    VISION_START_ID,
    load_core,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

QWEN_CORE = getattr(load_core(), "qwen_vl", None)


@unittest.skipUnless(
    QWEN_CORE and hasattr(QWEN_CORE, "mrope_image_only_py"),
    "sglang-mm native Qwen bindings not built",
)
class TestQwenPromptGeometry(CustomTestCase):
    def test_mrope_matches_model_reference(self):
        """The single Rust image-only M-RoPE must match ``get_rope_index``
        for every native Qwen family (image-only makes them coincide)."""
        from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

        grids = [(1, 4, 6), (1, 6, 4)]
        ids, items = [10], []
        for grid in grids:
            ids.extend((VISION_START_ID, IMAGE_TOKEN_ID))
            start = len(ids) - 1
            ids.extend([IMAGE_TOKEN_ID] * (np.prod(grid) // 4 - 1))
            items.append((start, len(ids) - 1, *grid))
            ids.extend((VISION_END_ID, 11))

        actual, delta = QWEN_CORE.mrope_image_only_py(len(ids), items, 2)
        actual = np.asarray(actual).reshape(3, -1)
        for model_type in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_5"):
            with self.subTest(model_type=model_type):
                expected, expected_delta = MRotaryEmbedding.get_rope_index(
                    spatial_merge_size=2,
                    image_token_id=IMAGE_TOKEN_ID,
                    video_token_id=VIDEO_TOKEN_ID,
                    vision_start_token_id=VISION_START_ID,
                    model_type=model_type,
                    tokens_per_second=2 if model_type == "qwen2_5_vl" else None,
                    input_ids=torch.tensor(ids).unsqueeze(0),
                    image_grid_thw=torch.tensor(grids),
                    video_grid_thw=None,
                )
                np.testing.assert_array_equal(actual, expected.squeeze(1).numpy())
                self.assertEqual(delta, int(expected_delta.item()))


if __name__ == "__main__":
    unittest.main()
