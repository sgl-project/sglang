"""Qwen placeholder expansion, offsets, and M-RoPE parity."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _utils import (  # noqa: E402
    IMAGE_TOKEN_ID,
    PROCESSOR_CONFIGS,
    VIDEO_TOKEN_ID,
    VISION_START_ID,
    image_bytes,
    request_payload,
    spec_json,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

try:
    from sglang.srt.multimodal import _core

    QWEN_CORE = _core.qwen_vl
except (AttributeError, ImportError):
    QWEN_CORE = None


@unittest.skipUnless(
    QWEN_CORE and hasattr(QWEN_CORE, "process_native_mm_payload"),
    "sglang-mm native Qwen driver not built",
)
class TestQwenPromptGeometry(CustomTestCase):
    def test_placeholder_expansion_and_offsets(self):
        from sglang.srt.multimodal.processors.base_processor import (
            BaseMultimodalProcessor,
        )

        config = PROCESSOR_CONFIGS["qwen2_5_vl"]
        for image_count in (1, 2):
            ids = [7]
            for _ in range(image_count):
                ids.extend((VISION_START_ID, IMAGE_TOKEN_ID, 902, 8))
            images = [image_bytes(96 + 8 * i, 80, i) for i in range(image_count)]
            with self.subTest(image_count=image_count):
                actual_ids, _, grids, _, offsets, _, _ = (
                    QWEN_CORE.process_native_mm_payload(
                        request_payload(ids, images), spec_json(config)
                    )
                )
                counts = [t * h * w // config["merge_size"] ** 2 for t, h, w in grids]
                expected_ids = BaseMultimodalProcessor._expand_input_ids(
                    ids, counts, IMAGE_TOKEN_ID
                )
                self.assertEqual(actual_ids, expected_ids)
                self.assertEqual(
                    offsets,
                    BaseMultimodalProcessor.get_mm_items_offset(
                        torch.tensor(expected_ids), IMAGE_TOKEN_ID
                    ),
                )

    def test_placeholder_mismatch_falls_back(self):
        with self.assertRaisesRegex(ValueError, "fallback.*placeholder"):
            QWEN_CORE.process_native_mm_payload(
                request_payload([7, 8], [image_bytes(80, 80)]),
                spec_json(PROCESSOR_CONFIGS["qwen2_5_vl"]),
            )

    def test_mrope_matches_model_reference(self):
        from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

        grids = [(1, 4, 6), (1, 6, 4)]
        ids, items = [10], []
        for grid in grids:
            ids.extend((VISION_START_ID, IMAGE_TOKEN_ID))
            start = len(ids) - 1
            ids.extend([IMAGE_TOKEN_ID] * (np.prod(grid) // 4 - 1))
            items.append((start, len(ids) - 1, *grid))
            ids.extend((902, 11))

        actual, delta = QWEN_CORE.mrope_image_only_py(len(ids), items, 2)
        expected, expected_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=2,
            image_token_id=IMAGE_TOKEN_ID,
            video_token_id=VIDEO_TOKEN_ID,
            vision_start_token_id=VISION_START_ID,
            model_type="qwen3_vl",
            tokens_per_second=None,
            input_ids=torch.tensor(ids).unsqueeze(0),
            image_grid_thw=torch.tensor(grids),
            video_grid_thw=None,
        )
        np.testing.assert_array_equal(
            np.asarray(actual).reshape(3, -1), expected.squeeze(1).numpy()
        )
        self.assertEqual(delta, int(expected_delta.item()))


if __name__ == "__main__":
    unittest.main()
