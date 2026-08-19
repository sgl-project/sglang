import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import gemma3_mm
from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeTritonAttnBackend(gemma3_mm.TritonAttnBackend):
    def __init__(self):
        self.forward_metadata = SimpleNamespace()


class FakeImageItem:
    offsets = [(1, 2)]

    def is_image(self):
        return True


class TestGemma3ForConditionalGeneration(CustomTestCase):
    @patch("sglang.srt.models.gemma3_mm.get_attn_backend")
    def test_prepare_attn_masks_skips_text_only_rows_in_mixed_batch(
        self, mock_get_attn_backend
    ):
        backend = FakeTritonAttnBackend()
        mock_get_attn_backend.return_value = backend

        model = object.__new__(Gemma3ForConditionalGeneration)
        forward_batch = SimpleNamespace(
            batch_size=2,
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=[4, 3],
            extend_prefix_lens=[0, 0],
            mm_inputs=[
                SimpleNamespace(mm_items=[FakeImageItem()]),
                None,
            ],
        )

        model.prepare_attn_masks(
            forward_batch,
            input_ids=torch.arange(7),
            mask_dtype=torch.bool,
        )

        expected_mask_size = 4 * 4 + 3 * 3
        self.assertEqual(
            backend.forward_metadata.custom_mask.numel(), expected_mask_size
        )
        self.assertEqual(
            backend.forward_metadata.mask_indptr.tolist(),
            [0, 16, expected_mask_size],
        )


if __name__ == "__main__":
    unittest.main()
