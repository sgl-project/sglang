from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    _cumulative_mask_offsets,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGemma4MaskOffsets(CustomTestCase):
    def test_prepare_attn_masks_uses_int64_offsets(self):
        backend = MagicMock(spec=TritonAttnBackend)
        backend.forward_metadata = SimpleNamespace()
        forward_batch = SimpleNamespace(
            batch_size=2,
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens=[2, 1],
            extend_prefix_lens=[1, 2],
            mm_inputs=[None, None],
        )

        with patch(
            "sglang.srt.models.gemma4_mm.get_attn_backend",
            return_value=backend,
        ):
            Gemma4ForConditionalGeneration.prepare_attn_masks(
                None,
                forward_batch,
                torch.tensor([1, 2, 3]),
                torch.bool,
            )

        mask_indptr = backend.forward_metadata.mask_indptr
        custom_mask = backend.forward_metadata.custom_mask
        self.assertEqual(mask_indptr.dtype, torch.int64)
        self.assertEqual(mask_indptr.tolist(), [0, 6, 9])
        self.assertTrue(torch.all(mask_indptr[1:] >= mask_indptr[:-1]))
        self.assertEqual(mask_indptr[-1], custom_mask.numel())

    def test_offsets_cross_int32_boundary(self):
        self.assertEqual(
            _cumulative_mask_offsets([2**31, 7]),
            [0, 2**31, 2**31 + 7],
        )

    def test_offsets_below_int32_boundary(self):
        self.assertEqual(
            _cumulative_mask_offsets([2**31 - 2, 1]),
            [0, 2**31 - 2, 2**31 - 1],
        )

    def test_offsets_follow_batch_order(self):
        self.assertEqual(
            _cumulative_mask_offsets([2**31, 7]),
            [0, 2**31, 2**31 + 7],
        )
        self.assertEqual(
            _cumulative_mask_offsets([7, 2**31]),
            [0, 7, 2**31 + 7],
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
