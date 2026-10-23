# SPDX-License-Identifier: Apache-2.0
"""Manual SM120 Sage correctness check against selected-block FP32 attention.

Requires an SM120 GPU and FlashInfer with the CuTe-DSL SM120 Sage backend.
Run: python test/manual/attention/test_subblock_sage_fp8_sm120.py
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires an SM120 GPU",
)
class TestSubBlockSageFp8Sm120(CustomTestCase):
    def test_ragged_sparse_plan_with_empty_rows(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse_attn import (
            _sm120_sage_fp8_sparse_attention,
        )

        torch.manual_seed(19)
        q = torch.randn(1, 65, 2, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 129, 2, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        index = torch.tensor(
            [[[[2, 0], [1, 0]], [[0, 2], [2, 1]]]], device="cuda", dtype=torch.int32
        )
        counts = torch.tensor([[[2, 1], [0, 2]]], device="cuda", dtype=torch.int32)
        mask = torch.zeros(1, 2, 65, 129, device="cuda", dtype=torch.bool)
        mask[0, 0, :64, :64] = True
        mask[0, 0, :64, 128:] = True
        mask[0, 0, 64:, 64:128] = True
        mask[0, 1, 64:, 64:] = True
        scale = 128**-0.5
        logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * scale
        probs = logits.masked_fill(~mask, -float("inf")).softmax(-1).nan_to_num()
        expected = torch.einsum("bhqk,bkhd->bqhd", probs, v.float())
        actual = _sm120_sage_fp8_sparse_attention(q, k, v, index, 2, scale, counts)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(torch.count_nonzero(actual[0, :64, 1]).item(), 0)
        torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
