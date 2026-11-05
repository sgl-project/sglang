"""bs == 1 masked SDPA output must be viewable as [tokens, heads * dim].

HunYuanAttention.forward does ``attn_output.view(q.shape[0], -1)`` on the
squeezed result, which fails if the transposed output is left non-contiguous.
"""

import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3_utils import (
    _masked_sdpa,
)
from sglang.test.test_utils import CustomTestCase


class TestMaskedSdpaContiguous(CustomTestCase):
    def test_bs1_output_viewable(self):
        seq, heads, kv_heads, dim = 5, 4, 2, 8
        q = torch.randn(1, seq, heads, dim)
        k = torch.randn(1, seq, kv_heads, dim)
        v = torch.randn(1, seq, kv_heads, dim)
        mask = torch.ones(1, 1, seq, seq, dtype=torch.bool)

        out = _masked_sdpa(q, k, v, mask)

        self.assertEqual(out.shape, (1, seq, heads, dim))
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out.squeeze(0).view(seq, -1).shape, (seq, heads * dim))


if __name__ == "__main__":
    unittest.main()
