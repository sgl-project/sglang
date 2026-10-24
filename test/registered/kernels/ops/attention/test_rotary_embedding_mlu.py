"""MLU rotary embedding layouts, partial rotary dimensions and cache refresh."""

import unittest

import torch
import torch_mlu  # noqa: F401

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=30, suite="pr-test-1-mlu")


class TestRotaryEmbeddingMLU(CustomTestCase):
    def test_rotary_layouts_and_cache_refresh(self):
        torch.manual_seed(42)
        for neox in (True, False):
            for rotary_dim in (32, 64):
                with self.subTest(neox=neox, rotary_dim=rotary_dim):
                    rope = RotaryEmbedding(
                        64, rotary_dim, 128, 10000, neox, torch.bfloat16
                    ).to("mlu")
                    for offset in (0, 7):
                        # On the second call, grow the warmed cache and query
                        # new rows. Reusing the old MLU cos/sin cache must fail.
                        start = 0
                        if offset:
                            start = rope.cos_sin_cache.shape[0]
                            rope._ensure_cos_sin_cache_length(start + offset + 6)
                        pos = torch.arange(start, start + 7, device="mlu")
                        offsets = torch.full_like(pos, offset)
                        q = torch.randn(7, 256, device="mlu", dtype=torch.bfloat16)
                        k = torch.randn(7, 128, device="mlu", dtype=torch.bfloat16)
                        actual = rope(pos, q.clone(), k.clone(), offsets)
                        expected = rope.forward_native(
                            pos, q.clone(), k.clone(), offsets
                        )
                        for a, e in zip(actual, expected):
                            torch.testing.assert_close(a, e, rtol=0.03, atol=0.03)


if __name__ == "__main__":
    unittest.main()
