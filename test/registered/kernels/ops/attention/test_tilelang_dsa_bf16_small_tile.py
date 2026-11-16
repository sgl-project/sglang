"""BF16 DSA v2 must load only the allocated KV tile and fit SM12x small-head shapes."""

import math
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.version.hip is None
    and torch.cuda.get_device_capability()[0] in (9, 10, 12),
    "CUDA SM90, SM10x or SM12x required",
)
class TestTileLangDSABF16SmallTile(CustomTestCase):
    def _check_attention(self, heads, direct_factory):
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            sparse_attention_fwd_kernel_v2,
            tilelang_sparse_fwd,
        )

        torch.manual_seed(7)
        d_v, d_tail, topk = 512, 64, 2048
        dim = d_v + d_tail
        scale = 1 / math.sqrt(dim)
        q = torch.randn(1, heads, dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(4096, 1, dim, device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(topk, device="cuda", dtype=torch.int32).view(1, 1, topk)
        # The final slots in a padded DSA index table must be masked.
        indices[..., -8:] = -1

        rows = indices[:, 0].long()
        selected = kv[rows.clamp_min(0), 0].float()
        scores = torch.einsum("thd,tkd->thk", q.float(), selected) * scale
        scores.masked_fill_(rows[:, None, :] < 0, float("-inf"))
        expected = torch.einsum(
            "thk,tkd->thd", scores.softmax(-1), selected[..., :d_v]
        ).to(torch.bfloat16)

        if direct_factory:
            kernel = sparse_attention_fwd_kernel_v2(
                heads, d_v, d_tail, topk, sm_scale=scale, block_I=32
            )
            actual = kernel(q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0))
        else:
            actual = tilelang_sparse_fwd(q, kv, indices, scale, d_v=d_v)
        torch.cuda.synchronize()
        self.assertEqual(actual.shape, (1, 1, heads, d_v))
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual.squeeze(0), expected, atol=0.04, rtol=0.04)

    def test_small_tile_matches_torch(self):
        """A 32-row KV tile must execute without illegal accesses and match torch."""
        self._check_attention(heads=8, direct_factory=True)

    def test_public_dispatch_matches_torch(self):
        """BF16 512+64 attention must fit SM12x shared memory and match torch."""
        # On SM12x this exercises the resource-safe tile; on Hopper/SM10x
        # the original 64-row selection must retain the same semantics.
        for heads in (8, 16):
            with self.subTest(heads=heads):
                self._check_attention(heads, direct_factory=False)


if __name__ == "__main__":
    unittest.main()
