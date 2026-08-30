import math
import unittest

import torch

from sglang.kernels.ops.attention.qsa_decode import (
    sparse_gqa_decode_physical_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestSparseGQADecodeKernel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is unavailable")

    def test_matches_grouped_query_reference(self):
        torch.manual_seed(7)
        device = torch.device("cuda")
        batch, cache_tokens, topk = 4, 4096, 2051
        q_heads, kv_heads, head_dim = 24, 2, 256
        q = torch.randn(batch, q_heads, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(
            cache_tokens, kv_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        v = torch.randn_like(k)
        slots = torch.randint(0, cache_tokens, (batch, topk), device=device)
        slots[:, -13:] = -1
        scale = 1 / math.sqrt(head_dim)

        actual = sparse_gqa_decode_physical_triton(q, k, v, slots, scale)
        valid = slots >= 0
        safe_slots = slots.clamp_min(0)
        selected_k = k[safe_slots]
        selected_v = v[safe_slots]
        group_size = q_heads // kv_heads
        q_grouped = q.view(batch, kv_heads, group_size, head_dim).float()
        scores = torch.einsum("bkgd,btkd->bkgt", q_grouped, selected_k.float())
        scores = scores.mul(scale).masked_fill(~valid[:, None, None, :], -torch.inf)
        probabilities = torch.softmax(scores, dim=-1)
        expected = torch.einsum(
            "bkgt,btkd->bkgd", probabilities, selected_v.float()
        ).to(q.dtype)
        expected = expected.reshape_as(q)

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)


if __name__ == "__main__":
    unittest.main()
