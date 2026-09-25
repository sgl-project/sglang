"""CUDA numerical checks for DiffusionGemma's inference optimizations."""

import unittest

import torch

from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_qkv_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGemma4SamplingCUDA(unittest.TestCase):
    def test_qkv_norm_with_full_and_sliding_head_dimensions(self):
        for num_q, num_kv, head_dim in ((8, 4, 256), (8, 1, 512)):
            with self.subTest(head_dim=head_dim):
                packed = torch.randn(
                    16,
                    (num_q + 2 * num_kv) * head_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                q, k, v = packed.split(
                    [num_q * head_dim, num_kv * head_dim, num_kv * head_dim], dim=-1
                )
                qw = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
                kw = torch.randn_like(qw)
                expected = []
                for x, heads, weight in (
                    (q, num_q, qw),
                    (k, num_kv, kw),
                    (v, num_kv, 1.0),
                ):
                    x = x.reshape(16, heads, head_dim).float()
                    normalized = x * torch.rsqrt(
                        x.square().mean(dim=-1, keepdim=True) + 1e-6
                    )
                    expected.append((normalized * weight).to(packed.dtype).flatten(1))
                gemma_qkv_rmsnorm(
                    q,
                    k,
                    v,
                    qw,
                    kw,
                    num_q_heads=num_q,
                    num_kv_heads=num_kv,
                    head_dim=head_dim,
                    eps=1e-6,
                )
                for got, want in zip((q, k, v), expected):
                    torch.testing.assert_close(got, want, rtol=8e-3, atol=8e-3)


if __name__ == "__main__":
    unittest.main()
