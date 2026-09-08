"""CUDA numerical checks for DiffusionGemma's inference optimizations."""

import unittest

import torch

from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_qkv_rmsnorm
from sglang.srt.dllm.algorithm.gemma4_renoise import (
    _compiled_denoiser_statistics,
    _denoiser_statistics,
    _sample_denoiser,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGemma4SamplingCUDA(unittest.TestCase):
    def test_full_vocabulary_statistics_and_sampling(self):
        generator = torch.Generator(device="cuda").manual_seed(123)
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                logits = torch.randn(
                    batch_size, 16, 262144, device="cuda", generator=generator
                )
                temperatures = torch.linspace(0.4, 0.8, batch_size, device="cuda")
                expected = _denoiser_statistics(logits, temperatures)
                actual = _compiled_denoiser_statistics(logits, temperatures)
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
                a = torch.Generator(device="cuda").manual_seed(42)
                b = torch.Generator(device="cuda").manual_seed(42)
                probabilities = actual[0].reshape(-1, logits.shape[-1])
                torch.testing.assert_close(
                    _sample_denoiser(probabilities, a),
                    torch.multinomial(probabilities, 1, generator=b).squeeze(-1),
                )
                torch.testing.assert_close(a.get_state(), b.get_state())

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
