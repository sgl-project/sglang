"""Packed QKV normalization parity for DiffusionGemma."""

import unittest

import torch

from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_qkv_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGemma4QKVNorm(CustomTestCase):
    @torch.inference_mode()
    def test_fanout_matches_three_rmsnorms(self):
        from flashinfer.norm import rmsnorm

        from sglang.kernels.ops.layernorm.rmsnorm_fanout import rmsnorm_fanout

        for dtype in (torch.bfloat16, torch.float16):
            for rows in (13, 512):
                with self.subTest(dtype=dtype, rows=rows):
                    x = torch.randn(rows, 2816, device="cuda", dtype=dtype)
                    weights = [
                        torch.randn(2816, device="cuda", dtype=dtype) for _ in range(3)
                    ]
                    expected = [rmsnorm(x, weight, 1e-6) for weight in weights]
                    actual = rmsnorm_fanout(x, *weights, 1e-6)
                    for value, reference in zip(actual, expected):
                        torch.testing.assert_close(value, reference, atol=0, rtol=0)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = rmsnorm_fanout(x, *weights, 1e-6)
                    x.mul_(0.75)
                    weights[1].add_(0.125)
                    graph.replay()
                    for value, weight in zip(actual, weights):
                        torch.testing.assert_close(
                            value, rmsnorm(x, weight, 1e-6), atol=0, rtol=0
                        )

    @torch.inference_mode()
    def test_norm_rope_matches_separate_inplace_operations(self):
        from sglang.kernels.ops.attention.gemma_qkv_norm_rope import gemma_qkv_norm_rope
        from sglang.kernels.ops.attention.rope import (
            apply_rope_with_cos_sin_cache_inplace,
        )

        for heads, kv_heads, dim, rotary in ((16, 8, 256, 256), (16, 2, 512, 128)):
            with self.subTest(dim=dim):
                rows = 257
                source = torch.randn(
                    rows,
                    (heads + 2 * kv_heads) * dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                actual, expected = source.clone(), source.clone()
                weights = [
                    torch.randn(dim, device="cuda", dtype=source.dtype)
                    for _ in range(2)
                ]
                positions = torch.arange(rows, device="cuda", dtype=torch.int64)
                cache = torch.randn(rows, rotary, device="cuda", dtype=torch.float32)

                def reference():
                    q, k, v = expected.split(
                        (heads * dim, kv_heads * dim, kv_heads * dim), -1
                    )
                    gemma_qkv_rmsnorm(q, k, v, *weights, heads, kv_heads, dim, 1e-6)
                    apply_rope_with_cos_sin_cache_inplace(
                        q.view(rows, heads, dim)[..., :rotary],
                        k.view(rows, kv_heads, dim)[..., :rotary],
                        cache,
                        positions,
                        is_neox=True,
                    )

                def fused():
                    q, k, v = actual.split(
                        (heads * dim, kv_heads * dim, kv_heads * dim), -1
                    )
                    gemma_qkv_norm_rope(
                        q,
                        k,
                        v,
                        *weights,
                        cache,
                        positions,
                        heads,
                        kv_heads,
                        dim,
                        rotary,
                        1e-6,
                    )

                reference()
                fused()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual.copy_(source)
                    fused()
                source.add_(0.25)
                positions.copy_(positions.flip(0))
                expected.copy_(source)
                reference()
                graph.replay()
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

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
