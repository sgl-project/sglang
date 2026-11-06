"""Fused Gemma normalization/NeoX RoPE with packed or shared KV."""

import unittest

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
    _xpu_fp16_intermediate,
)
from sglang.kernels.ops.attention.gemma_qkv_norm_rope import gemma_qkv_norm_rope
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_xpu_ci(est_time=20, stage="stage-b", runner_config="1-gpu-xpu")

DEVICE = "xpu" if torch.xpu.is_available() else "cuda"


@triton.jit
def _round_intermediate_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offsets, offsets < N, other=0)
    tl.store(Y + offsets, _xpu_fp16_intermediate(value), offsets < N)


def reference(x, weight, dim, cache=None, positions=None):
    dtype = x.dtype
    x = x.reshape(x.shape[0], -1, dim).float()
    normalized = (
        x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight
    ).to(dtype)
    if cache is None:
        return normalized.flatten(1)
    normalized = normalized.float()
    half = cache.shape[1] // 2
    cosine, sine = cache[positions].chunk(2, -1)
    cosine, sine = cosine[:, None, :], sine[:, None, :]
    first, second = normalized[..., :half], normalized[..., half : 2 * half]
    result = torch.cat(
        (
            first * cosine - second * sine,
            second * cosine + first * sine,
            normalized[..., 2 * half :],
        ),
        -1,
    )
    return result.to(dtype).flatten(1)


@unittest.skipUnless(
    torch.xpu.is_available() or torch.cuda.is_available(), "requires XPU or CUDA"
)
class TestGemmaQKVNormRope(CustomTestCase):
    def test_packed_strides_shared_kv_and_rounding(self):
        torch.manual_seed(44)
        for dim, rotary in ((128, 64), (256, 64), (256, 256), (512, 128), (512, 512)):
            for dtype in (torch.bfloat16, torch.float16):
                for shared in (False, True):
                    with self.subTest(
                        dim=dim, rotary=rotary, dtype=dtype, shared=shared
                    ):
                        # Include guards and a nontrivial token stride; each Q/K/V
                        # view has contiguous heads but is not itself contiguous.
                        storage = torch.randn(
                            (4, 10 * dim + 16), device=DEVICE, dtype=dtype
                        )
                        original = storage.clone()
                        q, k, v = storage[:, : 10 * dim].split((8 * dim, dim, dim), -1)
                        qw, kw = torch.randn((2, dim), device=DEVICE, dtype=dtype)
                        angles = torch.randn((256, rotary // 2), device=DEVICE)
                        cache = torch.cat((angles.cos(), angles.sin()), -1)
                        positions = torch.tensor(
                            [0, 7, 127, 255], device=DEVICE, dtype=torch.int64
                        )
                        expected_q = reference(q, qw, dim, cache, positions)
                        expected_k = reference(k, kw, dim, cache, positions)
                        expected_v = reference(v, 1.0, dim)
                        gemma_qkv_norm_rope(
                            q,
                            None if shared else k,
                            None if shared else v,
                            qw,
                            None if shared else kw,
                            cache,
                            positions,
                            8,
                            1,
                            dim,
                            rotary,
                            1e-6,
                        )
                        tolerance = (
                            dict(rtol=0.015625, atol=0.0078125)
                            if dtype == torch.bfloat16
                            # Two rounded norm operands feed a subtract in RoPE;
                            # cancellation needs an absolute bound of two FP16 eps.
                            else dict(rtol=0.002, atol=2 * torch.finfo(dtype).eps)
                        )
                        torch.testing.assert_close(q, expected_q, **tolerance)
                        if shared:
                            torch.testing.assert_close(
                                storage[:, 8 * dim :],
                                original[:, 8 * dim :],
                                rtol=0,
                                atol=0,
                            )
                        else:
                            torch.testing.assert_close(k, expected_k, **tolerance)
                            torch.testing.assert_close(v, expected_v, **tolerance)
                            torch.testing.assert_close(
                                storage[:, 10 * dim :],
                                original[:, 10 * dim :],
                                rtol=0,
                                atol=0,
                            )

    @unittest.skipUnless(torch.xpu.is_available(), "XPU rounding fence")
    def test_fp16_rounding_fence_edges(self):
        generator = torch.Generator().manual_seed(58)
        # Cover all exponent ranges and explicit overflow, subnormal, tie,
        # signed-zero and non-finite values, independently of normalization.
        bits = torch.randint(0, 2**32, (8192,), dtype=torch.int64, generator=generator)
        values = bits.to(torch.int32).view(torch.float32)
        edges = torch.tensor(
            [
                0.0,
                -0.0,
                2.0**-25,
                3 * 2.0**-25,
                2.0**-24,
                2.0**-14,
                65504.0,
                65520.0,
                -65520.0,
                float("inf"),
                -float("inf"),
                float("nan"),
            ]
        )
        values = torch.cat((values, edges)).to("xpu")
        expected = values.to(torch.float16).float()
        actual = torch.empty_like(values)
        _round_intermediate_kernel[(triton.cdiv(values.numel(), 256),)](
            values, actual, values.numel(), 256
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        zeros = expected == 0
        torch.testing.assert_close(
            torch.signbit(actual[zeros]), torch.signbit(expected[zeros])
        )

    @unittest.skipUnless(torch.xpu.is_available(), "native XPU reference")
    def test_native_xpu_arithmetic_is_preserved(self):
        from sglang.kernels.ops.layernorm.gemma4_fused_ops import gemma_qkv_rmsnorm
        from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

        torch.manual_seed(47)
        for dtype in (torch.bfloat16, torch.float16):
            for dim, rotary in ((256, 64), (256, 256), (512, 128), (512, 512)):
                rope = RotaryEmbedding(dim, rotary, 8192, 10000, True, dtype)
                rope.cos_sin_cache = rope.cos_sin_cache.to("xpu")
                qw, kw = torch.randn((2, dim), device="xpu", dtype=dtype)
                for batch in (1, 8, 128, 512):
                    for shared in (False, True):
                        with self.subTest(
                            dtype=dtype,
                            dim=dim,
                            rotary=rotary,
                            batch=batch,
                            shared=shared,
                        ):
                            storage = torch.randn(
                                (batch, 10 * dim), device="xpu", dtype=dtype
                            )
                            expected = storage.clone()
                            q, k, v = storage.split((8 * dim, dim, dim), -1)
                            qr, kr, vr = expected.split((8 * dim, dim, dim), -1)
                            positions = torch.randint(0, 8192, (batch,), device="xpu")
                            gemma_qkv_rmsnorm(
                                qr,
                                None if shared else kr,
                                None if shared else vr,
                                qw,
                                None if shared else kw,
                                8,
                                1,
                                dim,
                            )
                            rope.forward_xpu(
                                positions, qr, torch.zeros_like(kr) if shared else kr
                            )
                            gemma_qkv_norm_rope(
                                q,
                                None if shared else k,
                                None if shared else v,
                                qw,
                                None if shared else kw,
                                rope.cos_sin_cache,
                                positions,
                                8,
                                1,
                                dim,
                                rotary,
                                1e-6,
                            )
                            torch.testing.assert_close(
                                storage, expected, rtol=0, atol=0
                            )

    def test_norm_is_rounded_before_rotary_cancellation(self):
        dim, rotary = 256, 64
        for dtype, value, scale in (
            (torch.bfloat16, 0.99609375, 1.0078125),
            (torch.float16, 0.9990234375, 1.0009765625),
        ):
            with self.subTest(dtype=dtype):
                q = torch.ones((1, dim), device=DEVICE, dtype=dtype)
                q[:, rotary // 2] = value
                weight = torch.ones(dim, device=DEVICE, dtype=dtype)
                weight[rotary // 2] = scale
                cache = torch.ones((1, rotary), device=DEVICE, dtype=torch.float32)
                positions = torch.zeros(1, device=DEVICE, dtype=torch.int64)
                expected = reference(q, weight, dim, cache, positions)
                # Rounded norm operands both equal 1. Retaining the FP32
                # operands through RoPE instead produces a nonzero difference.
                self.assertEqual(expected[0, 0].item(), 0)
                gemma_qkv_norm_rope(
                    q,
                    None,
                    None,
                    weight,
                    None,
                    cache,
                    positions,
                    1,
                    1,
                    dim,
                    rotary,
                    1e-6,
                )
                self.assertEqual(q[0, 0].item(), 0)
                torch.testing.assert_close(q, expected, rtol=0, atol=0)

    def test_graph_replay_reads_updated_positions_and_inputs(self):
        dim, rotary = 256, 64
        q = torch.randn((8, 8 * dim), device=DEVICE, dtype=torch.bfloat16)
        initial = q.clone()
        weight = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16)
        positions = torch.zeros(8, device=DEVICE, dtype=torch.int32)
        angles = torch.randn((256, rotary // 2), device=DEVICE)
        cache = torch.cat((angles.cos(), angles.sin()), -1)

        def run():
            gemma_qkv_norm_rope(
                q, None, None, weight, None, cache, positions, 8, 1, dim, rotary, 1e-6
            )

        run()
        module = torch.xpu if DEVICE == "xpu" else torch.cuda
        module.synchronize()
        graph = module.XPUGraph() if DEVICE == "xpu" else module.CUDAGraph()
        with module.graph(graph):
            run()
        for sign, position in ((1, 7), (-1, 127)):
            q.copy_(initial * sign)
            positions.fill_(position)
            expected = reference(q, weight, dim, cache, positions)
            graph.replay()
            torch.testing.assert_close(q, expected, rtol=0.015625, atol=0.0078125)


if __name__ == "__main__":
    unittest.main()
