# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the per-token-block W4AFP8 grouped GEMM operator.

Guards the numerically-correct handling of DeepEP low-latency per-token-block
activation scales `[E, M, K // 128]`. The previous low-latency path collapsed
that scale to a single per-tensor value via ``a_scales.abs().max()``; these
tests pin the correct block-wise result and assert the collapse is materially
worse, so a regression back to it turns the suite red.
"""

import os
import unittest
from unittest.mock import patch

import torch
from sglang.srt.layers.moe.cutedsl_w4afp8_gemm_per_token_block import (
    SCALE_BLOCK_SIZE,
    cutedsl_w4afp8_gemm_per_token_block,
    deinterleave_w_scale,
    unpack_int4_weight,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

SKIP_TEST = not torch.cuda.is_available()


def _pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 values two-per-byte (low nibble = even index)."""
    values = values.to(torch.int8)
    low = values[..., 0::2]
    high = values[..., 1::2]
    return ((high << 4) | (low & 0x0F)).to(torch.int8)


def _independent_fp32_reference(
    a_fp8: torch.Tensor,
    a_scale: torch.Tensor,
    ref_w_int: torch.Tensor,
    w_scale: torch.Tensor,
    masked_m,
) -> torch.Tensor:
    """Block-by-block FP32 reference sharing no code with the operator.

    out[e,m,n] = sum_blk (a_blk * a_scale_blk) @ (w_blk * w_scale_blk).T
    """
    e, m, k = a_fp8.shape
    n = ref_w_int.shape[1]
    num_blocks = k // SCALE_BLOCK_SIZE
    out = torch.zeros(e, m, n, dtype=torch.float32, device=a_fp8.device)
    af = a_fp8.float()
    wf = ref_w_int.float()
    for expert in range(e):
        valid = m if masked_m is None else int(masked_m[expert])
        for blk in range(num_blocks):
            ks = blk * SCALE_BLOCK_SIZE
            ke = ks + SCALE_BLOCK_SIZE
            a_blk = af[expert, :valid, ks:ke] * a_scale[expert, :valid, blk : blk + 1]
            w_blk = wf[expert, :, ks:ke] * w_scale[expert, :, blk : blk + 1]
            out[expert, :valid] += a_blk @ w_blk.t()
    return out


def _make_case(e, m, n, k, seed, device="cuda", varied_scales=True):
    torch.manual_seed(seed)
    a = (torch.randn(e, m, k, device=device) * 0.5).to(torch.float8_e4m3fn)
    if varied_scales:
        # Widely varying block scales: the regime where a per-tensor collapse
        # loses the most precision.
        a_scale = (
            torch.rand(e, m, k // SCALE_BLOCK_SIZE, device=device) * 2.0 + 0.05
        ).float()
    else:
        a_scale = torch.ones(e, m, k // SCALE_BLOCK_SIZE, device=device)
    ref_w = torch.randint(-8, 8, (e, n, k), dtype=torch.int8, device=device)
    w = _pack_int4(ref_w.cpu()).to(device).view(e, n, k // 2)
    w_scale = (torch.randn(e, n, k // SCALE_BLOCK_SIZE, device=device) * 0.02).float()
    return a, a_scale, w, w_scale, ref_w


@unittest.skipIf(SKIP_TEST, "requires CUDA")
class TestPerTokenBlockW4AFP8Gemm(unittest.TestCase):
    def test_int4_unpack_round_trip(self):
        # Guards the packing convention shared with the CUTLASS kernel: a wrong
        # nibble order / sign extension silently corrupts every weight.
        ref = torch.randint(-8, 8, (3, 8, 64), dtype=torch.int8, device="cuda")
        packed = _pack_int4(ref.cpu()).to("cuda").view(3, 8, 32)
        self.assertTrue(torch.equal(unpack_int4_weight(packed), ref.float()))

    def test_deinterleave_known_layout(self):
        # Independent logical-index check, without importing the full model
        # quantization stack. Covers both interleaved and single-block layouts.
        for k in (128, 384, 512, 1024):
            e, n, blocks = 2, 17, k // SCALE_BLOCK_SIZE
            alignment = 4 if blocks % 4 == 0 else 1
            packed = torch.arange(
                e * n * blocks, device="cuda", dtype=torch.float32
            ).view(e, blocks // alignment, n * alignment)
            logical = deinterleave_w_scale(packed, n, k)
            for expert in range(e):
                for col in range(n):
                    for block in range(blocks):
                        self.assertEqual(
                            logical[expert, col, block].item(),
                            packed[
                                expert,
                                block // alignment,
                                col * alignment + block % alignment,
                            ].item(),
                        )

    def test_deinterleave_inverts_interleave(self):
        # deinterleave_w_scale must invert w4afp8.interleave_scales so the
        # operator can consume the on-layer packed weight scale.
        from sglang.srt.layers.quantization.w4afp8 import interleave_scales

        e, n, k = 2, 16, 512
        logical = torch.randn(e, n, k // SCALE_BLOCK_SIZE, device="cuda")
        packed = interleave_scales(logical)
        recovered = deinterleave_w_scale(packed.float(), n, k)
        torch.testing.assert_close(recovered, logical.float())

    def test_matches_fp32_reference_various_shapes(self):
        # Derived property: the operator equals a block-wise FP32 contraction up
        # to bf16 output rounding, across representative MoE shapes.
        for e, m, n, k in [
            (2, 16, 64, 256),
            (4, 8, 128, 512),
            (1, 32, 256, 128),
            (8, 4, 64, 1024),
            (3, 17, 96, 384),
        ]:
            with self.subTest(e=e, m=m, n=n, k=k):
                a, a_scale, w, w_scale, ref_w = _make_case(e, m, n, k, seed=e * 7 + k)
                out = torch.empty(e, m, n, dtype=torch.bfloat16, device="cuda")
                cutedsl_w4afp8_gemm_per_token_block(a, a_scale, w, w_scale, out, None)
                ref = _independent_fp32_reference(a, a_scale, ref_w, w_scale, None)
                rel = (out.float() - ref).abs().mean() / ref.abs().mean().clamp_min(
                    1e-6
                )
                self.assertLess(rel.item(), 5e-3)

    def test_masked_m_zeroes_invalid_rows(self):
        # Rows >= masked_m[e] must be zeroed; valid rows must match the reference.
        e, m, n, k = 4, 16, 64, 256
        a, a_scale, w, w_scale, ref_w = _make_case(e, m, n, k, seed=123)
        masked_m = torch.tensor([m, 0, 8, 3], dtype=torch.int32, device="cuda")
        out = torch.full((e, m, n), 7.0, dtype=torch.bfloat16, device="cuda")
        cutedsl_w4afp8_gemm_per_token_block(a, a_scale, w, w_scale, out, masked_m)
        ref = _independent_fp32_reference(a, a_scale, ref_w, w_scale, masked_m)
        for expert in range(e):
            valid = int(masked_m[expert])
            self.assertTrue(
                torch.all(out[expert, valid:] == 0),
                f"invalid rows of expert {expert} not zeroed",
            )
            if valid > 0:
                rel = (out[expert, :valid].float() - ref[expert, :valid]).abs().mean()
                rel = rel / ref[expert, :valid].abs().mean().clamp_min(1e-6)
                self.assertLess(rel.item(), 5e-3)

    def test_per_tensor_collapse_is_materially_worse(self):
        # Bug-regression guard: the removed a_scales.abs().max() collapse must be
        # far worse than the block-wise operator when block scales vary.
        e, m, n, k = 4, 32, 128, 512
        a, a_scale, w, w_scale, ref_w = _make_case(e, m, n, k, seed=1)
        ref = _independent_fp32_reference(a, a_scale, ref_w, w_scale, None)
        denom = ref.abs().mean().clamp_min(1e-6)

        out = torch.empty(e, m, n, dtype=torch.bfloat16, device="cuda")
        cutedsl_w4afp8_gemm_per_token_block(a, a_scale, w, w_scale, out, None)
        block_rel = ((out.float() - ref).abs().mean() / denom).item()

        global_scale = a_scale.abs().max()
        collapse = torch.zeros(e, m, n, dtype=torch.float32, device="cuda")
        for expert in range(e):
            w_dq = unpack_int4_weight(w[expert]) * w_scale[expert].repeat_interleave(
                SCALE_BLOCK_SIZE, dim=-1
            )
            collapse[expert] = (a[expert].float() * global_scale) @ w_dq.t()
        collapse_rel = ((collapse - ref).abs().mean() / denom).item()

        self.assertLess(block_rel, 5e-3)
        self.assertGreater(collapse_rel, 0.1)
        self.assertGreater(collapse_rel, 20 * block_rel)


@unittest.skipIf(
    SKIP_TEST or (not SKIP_TEST and torch.cuda.get_device_capability() != (9, 0)),
    "requires SM90",
)
class TestHopperPerTokenBlockW4AFP8Gemm(unittest.TestCase):
    def test_graph_fallback_fails_explicitly(self):
        a, sa, w, sw, _ = _make_case(1, 8, 64, 128, seed=43)
        out = torch.empty(1, 8, 64, device="cuda", dtype=torch.float32)
        mask = torch.tensor([8], device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(
            RuntimeError, "fallback does not support CUDA graph"
        ):
            with torch.cuda.graph(graph, stream=stream):
                cutedsl_w4afp8_gemm_per_token_block(a, sa, w, sw, out, mask)

    def test_int4_lookup_exhaustive_byte_pairs(self):
        # Every packed byte at every position of an 8-byte load. One-hot
        # activations expose decoded values directly, without cancellation.
        for m in (16, 32, 64):
            a = torch.zeros(1, m, 128, device="cuda")
            a[0, torch.arange(m), torch.arange(m) % 16] = 1
            a = a.to(torch.float8_e4m3fn)
            w = torch.arange(256, device="cuda").to(torch.int8)
            w = w.view(1, 256, 1).expand(1, 256, 64).contiguous()
            sa = torch.ones(1, m, 1, device="cuda")
            sw = torch.ones(1, 256, 1, device="cuda")
            mask = torch.tensor([m], device="cuda", dtype=torch.int32)
            out = torch.empty(1, m, 256, device="cuda", dtype=torch.bfloat16)
            self.kernel(a, sa, w, sw, out, mask)
            byte = torch.arange(256, device="cuda")
            low, high = byte % 16, byte // 16
            low = torch.where(low < 8, low, low - 16)
            high = torch.where(high < 8, high, high - 16)
            expected = torch.where(
                (torch.arange(m, device="cuda") % 2)[:, None] == 0, low, high
            )
            torch.testing.assert_close(out[0].float(), expected.float(), rtol=0, atol=0)

    def test_rejects_output_alias_and_internal_overlap(self):
        a, sa, w, sw, _ = _make_case(1, 16, 64, 128, seed=31)
        # FP8 input storage has exactly enough bytes for this BF16 output.
        alias = a.view(torch.bfloat16).view(1, 16, 64)
        mask = torch.tensor([16], device="cuda", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "separate storage"):
            self.kernel(a, sa, w, sw, alias, mask)
        overlap = torch.empty(1, 1, 64, device="cuda", dtype=torch.bfloat16).expand(
            1, 16, 64
        )
        with self.assertRaisesRegex(ValueError, "overlapping"):
            cutedsl_w4afp8_gemm_per_token_block(a, sa, w, sw, overlap, mask)

    def test_two_layer_ll_and_graph(self):
        from sglang.srt.layers.moe.cutedsl_w4afp8_gemm_hopper import (
            hopper_w4afp8_gemm_per_token_block,
        )
        from sglang.srt.layers.moe.cutedsl_w4afp8_gemm_per_token_block import (
            _get_hopper_kernel,
        )
        from sglang.srt.layers.moe.cutedsl_w4afp8_moe import (
            cutedsl_w4afp8_moe_deepep_ll,
        )

        # Do not permit the Torch fallback in this integration test.
        self.assertIs(_get_hopper_kernel(), hopper_w4afp8_gemm_per_token_block)
        for m in (17, 65):
            e, k, n = 3, 512, 512
            a, sa, w1, s1, rw1 = _make_case(e, m, 2 * n, k, seed=37)
            _, _, w2, s2, rw2 = _make_case(e, m, k, n, seed=41)
            # Match DeepEP-style token-minor scale strides, not just row-major.
            sa = sa.transpose(1, 2).contiguous().transpose(1, 2)
            mask = torch.tensor([m, 0, 9], device="cuda", dtype=torch.int32)

            def interleave(s):
                return (
                    s.view(e, s.shape[1], -1, 4)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                    .flatten(2)
                    .to(torch.bfloat16)
                )

            s1p, s2p = interleave(s1), interleave(s2)
            # Reference uses exactly the BF16 weight scales supplied to the LL API.
            s1, s2 = s1.to(torch.bfloat16).float(), s2.to(torch.bfloat16).float()
            topk = torch.zeros(e, m, 1, device="cuda", dtype=torch.int64)
            metadata = [
                torch.zeros(e, 3, device="cuda", dtype=torch.int64) for _ in range(8)
            ]
            offsets = torch.zeros(e + 1, device="cuda", dtype=torch.int64)
            sizes = [
                torch.full((e, 3), -7, device="cuda", dtype=torch.int32)
                for _ in range(2)
            ]

            def run():
                return cutedsl_w4afp8_moe_deepep_ll(
                    a, sa, w1, w2, s1p, s2p, topk, mask, *metadata, offsets, *sizes
                )

            def reference():
                first = _independent_fp32_reference(a, sa, rw1, s1, mask).to(
                    torch.bfloat16
                )
                gate, up = first.chunk(2, dim=-1)
                gate = torch.nn.functional.silu(gate.float()).to(torch.bfloat16)
                hidden = (gate * up).float().reshape(e, m, n // 128, 128)
                scale = hidden.abs().amax(-1).clamp_min(1e-10) / 448.0
                quant = (
                    (hidden / scale[..., None])
                    .clamp(-448, 448)
                    .to(torch.float8_e4m3fn)
                    .reshape(e, m, n)
                )
                return _independent_fp32_reference(quant, scale, rw2, s2, mask)

            def check(got):
                expected = reference()
                self.assertTrue(torch.isfinite(got).all().item())
                error = (got.float() - expected).abs()
                # Includes two BF16 GEMMs and the intermediate FP8 rounding.
                self.assertLess(
                    (error.mean() / expected.abs().mean().clamp_min(1e-6)).item(), 1e-2
                )
                self.assertLess(
                    (error.max() / expected.abs().max().clamp_min(1e-6)).item(), 3e-2
                )
                for expert in range(e):
                    self.assertEqual(
                        torch.count_nonzero(got[expert, int(mask[expert]) :]).item(), 0
                    )
                for size in sizes:
                    self.assertTrue(torch.all(size == -7).item())

            check(run())
            with self.assertRaisesRegex(ValueError, "weight shapes"):
                cutedsl_w4afp8_moe_deepep_ll(
                    a,
                    sa,
                    w1[:, :-1],
                    w2,
                    s1p,
                    s2p,
                    topk,
                    mask,
                    *metadata,
                    offsets,
                    *sizes,
                )
            with self.assertRaisesRegex(ValueError, "interleaved"):
                cutedsl_w4afp8_moe_deepep_ll(
                    a,
                    sa,
                    w1,
                    w2,
                    s1p[..., :-1],
                    s2p,
                    topk,
                    mask,
                    *metadata,
                    offsets,
                    *sizes,
                )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = run()
                sa.mul_(0.75)
                mask.copy_(
                    torch.tensor([3, m - 1, 0], device="cuda", dtype=torch.int32)
                )
                graph.replay()
            torch.cuda.current_stream().wait_stream(stream)
            check(out)

    def test_large_contraction_and_token_boundaries(self):
        for m, n, k in [
            (1, 65, 128),
            (16, 73, 384),
            (17, 73, 384),
            (32, 73, 384),
            (33, 73, 384),
            (64, 2048, 7168),
            (8, 7168, 2048),
        ]:
            with self.subTest(shape=(m, n, k)):
                a, sa, w, sw, rw = _make_case(1, m, n, k, seed=19)
                mask = torch.tensor([m], device="cuda", dtype=torch.int32)
                out = torch.empty(1, m, n, device="cuda", dtype=torch.bfloat16)
                self.kernel(a, sa, w, sw, out, mask)
                self.check_result(out, _independent_fp32_reference(a, sa, rw, sw, mask))

    def test_reference_fallback_output_layouts(self):
        a, sa, w, sw, rw = _make_case(2, 17, 73, 256, seed=29)
        mask = torch.tensor([17, 3], device="cuda", dtype=torch.int32)
        ref = _independent_fp32_reference(a, sa, rw, sw, mask)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            out = torch.empty(2, 73, 17, device="cuda", dtype=dtype).transpose(1, 2)
            cutedsl_w4afp8_gemm_per_token_block(a, sa, w, sw, out, mask)
            self.check_result(out, ref)
        with patch(
            "sglang.srt.layers.moe.cutedsl_w4afp8_gemm_per_token_block._get_hopper_kernel",
            return_value=None,
        ):
            out = torch.empty(2, 17, 73, device="cuda", dtype=torch.bfloat16)
            cutedsl_w4afp8_gemm_per_token_block(a, sa, w, sw, out, mask)
            self.check_result(out, ref)

    @staticmethod
    def kernel(*args):
        # Direct entry: missing toolchain or a broken kernel must FAIL, not
        # silently fall back to the reference that is being tested against.
        from sglang.srt.layers.moe.cutedsl_w4afp8_gemm_hopper import (
            hopper_w4afp8_gemm_per_token_block,
        )

        hopper_w4afp8_gemm_per_token_block(*args)

    def check_result(self, got, ref):
        self.assertTrue(torch.isfinite(got).all().item())
        error = (got.float() - ref).abs()
        self.assertLess((error.mean() / ref.abs().mean().clamp_min(1e-6)).item(), 5e-3)
        # Also bound individual errors; mean-only checks can hide bad tail rows.
        self.assertLess((error.max() / ref.abs().max().clamp_min(1e-6)).item(), 1e-2)

    def test_direct_kernel_tails_and_poisoned_mask(self):
        for e, m, n, k in [(3, 65, 73, 384), (2, 129, 129, 128), (4, 8, 17, 1024)]:
            with self.subTest(shape=(e, m, n, k)):
                a, sa, w, sw, rw = _make_case(e, m, n, k, seed=7)
                mask = torch.tensor(
                    [0] + [m - 1] * (e - 1), device="cuda", dtype=torch.int32
                )
                for expert in range(e):
                    valid = int(mask[expert])
                    sa[expert, valid:] = float("nan")
                    # FP8 cannot be filled directly on all supported PyTorch versions.
                    a.view(torch.int8)[expert, valid:] = 127
                out = torch.full(
                    (e, m, n), float("nan"), device="cuda", dtype=torch.bfloat16
                )
                self.kernel(a, sa, w, sw, out, mask)
                ref = _independent_fp32_reference(a, sa, rw, sw, mask)
                self.check_result(out, ref)
                for expert in range(e):
                    self.assertEqual(
                        torch.count_nonzero(out[expert, int(mask[expert]) :]).item(), 0
                    )

    def test_nondefault_stream_and_graph_replay(self):
        a, sa, w, sw, rw = _make_case(2, 17, 73, 384, seed=17)
        mask = torch.tensor([17, 4], dtype=torch.int32, device="cuda")
        out = torch.empty(2, 17, 73, dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.kernel(a, sa, w, sw, out, mask)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.kernel(a, sa, w, sw, out, mask)
            # New values and a new mask must be read on every replay.
            sa.mul_(2)
            mask.fill_(9)
            out.fill_(float("nan"))
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        ref = _independent_fp32_reference(a, sa, rw, sw, mask)
        self.check_result(out, ref)
        self.assertEqual(torch.count_nonzero(out[:, 9:]).item(), 0)

    def test_unaligned_payload_and_noncontiguous_inputs(self):
        a, sa, w, sw, rw = _make_case(2, 17, 73, 256, seed=13)

        # Contiguous, but deliberately not aligned to the vector copy width.
        def offset_copy(t):
            storage = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)
            result = storage[1:].view(t.shape)
            result.copy_(t)
            return result

        a, w = offset_copy(a), offset_copy(w)
        sa = sa.transpose(1, 2).contiguous().transpose(1, 2)
        mask = torch.full((2,), 17, dtype=torch.int64, device="cuda")
        out = torch.empty(2 * 17 * 73 + 1, dtype=torch.bfloat16, device="cuda")[
            1:
        ].view(2, 17, 73)
        self.kernel(a, sa, w, sw, out, mask)
        self.check_result(out, _independent_fp32_reference(a, sa, rw, sw, mask))

    def test_all_int4_values_and_zero_scales(self):
        a, sa, w, sw, rw = _make_case(1, 16, 64, 256, seed=11)
        rw.copy_(
            torch.arange(-8, 8, device="cuda", dtype=torch.int8)
            .repeat(rw.numel() // 16)
            .view_as(rw)
        )
        w.copy_(_pack_int4(rw))
        sa[:, 0] = 0
        sw[:, 0] = 0
        mask = torch.tensor([16], dtype=torch.int32, device="cuda")
        out = torch.empty(1, 16, 64, dtype=torch.bfloat16, device="cuda")
        self.kernel(a, sa, w, sw, out, mask)
        self.check_result(out, _independent_fp32_reference(a, sa, rw, sw, mask))
        self.assertEqual(torch.count_nonzero(out[:, 0]).item(), 0)
        self.assertEqual(torch.count_nonzero(out[:, :, 0]).item(), 0)

    def test_runtime_failures_are_not_swallowed(self):
        a, sa, w, sw, _ = _make_case(1, 8, 64, 128, seed=3)
        out = torch.empty(1, 8, 64, dtype=torch.bfloat16, device="cuda")
        with patch(
            "sglang.srt.layers.moe.cutedsl_w4afp8_gemm_per_token_block._get_hopper_kernel",
            return_value=unittest.mock.Mock(side_effect=RuntimeError("kernel failure")),
        ):
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                cutedsl_w4afp8_gemm_per_token_block(a, sa, w, sw, out)

    def test_empty_dimensions_and_invalid_metadata(self):
        for e, m, n, k in [
            (0, 8, 64, 128),
            (2, 0, 64, 128),
            (2, 8, 0, 128),
            (2, 8, 64, 0),
        ]:
            a, sa, w, sw, _ = _make_case(e, m, n, k, seed=1)
            mask = torch.full((e,), m, dtype=torch.int32, device="cuda")
            out = torch.full((e, m, n), 7, dtype=torch.bfloat16, device="cuda")
            self.kernel(a, sa, w, sw, out, mask)
            self.assertEqual(torch.count_nonzero(out).item(), 0)
        a, sa, w, sw, _ = _make_case(2, 8, 64, 128, seed=1)
        out = torch.empty(2, 8, 64, dtype=torch.bfloat16, device="cuda")
        with self.assertRaises(ValueError):
            cutedsl_w4afp8_gemm_per_token_block(a, sa, w[:1], sw, out)
        with self.assertRaises(ValueError):
            cutedsl_w4afp8_gemm_per_token_block(
                a, sa, w, sw, out, torch.zeros(1, device="cuda", dtype=torch.int32)
            )
        with self.assertRaises(ValueError):
            cutedsl_w4afp8_gemm_per_token_block(a, sa.cpu(), w, sw, out)

    @unittest.skipUnless(torch.cuda.device_count() > 1, "requires two CUDA devices")
    def test_device_context_and_cache(self):
        for device in [0, 1, 0]:
            if torch.cuda.get_device_capability(device) != (9, 0):
                continue
            a, sa, w, sw, rw = _make_case(
                1, 8, 64, 128, seed=3, device=f"cuda:{device}"
            )
            mask = torch.tensor([8], dtype=torch.int32, device=a.device)
            out = torch.empty(1, 8, 64, dtype=torch.bfloat16, device=a.device)
            with torch.cuda.device(1 - device):
                self.kernel(a, sa, w, sw, out, mask)
                self.assertEqual(torch.cuda.current_device(), 1 - device)
            self.check_result(out, _independent_fp32_reference(a, sa, rw, sw, mask))

    @unittest.skipUnless(
        os.environ.get("SGLANG_BENCH_W4AFP8") == "1", "opt-in benchmark"
    )
    def test_benchmark(self):
        # Opt-in only: CI should not impose hardware-dependent latency limits.
        # CUDA graph events exclude Python submission overhead, compilation and
        # allocations. Five independent measurements expose timing variation.
        import statistics

        for e, m, n, k in [
            (8, 128, 2048, 7168),
            (8, 128, 7168, 2048),
            (32, 64, 2048, 7168),
            (64, 32, 2048, 7168),
            (256, 8, 2048, 7168),
        ]:
            a, sa, w, sw, _ = _make_case(e, m, n, k, seed=0)
            mask = torch.full((e,), m, dtype=torch.int32, device="cuda")
            out = torch.empty(e, m, n, dtype=torch.bfloat16, device="cuda")
            for _ in range(5):
                self.kernel(a, sa, w, sw, out, mask)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(20):
                    self.kernel(a, sa, w, sw, out, mask)
            timings = []
            for _ in range(5):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end) * 1000 / 20)
            us = statistics.median(timings)
            tflops = 2 * e * m * n * k / (us * 1e6)
            print(
                f"E={e} M={m} K={k} N={n}: median={us:.1f} us range=[{min(timings):.1f},{max(timings):.1f}] {tflops:.1f} TFLOP/s",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
