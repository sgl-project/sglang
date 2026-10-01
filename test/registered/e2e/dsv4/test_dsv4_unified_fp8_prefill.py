# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prefill read path over the two-pool fp8 unified_kv (aiter's opus kernel).

Two regions per token: the paged prefix pools and this chunk's flat extend pair.
What these pin is that both regions are addressed with the same row layout and
that the pair guards fire before the launch -- the reference attends over the
*dequantized* pools, so a mismatch is the wiring rather than the fp8 round-trip.

The quantization helpers are the ones from the decode test rather than a shared
module: files under test/registered/ are collected standalone (no __init__.py,
no conftest), so importing across them breaks in CI.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DSV4_FP8_NOPE_ROW_BYTES
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEVICE = torch.device("cuda")

NOPE_ROW_BYTES = DSV4_FP8_NOPE_ROW_BYTES
NOPE_DIM = 448  # fp8 values per row, in elements
ROPE_DIM = 64
QUANT_TILE = 64
NUM_TILES = NOPE_DIM // QUANT_TILE  # 7
SCALE_OFF = NOPE_DIM
# latent element count; the same number as NOPE_ROW_BYTES, different unit
V_HEAD_DIM = NOPE_DIM + ROPE_DIM
SOFTMAX_SCALE = V_HEAD_DIM**-0.5

_needs_gfx950 = unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "two-pool fp8 prefill runs on the gfx950 opus kernel",
)


def _pow2_ceil_scale(amax: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.clamp_min(amax, 1e-4).log2().ceil()).to(torch.float32)


def _pow2_to_e8m0(pow2: torch.Tensor) -> torch.Tensor:
    biased = torch.log2(pow2).round().to(torch.int32) + 127
    return torch.clamp(biased, 0, 254).to(torch.uint8)


def _quantize_nope(nope_fp32: torch.Tensor):
    """[..., 448] fp32 -> (fp8 values, [..., 7] e8m0 bytes, bf16 round-trip)"""
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    leading = nope_fp32.shape[:-1]
    tiled = nope_fp32.reshape(*leading, NUM_TILES, QUANT_TILE)
    scale = _pow2_ceil_scale(tiled.abs().amax(dim=-1) / fp8_max)
    values = (tiled / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    dequant = (values.to(torch.float32) * scale.unsqueeze(-1)).reshape(
        *leading, NOPE_DIM
    )
    return (
        values.reshape(*leading, NOPE_DIM),
        _pow2_to_e8m0(scale),
        dequant.to(torch.bfloat16),
    )


def _pack(values: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """448 values + each tile scale twice + pad, as one NOPE_ROW_BYTES fp8 row

    Pad bytes get garbage on purpose, same reason as the decode test: production
    allocates these with new_empty(), so a reader that walks past the scales
    should fail here rather than as an accuracy drift.
    """
    leading = values.shape[:-1]
    row = torch.randint(
        1, 256, (*leading, NOPE_ROW_BYTES), dtype=torch.uint8, device=values.device
    )
    row[..., :NOPE_DIM] = values.view(torch.uint8)
    dup = scale_e8m0.unsqueeze(-1).expand(*scale_e8m0.shape, 2).reshape(*leading, -1)
    row[..., SCALE_OFF : SCALE_OFF + 2 * NUM_TILES] = dup
    return row.view(torch.float8_e4m3fn)


def _make_latent(*leading: int):
    """Return (packed fp8 rows, bf16 rope, bf16 latent the kernel effectively sees)."""
    nope = torch.randn(*leading, NOPE_DIM, device=DEVICE, dtype=torch.float32)
    rope = torch.randn(*leading, ROPE_DIM, device=DEVICE, dtype=torch.bfloat16)
    values, scale, nope_bf16 = _quantize_nope(nope)
    silver = torch.cat([nope_bf16, rope], dim=-1)
    return _pack(values, scale).contiguous(), rope.contiguous(), silver


def _ragged(lengths, rows):
    """per-token row lists -> (flat int32 indices, int32 indptr)"""
    indptr = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=DEVICE)
    indptr[1:] = torch.cumsum(
        torch.tensor(lengths, dtype=torch.int32, device=DEVICE), dim=0
    )
    parts = [
        torch.randperm(rows, device=DEVICE)[:n].to(torch.int32)
        for n in lengths
        if n > 0
    ]
    flat = (
        torch.cat(parts) if parts else torch.empty(0, dtype=torch.int32, device=DEVICE)
    )
    return flat.contiguous(), indptr


def _reference(q_silver, sources, sink, scale):
    """Ragged two-region attention in fp32; V is the full latent, sink V is zero.

    ``sources`` is [(silver, indices, indptr), ...]. The kernel shares one online
    softmax across the regions, so order does not matter and this just
    concatenates whatever each region selected.
    """
    T, H, _ = q_silver.shape
    out = torch.zeros(T, H, V_HEAD_DIM, device=q_silver.device, dtype=torch.float32)
    q = q_silver.float()
    sink_f = sink.float()
    for t in range(T):
        keys = []
        for silver, indices, indptr in sources:
            lo, hi = int(indptr[t]), int(indptr[t + 1])
            if hi > lo:
                keys.append(silver[indices[lo:hi].long()].float())
        if not keys:
            # only the sink is left: it contributes to the denominator and has
            # V = 0, so the row is exactly zero
            continue
        k = torch.cat(keys, dim=0)
        logits = q[t] @ k.transpose(0, 1) * scale
        m = torch.cat([logits, sink_f.unsqueeze(1)], dim=1).amax(dim=1, keepdim=True)
        p = torch.exp(logits - m)
        denom = p.sum(dim=1, keepdim=True) + torch.exp(sink_f.unsqueeze(1) - m)
        out[t] = (p @ k) / denom
    return out


class TestUnifiedFp8Prefill(CustomTestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.rows = 256

    def _run(self, prefix_lens, extend_lens, num_heads=16, scale=SOFTMAX_SCALE):
        T = len(prefix_lens)
        self.assertEqual(T, len(extend_lens))
        extend_rows = max(max(extend_lens), 1)
        pool_nope, pool_rope, pool_silver = _make_latent(self.rows)
        ext_nope, ext_rope, ext_silver = _make_latent(extend_rows)
        q_packed, q_rope, q_silver = _make_latent(T, num_heads)
        pre_i, pre_p = _ragged(prefix_lens, self.rows)
        ext_i, ext_p = _ragged(extend_lens, extend_rows)
        sink = torch.randn(num_heads, device=DEVICE, dtype=torch.float32)

        got = runtime.prefill_fp8_2buff(
            q=q_packed,
            q_rope=q_rope,
            unified_kv=pool_nope,
            unified_kv_rope=pool_rope,
            kv_indices_prefix=pre_i,
            kv_indptr_prefix=pre_p,
            kv_extend=ext_nope,
            kv_extend_rope=ext_rope,
            kv_indices_extend=ext_i,
            kv_indptr_extend=ext_p,
            attn_sink=sink,
            softmax_scale=scale,
            v_head_dim=V_HEAD_DIM,
        )
        want = _reference(
            q_silver,
            [(pool_silver, pre_i, pre_p), (ext_silver, ext_i, ext_p)],
            sink,
            scale,
        )
        return got.float(), want

    def _assert_close(self, got, want, atol=3e-2, rtol=3e-2):
        """torch-style combined bound, same reasoning as the decode test.

        A pure relative bound is useless here: the latent's outputs straddle
        zero, so the absolute error bf16 accumulation costs reads as a huge
        relative one on the rows that land near zero.
        """
        diff = (got - want).abs()
        outside = diff > atol + rtol * want.abs()
        self.assertEqual(
            outside.sum().item(),
            0,
            f"{outside.sum().item()}/{outside.numel()} elements outside "
            f"the bound, max abs {diff.max().item():.4g}",
        )

    @_needs_gfx950
    def test_matches_dequantized_reference(self):
        cases = (
            ([64, 64, 64, 64], [1, 2, 3, 4]),
            ([17, 5, 128, 1], [4, 4, 4, 4]),
            ([200] * 6, [1, 3, 6, 2, 5, 4]),
        )
        for prefix_lens, extend_lens in cases:
            with self.subTest(prefix=prefix_lens, extend=extend_lens):
                got, want = self._run(prefix_lens, extend_lens)
                self._assert_close(got, want)

    @_needs_gfx950
    def test_first_chunk_has_an_empty_prefix_for_every_token(self):
        """the real shape of chunk 0: nothing committed yet, extend is all there is"""
        got, want = self._run([0, 0, 0, 0], [1, 2, 3, 4])
        self.assertTrue(bool(got.isfinite().all()))
        self._assert_close(got, want)

    @_needs_gfx950
    def test_a_token_with_neither_region_comes_back_zero(self):
        """Not NaN, which is where this differs from the asm decode reader.

        decode_fp8_2buff has to mask that case itself; this kernel already
        returns zeros, so there is deliberately no mask on this path. The
        reference skips those rows for the same reason: with only the sink left
        the numerator is zero.
        """
        got, want = self._run([64, 0, 64], [2, 0, 2])
        self.assertTrue(torch.equal(got[1], torch.zeros_like(got[1])))
        self._assert_close(got, want)

    @_needs_gfx950
    def test_head_count_64(self):
        got, want = self._run([48, 96], [3, 5], num_heads=64)
        self._assert_close(got, want)

    @_needs_gfx950
    def test_scale_is_passed_through(self):
        """unlike the decode reader, this kernel takes the scale as an argument"""
        got, want = self._run([32, 32], [2, 2], scale=0.5 * SOFTMAX_SCALE)
        self._assert_close(got, want)

    @_needs_gfx950
    def test_extend_row_narrower_than_the_pool_is_rejected(self):
        """the two regions are walked with one row layout, so a short row would
        read the next token's bytes as this one's scales"""
        pool_nope, pool_rope, _ = _make_latent(self.rows)
        q_packed, q_rope, _ = _make_latent(2, 16)
        ext_nope, ext_rope, _ = _make_latent(4)
        pre_i, pre_p = _ragged([8, 8], self.rows)
        ext_i, ext_p = _ragged([1, 1], 4)
        with self.assertRaisesRegex(AssertionError, "extend nope row"):
            runtime.prefill_fp8_2buff(
                q=q_packed,
                q_rope=q_rope,
                unified_kv=pool_nope,
                unified_kv_rope=pool_rope,
                kv_indices_prefix=pre_i,
                kv_indptr_prefix=pre_p,
                kv_extend=ext_nope[:, : NOPE_ROW_BYTES // 2].contiguous(),
                kv_extend_rope=ext_rope,
                kv_indices_extend=ext_i,
                kv_indptr_extend=ext_p,
                attn_sink=torch.randn(16, device=DEVICE, dtype=torch.float32),
                softmax_scale=SOFTMAX_SCALE,
                v_head_dim=V_HEAD_DIM,
            )

    @_needs_gfx950
    def test_mismatched_pools_are_rejected_before_the_launch(self):
        pool_nope, _, _ = _make_latent(self.rows)
        short_rope = torch.zeros(
            self.rows // 2, ROPE_DIM, dtype=torch.bfloat16, device=DEVICE
        )
        q_packed, q_rope, _ = _make_latent(2, 16)
        ext_nope, ext_rope, _ = _make_latent(4)
        pre_i, pre_p = _ragged([8, 8], self.rows)
        ext_i, ext_p = _ragged([1, 1], 4)
        with self.assertRaisesRegex(AssertionError, "pool rows differ"):
            runtime.prefill_fp8_2buff(
                q=q_packed,
                q_rope=q_rope,
                unified_kv=pool_nope,
                unified_kv_rope=short_rope,
                kv_indices_prefix=pre_i,
                kv_indptr_prefix=pre_p,
                kv_extend=ext_nope,
                kv_extend_rope=ext_rope,
                kv_indices_extend=ext_i,
                kv_indptr_extend=ext_p,
                attn_sink=torch.randn(16, device=DEVICE, dtype=torch.float32),
                softmax_scale=SOFTMAX_SCALE,
                v_head_dim=V_HEAD_DIM,
            )


if __name__ == "__main__":
    unittest.main()
