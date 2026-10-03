"""Decode read path over the two-pool fp8 unified_kv (aiter's v4 nm asm kernel).

What these pin is the reader-side plumbing, not the kernel's arithmetic: the
packed 512 B nope row and the bf16 rope pool addressed by one shared row index,
a per-token ``qo_indptr``, and the ragged ``kv_indptr`` the existing index
builders emit -- including what they emit for a cuda-graph padded row. The
reference attends over the *dequantized* pools, so a mismatch is the wiring
rather than the fp8 round-trip.

The quantization helpers mirror aiter's own reference
(``op_tests/test_mla_v40_persistent.py``: ``quantize_v4_nope_bpad8`` /
``pack_v4_nope_scale``). They are duplicated rather than imported because that
file is a test, not part of the aiter package.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DSV4_FP8_NOPE_ROW_BYTES
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# the asm shader is only shipped for gfx950
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEVICE = torch.device("cuda")

NOPE_ROW_BYTES = DSV4_FP8_NOPE_ROW_BYTES
NOPE_DIM = 448  # fp8 values per row, in elements
ROPE_DIM = 64
QUANT_TILE = 64
NUM_TILES = NOPE_DIM // QUANT_TILE  # 7
SCALE_OFF = NOPE_DIM  # scales start where the values end
# latent element count; the same number as NOPE_ROW_BYTES, different unit
V_HEAD_DIM = NOPE_DIM + ROPE_DIM
SOFTMAX_SCALE = V_HEAD_DIM**-0.5  # what the kernel hardcodes

_needs_gfx950 = unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "two-pool fp8 decode runs on the gfx950 asm shader",
)


def _pow2_ceil_scale(amax: torch.Tensor) -> torch.Tensor:
    """amax/fp8_max -> the next power of two at or above it, as fp32"""
    return torch.pow(2.0, torch.clamp_min(amax, 1e-4).log2().ceil()).to(torch.float32)


def _pow2_to_e8m0(pow2: torch.Tensor) -> torch.Tensor:
    """byte B encodes 2^(B-127); 0 means 0.0 and 255 means inf, so clamp to 254"""
    biased = torch.log2(pow2).round().to(torch.int32) + 127
    return torch.clamp(biased, 0, 254).to(torch.uint8)


def _e8m0_to_fp32(byte: torch.Tensor) -> torch.Tensor:
    return torch.exp2((byte.to(torch.int32) - 127).to(torch.float32))


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

    The 50 pad bytes get garbage on purpose. Production allocates Q with
    nope_pool.new_empty(), so a reader that ever starts looking past the scales
    should fail here and not in a bf16-vs-fp8 accuracy chase.
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


def _ragged(lengths, rows, device=DEVICE):
    """per-token row lists -> (flat int32 indices, int32 indptr)"""
    indptr = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(
        torch.tensor(lengths, dtype=torch.int32, device=device), dim=0
    )
    flat = torch.cat(
        [
            torch.randperm(rows, device=device)[:n].to(torch.int32)
            for n in lengths
            if n > 0
        ]
        or [torch.empty(0, dtype=torch.int32, device=device)]
    )
    return flat.contiguous(), indptr


def _reference(q_silver, kv_silver, indices, indptr, sink):
    """Ragged sparse attention in fp32; V is the full latent, sink has zero V."""
    T, H, _ = q_silver.shape
    out = torch.zeros(T, H, V_HEAD_DIM, device=q_silver.device, dtype=torch.float32)
    q = q_silver.float()
    sink_f = sink.float()
    for t in range(T):
        lo, hi = int(indptr[t]), int(indptr[t + 1])
        k = kv_silver[indices[lo:hi].long()].float()  # [L, 512]
        logits = q[t] @ k.transpose(0, 1) * SOFTMAX_SCALE  # [H, L]
        aug = torch.cat([logits, sink_f.unsqueeze(1)], dim=1)
        m = aug.amax(dim=1, keepdim=True)
        p = torch.exp(logits - m)
        denom = p.sum(dim=1, keepdim=True) + torch.exp(sink_f.unsqueeze(1) - m)
        out[t] = (p @ k) / denom
    return out


class TestUnifiedFp8Decode(CustomTestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.rows = 256

    def _run(self, lengths, num_heads):
        T = len(lengths)
        pool_nope, pool_rope, kv_silver = _make_latent(self.rows)
        q_packed, q_rope, q_silver = _make_latent(T, num_heads)
        indices, indptr = _ragged(lengths, self.rows)
        sink = torch.randn(num_heads, device=DEVICE, dtype=torch.float32)

        got = runtime.decode_fp8_2buff(
            q=q_packed,
            q_rope=q_rope,
            unified_kv=pool_nope,
            unified_kv_rope=pool_rope,
            kv_indices=indices,
            kv_indptr=indptr,
            attn_sink=sink,
            v_head_dim=V_HEAD_DIM,
        )
        want = _reference(q_silver, kv_silver, indices, indptr, sink)
        return got.float(), want

    def _assert_close(self, got, want, atol=3e-2, rtol=3e-2):
        """torch-style combined bound.

        A pure relative bound is useless here: the latent's outputs straddle
        zero, so an absolute error of 4e-3 -- which is what bf16 accumulation
        costs -- reads as 47% relative on the rows that land near zero.
        """
        diff = (got - want).abs()
        outside = diff > atol + rtol * want.abs()
        self.assertEqual(
            outside.sum().item(),
            0,
            f"{outside.sum().item()}/{outside.numel()} elements outside "
            f"{atol}+{rtol}|ref|, max abs {diff.max().item():.4g}",
        )

    @_needs_gfx950
    def test_matches_dequantized_reference(self):
        for lengths in ([64] * 4, [17, 5, 128, 1], [200] * 8):
            with self.subTest(lengths=lengths):
                got, want = self._run(lengths, num_heads=16)
                self._assert_close(got, want)

    @_needs_gfx950
    def test_head_count_64(self):
        got, want = self._run([48, 96], num_heads=64)
        self._assert_close(got, want)

    @_needs_gfx950
    def test_cuda_graph_pad_reads_only_the_reserved_ring_row(self):
        """What the real builder emits for a cuda-graph padded row.

        Not an empty segment: both dsv4 backends fill padded ``seq_lens`` with 1,
        so ``clamp(seq_lens, max=win)`` leaves the pad one row long. It lands on
        ring row 0, the slot ReqToTokenPool reserves for exactly this
        (``free_slots`` starts at 1), so a pad only ever reads and writes there.
        """
        win = ring = 64
        seq_lens = torch.tensor([37, 55, 1, 1], dtype=torch.int32, device=DEVICE)
        state_slot = torch.tensor([1, 2, 0, 0], dtype=torch.int32, device=DEVICE)
        n = seq_lens.numel()
        zero = torch.zeros(n, dtype=torch.int32, device=DEVICE)

        indices, indptr = runtime.build_decode_streams(
            state_slot=state_slot,
            positions=seq_lens - 1,  # raw_positions, as the backend derives it
            swa_len=torch.clamp(seq_lens, max=win),
            hca_len=zero,
            csa_len=zero,
            hca_page_indices=torch.zeros(n, 1, dtype=torch.int32, device=DEVICE),
            csa_width=1,
            win=win,
            ring_stride=ring,
            swa_pages=self.rows,
        )[:2]

        seg = (indptr[1 : n + 1] - indptr[:n]).tolist()
        self.assertEqual(seg, [37, 55, 1, 1])
        for pad in (2, 3):
            self.assertEqual(indices[int(indptr[pad])].item(), 0)
        live = indices[: int(indptr[2])]
        self.assertGreaterEqual(int(live.min()), ring, "live rows hit slot 0's block")

        pool_nope, pool_rope, _ = _make_latent(self.rows)
        q_packed, q_rope, _ = _make_latent(n, 16)
        out = runtime.decode_fp8_2buff(
            q=q_packed,
            q_rope=q_rope,
            unified_kv=pool_nope,
            unified_kv_rope=pool_rope,
            kv_indices=indices.contiguous(),
            kv_indptr=indptr,
            attn_sink=torch.randn(16, device=DEVICE, dtype=torch.float32),
            v_head_dim=V_HEAD_DIM,
        )
        # the mask never fires here, so what matters is the reserved row keeping
        # the pad finite rather than it coming back zeroed
        self.assertTrue(bool(out.isfinite().all()))

    @_needs_gfx950
    def test_empty_segment_comes_back_nonfinite(self):
        """Guard for a shape the builders do not reach today.

        Padded seq_lens are always filled with 1 (see
        test_cuda_graph_pad_reads_only_the_reserved_ring_row), so an empty segment
        can only come from a builder change -- and it comes back NaN, not zero,
        since the asm kernel divides by an all-sink denominator.
        """
        got, want = self._run([32, 0, 32], num_heads=16)
        self.assertTrue(bool(torch.isnan(got[1]).any()))
        for t in (0, 2):
            self._assert_close(got[t], want[t])

    @_needs_gfx950
    def test_split_tail_override_matches_reference(self):
        """past 40 tokens runtime overrides the split count, moving the kernel onto
        a different stage-2 merge partition -- must still match the reference
        """
        lengths = [200, 64] * 24  # 48 tokens, both layer flavours' segment lengths
        self.assertGreater(len(lengths), 40)
        got, want = self._run(lengths, num_heads=16)
        self._assert_close(got, want)

    @_needs_gfx950
    def test_rejects_pool_that_is_not_a_pair(self):
        pool_nope, pool_rope, _ = _make_latent(self.rows)
        q_packed, q_rope, _ = _make_latent(2, 16)
        indices, indptr = _ragged([4, 4], self.rows)
        sink = torch.zeros(16, device=DEVICE, dtype=torch.float32)
        with self.assertRaises(AssertionError):
            runtime.decode_fp8_2buff(
                q=q_packed,
                q_rope=q_rope,
                unified_kv=pool_nope,
                unified_kv_rope=pool_rope[: self.rows // 2],
                kv_indices=indices,
                kv_indptr=indptr,
                attn_sink=sink,
                v_head_dim=V_HEAD_DIM,
            )

    @_needs_gfx950
    def test_rejects_q_row_wider_than_the_pool_row(self):
        pool_nope, pool_rope, _ = _make_latent(self.rows)
        q_packed, q_rope, _ = _make_latent(2, 16)
        indices, indptr = _ragged([4, 4], self.rows)
        sink = torch.zeros(16, device=DEVICE, dtype=torch.float32)
        wider = torch.zeros(
            2, 16, NOPE_ROW_BYTES + 64, device=DEVICE, dtype=torch.float8_e4m3fn
        )
        wider[..., :NOPE_ROW_BYTES] = q_packed
        with self.assertRaises(AssertionError):
            runtime.decode_fp8_2buff(
                q=wider,
                q_rope=q_rope,
                unified_kv=pool_nope,
                unified_kv_rope=pool_rope,
                kv_indices=indices,
                kv_indptr=indptr,
                attn_sink=sink,
                v_head_dim=V_HEAD_DIM,
            )


if __name__ == "__main__":
    unittest.main()
