"""Two-pool fp8 store tests for the fused QK norm+RoPE kernel wrapper.

Under SGLANG_DSV4_UNIFIED_KV_FP8 ``fused_qk_norm_rope_swa_store`` delegates to
aiter, which packs K into a 512 B fp8 nope row (448 B payload + 14 B duplicated
E8M0 tile scales) plus a bf16 rope row and scatters both into the SWA ring.
These tests pin that layout, which the decode reader depends on, and that Q
comes back unquantized.
"""

import unittest

import torch

from sglang.kernels.ops.attention.fused_qk_norm_rope_store import (
    _HAS_GROUP_QUANT,
    fused_qk_norm_rope_swa_store,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV4_FP8_NOPE_ROW_BYTES,
    DSV4_FP8_QUANT_TILE,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# aiter's group-quant path is gfx95-only, so on the default mi300 runner every case
# here would skip
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEVICE = torch.device("cuda")

NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
NUM_TILES = NOPE_DIM // DSV4_FP8_QUANT_TILE
SCALE_OFF = NOPE_DIM
SCALE_BYTES = 2 * NUM_TILES

NUM_HEADS = 4
EPS = 1e-6
MAX_POS = 256
RING_STRIDE = 16


def _cos_sin():
    inv = 1.0 / (
        10000 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM)
    )
    ang = torch.arange(MAX_POS, dtype=torch.float32)[:, None] * inv[None, :]
    return (
        ang.cos().to(torch.bfloat16).to(DEVICE),
        ang.sin().to(torch.bfloat16).to(DEVICE),
    )


def _ref_norm_rope(kv, weight, cos, sin, positions):
    """rmsnorm over the whole latent, then GPT-J rope on the trailing pe half."""
    x = kv.float()
    scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)
    normed = x * scale * weight.float()
    nope, pe = normed[:, :NOPE_DIM], normed[:, NOPE_DIM:]
    c = cos.float()[positions]
    s = sin.float()[positions]
    even, odd = pe[:, 0::2], pe[:, 1::2]
    out = torch.empty_like(pe)
    out[:, 0::2] = even * c - odd * s
    out[:, 1::2] = odd * c + even * s
    return nope, out


def _ref_tile_scales(nope):
    """e8m0 exponent byte per 1x64 tile, from the fp32 reference nope."""
    tiles = nope.reshape(nope.shape[0], NUM_TILES, DSV4_FP8_QUANT_TILE)
    absmax = tiles.abs().amax(-1).clamp_min(1e-8)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    return torch.ceil(torch.log2(absmax / fp8_max))


def _pools(n_rows):
    nope_pool = torch.zeros(
        n_rows, DSV4_FP8_NOPE_ROW_BYTES, dtype=torch.float8_e4m3fn, device=DEVICE
    )
    rope_pool = torch.zeros(n_rows, ROPE_DIM, dtype=torch.bfloat16, device=DEVICE)
    return nope_pool, rope_pool


class _StoreCase(CustomTestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.T = 6
        self.cos, self.sin = _cos_sin()
        self.weight = torch.randn(HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
        self.kv = torch.randn(self.T, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
        self.q = torch.randn(
            self.T, NUM_HEADS * HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
        )
        self.positions = torch.arange(self.T, device=DEVICE, dtype=torch.int64)
        # distinct ring rows so each row has one unambiguous writer
        self.swa_loc = (
            self.positions.to(torch.int32) % RING_STRIDE + RING_STRIDE
        ).contiguous()


@unittest.skipUnless(_HAS_GROUP_QUANT, "needs aiter's group-quant kernel on gfx95x")
class TestUnifiedFp8QkNormRope(_StoreCase):
    def _call(self, nope_pool=None, rope_pool=None, k_nope=None, k_rope=None):
        return fused_qk_norm_rope_swa_store(
            q=self.q,
            kv=self.kv,
            q_norm_weight=None,
            kv_norm_weight=self.weight,
            q_rms_eps=EPS,
            kv_rms_eps=EPS,
            rope_head_dim=ROPE_DIM,
            cos_cache=self.cos,
            sin_cache=self.sin,
            positions=self.positions,
            swa_cache=nope_pool,
            swa_loc=None if nope_pool is None else self.swa_loc,
            swa_page_size=1,
            dtype=torch.bfloat16,
            fp8_2buff=True,
            swa_rope_cache=rope_pool,
            k_nope_out=k_nope,
            k_rope_out=k_rope,
        )

    def test_pool_rows_equal_the_dense_packed_output(self):
        """the ring write and the dense K buffers come from the same values"""
        nope_pool, rope_pool = _pools(2 * RING_STRIDE)
        k_nope = torch.empty(
            self.T, 1, DSV4_FP8_NOPE_ROW_BYTES, dtype=torch.float8_e4m3fn, device=DEVICE
        )
        k_rope = torch.empty(self.T, 1, ROPE_DIM, dtype=torch.bfloat16, device=DEVICE)
        self._call(nope_pool, rope_pool, k_nope, k_rope)

        rows = self.swa_loc.long()
        pool_bytes = nope_pool.view(torch.uint8)[rows, : SCALE_OFF + SCALE_BYTES]
        dense_bytes = k_nope.view(torch.uint8)[:, 0, : SCALE_OFF + SCALE_BYTES]
        self.assertTrue(torch.equal(pool_bytes, dense_bytes))
        self.assertTrue(torch.equal(rope_pool[rows], k_rope[:, 0]))

    def test_scale_bytes_are_duplicated_e8m0(self):
        """the asm reader reads each tile scale twice, so the 14 B must be 7 equal pairs"""
        nope_pool, rope_pool = _pools(2 * RING_STRIDE)
        self._call(nope_pool, rope_pool)

        rows = self.swa_loc.long()
        scales = nope_pool.view(torch.uint8)[
            rows, SCALE_OFF : SCALE_OFF + SCALE_BYTES
        ].reshape(self.T, NUM_TILES, 2)
        self.assertTrue(torch.equal(scales[..., 0], scales[..., 1]))

        ref_nope, _ = _ref_norm_rope(
            self.kv, self.weight, self.cos, self.sin, self.positions
        )
        expected = (_ref_tile_scales(ref_nope) + 127).to(torch.uint8)
        self.assertTrue(torch.equal(scales[..., 0], expected))

    def test_dequantized_nope_tracks_the_reference(self):
        nope_pool, rope_pool = _pools(2 * RING_STRIDE)
        self._call(nope_pool, rope_pool)

        ref_nope, _ = _ref_norm_rope(
            self.kv, self.weight, self.cos, self.sin, self.positions
        )
        exps = _ref_tile_scales(ref_nope)
        payload = nope_pool[self.swa_loc.long(), :NOPE_DIM].float()
        deq = (
            payload.reshape(self.T, NUM_TILES, DSV4_FP8_QUANT_TILE)
            * torch.exp2(exps)[..., None]
        ).reshape(self.T, NOPE_DIM)

        # e4m3 carries 3 mantissa bits, so the worst case is ~2^-4 of the tile's
        # own absmax. Anything beyond that means the scale or the payload is off,
        # not rounding.
        tile_absmax = (
            ref_nope.reshape(self.T, NUM_TILES, DSV4_FP8_QUANT_TILE)
            .abs()
            .amax(-1)
            .repeat_interleave(DSV4_FP8_QUANT_TILE, dim=1)
        )
        self.assertTrue(torch.all((deq - ref_nope).abs() <= 0.07 * tile_absmax))

    def test_rope_pool_matches_the_bf16_reference(self):
        nope_pool, rope_pool = _pools(2 * RING_STRIDE)
        self._call(nope_pool, rope_pool)

        _, ref_pe = _ref_norm_rope(
            self.kv, self.weight, self.cos, self.sin, self.positions
        )
        got = rope_pool[self.swa_loc.long()].float()
        torch.testing.assert_close(got, ref_pe, rtol=2e-2, atol=2e-2)

    def test_q_stays_bf16_and_rotated(self):
        q_out = self._call()
        self.assertEqual(q_out.dtype, torch.bfloat16)
        self.assertEqual(tuple(q_out.shape), (self.T, NUM_HEADS, HEAD_DIM))

        head = self.q.view(self.T, NUM_HEADS, HEAD_DIM)[:, 0]
        ones = torch.ones(HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
        ref_nope, ref_pe = _ref_norm_rope(
            head, ones, self.cos, self.sin, self.positions
        )
        got = q_out[:, 0].float()
        torch.testing.assert_close(got[:, :NOPE_DIM], ref_nope, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(got[:, NOPE_DIM:], ref_pe, rtol=2e-2, atol=2e-2)

    def test_strided_q_out_is_filled_without_touching_the_padding(self):
        """attn_tp_size > 1 hands us a slice of a head-padded [T, 64, D] buffer

        The zero-init is this test's way of seeing whether the staging copy strays
        outside the slice. gfx950 allocates that buffer with new_empty, so in
        production the padding holds garbage, not zeros -- what matters is only that
        nobody writes it.
        """
        padded = torch.zeros(self.T, 64, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
        q_out = padded[:, :NUM_HEADS, :]
        self.assertFalse(q_out.is_contiguous())
        packed = self._call()

        got = fused_qk_norm_rope_swa_store(
            q=self.q,
            kv=self.kv,
            q_norm_weight=None,
            kv_norm_weight=self.weight,
            q_rms_eps=EPS,
            kv_rms_eps=EPS,
            rope_head_dim=ROPE_DIM,
            cos_cache=self.cos,
            sin_cache=self.sin,
            positions=self.positions,
            q_out=q_out,
            dtype=torch.bfloat16,
            fp8_2buff=True,
        )
        self.assertIs(got, q_out)
        self.assertTrue(torch.all(padded[:, NUM_HEADS:, :] == 0))
        # staging must not reorder the heads, so the strided destination has to
        # hold exactly what the contiguous call produced
        self.assertTrue(torch.equal(q_out, packed))

    def test_negative_position_skips_both_pools(self):
        """a stale/pad token must leave both pools alone, not half-write a row"""
        nope_pool, rope_pool = _pools(2 * RING_STRIDE)
        self.positions[2] = -1
        self._call(nope_pool, rope_pool)

        row = self.swa_loc[2].item()
        self.assertEqual(nope_pool.view(torch.uint8)[row].max().item(), 0)
        self.assertEqual(rope_pool[row].abs().max().item(), 0)

    def test_rope_pool_is_required_with_the_nope_pool(self):
        nope_pool, _ = _pools(2 * RING_STRIDE)
        with self.assertRaises(AssertionError):
            self._call(nope_pool, None)

    def test_mismatched_pools_are_rejected_before_the_launch(self):
        """aiter aborts the process on a short pool, so these must fail in python"""
        nope_pool, rope_pool = _pools(2 * RING_STRIDE)
        short_rope = rope_pool[:RING_STRIDE].contiguous()
        cases = {
            "fewer rope rows": (nope_pool, short_rope),
            "rope dtype": (nope_pool, rope_pool.to(torch.float16)),
            "rope width": (nope_pool, rope_pool[:, : ROPE_DIM // 2].contiguous()),
            "nope row bytes": (nope_pool[:, :NOPE_DIM].contiguous(), rope_pool),
        }
        for name, (nope, rope) in cases.items():
            with self.subTest(name), self.assertRaises(AssertionError):
                self._call(nope, rope)

    def test_bf16_store_is_a_different_store(self):
        nope_pool, rope_pool = _pools(2 * RING_STRIDE)
        with self.assertRaises(AssertionError):
            fused_qk_norm_rope_swa_store(
                q=self.q,
                kv=self.kv,
                q_norm_weight=None,
                kv_norm_weight=self.weight,
                q_rms_eps=EPS,
                kv_rms_eps=EPS,
                rope_head_dim=ROPE_DIM,
                cos_cache=self.cos,
                sin_cache=self.sin,
                positions=self.positions,
                swa_cache=nope_pool,
                swa_loc=self.swa_loc,
                swa_page_size=1,
                dtype=torch.bfloat16,
                bf16_store=True,
                fp8_2buff=True,
                swa_rope_cache=rope_pool,
            )


class TestBf16StoreStillWorks(_StoreCase):
    """fp8_2buff returns before the Triton kernel, so pin the branch it skips"""

    def test_bf16_store_writes_the_whole_row(self):
        pool = torch.zeros(
            2 * RING_STRIDE, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
        )
        ref_nope, ref_pe = _ref_norm_rope(
            self.kv, self.weight, self.cos, self.sin, self.positions
        )
        q_out = fused_qk_norm_rope_swa_store(
            q=self.q,
            kv=self.kv,
            q_norm_weight=None,
            kv_norm_weight=self.weight,
            q_rms_eps=EPS,
            kv_rms_eps=EPS,
            rope_head_dim=ROPE_DIM,
            cos_cache=self.cos,
            sin_cache=self.sin,
            positions=self.positions,
            swa_cache=pool,
            swa_loc=self.swa_loc,
            swa_page_size=1,
            dtype=torch.bfloat16,
            bf16_store=True,
        )
        self.assertEqual(q_out.dtype, torch.bfloat16)
        self.assertEqual(tuple(q_out.shape), (self.T, NUM_HEADS, HEAD_DIM))

        rows = self.swa_loc.long()
        got = pool[rows].float()
        torch.testing.assert_close(got[:, :NOPE_DIM], ref_nope, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(got[:, NOPE_DIM:], ref_pe, rtol=2e-2, atol=2e-2)

        untouched = torch.ones(pool.shape[0], dtype=torch.bool, device=DEVICE)
        untouched[rows] = False
        self.assertEqual(pool[untouched].abs().max().item(), 0)


@unittest.skipUnless(_HAS_GROUP_QUANT, "needs aiter's group-quant kernel on gfx95x")
class TestUnifiedFp8SwaRingWrap(CustomTestCase):
    """What the ring holds once a slot gets written a second time.

    The two pools have to turn over together. A row whose nope came from the new
    token but whose rope is still the old one decodes against the wrong angle,
    and nothing downstream can notice -- both halves are individually
    well-formed.

    Wrap is driven across calls, not inside one. Within a launch the
    out-of-window tokens carry loc -1 and get skipped, so every live row has a
    single writer; two writers to one row in one launch would be a race with no
    defined winner to assert on.
    """

    def setUp(self):
        torch.manual_seed(11)
        self.T = 6
        self.cos, self.sin = _cos_sin()
        self.weight = torch.randn(HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
        self.q = torch.randn(
            self.T, NUM_HEADS * HEAD_DIM, device=DEVICE, dtype=torch.bfloat16
        )

    def _store(self, kv, positions, swa_loc, nope_pool, rope_pool):
        """one launch; hands back the dense K pair as the per-token truth"""
        k_nope = torch.empty(
            self.T, 1, DSV4_FP8_NOPE_ROW_BYTES, dtype=torch.float8_e4m3fn, device=DEVICE
        )
        k_rope = torch.empty(self.T, 1, ROPE_DIM, dtype=torch.bfloat16, device=DEVICE)
        fused_qk_norm_rope_swa_store(
            q=self.q,
            kv=kv,
            q_norm_weight=None,
            kv_norm_weight=self.weight,
            q_rms_eps=EPS,
            kv_rms_eps=EPS,
            rope_head_dim=ROPE_DIM,
            cos_cache=self.cos,
            sin_cache=self.sin,
            positions=positions,
            swa_cache=nope_pool,
            swa_loc=swa_loc,
            swa_page_size=1,
            dtype=torch.bfloat16,
            fp8_2buff=True,
            swa_rope_cache=rope_pool,
            k_nope_out=k_nope,
            k_rope_out=k_rope,
        )
        return k_nope.view(torch.uint8)[:, 0, : SCALE_OFF + SCALE_BYTES], k_rope[:, 0]

    def _pass(self, step, nope_pool, rope_pool, count=None):
        """step 0 fills the ring, step 1 comes back around onto the same slots"""
        count = self.T if count is None else count
        kv = torch.randn(self.T, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
        positions = (
            torch.arange(self.T, device=DEVICE, dtype=torch.int64) + step * RING_STRIDE
        )
        swa_loc = (positions.to(torch.int32) % RING_STRIDE + RING_STRIDE).contiguous()
        # tokens past `count` fall out of window on this pass, like a short step
        if count < self.T:
            positions = positions.clone()
            positions[count:] = -1
        nope, rope = self._store(kv, positions, swa_loc, nope_pool, rope_pool)
        return swa_loc.long(), nope.clone(), rope.clone()

    def test_wrap_turns_over_both_pools(self):
        nope_pool, rope_pool = _pools(2 * RING_STRIDE)
        rows, old_nope, _ = self._pass(0, nope_pool, rope_pool)
        rows2, new_nope, new_rope = self._pass(1, nope_pool, rope_pool)
        self.assertTrue(torch.equal(rows, rows2), "the wrap must reuse the same slots")
        # only meaningful if pass 1 actually changed the bytes
        self.assertFalse(torch.equal(old_nope, new_nope))

        pool_bytes = nope_pool.view(torch.uint8)[rows, : SCALE_OFF + SCALE_BYTES]
        self.assertTrue(torch.equal(pool_bytes, new_nope))
        self.assertTrue(torch.equal(rope_pool[rows], new_rope))

    def test_a_slot_the_wrap_skipped_keeps_its_old_pair(self):
        """a short second pass must leave the rows it didn't address alone"""
        keep = 2
        nope_pool, rope_pool = _pools(2 * RING_STRIDE)
        rows, old_nope, old_rope = self._pass(0, nope_pool, rope_pool)
        _, new_nope, new_rope = self._pass(1, nope_pool, rope_pool, count=keep)

        pool_bytes = nope_pool.view(torch.uint8)[rows, : SCALE_OFF + SCALE_BYTES]
        got_rope = rope_pool[rows]
        self.assertTrue(torch.equal(pool_bytes[:keep], new_nope[:keep]))
        self.assertTrue(torch.equal(got_rope[:keep], new_rope[:keep]))
        self.assertTrue(torch.equal(pool_bytes[keep:], old_nope[keep:]))
        self.assertTrue(torch.equal(got_rope[keep:], old_rope[keep:]))


if __name__ == "__main__":
    unittest.main()
