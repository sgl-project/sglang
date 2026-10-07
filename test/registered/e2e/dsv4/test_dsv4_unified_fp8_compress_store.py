"""Two-pool fp8 store tests for the compressor's norm+rope kernel.

Under SGLANG_DSV4_UNIFIED_KV_FP8 the c4/c128 compressor writes its compressed
latent through ``forward_fp8_2buff``: a 512 B fp8 nope row (448 B payload + 7
UE8M0 tile scales stored twice) in the unified_kv pool, plus a bf16 rope row in
the second pool, both at ``out_loc``. These tests pin that layout for both
compress ratios, against the bf16 store of the same kernel (which shares the
norm+rope math, so the comparison is byte-exact) and against a torch reference.

Both plans are covered. Most cases run the decode plan; the extend arm (what
prefill takes) gets the bf16 comparison only, since its plan check and its
out_loc bound are hand-copied from decode's and nothing else exercises them.
"""

import unittest

import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis
from sglang.kernels.ops.attention.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_norm_rope_store,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV4_FP8_NOPE_ROW_BYTES,
    DSV4_FP8_QUANT_TILE,
)
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# the kernel takes E4M3FN vs E4M3FNUZ from the arch and the two-pool layout is only
# ever allocated on gfx95, so on the default mi300 runner every case here would skip
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEVICE = torch.device("cuda")

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
NUM_TILES = NOPE_DIM // DSV4_FP8_QUANT_TILE
SCALE_OFF = NOPE_DIM
SCALE_BYTES = 2 * NUM_TILES

NUM_TOKENS = 6
POOL_ROWS = 32
EPS = 1e-6
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
RATIOS = (4, 128)


def _inputs(compress_ratio, seq_lens=None):
    torch.manual_seed(compress_ratio)
    kv = torch.randn(NUM_TOKENS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    if seq_lens is None:
        seq_lens = (
            torch.arange(1, NUM_TOKENS + 1, device=DEVICE, dtype=torch.int64)
            * compress_ratio
        )
    plan = CompressorDecodePlan.generate_legacy(
        compress_ratio,
        torch.arange(NUM_TOKENS, device=DEVICE, dtype=torch.int64),
        seq_lens,
    )
    # every other row, so a row that gets written always has an untouched neighbour
    out_loc = torch.arange(1, 2 * NUM_TOKENS + 1, 2, device=DEVICE, dtype=torch.int64)
    freqs_cis = precompute_freqs_cis(
        ROPE_DIM, int(seq_lens.max().item()) + 1, 0, 10000, 1, 32, 1
    ).to(DEVICE)
    return kv, weight, seq_lens, plan, out_loc, freqs_cis


def _extend_inputs(compress_ratio):
    """one request whose extend spans several compress boundaries"""
    torch.manual_seed(compress_ratio + 1)
    total = compress_ratio * NUM_TOKENS
    seq_lens = torch.tensor([total], dtype=torch.int64)
    plan = CompressorPrefillPlan.generate_legacy(
        compress_ratio,
        torch.zeros(1, dtype=torch.int64, device=DEVICE),
        seq_lens,
        seq_lens.clone(),  # the whole sequence is the extend
        total,
        DEVICE,
    )
    # the kernel binds its token count off the input and then requires the plan to
    # have that many rows, so the fixture has to follow whatever the planner emitted
    num_c = plan.plan_c.shape[0]
    kv = torch.randn(num_c, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    # unlike decode, extend indexes out_loc by ragged_id -- one entry per q token, of
    # which only the compress boundaries are ever read. Sizing this num_c long instead
    # reads off the end and stores to whatever row index it finds there.
    written = torch.arange(1, 2 * num_c + 1, 2, device=DEVICE, dtype=torch.int64)
    out_loc = torch.zeros(total, dtype=torch.int64, device=DEVICE)
    out_loc[compress_ratio - 1 :: compress_ratio] = written
    freqs_cis = precompute_freqs_cis(ROPE_DIM, total + 1, 0, 10000, 1, 32, 1).to(DEVICE)
    return kv, weight, plan, out_loc, written, freqs_cis


def _ref_norm_rope(kv, weight, freqs_cis, positions):
    """rmsnorm over the latent, then rope on the trailing 64, as the kernel does."""
    x = kv.float()
    x = x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / HEAD_DIM + EPS)
    x = x * weight.float()
    nope, pe = x[:, :NOPE_DIM], x[:, NOPE_DIM:]

    freqs = torch.view_as_real(freqs_cis).flatten(-2)[positions]
    freqs = freqs.reshape(-1, ROPE_DIM // 2, 2).float()
    pairs = pe.reshape(-1, ROPE_DIM // 2, 2)
    out = torch.empty_like(pairs)
    out[..., 0] = pairs[..., 0] * freqs[..., 0] - pairs[..., 1] * freqs[..., 1]
    out[..., 1] = pairs[..., 0] * freqs[..., 1] + pairs[..., 1] * freqs[..., 0]
    # the quant warps round through bf16 first, so the scales come off bf16 values
    return nope.to(torch.bfloat16).float(), out.reshape(-1, ROPE_DIM)


def _tile_scale_bytes(nope):
    """cast_to_ue8m0(max(absmax, 1e-4) / fp8_max) per 1x64 tile."""
    tiles = nope.reshape(nope.shape[0], NUM_TILES, DSV4_FP8_QUANT_TILE)
    scale_raw = tiles.abs().amax(-1).clamp_min(1e-4) / FP8_MAX
    bits = scale_raw.contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return exp.to(torch.uint8)


@unittest.skipUnless(is_gfx95_supported(), "needs an AMD gfx95 GPU for e4m3fn")
class TestUnifiedFp8CompressStore(CustomTestCase):
    def _store_fp8(self, compress_ratio, *, seq_lens=None, rope_rows=POOL_ROWS):
        kv, weight, seq_lens, plan, out_loc, freqs_cis = _inputs(
            compress_ratio, seq_lens
        )
        nope_pool = torch.zeros(
            POOL_ROWS, DSV4_FP8_NOPE_ROW_BYTES, dtype=torch.float8_e4m3fn, device=DEVICE
        )
        rope_pool = torch.zeros(
            rope_rows, ROPE_DIM, dtype=torch.bfloat16, device=DEVICE
        )
        compress_norm_rope_store(
            kv.clone(),
            plan,
            norm_weight=weight,
            norm_eps=EPS,
            freq_cis=freqs_cis,
            out_loc=out_loc,
            kvcache=nope_pool.view(torch.uint8),
            page_size=1,
            fp8_2buff=True,
            kvcache_rope=rope_pool.view(torch.uint8),
        )
        ref = _ref_norm_rope(kv, weight, freqs_cis, (seq_lens - compress_ratio).long())
        return nope_pool, rope_pool, out_loc, ref

    def _store_bf16(self, compress_ratio):
        """same inputs through the bf16 store, i.e. the values before quantization"""
        kv, weight, _, plan, out_loc, freqs_cis = _inputs(compress_ratio)
        cache = torch.zeros(POOL_ROWS, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
        compress_norm_rope_store(
            kv.clone(),
            plan,
            norm_weight=weight,
            norm_eps=EPS,
            freq_cis=freqs_cis,
            out_loc=out_loc,
            kvcache=cache.view(torch.uint8),
            page_size=1,
            bf16_store=True,
        )
        return cache[out_loc]

    def _store_extend(self, compress_ratio, *, fp8):
        kv, weight, plan, out_loc, written, freqs_cis = _extend_inputs(compress_ratio)
        rows = int(written.max().item()) + 2
        common = dict(
            norm_weight=weight,
            norm_eps=EPS,
            freq_cis=freqs_cis,
            out_loc=out_loc,
            page_size=1,
        )
        if not fp8:
            cache = torch.zeros(rows, HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
            compress_norm_rope_store(
                kv.clone(),
                plan,
                kvcache=cache.view(torch.uint8),
                bf16_store=True,
                **common,
            )
            return cache[written]

        nope_pool = torch.zeros(
            rows, DSV4_FP8_NOPE_ROW_BYTES, dtype=torch.float8_e4m3fn, device=DEVICE
        )
        rope_pool = torch.zeros(rows, ROPE_DIM, dtype=torch.bfloat16, device=DEVICE)
        compress_norm_rope_store(
            kv.clone(),
            plan,
            kvcache=nope_pool.view(torch.uint8),
            fp8_2buff=True,
            kvcache_rope=rope_pool.view(torch.uint8),
            **common,
        )
        return nope_pool, rope_pool, written

    def test_extend_plan_stores_the_same_rows(self):
        for ratio in RATIOS:
            with self.subTest(compress_ratio=ratio):
                nope_pool, rope_pool, written = self._store_extend(ratio, fp8=True)
                pre_quant = self._store_extend(ratio, fp8=False)

                nope = pre_quant[:, :NOPE_DIM].float()
                num_c = nope.shape[0]
                scale_bytes = _tile_scale_bytes(nope)
                scale = torch.exp2((scale_bytes.to(torch.int32) - 127).float())
                want = (
                    nope.reshape(num_c, NUM_TILES, DSV4_FP8_QUANT_TILE)
                    / scale[..., None]
                ).to(torch.float8_e4m3fn)

                self.assertTrue(
                    torch.equal(
                        nope_pool[written][:, :NOPE_DIM].view(torch.uint8),
                        want.view(torch.uint8).reshape(num_c, NOPE_DIM),
                    )
                )
                self.assertTrue(
                    torch.equal(rope_pool[written], pre_quant[:, NOPE_DIM:])
                )

    def test_row_matches_the_bf16_store_byte_for_byte(self):
        for ratio in RATIOS:
            with self.subTest(compress_ratio=ratio):
                nope_pool, rope_pool, out_loc, _ = self._store_fp8(ratio)
                pre_quant = self._store_bf16(ratio)

                nope = pre_quant[:, :NOPE_DIM].float()
                scale_bytes = _tile_scale_bytes(nope)
                scale = torch.exp2((scale_bytes.to(torch.int32) - 127).float())
                want = (
                    nope.reshape(NUM_TOKENS, NUM_TILES, DSV4_FP8_QUANT_TILE)
                    / scale[..., None]
                ).to(torch.float8_e4m3fn)

                # the fixture has to reach the top e4m3 exponent, otherwise it would
                # not notice a cast that saturates everything above 256
                self.assertTrue(bool((want.float().abs() >= 256).any()))

                row = nope_pool[out_loc]
                self.assertTrue(
                    torch.equal(
                        row[:, :NOPE_DIM].view(torch.uint8),
                        want.view(torch.uint8).reshape(NUM_TOKENS, NOPE_DIM),
                    )
                )
                got_scales = row.view(torch.uint8)[
                    :, SCALE_OFF : SCALE_OFF + SCALE_BYTES
                ].reshape(NUM_TOKENS, NUM_TILES, 2)
                self.assertTrue(torch.equal(got_scales[..., 0], scale_bytes))
                self.assertTrue(torch.equal(got_scales[..., 1], scale_bytes))
                self.assertTrue(
                    torch.equal(rope_pool[out_loc], pre_quant[:, NOPE_DIM:])
                )

    def test_scale_bytes_track_the_torch_reference(self):
        for ratio in RATIOS:
            with self.subTest(compress_ratio=ratio):
                nope_pool, _, out_loc, (ref_nope, _) = self._store_fp8(ratio)
                got = nope_pool.view(torch.uint8)[
                    out_loc, SCALE_OFF : SCALE_OFF + SCALE_BYTES
                ].reshape(NUM_TOKENS, NUM_TILES, 2)
                self.assertTrue(torch.equal(got[..., 0], _tile_scale_bytes(ref_nope)))

    def test_dequantized_nope_tracks_the_reference(self):
        for ratio in RATIOS:
            with self.subTest(compress_ratio=ratio):
                nope_pool, _, out_loc, (ref_nope, _) = self._store_fp8(ratio)
                exps = _tile_scale_bytes(ref_nope).to(torch.int32) - 127
                payload = nope_pool[out_loc, :NOPE_DIM].float()
                deq = (
                    payload.reshape(NUM_TOKENS, NUM_TILES, DSV4_FP8_QUANT_TILE)
                    * torch.exp2(exps.float())[..., None]
                ).reshape(NUM_TOKENS, NOPE_DIM)

                # e4m3 carries 3 mantissa bits, so half a step is at most ~2^-4 of
                # the tile's own absmax; beyond that the scale or the payload is off
                tile_absmax = (
                    ref_nope.reshape(NUM_TOKENS, NUM_TILES, DSV4_FP8_QUANT_TILE)
                    .abs()
                    .amax(-1)
                    .repeat_interleave(DSV4_FP8_QUANT_TILE, dim=1)
                )
                self.assertTrue(torch.all((deq - ref_nope).abs() <= 0.07 * tile_absmax))

    def test_rope_pool_matches_the_bf16_reference(self):
        for ratio in RATIOS:
            with self.subTest(compress_ratio=ratio):
                _, rope_pool, out_loc, (_, ref_pe) = self._store_fp8(ratio)
                torch.testing.assert_close(
                    rope_pool[out_loc].float(), ref_pe, rtol=2e-2, atol=2e-2
                )

    def test_pad_and_neighbour_rows_untouched(self):
        nope_pool, rope_pool, out_loc, _ = self._store_fp8(4)
        nope_bytes = nope_pool.view(torch.uint8)
        self.assertTrue(torch.all(nope_bytes[out_loc, SCALE_OFF + SCALE_BYTES :] == 0))

        untouched = torch.ones(POOL_ROWS, dtype=torch.bool, device=DEVICE)
        untouched[out_loc] = False
        self.assertTrue(torch.all(nope_bytes[untouched] == 0))
        self.assertTrue(torch.all(rope_pool[untouched] == 0))

    def test_non_boundary_decode_is_skipped(self):
        # only sequences whose length is a multiple of the ratio produce a token
        seq_lens = torch.full(
            (NUM_TOKENS,), 4 * 128 + 1, device=DEVICE, dtype=torch.int64
        )
        nope_pool, rope_pool, _, _ = self._store_fp8(128, seq_lens=seq_lens)
        self.assertTrue(torch.all(nope_pool.view(torch.uint8) == 0))
        self.assertTrue(torch.all(rope_pool == 0))

    def test_short_rope_pool_rejected(self):
        # one row index addresses both pools, so a short rope pool has to be caught
        # before either pool is written
        with self.assertRaises(RuntimeError):
            self._store_fp8(4, rope_rows=POOL_ROWS // 2)


if __name__ == "__main__":
    unittest.main()
