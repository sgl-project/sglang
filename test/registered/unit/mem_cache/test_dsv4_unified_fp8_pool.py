import contextlib
import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV4_FP8_NOPE_ROW_BYTES,
    DSV4_FP8_QUANT_TILE,
    DeepSeekV4UnifiedKVPool,
    dsv4_unified_row_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# DeepSeek-V4-Pro geometry.
NOPE_DIM = 448
ROPE_DIM = 64


class _StubMemorySaver:
    def region(self, _tag):
        return contextlib.nullcontext()


class TestDSV4UnifiedRowBytes(CustomTestCase):
    """Row width drives both `bytes_per_full_token` and `_fixed_swa_bytes`, so the
    capacity claim for the fp8 pool is only as good as this arithmetic."""

    def test_bf16_row_is_the_whole_latent(self):
        self.assertEqual(
            dsv4_unified_row_bytes(NOPE_DIM, ROPE_DIM, fp8=False),
            (NOPE_DIM + ROPE_DIM) * 2,
        )

    def test_fp8_row_is_padded_nope_plus_bf16_rope(self):
        self.assertEqual(
            dsv4_unified_row_bytes(NOPE_DIM, ROPE_DIM, fp8=True),
            DSV4_FP8_NOPE_ROW_BYTES + ROPE_DIM * 2,
        )

    def test_fp8_saves_exactly_three_eighths(self):
        """0.625x is where the >=1.40x capacity target comes from; the remaining
        dilution is the fixed SWA/c4-state bias, not the row."""
        bf16 = dsv4_unified_row_bytes(NOPE_DIM, ROPE_DIM, fp8=False)
        fp8 = dsv4_unified_row_bytes(NOPE_DIM, ROPE_DIM, fp8=True)
        self.assertEqual((bf16, fp8), (1024, 640))
        self.assertAlmostEqual(fp8 / bf16, 0.625)

    def test_scales_and_latent_fit_the_asm_stride(self):
        """7 tiles written twice = 14 B; 448 + 14 leaves 50 B the reader never
        touches. If a future head_dim broke this the pack would silently overlap."""
        num_tiles = NOPE_DIM // DSV4_FP8_QUANT_TILE
        self.assertEqual(num_tiles, 7)
        self.assertLessEqual(NOPE_DIM + 2 * num_tiles, DSV4_FP8_NOPE_ROW_BYTES)

    def test_oversized_latent_is_rejected(self):
        # ValueError, not assert: sizing has to keep checking under python -O
        with self.assertRaises(ValueError):
            dsv4_unified_row_bytes(DSV4_FP8_NOPE_ROW_BYTES, ROPE_DIM, fp8=True)


class TestDSV4UnifiedFp8PoolAllocation(CustomTestCase):
    """The sizing formula and the allocation are two separate code paths; this pins
    them to the same row width so a change to one cannot silently outrun the other."""

    STAGE_RATIOS = [4, 128]
    NUM_SLOTS = 3
    NUM_BLOCKS = 5
    PAGE_SIZE = 256
    SWA_RING = 8

    def _pool(self, fp8):
        return DeepSeekV4UnifiedKVPool(
            stage_ratios=self.STAGE_RATIOS,
            num_slots=self.NUM_SLOTS,
            num_blocks=self.NUM_BLOCKS,
            page_size=self.PAGE_SIZE,
            qk_nope_head_dim=NOPE_DIM,
            qk_rope_head_dim=ROPE_DIM,
            device="cpu",
            memory_saver_adapter=_StubMemorySaver(),
            custom_mem_pool=None,
            swa_ring_size=self.SWA_RING,
            fp8=fp8,
        )

    def test_bf16_pool_is_unchanged(self):
        """fp8 defaults off, so the bf16 arm must keep one pool of bf16 latents."""
        pool = self._pool(fp8=False)
        for buf, rope in zip(pool.kv_buffer, pool.kv_buffer_rope):
            self.assertEqual(buf.dtype, torch.bfloat16)
            self.assertEqual(buf.shape[1], NOPE_DIM + ROPE_DIM)
            self.assertIsNone(rope)

    def test_fp8_pool_row_counts_match_across_both_pools(self):
        """A row index addresses the SWA ring and the compressed region in both
        pools, so the two must have identical row counts."""
        pool = self._pool(fp8=True)
        for buf, rope in zip(pool.kv_buffer, pool.kv_buffer_rope):
            self.assertEqual(buf.dtype, torch.float8_e4m3fn)
            self.assertEqual(rope.dtype, torch.bfloat16)
            self.assertEqual(buf.shape[0], rope.shape[0])
            self.assertEqual(buf.shape[1], DSV4_FP8_NOPE_ROW_BYTES)
            self.assertEqual(rope.shape[1], ROPE_DIM)

    def test_fp8_pool_bytes_match_the_sizing_row_width(self):
        bf16, fp8 = self._pool(fp8=False), self._pool(fp8=True)
        for layer, buf in enumerate(bf16.kv_buffer):
            rows = buf.shape[0]
            self.assertEqual(fp8.kv_buffer[layer].shape[0], rows)
            self.assertEqual(
                buf.nbytes,
                rows * dsv4_unified_row_bytes(NOPE_DIM, ROPE_DIM, fp8=False),
            )
            self.assertEqual(
                fp8.kv_buffer[layer].nbytes + fp8.kv_buffer_rope[layer].nbytes,
                rows * dsv4_unified_row_bytes(NOPE_DIM, ROPE_DIM, fp8=True),
            )

    def test_rope_accessor_rejects_the_bf16_pool(self):
        with self.assertRaises(AssertionError):
            self._pool(fp8=False).get_unified_kv_rope(0)


if __name__ == "__main__":
    unittest.main()
