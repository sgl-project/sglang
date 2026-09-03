import contextlib
import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV4_FP8_NOPE_ROW_BYTES,
    DSV4_FP8_QUANT_TILE,
    DeepSeekV4TokenToKVPool,
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


class _StubTokenToKVPool:
    """
    Only the attributes the region math reads, so it can be exercised without standing
    up a whole DeepSeekV4TokenToKVPool.
    """

    _unified_page_views = DeepSeekV4TokenToKVPool._unified_page_views
    unified_region_buffers = DeepSeekV4TokenToKVPool.unified_region_buffers
    unified_rope_region_buffers = DeepSeekV4TokenToKVPool.unified_rope_region_buffers

    def __init__(self, unified_kv_pool, page_size, stage_ratios, fp8):
        self.unified_kv_pool = unified_kv_pool
        self.page_size = page_size
        self.compression_ratios = list(stage_ratios)
        self._stage_start = 0
        self._stage_end = len(stage_ratios)
        self._unified_kv = True
        self._unified_kv_fp8 = fp8


class TestDSV4UnifiedRegionBuffers(CustomTestCase):
    """
    HiCache mirrors the compressed region of every unified_kv layer. Under fp8
    that region lives in two pools, and offloading only the nope half refills a
    fetched page with stale rope (wrong output, no crash), so the pairing is what
    these tests pin.
    """

    # Two c4 layers to cover the per-ratio layer filter, one c128 layer. Blocks
    # must be even: a c4 layer holds `num_blocks * 32` compressed rows plus one
    # padding page of 64, which only divides into whole pages when it is.
    STAGE_RATIOS = [4, 4, 128]
    NUM_SLOTS = 3
    NUM_BLOCKS = 6
    PAGE_SIZE = 256
    SWA_RING = 8
    NUM_PAGES = 4

    def _stub(self, fp8):
        pool = DeepSeekV4UnifiedKVPool(
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
        return pool, _StubTokenToKVPool(pool, self.PAGE_SIZE, self.STAGE_RATIOS, fp8)

    def test_bf16_keeps_the_whole_row_in_one_region(self):
        """
        The bf16 arm must not grow a second host pool: its row is undivided,
        and a stray rope region would mirror a buffer that does not exist.
        """
        _, stub = self._stub(fp8=False)
        for ratio, num_layers in ((4, 2), (128, 1)):
            views, item_bytes = stub.unified_region_buffers(ratio)
            rows_per_page = self.PAGE_SIZE // ratio
            self.assertEqual(len(views), num_layers)
            self.assertEqual(item_bytes, rows_per_page * (NOPE_DIM + ROPE_DIM) * 2)
            self.assertIsNone(stub.unified_rope_region_buffers(ratio))

    def test_fp8_pairs_a_rope_region_with_the_nope_one(self):
        _, stub = self._stub(fp8=True)
        for ratio, num_layers in ((4, 2), (128, 1)):
            rows_per_page = self.PAGE_SIZE // ratio
            for (views, item_bytes), row_bytes in (
                (stub.unified_region_buffers(ratio), DSV4_FP8_NOPE_ROW_BYTES),
                (stub.unified_rope_region_buffers(ratio), ROPE_DIM * 2),
            ):
                self.assertEqual(len(views), num_layers)
                self.assertEqual(item_bytes, rows_per_page * row_bytes)
                for view in views:
                    self.assertEqual(view.dtype, torch.uint8)
                    self.assertEqual(list(view.shape), [self.NUM_PAGES, item_bytes])

    def test_regions_start_past_the_swa_ring(self):
        """
        The host pool takes each view's data_ptr as the device base, so a view
        that still covered the ring would offload ring rows as page 0.
        """
        pool, stub = self._stub(fp8=True)
        swa_pages = pool.swa_pages
        for (views, _), device_buffers in (
            (stub.unified_region_buffers(4), pool.kv_buffer),
            (stub.unified_rope_region_buffers(4), pool.kv_buffer_rope),
        ):
            for view, buf in zip(views, device_buffers):
                row_bytes = buf.shape[1] * buf.element_size()
                self.assertEqual(
                    view.data_ptr(), buf.data_ptr() + swa_pages * row_bytes
                )

    def test_fp8_regions_cover_the_same_bytes_as_bf16(self):
        """
        Both halves together must mirror the whole row; the 0.625x is the same
        saving the row-width test pins.
        """
        _, bf16 = self._stub(fp8=False)
        _, fp8 = self._stub(fp8=True)
        for ratio in (4, 128):
            _, whole = bf16.unified_region_buffers(ratio)
            _, nope = fp8.unified_region_buffers(ratio)
            _, rope = fp8.unified_rope_region_buffers(ratio)
            self.assertAlmostEqual((nope + rope) / whole, 0.625)


if __name__ == "__main__":
    unittest.main()
