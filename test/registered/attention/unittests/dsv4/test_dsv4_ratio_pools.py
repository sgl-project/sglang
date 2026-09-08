"""Compressed pools are built and addressed by compress ratio.

DeepSeek-V4 (ratios 4/128, every compressed layer owns its storage) and
DeepSeek-V4.1 (ratios 1/2, only the kv_source layers own storage, later layers
read the nearest one) share one construction and addressing path, so a ratio
that is mis-sized, built for an absent ratio, or dropped from the PD export on
one layout has no second code path to hide behind.
"""

import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV41_INDEX_PAGE_SIZE,
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

PAGE_SIZE = 256
FULL_SIZE = 2 * PAGE_SIZE

V4_RATIOS = [0, 4, 128, 4, 128, 0]
V41_RATIOS = [0] * 2 + [2] * 18 + [1] * 20
V41_SOURCES = [2, 8, 14, 20]


def _make_pool(*, compression_ratios, kv_source_layers=(), c4_size=0, c128_size=0):
    return DeepSeekV4TokenToKVPool(
        max_num_reqs=1,
        swa_size=FULL_SIZE,
        c4_size=c4_size,
        c128_size=c128_size,
        c4_state_pool_size=8 if c4_size else 0,
        c128_state_pool_size=8 if c128_size else 0,
        page_size=PAGE_SIZE,
        swa_page_size=128,
        dtype=torch.float8_e4m3fn,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=len(compression_ratios),
        device="cuda",
        enable_memory_saver=False,
        compression_ratios=compression_ratios,
        kv_source_layers=kv_source_layers,
        full_size=FULL_SIZE,
    )


@unittest.skipUnless(torch.cuda.is_available(), "allocates device KV buffers")
class TestDeepSeekV4RatioPools(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=PAGE_SIZE)
        )
        cls.v4 = _make_pool(
            compression_ratios=V4_RATIOS,
            c4_size=FULL_SIZE // 4,
            c128_size=FULL_SIZE // 128,
        )
        cls.v41 = _make_pool(
            compression_ratios=V41_RATIOS, kv_source_layers=V41_SOURCES
        )

    def test_only_present_ratios_get_pools(self):
        self.assertEqual(sorted(self.v4.kv_pools), [4, 128])
        self.assertEqual(sorted(self.v4.index_pools), [4])
        self.assertEqual(sorted(self.v41.kv_pools), [1, 2])
        self.assertEqual(sorted(self.v41.index_pools), [1, 2])

        # The names HiSparse / HiCache / NPU read must track the registries.
        self.assertIs(self.v4.c4_kv_pool, self.v4.kv_pools[4])
        self.assertIs(self.v4.c128_kv_pool, self.v4.kv_pools[128])
        self.assertIs(self.v4.c4_indexer_kv_pool, self.v4.index_pools[4])
        self.assertIsNone(self.v41.c4_kv_pool)
        self.assertIsNone(self.v41.c128_kv_pool)
        self.assertIsNone(self.v41.c4_indexer_kv_pool)

        self.assertEqual(self.v4.sources_by_ratio, {4: [1, 3], 128: [2, 4]})
        self.assertEqual(self.v41.sources_by_ratio, {1: [20], 2: [2, 8, 14]})

    def test_layer_mapping_resolves_to_the_owning_layer(self):
        expected_v4 = [(0, 0), (4, 0), (128, 0), (4, 1), (128, 1), (0, 1)]
        for layer_id, (ratio, slot) in enumerate(expected_v4):
            item = self.v4.layer_mapping[layer_id]
            self.assertEqual(
                (item.compress_ratio, item.compress_layer_id), (ratio, slot)
            )
            self.assertIs(item.compress_kv_pool, self.v4.kv_pools.get(ratio))

        # Ratio-2 sources 2/8/14 own slots 0/1/2; the ratio-1 source 20 owns slot 0.
        expected_v41 = (
            [(0, 0), (0, 1)]
            + [(2, 0)] * 6
            + [(2, 1)] * 6
            + [(2, 2)] * 6
            + [(1, 0)] * 20
        )
        for layer_id, (ratio, slot) in enumerate(expected_v41):
            item = self.v41.layer_mapping[layer_id]
            self.assertEqual(
                (item.compress_ratio, item.compress_layer_id), (ratio, slot)
            )
            self.assertIs(item.compress_kv_pool, self.v41.kv_pools.get(ratio))

    def test_consumer_layers_share_their_source_storage(self):
        for source, consumer in ((2, 7), (8, 13), (14, 19), (20, 39)):
            with self.subTest(source=source):
                self.assertEqual(self.v41.source_layer_of(consumer), source)
                self.assertEqual(
                    self.v41.get_extra_key_buffer(consumer).data_ptr(),
                    self.v41.get_extra_key_buffer(source).data_ptr(),
                )
                self.assertIs(
                    self.v41.get_index_k_with_scale_buffer(consumer),
                    self.v41.get_index_k_with_scale_buffer(source),
                )

    def test_pool_extents_follow_the_ratio_formulas(self):
        for ratio, size in ((4, FULL_SIZE // 4), (128, FULL_SIZE // 128)):
            with self.subTest(ratio=ratio):
                kv_pool = self.v4.kv_pools[ratio]
                self.assertEqual(
                    (kv_pool.size, kv_pool.page_size), (size, PAGE_SIZE // ratio)
                )
        c4_index = self.v4.index_pools[4]
        self.assertEqual(c4_index.size, self.v4.c128_size * 32)
        self.assertEqual(c4_index.page_size, PAGE_SIZE // 4)
        self.assertFalse(c4_index.index_k_rne)

        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                kv_pool = self.v41.kv_pools[ratio]
                self.assertEqual(
                    (kv_pool.size, kv_pool.page_size),
                    (FULL_SIZE // ratio, PAGE_SIZE // ratio),
                )
                index_pool = self.v41.index_pools[ratio]
                # FULL page 0 is reserved, so slots run past FULL_SIZE // ratio.
                self.assertEqual(index_pool.size, (FULL_SIZE + PAGE_SIZE) // ratio)
                self.assertEqual(index_pool.page_size, DSV41_INDEX_PAGE_SIZE)
                self.assertTrue(index_pool.use_fp4_indexer)
                self.assertTrue(index_pool.index_k_rne)

    def test_pd_export_lists_every_pool_once_in_ratio_order(self):
        for pool, groups in (
            (self.v4, ((4, True), (128, False))),
            (self.v41, ((1, True), (2, True))),
        ):
            with self.subTest(pool=sorted(pool.kv_pools)):
                expected_ptrs = []
                expected_item_lens = []
                for ratio, has_index in groups:
                    kv_pool = pool.kv_pools[ratio]
                    expected_ptrs += [b.data_ptr() for b in kv_pool.kv_buffer]
                    expected_item_lens += [kv_pool.bytes_per_page_padded] * len(
                        kv_pool.kv_buffer
                    )
                    if not has_index:
                        continue
                    index_pool = pool.index_pools[ratio]
                    index_bufs = index_pool.contiguous_page_row_buffers()
                    expected_ptrs += [b.data_ptr() for b in index_bufs]
                    expected_item_lens += [
                        (PAGE_SIZE // ratio) * index_pool.get_bytes_per_token()
                    ] * len(index_bufs)

                data_ptrs, data_lens, item_lens = pool.get_contiguous_buf_infos()
                self.assertEqual(data_ptrs, expected_ptrs)
                self.assertEqual(item_lens, expected_item_lens)
                self.assertEqual(len(set(data_ptrs)), len(data_ptrs))
                self.assertTrue(all(n > 0 for n in data_lens))

    def test_index_k_export_item_is_one_full_page_of_index_pages(self):
        # The transfer addresses the index pools by FULL page id, so the bytes
        # at [page * item_len, (page + 1) * item_len) must be exactly the index
        # pages holding that FULL page's slots (slot s lives in index page
        # s // index_pool.page_size).
        for pool in (self.v4, self.v41):
            data_ptrs, _, item_lens = pool.get_contiguous_buf_infos()
            item_len_of = dict(zip(data_ptrs, item_lens))
            for ratio, index_pool in pool.index_pools.items():
                slots_per_full_page = PAGE_SIZE // ratio
                for buf in index_pool.contiguous_page_row_buffers():
                    item_len = item_len_of[buf.data_ptr()]
                    row_bytes = buf[0].nbytes
                    for full_page in range(2):
                        slots = range(
                            full_page * slots_per_full_page,
                            (full_page + 1) * slots_per_full_page,
                        )
                        index_pages = {s // index_pool.page_size for s in slots}
                        lo = min(index_pages) * row_bytes
                        hi = (max(index_pages) + 1) * row_bytes
                        with self.subTest(ratio=ratio, full_page=full_page):
                            self.assertEqual(
                                (lo, hi),
                                (full_page * item_len, (full_page + 1) * item_len),
                            )

    def test_pair_state_is_owned_by_the_ratio_2_sources_alone(self):
        pools = self.v41.compress_state_pools
        owned = [i for i, pool in enumerate(pools) if pool is not None]
        # Ratio-1 layers pair nothing, and a ratio-2 layer that is not a kv_source
        # reads its source's ring instead of keeping one.
        self.assertEqual(owned, self.v41.sources_by_ratio[2])
        # V4 keeps its c4 / c128 rings; none of them is a pair ring.
        self.assertEqual(
            [p.ratio for p in self.v4.compress_state_pools if p], [4, 128, 4, 128]
        )

        ring_size = self.v41.get_ring_size(2)
        self.assertEqual(ring_size, 2)
        for layer_id in owned:
            state = pools[layer_id]
            self.assertEqual((state.ratio, state.ring_size), (2, ring_size))
            self.assertGreaterEqual(
                state.kv_score_buffer.kv_score.shape[0],
                self.v41.num_req_slots * ring_size,
            )

    def test_pair_state_ships_as_request_scoped_state(self):
        pools = self.v41.compress_state_pools
        owned = self.v41.sources_by_ratio[2]
        ring_size = self.v41.get_ring_size(2)

        data_ptrs, data_lens, item_lens = self.v41.get_c128_state_buf_infos()
        buffers = [pools[layer_id].kv_score_buffer.kv_score for layer_id in owned]
        self.assertEqual(data_ptrs, [b.data_ptr() for b in buffers])
        self.assertEqual(data_lens, [b.nbytes for b in buffers])
        # One item is one request's whole ring: the payload is keyed by req slot.
        self.assertEqual(item_lens, [b[0].nbytes * ring_size for b in buffers])

        # The SWA-addressed export must not carry request-scoped state as well.
        swa_ptrs, _, _ = self.v41.get_state_buf_infos()
        self.assertFalse(set(swa_ptrs) & set(data_ptrs))

    def test_clear_req_state_touches_only_that_request_ring(self):
        ring_size = self.v41.get_ring_size(2)
        source = self.v41.sources_by_ratio[2][0]
        buf = self.v41.compress_state_pools[source].kv_score_buffer.kv_score
        saved = buf.clone()
        self.addCleanup(buf.copy_, saved)

        buf.fill_(1.0)
        self.v41.clear_c128_req_state(1)

        half = buf.shape[-1] // 2
        cleared = buf[ring_size : 2 * ring_size]
        self.assertTrue(bool((cleared[:, :half] == 0).all()))
        self.assertTrue(bool((cleared[:, half:] == -torch.inf).all()))
        self.assertTrue(bool((buf[:ring_size] == 1.0).all()))
        self.assertTrue(bool((buf[2 * ring_size :] == 1.0).all()))


if __name__ == "__main__":
    unittest.main()
