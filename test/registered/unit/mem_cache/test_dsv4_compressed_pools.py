import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import (
    KVLayout,
    is_valid_kv_layout_pair,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
    _CompressedPoolConfig,
    _num_dsv4_physical_kv_pages,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV4CompressedPools(CustomTestCase):
    def test_physical_kv_pages_cover_reserved_logical_page(self):
        size = 8192
        self.assertEqual(_num_dsv4_physical_kv_pages(size, 256, 256), 33)
        self.assertEqual(_num_dsv4_physical_kv_pages(size, 64, 256), 132)
        self.assertGreaterEqual(
            _num_dsv4_physical_kv_pages(size, 64, 256) * 64,
            size + 256,
        )

    def test_state_buf_item_covers_one_logical_swa_page(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.swa_page_size = 256
        pool.compress_state_pools = []
        pool.indexer_compress_state_pools = []
        for physical_page_size in (256, 64):
            with self.subTest(physical_page_size=physical_page_size):
                row_bytes = physical_page_size * 4
                buf = torch.empty((8, row_bytes), dtype=torch.uint8)
                pool.swa_kv_pool = SimpleNamespace(
                    page_size=physical_page_size, kv_buffer=[buf]
                )
                data_ptrs, data_lens, item_lens = pool.get_state_buf_infos()
                self.assertEqual(data_ptrs, [buf.data_ptr()])
                self.assertEqual(data_lens, [buf.nbytes])
                self.assertEqual(item_lens, [256 * 4])

    def test_state_buf_infos_without_paged_swa(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.swa_kv_pool = None
        pool.compress_state_pools = []
        pool.indexer_compress_state_pools = []
        self.assertEqual(pool.get_state_buf_infos(), ([], [], []))

    def test_pp_mapping_and_pd_buffer_order(self):
        for unified, stage_ratios in product(
            (False, True), ([4, 0, 128, 4], [128], [0])
        ):
            with self.subTest(unified=unified, stage_ratios=stage_ratios):
                pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
                pool._unified_kv = unified
                pool.uniform_fp8 = False
                pool.kv_layout = KVLayout.V4
                pool.compressed_kv_layout_option = None
                pool.compressed_pool_configs = {
                    4: _CompressedPoolConfig(
                        256, 64, torch.bfloat16, indexer_size=1024
                    ),
                    128: _CompressedPoolConfig(512, 8, torch.float32),
                }
                pool.indexer_head_dim = 128
                pool.page_size = 256
                pool.compression_ratios = [128] + stage_ratios + [4]
                pool._stage_start = 1
                pool._stage_end = 1 + len(stage_ratios)

                def kv_factory(**kwargs):
                    return SimpleNamespace(
                        kv_buffer=[
                            torch.empty((3, kwargs["page_size"]), dtype=torch.uint8)
                            for _ in range(kwargs["layer_num"])
                        ]
                    )

                # Separate payload/scale buffers exercise the FP4 transfer contract.
                indexer_buffers = [
                    torch.empty((3, width), dtype=torch.uint8)
                    for _ in range(stage_ratios.count(4))
                    for width in (32, 2)
                ]
                indexer = SimpleNamespace(
                    contiguous_page_row_buffers=lambda: indexer_buffers
                )
                with (
                    patch.object(pool, "_make_kv_pool", side_effect=kv_factory),
                    patch.object(pool, "_make_indexer_pool", return_value=indexer),
                ):
                    pool._init_compressed_pools(
                        stage_ratios=stage_ratios,
                        page_size=256,
                        dtype=torch.float8_e4m3fn,
                        device="cpu",
                        enable_memory_saver=False,
                        enable_hisparse=False,
                        kv_pool_cls=DeepSeekV4SingleKVPool,
                    )
                pool._init_compressed_layer_mapping()
                self.assertIsNone(pool.layer_mapping[0])
                self.assertIsNone(pool.layer_mapping[-1])
                for local_id, ratio in enumerate(stage_ratios):
                    item = pool.layer_mapping[local_id + 1]
                    self.assertEqual(
                        item.compress_layer_id, stage_ratios[:local_id].count(ratio)
                    )
                    self.assertIs(item.compress_kv_pool, pool.kv_pools.get(ratio))
                self.assertIs(pool.c4_kv_pool, pool.kv_pools[4])
                self.assertIs(pool.c128_kv_pool, pool.kv_pools[128])
                self.assertIs(pool.c4_indexer_kv_pool, pool.index_pools[4])

                if unified:
                    buffers = [
                        torch.empty((9, 8), dtype=torch.uint8) for _ in stage_ratios
                    ]
                    pool.unified_kv_pool = SimpleNamespace(
                        swa_pages=2, kv_buffer=buffers
                    )

                    def kv_entries(ratio):
                        return [
                            (buf.data_ptr() + 16, 56, 256 // ratio * 8)
                            for buf, r in zip(buffers, stage_ratios)
                            if r == ratio
                        ]
                else:

                    def kv_entries(ratio):
                        return [
                            (b.data_ptr(), b.nbytes, b[0].nbytes)
                            for b in pool.kv_pools[ratio].kv_buffer
                        ]

                indexer_entries = [
                    (b.data_ptr(), b.nbytes, b[0].nbytes) for b in indexer_buffers
                ]
                expected = kv_entries(4) + indexer_entries + kv_entries(128)
                actual = list(zip(*pool.get_contiguous_buf_infos()))
                self.assertEqual(actual, expected)

    def test_shared_state_factory_preserves_layouts(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.compressed_pool_configs = {
            4: _CompressedPoolConfig(256, 64, torch.bfloat16, indexer_size=1024),
            128: _CompressedPoolConfig(512, 8, torch.float32),
        }
        pool.compression_ratios = [0, 4, 128]
        pool._stage_start, pool._stage_end = 0, 3
        pool.index_pools = {4: object()}
        pool.qk_nope_head_dim, pool.qk_rope_head_dim = 448, 64
        pool.indexer_head_dim = 128
        pool.device = "cpu"
        pool.swa_page_size = 128
        pool.online_mtp_max_draft_tokens = 3
        for online in (False, True):
            with (
                self.subTest(online=online),
                patch(
                    "sglang.srt.mem_cache.deepseek_v4_memory_pool.ONLINE_C128", online
                ),
                patch.object(
                    pool,
                    "get_ring_size",
                    side_effect=lambda r: 8 if r == 4 else (1 if online else 128),
                ),
            ):
                pool._init_paged_compress_states(False)
            c4 = pool.compress_state_pools[1].kv_score_buffer.kv_score
            indexer = pool.indexer_compress_state_pools[1].kv_score_buffer.kv_score
            c128 = pool.compress_state_pools[2].kv_score_buffer.kv_score
            self.assertEqual(c4.shape, (76, 2048))
            self.assertEqual(indexer.shape, (76, 512))
            self.assertEqual(c128.shape, (40, 1536) if online else (256, 1024))
            self.assertEqual(c4.dtype, torch.bfloat16)
            self.assertEqual(indexer.dtype, torch.bfloat16)
            self.assertEqual(c128.dtype, torch.float32)
            self.assertNotEqual(c4.data_ptr(), indexer.data_ptr())
            self.assertIsNone(pool.compress_state_pools[0])
            self.assertIsNone(pool.indexer_compress_state_pools[2])

    def test_indexer_access_uses_layer_ratio_and_waits_only_for_reads(self):
        pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
        pool.kv_pools = {4: None, 128: None}
        pool.compression_ratios = [4, 128, 4]
        pool._stage_start, pool._stage_end = 0, 3
        pool._init_compressed_layer_mapping()
        indexer = MagicMock(page_size=32)
        pool.index_pools = {4: indexer}
        trace = MagicMock()
        trace.attach_mock(indexer, "indexer")
        with patch.object(pool, "wait_layer_transfer") as wait:
            trace.attach_mock(wait, "wait")
            pool.get_index_k_fp4_payload_buffer(2)
            pool.set_index_k_fp4(2, "loc", "cache")
        self.assertEqual(
            trace.mock_calls,
            [
                unittest.mock.call.wait(2),
                unittest.mock.call.indexer.get_index_k_fp4_payload_buffer(1),
                unittest.mock.call.indexer.set_index_fp4(1, "loc", "cache"),
            ],
        )
        self.assertEqual(pool.get_index_k_page_size(4), 32)
        with self.assertRaisesRegex(AssertionError, "No indexer pool"):
            pool.get_index_k_page_size(128)


HEAD_DIM = 512
ROPE_DIM = 64
PAGE_SIZE = 256
FULL_SIZE = 4 * PAGE_SIZE


class TestV41KVPoolLayouts(CustomTestCase):
    """A V4.1-layout pool hands the attention kernel page-aligned buffers and
    picks the compressed layout each ratio asks for."""

    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(page_size=PAGE_SIZE)
        override.install()
        self.addCleanup(override.restore)

    def make_pool(self, ratios, kv_source_layers, kv_layout, compressed=None, **sizes):
        return DeepSeekV4TokenToKVPool(
            max_num_reqs=16,
            swa_size=FULL_SIZE,
            c4_size=sizes.get("c4_size", 0),
            c128_size=sizes.get("c128_size", 0),
            c4_state_pool_size=sizes.get("c4_state_pool_size", 0),
            c128_state_pool_size=sizes.get("c128_state_pool_size", 0),
            page_size=PAGE_SIZE,
            swa_page_size=PAGE_SIZE,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=HEAD_DIM - ROPE_DIM,
            qk_rope_head_dim=ROPE_DIM,
            indexer_head_dim=128,
            layer_num=len(ratios),
            device="cpu",
            enable_memory_saver=False,
            compression_ratios=ratios,
            kv_source_layers=kv_source_layers,
            full_size=FULL_SIZE,
            kv_layout=kv_layout,
            compressed_kv_layout=compressed,
        )

    def assert_kernel_requirements(self, pool, layout):
        """Pages start on the kernel's alignment, and its
        (num_pages, page_size, 1, bytes_per_token) view walks one token per row."""
        for buf in pool.kv_buffer:
            self.assertEqual(buf.stride(0) % layout.page_align, 0)
            bpt = layout.bytes_per_token
            view = buf[:, : pool.page_size * bpt].view(
                buf.shape[0], pool.page_size, 1, bpt
            )
            self.assertEqual(view.stride(1), bpt)
            self.assertEqual(view.stride(0), pool.bytes_per_page_padded)

    def test_v41_pool_buffers(self):
        for option, expect in ((None, KVLayout.V41_FP4), ("fp8", KVLayout.V41)):
            with self.subTest(compressed=option):
                pool = self.make_pool([0, 0, 2, 1, 1], [2, 3], KVLayout.V41, option)
                self.assert_kernel_requirements(pool.swa_kv_pool, KVLayout.V41)
                self.assertEqual(pool.get_swa_key_bytes_per_token(), 528)
                for ratio in (1, 2):
                    layer_id = pool.sources_by_ratio[ratio][0]
                    self.assertIs(pool.get_extra_key_layout(layer_id), expect)
                    self.assertEqual(
                        pool.get_extra_key_bytes_per_token(layer_id),
                        expect.bytes_per_token,
                    )
                    self.assertTrue(is_valid_kv_layout_pair(pool.kv_layout, expect))
                    self.assert_kernel_requirements(pool.kv_pools[ratio], expect)
        # A pool of the fp4 layout cannot be the main cache.
        with self.assertRaises(AssertionError):
            self.make_pool([0], [], KVLayout.V41_FP4)

    def test_v41_pool_with_c4_c128(self):
        pool = self.make_pool(
            [0, 4, 128],
            [],
            KVLayout.V41,
            c4_size=PAGE_SIZE,
            c128_size=PAGE_SIZE,
            c4_state_pool_size=16,
            c128_state_pool_size=16,
        )
        for ratio in (4, 128):
            self.assertEqual(pool.kv_pools[ratio].page_size, PAGE_SIZE // ratio)
        # The 2-token c128 page is the only production page that pads.
        self.assertEqual(pool.kv_pools[128].bytes_per_page_padded, 1536)


class TestPagedDSparkWithEncoderReplay(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            enable_encoder_swa_bounded_replay=True,
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=6,
            speculative_dspark_block_size=5,
            page_size=256,
            max_running_requests=2,
            chunked_prefill_size=256,
        )
        override.install()
        self.addCleanup(override.restore)

    def make_pool(self, *, draft):
        return DeepSeekV4TokenToKVPool(
            max_num_reqs=2,
            num_req_slots=3,
            swa_size=1024,
            c4_size=0,
            c128_size=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            page_size=256,
            swa_page_size=256,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.bfloat16,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=3,
            device="cpu",
            enable_memory_saver=False,
            compression_ratios=[0, 0, 0],
            online_mtp_max_draft_tokens=6,
            full_size=2048,
            is_draft_worker=draft,
        )

    def test_target_window_and_draft_paged_storage_share_allocator_mapping(self):
        target = self.make_pool(draft=False)
        draft = self.make_pool(draft=True)
        allocator = SWATokenToKVPoolAllocator(
            2048, 1024, 256, torch.float8_e4m3fn, "cpu", target, False
        )
        draft.register_mapping(allocator.full_to_swa_index_mapping)
        allocator.full_to_swa_index_mapping[256:512] = torch.arange(768, 1024)
        self.assertEqual(
            draft.translate_loc_from_full_to_swa(
                torch.tensor([256, 300, 511])
            ).tolist(),
            [768, 812, 1023],
        )


if __name__ == "__main__":
    unittest.main()
