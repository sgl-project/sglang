import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
    _CompressedPoolConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV4CompressedPools(CustomTestCase):
    def test_pp_mapping_and_pd_buffer_order(self):
        for unified, stage_ratios in product(
            (False, True), ([4, 0, 128, 4], [128], [0])
        ):
            with self.subTest(unified=unified, stage_ratios=stage_ratios):
                pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
                pool._unified_kv = unified
                pool.uniform_fp8 = False
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


if __name__ == "__main__":
    unittest.main()
