import types
import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
    DeepSeekV4UnifiedKVPool,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _single_pool(*shapes):
    pool = object.__new__(DeepSeekV4SingleKVPool)
    pool.kv_buffer = [torch.empty(shape, dtype=torch.uint8) for shape in shapes]
    return pool


def _indexer_pool(*named_buffers):
    pool = object.__new__(DeepSeekV4IndexerPool)
    for name, buffers in named_buffers:
        setattr(
            pool,
            name,
            [torch.empty(shape, dtype=torch.uint8) for shape in buffers],
        )
    return pool


def _state_pool(shape):
    return types.SimpleNamespace(
        kv_score_buffer=types.SimpleNamespace(
            kv_score=torch.empty(shape, dtype=torch.float32)
        )
    )


class TestDeepSeekV4MemoryUsage(CustomTestCase):
    """Allocated KV and compressor storage must be reflected in memory metrics."""

    def test_single_pool_reports_allocated_bytes(self):
        pool = _single_pool((2, 3), (4, 5))

        self.assertEqual(pool.get_kv_size_bytes(), 2 * 3 + 4 * 5)

    def test_indexer_reports_each_active_layout_buffer(self):
        layouts = (
            ((("index_k_with_scale_buffer", [(2, 3)]),), 2 * 3),
            (
                (
                    ("index_k_payload_buffer", [(4, 5)]),
                    ("index_k_scale_buffer", [(6, 7)]),
                ),
                4 * 5 + 6 * 7,
            ),
            (
                (
                    ("index_k_with_scale_buffer", [(8, 9)]),
                    ("index_k_buffer", [(10, 11)]),
                    ("index_scale_buffer", [(12, 13)]),
                ),
                8 * 9 + 10 * 11 + 12 * 13,
            ),
        )
        for buffers, expected in layouts:
            with self.subTest(buffers=buffers):
                pool = _indexer_pool(*buffers)
                self.assertEqual(pool.get_kv_size_bytes(), expected)

    def test_indexer_finalizes_after_backend_allocation(self):
        """Backend buffers allocated after the base buffer must reach mem_usage."""

        class BackendIndexerPool(DeepSeekV4IndexerPool):
            def _create_buffer(self):
                super()._create_buffer()
                self.index_k_buffer = [torch.empty((10,), dtype=torch.int8)]
                self.index_scale_buffer = [torch.empty((5,), dtype=torch.float16)]

        pool = BackendIndexerPool(
            size=8,
            page_size=4,
            dtype=torch.uint8,
            index_head_dim=128,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
            use_fp4_indexer=False,
        )

        self.assertEqual(pool.get_kv_size_bytes(), 3 * 4 * 132 + 10 + 5 * 2)
        self.assertEqual(pool.mem_usage, pool.get_kv_size_bytes() / (1024**3))

    def test_top_level_pool_aggregates_separate_layout_and_state_storage(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.swa_kv_pool = _single_pool((2, 3))
        pool.c4_kv_pool = _single_pool((4, 5))
        pool.c128_kv_pool = _single_pool((6, 7))
        pool.c4_indexer_kv_pool = _indexer_pool(("index_k_with_scale_buffer", [(8, 9)]))
        pool.kv_pools = {4: pool.c4_kv_pool, 128: pool.c128_kv_pool}
        pool.index_pools = {4: pool.c4_indexer_kv_pool}
        pool.request_window = None
        pool.compress_state_pools = [_state_pool((10, 11)), None]
        pool.indexer_compress_state_pools = [None, _state_pool((12, 13))]

        expected = 2 * 3 + 4 * 5 + 6 * 7 + 8 * 9
        expected += 10 * 11 * 4 + 12 * 13 * 4
        self.assertEqual(pool.get_kv_size_bytes(), expected)

    def test_top_level_pool_aggregates_unified_storage(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = True
        pool.unified_kv_pool = object.__new__(DeepSeekV4UnifiedKVPool)
        pool.unified_kv_pool.kv_buffer = [
            torch.empty((2, 3), dtype=torch.bfloat16),
            torch.empty((4, 5), dtype=torch.bfloat16),
        ]
        pool.unified_kv_pool.kv_buffer_rope = [None, None]
        pool.c4_indexer_kv_pool = _indexer_pool(("index_k_with_scale_buffer", [(6, 7)]))
        pool.index_pools = {4: pool.c4_indexer_kv_pool}
        pool.compress_state_pools = [_state_pool((8, 9))]
        pool.indexer_compress_state_pools = []
        expected = (2 * 3 + 4 * 5) * 2 + 6 * 7 + 8 * 9 * 4
        self.assertEqual(pool.get_kv_size_bytes(), expected)

    def test_constructors_expose_allocated_gib(self):
        """Construction must publish nonzero memory usage without manual finalization."""
        override = get_context().override_server_args(page_size=256)
        override.install()
        self.addCleanup(override.restore)
        pool = DeepSeekV4TokenToKVPool(
            max_num_reqs=1,
            swa_size=256,
            c4_size=0,
            c128_size=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            page_size=256,
            swa_page_size=256,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
            compression_ratios=[0],
        )

        expected = sum(buffer.nbytes for buffer in pool.swa_kv_pool.kv_buffer)
        self.assertGreater(expected, 0)
        self.assertEqual(pool.mem_usage, expected / (1024**3))
        self.assertEqual(pool.swa_kv_pool.mem_usage, expected / (1024**3))
        self.assertEqual(pool.get_kv_size_bytes(), expected)

    def test_unified_fp8_counts_separate_rope_storage(self):
        pool = object.__new__(DeepSeekV4UnifiedKVPool)
        pool.kv_buffer = [torch.empty((3, 512), dtype=torch.float8_e4m3fn)]
        pool.kv_buffer_rope = [torch.empty((3, 64), dtype=torch.bfloat16)]

        self.assertEqual(pool.get_kv_size_bytes(), 3 * (512 + 64 * 2))

    def test_top_level_counts_low_ratio_pools_once(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.request_window = None
        pool.swa_kv_pool = _single_pool((2, 3))
        pool.kv_pools = {
            4: _single_pool((4, 5)),
            128: None,
            1: _single_pool((6, 7)),
            2: _single_pool((8, 9)),
        }
        pool.index_pools = {
            1: _indexer_pool(("index_k_with_scale_buffer", [(10, 11)])),
            2: _indexer_pool(("index_k_with_scale_buffer", [(12, 13)])),
        }
        pool.c4_kv_pool = pool.kv_pools[4]
        pool.c128_kv_pool = None
        pool.c4_indexer_kv_pool = None
        pool.compress_state_pools = [None, _state_pool((14, 15))]
        pool.indexer_compress_state_pools = []

        self.assertEqual(
            pool.get_kv_size_bytes(),
            2 * 3 + 4 * 5 + 6 * 7 + 8 * 9 + 10 * 11 + 12 * 13 + 14 * 15 * 4,
        )

    def test_top_level_counts_request_window_storage(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.swa_kv_pool = None
        pool.kv_pools = {}
        pool.index_pools = {}
        pool.compress_state_pools = []
        pool.indexer_compress_state_pools = []
        pool.request_window = types.SimpleNamespace(
            state=_single_pool((2, 3)),
            workspace=None,
            tags=torch.empty((4, 5), dtype=torch.int64),
        )

        self.assertEqual(pool.get_kv_size_bytes(), 2 * 3 + 4 * 5 * 8)
        pool.request_window.workspace = _single_pool((6, 7))
        self.assertEqual(pool.get_kv_size_bytes(), 2 * 3 + 4 * 5 * 8 + 6 * 7)


if __name__ == "__main__":
    unittest.main()
