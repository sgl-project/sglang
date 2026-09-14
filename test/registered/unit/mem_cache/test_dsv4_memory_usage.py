import types
import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
    DeepSeekV4UnifiedKVPool,
)
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

    def test_top_level_pool_aggregates_separate_layout_and_state_storage(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = False
        pool.swa_kv_pool = _single_pool((2, 3))
        pool.c4_kv_pool = _single_pool((4, 5))
        pool.c128_kv_pool = _single_pool((6, 7))
        pool.c4_indexer_kv_pool = _indexer_pool(("index_k_with_scale_buffer", [(8, 9)]))
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
        pool.c4_indexer_kv_pool = _indexer_pool(("index_k_with_scale_buffer", [(6, 7)]))
        pool.compress_state_pools = [_state_pool((8, 9))]
        pool.indexer_compress_state_pools = []
        expected = (2 * 3 + 4 * 5) * 2 + 6 * 7 + 8 * 9 * 4
        self.assertEqual(pool.get_kv_size_bytes(), expected)

    def test_top_level_allocation_log_exposes_gib(self):
        pool = object.__new__(DeepSeekV4TokenToKVPool)
        pool._unified_kv = True
        pool.unified_kv_pool = object.__new__(DeepSeekV4UnifiedKVPool)
        pool.unified_kv_pool.kv_buffer = [torch.empty((1024,), dtype=torch.uint8)]
        pool.c4_indexer_kv_pool = _indexer_pool()
        pool.compress_state_pools = []
        pool.indexer_compress_state_pools = []
        pool.allocation_label = None

        pool._finalize_allocation_log(1)

        self.assertAlmostEqual(pool.mem_usage, 1024 / (1024**3))


if __name__ == "__main__":
    unittest.main()
