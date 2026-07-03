"""Unit tests for SGLANG_DSA_COMPACT_INDEXER (skip-topk indexer buffer dedup).

DSA models with index_topk_freq > 1 (e.g. GLM-5.2) run the indexer only on a
subset of layers; the remaining "shared" layers carry no indexer weights and
never read or write their index_k buffer. The compact mode replaces those dead
per-layer buffers with a single shared alias tensor.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    dsa_compact_indexer_layer_mask,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ENV = "SGLANG_DSA_COMPACT_INDEXER"


def _cfg(**kwargs):
    return SimpleNamespace(**kwargs)


class TestDsaCompactIndexerMask(CustomTestCase):
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_ENV, None)
            cfg = _cfg(num_hidden_layers=8, index_topk_freq=4)
            self.assertIsNone(dsa_compact_indexer_layer_mask(cfg, 8))

    def test_freq_derivation(self):
        with patch.dict(os.environ, {_ENV: "1"}):
            cfg = _cfg(num_hidden_layers=8, index_topk_freq=4)
            mask = dsa_compact_indexer_layer_mask(cfg, 8)
            # max(i - 1, 0) % freq == 0 -> layers 0, 1, 5 are full
            self.assertEqual(
                mask, [True, True, False, False, False, True, False, False]
            )

    def test_pattern_takes_priority_over_freq(self):
        with patch.dict(os.environ, {_ENV: "1"}):
            cfg = _cfg(
                num_hidden_layers=4,
                index_topk_pattern="FSFS",
                index_topk_freq=1,  # would return None if consulted
            )
            self.assertEqual(
                dsa_compact_indexer_layer_mask(cfg, 4), [True, False, True, False]
            )

    def test_freq_one_returns_none(self):
        # All layers run the indexer -> nothing to compact.
        with patch.dict(os.environ, {_ENV: "1"}):
            cfg = _cfg(num_hidden_layers=4, index_topk_freq=1)
            self.assertIsNone(dsa_compact_indexer_layer_mask(cfg, 4))

    def test_partial_pool_returns_none(self):
        # A pool that does not span the full model (e.g. the NEXTN draft pool,
        # whose single layer owns real indexer weights) must not be compacted.
        with patch.dict(os.environ, {_ENV: "1"}):
            cfg = _cfg(num_hidden_layers=8, index_topk_freq=4)
            self.assertIsNone(dsa_compact_indexer_layer_mask(cfg, 1))
            self.assertIsNone(dsa_compact_indexer_layer_mask(cfg, 8, start_layer=2))

    def test_indexer_types_cross_check_mismatch_fails_safe(self):
        with patch.dict(os.environ, {_ENV: "1"}):
            cfg = _cfg(
                num_hidden_layers=4,
                index_topk_freq=2,
                # Derivation says [T, T, F, T]; contradictory config below.
                indexer_types=["full", "full", "full", "full"],
            )
            self.assertIsNone(dsa_compact_indexer_layer_mask(cfg, 4))

    def test_indexer_types_cross_check_match(self):
        with patch.dict(os.environ, {_ENV: "1"}):
            cfg = _cfg(
                num_hidden_layers=4,
                index_topk_freq=2,
                indexer_types=["full", "full", "shared", "full"],
            )
            self.assertEqual(
                dsa_compact_indexer_layer_mask(cfg, 4), [True, True, False, True]
            )


class TestDsaCompactIndexerAllocation(CustomTestCase):
    def _make_pool(self, mask):
        return DSATokenToKVPool(
            size=256,
            page_size=64,
            dtype=torch.float16,
            kv_lora_rank=128,
            qk_rope_head_dim=64,
            layer_num=len(mask) if mask else 4,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=192,
            indexer_layer_mask=mask,
        )

    def test_compact_allocation_shares_alias(self):
        mask = [True, False, False, True]
        pool = self._make_pool(mask)
        bufs = pool.index_k_with_scale_buffer
        self.assertEqual(len(bufs), 4)
        # Skip-topk layers share one alias tensor; full layers stay distinct.
        self.assertIs(bufs[1], bufs[2])
        self.assertIsNot(bufs[0], bufs[3])
        self.assertIsNot(bufs[0], bufs[1])
        self.assertEqual(pool.indexer_layer_ids, [0, 3])
        # Every buffer keeps the full shape, so full-layer sweeps stay in-bounds.
        self.assertEqual(bufs[1].shape, bufs[0].shape)

    def test_stock_allocation_unchanged_without_mask(self):
        pool = self._make_pool(None)
        bufs = pool.index_k_with_scale_buffer
        self.assertEqual(len({id(b) for b in bufs}), len(bufs))

    def test_kv_size_bytes_dedupes_alias(self):
        compact = self._make_pool([True, False, False, True])
        stock = self._make_pool(None)
        per_layer_index_bytes = (
            stock.index_k_with_scale_buffer[0].numel()
            * stock.index_k_with_scale_buffer[0].element_size()
        )
        total_stock = stock.get_kv_size_bytes()
        total_compact = compact.get_kv_size_bytes()
        # 4 real buffers -> 2 real + 1 alias
        self.assertEqual(total_stock - total_compact, per_layer_index_bytes)


if __name__ == "__main__":
    unittest.main()
