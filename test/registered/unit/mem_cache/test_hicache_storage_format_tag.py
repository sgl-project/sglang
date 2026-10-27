"""A host pool with its own page byte format namespaces its storage keys.

The unified pool's host mirror stores whole page envelopes, whose bytes are
token-major entries. A page persisted under another format has the same size,
so only the key can keep it from being read back as a unified page: the
storage model name carries the pool's format tag, and an untagged page misses.

CPU-only.

    python -m pytest test/registered/unit/mem_cache/test_hicache_storage_format_tag.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import shutil
import tempfile
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.cache_controller import storage_model_name
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.unified import (
    UnifiedPageEnvelopeHostPool,
    _SharedPageEnvelopeHostBacking,
)
from sglang.test.test_utils import CustomTestCase


def _config(model_name):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model_name,
    )


class TestStorageModelName(CustomTestCase):
    def test_default_pools_keep_the_model_name(self):
        self.assertIsNone(HostKVCache.storage_format_tag)
        pool = SimpleNamespace(storage_format_tag=None)
        self.assertEqual(storage_model_name("org/model", pool), "org/model")
        self.assertIsNone(storage_model_name(None, pool))

    def test_logical_anchor_pool_keeps_the_model_name(self):
        """A LogicalHostPool storage anchor keeps the untagged model name."""
        pool = LogicalHostPool(size=4, page_size=2)
        self.assertEqual(storage_model_name("org/model", pool), "org/model")

    def test_unified_pool_tags_the_model_name(self):
        tag = UnifiedPageEnvelopeHostPool.storage_format_tag
        self.assertIsNotNone(tag)
        self.assertEqual(
            storage_model_name("org/model", UnifiedPageEnvelopeHostPool),
            f"org/model-{tag}",
        )
        self.assertEqual(storage_model_name(None, UnifiedPageEnvelopeHostPool), tag)

    def test_untagged_page_misses_under_the_unified_name(self):
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        untagged = HiCacheFile(_config("org/model"), file_path=root)
        tagged = HiCacheFile(
            _config(storage_model_name("org/model", UnifiedPageEnvelopeHostPool)),
            file_path=root,
        )
        self.assertTrue(untagged.set("page-0", torch.zeros(64, dtype=torch.uint8)))
        self.assertTrue(untagged.exists("page-0"))
        self.assertFalse(tagged.exists("page-0"))

    def test_unified_backing_partial_hole_selection_and_extent_reservation(self):
        extents = [(0, 4), (6, 10), (12, 16)]
        reserved = _SharedPageEnvelopeHostBacking._reserve_logical_pages(extents, 2)
        self.assertEqual(reserved, [0, 1])
        self.assertEqual(extents, [(2, 4), (6, 10), (12, 16)])

        reserved_spanning = _SharedPageEnvelopeHostBacking._reserve_logical_pages(
            extents, 5
        )
        self.assertEqual(reserved_spanning, [2, 3, 6, 7, 8])
        self.assertEqual(extents, [(9, 10), (12, 16)])

        up_side = _SharedPageEnvelopeHostBacking._Side(
            page_size=2, page_bytes=16, page_num=16, grow_direction="up"
        )
        up_side.watermark = 10
        up_side.free_physical_pages = {7, 2, 9, 4}
        backing = object.__new__(_SharedPageEnvelopeHostBacking)
        self.assertEqual(backing._take_physical_pages(up_side, 0), [])
        self.assertEqual(backing._take_physical_pages(up_side, 2), [2, 4])
        self.assertEqual(up_side.free_physical_pages, {7, 9})

        down_side = _SharedPageEnvelopeHostBacking._Side(
            page_size=2, page_bytes=16, page_num=16, grow_direction="down"
        )
        down_side.watermark = 5
        down_side.free_physical_pages = {7, 14, 9, 12}
        self.assertEqual(backing._take_physical_pages(down_side, 2), [14, 12])
        self.assertEqual(down_side.free_physical_pages, {7, 9})

        import threading

        side = _SharedPageEnvelopeHostBacking._Side(
            page_size=2, page_bytes=16, page_num=8, grow_direction="up"
        )
        side.free_logical_extents = [(0, 1), (3, 4), (6, 8)]
        side.logical_to_physical = {1: 10, 2: 11, 4: 12, 5: 13}
        side.physical_to_logical = {10: 1, 11: 2, 12: 4, 13: 5}
        side.watermark = 14
        backing.lock = threading.RLock()
        backing.sides = {"full": side}
        freed = backing.free(
            "full", torch.tensor([8, 9, 2, 3], dtype=torch.int64)
        )
        self.assertEqual(freed, 4)
        self.assertEqual(side.free_logical_extents, [(0, 2), (3, 5), (6, 8)])


if __name__ == "__main__":
    unittest.main()
