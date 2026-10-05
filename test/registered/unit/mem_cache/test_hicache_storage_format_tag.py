"""A host pool with its own page byte format namespaces its storage keys.

The unified pool's host mirror stores whole page envelopes, whose bytes are
token-major entries. A page persisted under another format has the same size,
so only the key can keep it from being read back as a unified page: the
storage model name carries the pool's format tag, and an untagged page misses.

CPU-only.

    python -m pytest test/registered/unit/mem_cache/test_hicache_storage_format_tag.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import shutil
import tempfile
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.cache_controller import storage_model_name
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.memory_pool_host import LogicalHostPool
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.mem_cache.pool_host.unified import UnifiedPageEnvelopeHostPool
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


if __name__ == "__main__":
    unittest.main()
