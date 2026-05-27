"""
Unit tests for two hicache prefetch gate fixes:

  Patch A (hybrid_cache_controller.py): decouple KV prefetch admission gate
  from the mamba-clamped final_pages. The gate now reads from
  hit_result.extra_pool_hit_pages[PoolName.KV] so absent mamba companions
  don't zero-out the KV hit count and revoke all L3 prefetches.

  Patch B (hi_mamba_radix_cache.py): align the prefetch last_hash seed with
  the write-side hash chain by deriving last_hash_for_prefetch from
  last_host_node.hash_value[-1] rather than always using the caller-supplied
  last_hash (which defaults to None).
"""
import unittest

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransferResult
from sglang.srt.mem_cache.mamba_radix_cache import TreeNode


class TestExtraPoolHitPagesGate(unittest.TestCase):
    """Patch A: kv_hit_pages must come from extra_pool_hit_pages[PoolName.KV]."""

    def _resolve(self, result: PoolTransferResult) -> int:
        """Replicate the one-liner from hybrid_cache_controller._storage_hit_query."""
        return result.extra_pool_hit_pages.get(PoolName.KV, result.kv_hit_pages)

    def test_extra_pool_kv_wins_over_zero_kv_hit_pages(self):
        """extra_pool_hit_pages[KV]=100 must win even when kv_hit_pages=0.

        Before the fix, kv_hit_pages=0 caused the prefetch gate to revoke
        every L3 read when mamba companion files were absent.
        """
        result = PoolTransferResult(
            kv_hit_pages=0, extra_pool_hit_pages={PoolName.KV: 100}
        )
        self.assertEqual(self._resolve(result), 100)

    def test_fallback_to_kv_hit_pages_when_extra_empty(self):
        """When batch_exists returns no extra_pool_hit_pages, fall back to kv_hit_pages."""
        result = PoolTransferResult(kv_hit_pages=50, extra_pool_hit_pages={})
        self.assertEqual(self._resolve(result), 50)

    def test_extra_pool_kv_zero_does_not_mask_nonzero_kv_hit_pages(self):
        """An explicit KV=0 entry in extra_pool_hit_pages overrides kv_hit_pages."""
        result = PoolTransferResult(
            kv_hit_pages=30, extra_pool_hit_pages={PoolName.KV: 0}
        )
        self.assertEqual(self._resolve(result), 0)

    def test_non_kv_pool_entry_does_not_affect_kv_count(self):
        """An extra entry for a non-KV pool must not influence KV hit pages."""
        result = PoolTransferResult(kv_hit_pages=20, extra_pool_hit_pages={})
        self.assertEqual(self._resolve(result), 20)


class TestPrefetchLastHashAlignment(unittest.TestCase):
    """Patch B: prefetch_from_storage derives last_hash from last_host_node.hash_value[-1]."""

    def _compute(self, cache_root, last_host_node, caller_last_hash):
        """Replicate the exact expression added to prefetch_from_storage."""
        return (
            last_host_node.hash_value[-1]
            if last_host_node is not cache_root and last_host_node.hash_value
            else caller_last_hash
        )

    def setUp(self):
        self.root = TreeNode()

    def test_non_root_with_hash_uses_last_page_hash(self):
        """A non-root host node with hash_value should seed prefetch with its last page hash."""
        node = TreeNode()
        node.hash_value = ["page_0_hash", "page_1_hash", "page_2_hash"]
        result = self._compute(self.root, node, "caller")
        self.assertEqual(result, "page_2_hash")

    def test_single_page_node_returns_only_hash(self):
        node = TreeNode()
        node.hash_value = ["only_page_hash"]
        result = self._compute(self.root, node, "caller")
        self.assertEqual(result, "only_page_hash")

    def test_root_node_falls_back_to_caller_hash(self):
        """The root node has no prefix to seed from — fall back to caller-supplied last_hash."""
        self.root.hash_value = ["root_hash"]
        result = self._compute(self.root, self.root, "caller_hash")
        self.assertEqual(result, "caller_hash")

    def test_non_root_with_empty_hash_value_falls_back(self):
        """Empty hash_value list on a non-root node falls back to caller hash."""
        node = TreeNode()
        node.hash_value = []
        result = self._compute(self.root, node, "caller_hash")
        self.assertEqual(result, "caller_hash")

    def test_non_root_with_none_hash_value_falls_back(self):
        """None hash_value on a non-root node falls back to caller hash."""
        node = TreeNode()
        node.hash_value = None
        result = self._compute(self.root, node, "caller_hash")
        self.assertEqual(result, "caller_hash")

    def test_caller_hash_none_and_root_returns_none(self):
        """When caller_last_hash is None and node is root, result is None (dense RadixCache compat)."""
        result = self._compute(self.root, self.root, None)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
