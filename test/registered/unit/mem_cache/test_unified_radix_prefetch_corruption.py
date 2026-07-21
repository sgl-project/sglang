"""Regression tests for a UnifiedRadixCache HiCache-prefetch corruption.

The L3-prefetch commit (`_insert_helper_host`) could hang a backed-up host child
under a device-only, un-backed-up parent, breaking the write-through invariant
"child backed up => parent backed up". That surfaced two ways:
  - idle sanity: "node N backed up but parent M not backed up"
  - eviction crash: _remove_leaf_from_parent -> assert v == node (the unbacked
    parent was deleted from under its still-attached host child)
Fix: drop the best-effort refill when the anchor isn't backed up. Each test is
red before the fix, green after.
"""

import unittest
from array import array
from unittest import mock

import torch
from test_unified_radix_cache_unittest import CacheConfig, build_fixture

from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Pure tree-bookkeeping logic; no backend-specific kernels, so CUDA-only (the
# fixtures still allocate real KV pools, which need a GPU).
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

PS = 16
_CFG = CacheConfig(
    page_size=PS,
    components=(ComponentType.FULL,),
    kv_size=4096,
    max_context_len=4096,
)


class TestUnifiedRadixPrefetchCorruption(CustomTestCase):
    cfg = _CFG

    # ---- fixture helpers ----
    def _init_hicache(self, cache):
        """Enable write-through HiCache with a non-pinned host pool.

        write_through_threshold is raised so plain inserts do NOT auto-backup,
        letting a test build the device-only-unbacked parent both incidents need.
        """
        import sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler as assembler

        orig_get_mha_host_pool_cls = assembler.get_mha_host_pool_cls

        def get_mha_host_pool_cls_wrapper(device_pool):
            host_pool_cls = orig_get_mha_host_pool_cls(device_pool)

            def kv_host_pool_wrapper(*args, **kwargs):
                kwargs["pin_memory"] = False
                return host_pool_cls(*args, **kwargs)

            return kv_host_pool_wrapper

        patcher = mock.patch.object(
            assembler,
            "get_mha_host_pool_cls",
            side_effect=get_mha_host_pool_cls_wrapper,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        server_args = ServerArgs(
            model_path="dummy",
            page_size=self.cfg.page_size,
            hicache_io_backend="direct",
            hicache_write_policy="write_through",
        )
        server_args._mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, self.cfg.page_size)
        set_global_server_args_for_scheduler(server_args)
        cache.init_hicache(server_args, cache.cache_init_params)
        cache.write_through_threshold = 1 << 30
        cache.load_back_threshold = 0

    def _insert_device(self, cache, allocator, ids):
        """Insert a device-only chain (no auto-backup) and return its leaf."""
        key = RadixKey(array("q", ids)).page_aligned(PS)
        val = allocator.alloc(len(key))
        self.assertIsNotNone(val)
        val = val.to(dtype=torch.int64)
        cache.insert(InsertParams(key=key, value=val, prev_prefix_len=0))
        m = cache.match_prefix(MatchPrefixParams(key=key))
        return m.last_device_node

    def _host_indices(self, cache, n):
        idx = cache.cache_controller.mem_pool_host.alloc(n)
        self.assertIsNotNone(idx, "host pool alloc failed")
        return idx.to(dtype=torch.int64) if hasattr(idx, "to") else idx

    def _attach_host_child(self, cache, parent, start_token):
        """Mimic a prefetch commit: hang a backed-up host chain under `parent`."""
        child_ids = list(range(start_token, start_token + 2 * PS))
        child_key = RadixKey(array("q", child_ids)).page_aligned(PS)
        host_idx = self._host_indices(cache, len(child_key))
        hashes = [f"h{i}" for i in range(len(child_key) // PS)]
        res = cache._insert_helper_host(parent, child_key, host_idx, hashes)
        return res.inserted_host_node

    # ---- Incident A ----
    def test_prefetch_refill_under_unbacked_parent_is_dropped(self):
        """Refill under an un-backed-up parent is dropped, keeping the invariant.

        Pre-fix the child was attached and idle sanity raised
        "node ... backed up but parent ... not backed up".
        """
        cache, allocator, _ = build_fixture(self.cfg)
        self._init_hicache(cache)

        parent = self._insert_device(cache, allocator, list(range(1, 1 + 3 * PS)))
        self.assertFalse(parent.backuped, "parent must start device-only, unbacked")

        child = self._attach_host_child(cache, parent, start_token=1000)
        self.assertIsNone(child, "refill under an un-backed-up parent must be dropped")
        self.assertFalse(parent.backuped)
        self.assertEqual(len(parent.children), 0)
        cache.sanity_check()

    # ---- Incident B ----
    def test_prefetch_refill_leaves_eviction_path_uncorrupted(self):
        """Eviction stays consistent after a refill under an un-backed-up parent.

        Pre-fix the unbacked parent (carrying a backed-up child) hit the
        write-through delete branch and detached itself from under the child
        (assert v == node). Post-fix the refill is dropped, so eviction is clean.
        """
        cache, allocator, _ = build_fixture(self.cfg)
        self._init_hicache(cache)

        parent = self._insert_device(cache, allocator, list(range(1, 1 + 3 * PS)))
        self._attach_host_child(cache, parent, start_token=1000)

        cache.evict(EvictParams(num_tokens=10 * PS))
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
