"""Tests for KV cache CPU backup during sleep/wakeup cycles.

These tests verify two layers of the feature:

1. Unit tests for ``RadixCache.export_snapshot`` / ``import_snapshot`` that run
   without a GPU or a running sglang server.
2. An integration test (requires GPU + enable_memory_saver) that starts a real
   ``sgl.Engine``, populates the prefix cache, sleeps, wakes up, and verifies that
   prefix cache hits are restored.
"""

import os
import time
import unittest

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import (
    RadixCache,
    RadixKey,
    InsertParams,
    MatchPrefixParams,
)
from sglang.test.test_utils import CustomTestCase, DEFAULT_SMALL_MODEL_NAME_FOR_TEST


# ---------------------------------------------------------------------------
# Unit tests for RadixCache snapshot / restore (no GPU required)
# ---------------------------------------------------------------------------


class _MockAllocator:
    """Minimal allocator stub for snapshot unit tests."""

    def __init__(self, size: int = 64):
        self.size = size
        self.device = torch.device("cpu")
        self.free_pages = torch.arange(1, size + 1, dtype=torch.int64)
        self.release_pages = torch.empty((0,), dtype=torch.int64)


class TestRadixCacheSnapshot(CustomTestCase):
    """Unit tests for export_snapshot / import_snapshot on RadixCache."""

    def _make_cache(self, size: int = 64) -> RadixCache:
        mock_alloc = _MockAllocator(size=size)
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=mock_alloc,
            page_size=1,
            enable_kv_cache_events=False,
        )
        return RadixCache(params)

    def test_export_empty_tree(self):
        cache = self._make_cache()
        snapshot = cache.export_snapshot()
        self.assertFalse(snapshot["disabled"])
        # Only the root node should be in the snapshot
        self.assertEqual(len(snapshot["nodes"]), 1)
        root_data = snapshot["nodes"][0]
        self.assertTrue(root_data["is_root"])

    def test_snapshot_round_trip(self):
        """Insert some entries, snapshot, reset, import, verify tree is identical."""
        cache = self._make_cache(size=128)

        # Insert some token sequences
        seq_a = list(range(1, 6))
        seq_b = list(range(1, 6)) + list(range(10, 15))
        seq_c = list(range(20, 27))

        slots_a = torch.arange(1, 6, dtype=torch.int64)
        slots_b = torch.arange(6, 11, dtype=torch.int64)
        slots_c = torch.arange(11, 18, dtype=torch.int64)

        cache.insert(
            InsertParams(key=RadixKey(token_ids=seq_a, extra_key=None), value=slots_a)
        )
        cache.insert(
            InsertParams(key=RadixKey(token_ids=seq_b, extra_key=None), value=slots_b)
        )
        cache.insert(
            InsertParams(key=RadixKey(token_ids=seq_c, extra_key=None), value=slots_c)
        )

        size_before = cache.total_size()

        snapshot = cache.export_snapshot()

        # Reset tree
        cache.reset()
        cache.token_to_kv_pool_allocator.free_pages = torch.arange(
            1, 129, dtype=torch.int64
        )
        self.assertEqual(cache.total_size(), 0)

        # Restore
        cache.import_snapshot(snapshot)

        size_after = cache.total_size()
        self.assertEqual(size_before, size_after)

    def test_prefix_match_after_restore(self):
        """After import_snapshot, match_prefix should return the correct slots."""
        cache = self._make_cache(size=64)

        tokens = list(range(1, 9))
        slots = torch.arange(1, 9, dtype=torch.int64)
        cache.insert(
            InsertParams(key=RadixKey(token_ids=tokens, extra_key=None), value=slots)
        )

        snapshot = cache.export_snapshot()

        # Reset and restore
        cache.reset()
        cache.token_to_kv_pool_allocator.free_pages = torch.arange(
            1, 65, dtype=torch.int64
        )
        cache.import_snapshot(snapshot)

        # Full prefix match should return all 8 slots
        result = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(token_ids=tokens, extra_key=None))
        )
        self.assertEqual(len(result.device_indices), 8)
        self.assertTrue(
            torch.equal(result.device_indices, slots),
            f"Expected {slots}, got {result.device_indices}",
        )

    def test_allocator_free_pages_updated(self):
        """After import_snapshot the allocator free_pages must not include used slots."""
        cache = self._make_cache(size=32)

        tokens = list(range(1, 6))
        slots = torch.arange(1, 6, dtype=torch.int64)  # slots 1..5 used
        cache.insert(
            InsertParams(key=RadixKey(token_ids=tokens, extra_key=None), value=slots)
        )

        snapshot = cache.export_snapshot()

        cache.reset()
        cache.token_to_kv_pool_allocator.free_pages = torch.arange(
            1, 33, dtype=torch.int64
        )
        cache.import_snapshot(snapshot)

        alloc = cache.token_to_kv_pool_allocator
        all_free = torch.cat([alloc.free_pages, alloc.release_pages]).tolist()
        for used_slot in range(1, 6):
            self.assertNotIn(
                used_slot,
                all_free,
                f"Slot {used_slot} should be allocated, not free",
            )

    def test_import_disabled_snapshot_is_noop(self):
        """import_snapshot on a disabled-cache snapshot should not raise and not modify tree."""
        cache = self._make_cache()
        disabled_snapshot = {"disabled": True, "nodes": [], "root_id": None}
        cache.import_snapshot(disabled_snapshot)
        # Tree should still be in reset state (only root)
        self.assertEqual(cache.total_size(), 0)

    def test_import_empty_nodes_is_noop(self):
        """import_snapshot with empty nodes list should not raise."""
        cache = self._make_cache()
        empty_snapshot = {"disabled": False, "nodes": [], "root_id": None}
        cache.import_snapshot(empty_snapshot)
        self.assertEqual(cache.total_size(), 0)


# ---------------------------------------------------------------------------
# Integration test: full engine sleep/wakeup with KV cache CPU backup
# ---------------------------------------------------------------------------


def _get_gpu_memory_gb():
    return torch.cuda.device_memory_used() / 1024**3


class TestKVCacheCPUBackupIntegration(CustomTestCase):
    """Integration tests that require a real GPU and torch-memory-saver."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        try:
            import torch_memory_saver  # noqa: F401

            cls._memory_saver_available = True
        except ImportError:
            cls._memory_saver_available = False

    def _skip_if_no_memory_saver(self):
        if not self._memory_saver_available:
            self.skipTest(
                "torch-memory-saver not installed; skipping KV cache CPU backup test. "
                "Install with: pip install torch-memory-saver"
            )

    def _setup_engine(self, enable_kv_cache_cpu_backup: bool = True):
        import sglang as sgl

        os.environ["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] = "1"
        return sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            random_seed=42,
            enable_memory_saver=True,
            mem_fraction_static=0.6,
            enable_kv_cache_cpu_backup=enable_kv_cache_cpu_backup,
        )

    def test_kv_cache_cpu_backup_basic_no_crash(self):
        """Engine should start, release KV cache to CPU, resume, and generate correctly."""
        self._skip_if_no_memory_saver()
        import sglang as sgl
        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

        engine = self._setup_engine(enable_kv_cache_cpu_backup=True)
        prompt = "Today is a sunny day and I like"
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        # Warm up - generate before sleep
        out_before = engine.generate(prompt, sampling_params)["text"]
        print(f"Output before sleep: {out_before!r}")
        self.assertIsInstance(out_before, str)
        self.assertGreater(len(out_before), 0)

        # Measure GPU memory before release
        mem_before = _get_gpu_memory_gb()

        # Sleep: release KV cache to CPU
        t0 = time.perf_counter()
        engine.release_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        mem_after_release = _get_gpu_memory_gb()
        print(
            f"Sleep took {time.perf_counter() - t0:.2f}s, "
            f"GPU memory: {mem_before:.2f} GB → {mem_after_release:.2f} GB"
        )

        # GPU memory should decrease after releasing KV cache
        self.assertLess(
            mem_after_release,
            mem_before,
            "GPU memory should decrease after releasing KV cache",
        )

        # Wakeup: resume KV cache from CPU
        t1 = time.perf_counter()
        engine.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        mem_after_resume = _get_gpu_memory_gb()
        print(
            f"Wakeup took {time.perf_counter() - t1:.2f}s, "
            f"GPU memory after resume: {mem_after_resume:.2f} GB"
        )

        # Generate after wakeup - should produce the same output as before
        out_after = engine.generate(prompt, sampling_params)["text"]
        print(f"Output after wakeup: {out_after!r}")
        self.assertEqual(
            out_before,
            out_after,
            "Output after wakeup should match output before sleep",
        )

        engine.shutdown()

    def test_kv_cache_cpu_backup_prefix_cache_restored(self):
        """After wakeup the prefix cache should be restored so cache hits fire."""
        self._skip_if_no_memory_saver()
        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

        engine = self._setup_engine(enable_kv_cache_cpu_backup=True)

        # Use a long prompt to ensure prefix cache population
        prompt = " ".join(str(i) for i in range(100))
        sampling_params = {"temperature": 0, "max_new_tokens": 4}

        # First generation populates the cache
        engine.generate(prompt, sampling_params)

        # Sleep
        engine.release_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])

        # Wakeup
        engine.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])

        # Second generation with the same prompt should hit the restored prefix cache.
        # The per-request "cached_tokens" in meta_info tells us how many tokens
        # were served from the prefix cache instead of being re-prefilled.
        result = engine.generate(prompt, sampling_params)
        cached_tokens = result.get("meta_info", {}).get("cached_tokens", 0)

        print(f"Cache hit (cached_tokens) after wakeup: {cached_tokens}")
        self.assertGreater(
            cached_tokens,
            0,
            "Prefix cache should have been restored after wakeup, "
            f"but cached_tokens={cached_tokens}. "
            "This means the radix tree snapshot was not properly imported.",
        )

        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
