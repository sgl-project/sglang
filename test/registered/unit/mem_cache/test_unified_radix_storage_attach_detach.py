"""Unit tests for runtime HiCache storage attach/detach on UnifiedRadixCache."""

import inspect
import unittest
from queue import Queue
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 64


class FakeHostPool:
    def __init__(self):
        self.freed = []
        self.page_size = PAGE_SIZE

    def free(self, indices):
        self.freed.append(indices)


class FakeHybridCacheController:
    """Stands in for HybridCacheController on the control path only."""

    def __init__(self, storage_backend=None):
        self.enable_storage = storage_backend is not None
        self.storage_backend_type = storage_backend
        self.write_policy = "write_through_selective"
        self.mem_pool_host = FakeHostPool()
        self.prefetch_revoke_queue = Queue()
        self.prefetch_hit_queue = Queue()
        self.ack_backup_queue = Queue()
        self.host_mem_release_queue = Queue()
        self.extra_host_mem_release_queues = {}
        self.prefetch_tokens_occupied = 0
        self.attach_calls = []
        self.detach_calls = 0
        self.released_extra_pools = []

    def attach_storage_backend(
        self,
        storage_backend,
        prefetch_threshold=256,
        model_name=None,
        storage_backend_extra_config=None,
        host_pools=None,
    ):
        if self.enable_storage:
            raise RuntimeError("Storage backend already attached.")
        self.attach_calls.append(
            {
                "storage_backend": storage_backend,
                "prefetch_threshold": prefetch_threshold,
                "model_name": model_name,
                "storage_backend_extra_config": storage_backend_extra_config,
                "host_pools": host_pools,
            }
        )
        self.storage_backend_type = storage_backend
        self.enable_storage = True

    def detach_storage_backend(self):
        self.detach_calls += 1
        self.storage_backend_type = None
        self.enable_storage = False

    def append_host_mem_release(self, host_indices=None, extra_pools=None):
        if host_indices is not None:
            self.host_mem_release_queue.put(host_indices)
        self.released_extra_pools.append(list(extra_pools or []))


class FakeTreeCore:
    """Holds the tree-owned state the UnifiedRadixCache facade delegates to."""

    def __init__(self):
        self.page_size = PAGE_SIZE
        self.enable_storage = False
        self.write_through_threshold = 2
        self.is_write_back = False
        self.dec_host_lock_ref_calls = []

    def dec_host_lock_ref(self, node_id, params=None):
        self.dec_host_lock_ref_calls.append((node_id, params))
        return None


def make_cache(storage_backend=None):
    """Build a UnifiedRadixCache carrying only the state the control path reads."""
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.disable = False
    cache.tree_core = FakeTreeCore()
    cache.tree_core.enable_storage = storage_backend is not None
    cache.cache_controller = FakeHybridCacheController(storage_backend)
    cache.host_pool_group = SimpleNamespace(entries=["kv-entry"])
    cache.ongoing_prefetch = {}
    cache.ongoing_backup = {}
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_stop_policy = "best_effort"
    cache.enable_storage_metrics = False
    cache.storage_metrics_collector = None
    cache._enable_metrics_flag = False
    cache.extra_metric_labels = None
    return cache


def make_hiradix_stub(storage_backend=None):
    """Duck-typed stand-in for HiRadixCache on the same control path.

    Used to compare UnifiedRadixCache's results against the reference
    implementation by calling the unbound HiRadixCache methods on it.
    """
    stub = SimpleNamespace(
        enable_storage=storage_backend is not None,
        enable_storage_metrics=False,
        prefetch_stop_policy="best_effort",
        write_through_threshold=2,
        cache_controller=SimpleNamespace(
            storage_backend_type=storage_backend,
            write_policy="write_through_selective",
            detach_storage_backend=lambda: None,
        ),
    )
    stub._drain_storage_control_queues_local = lambda: None
    stub._force_release_pending_storage_ops = lambda: None
    return stub


class TestUnifiedRadixStorageAttach(unittest.TestCase):
    def test_attach_without_storage_succeeds(self):
        cache = make_cache()
        self.assertFalse(cache.enable_storage)

        ok, msg = cache.attach_storage_backend(
            storage_backend="file",
            storage_backend_extra_config_json='{"prefetch_threshold": 128}',
            served_model_name="my-model",
            hicache_storage_prefetch_policy="timeout",
            hicache_write_policy="write_through",
        )

        self.assertTrue(ok, msg)
        self.assertTrue(cache.enable_storage)
        self.assertTrue(cache.cache_controller.enable_storage)
        self.assertEqual(cache.cache_controller.storage_backend_type, "file")
        self.assertEqual(len(cache.cache_controller.attach_calls), 1)
        call = cache.cache_controller.attach_calls[0]
        self.assertEqual(call["storage_backend"], "file")
        self.assertEqual(call["model_name"], "my-model")
        self.assertEqual(call["prefetch_threshold"], 128)
        self.assertEqual(call["storage_backend_extra_config"], {})
        self.assertEqual(call["host_pools"], ["kv-entry"])
        # Policies land on the tree and the controller.
        self.assertEqual(cache.prefetch_stop_policy, "timeout")
        self.assertEqual(cache.cache_controller.write_policy, "write_through")
        self.assertEqual(cache.write_through_threshold, 1)
        self.assertFalse(cache.is_write_back)
        # Runtime config derived from the parsed extra config.
        self.assertEqual(cache.prefetch_threshold, 128)
        self.assertEqual(cache.prefetch_timeout_base, 1.0)
        self.assertFalse(cache.hicache_storage_pass_prefix_keys)

    def test_attach_write_back_policy_sets_tree_flag(self):
        cache = make_cache()
        ok, _ = cache.attach_storage_backend(
            storage_backend="file",
            hicache_write_policy="write_back",
        )
        self.assertTrue(ok)
        self.assertTrue(cache.is_write_back)
        self.assertEqual(cache.write_through_threshold, 2)

    def test_attach_rejects_invalid_policies(self):
        cache = make_cache()
        ok, msg = cache.attach_storage_backend(
            storage_backend="file", hicache_storage_prefetch_policy="nope"
        )
        self.assertFalse(ok)
        self.assertIn("Invalid hicache_storage_prefetch_policy", msg)

        ok, msg = cache.attach_storage_backend(
            storage_backend="file", hicache_write_policy="nope"
        )
        self.assertFalse(ok)
        self.assertIn("Invalid hicache_write_policy", msg)
        # Nothing was attached by the rejected calls.
        self.assertFalse(cache.enable_storage)
        self.assertEqual(cache.cache_controller.attach_calls, [])

    def test_attach_same_backend_while_attached_matches_hiradix(self):
        cache = make_cache(storage_backend="file")
        unified = cache.attach_storage_backend(
            storage_backend="file",
            hicache_storage_prefetch_policy="wait_complete",
            hicache_write_policy="write_through",
        )
        hiradix = HiRadixCache.attach_storage_backend(
            make_hiradix_stub(storage_backend="file"),
            storage_backend="file",
            hicache_storage_prefetch_policy="wait_complete",
            hicache_write_policy="write_through",
        )

        self.assertEqual(unified, hiradix)
        self.assertTrue(unified[0])
        # Same-backend attach only refreshes policies, it does not re-attach.
        self.assertEqual(cache.cache_controller.attach_calls, [])
        self.assertEqual(cache.prefetch_stop_policy, "wait_complete")
        self.assertEqual(cache.write_through_threshold, 1)

    def test_attach_different_backend_while_attached_matches_hiradix(self):
        cache = make_cache(storage_backend="file")
        unified = cache.attach_storage_backend(storage_backend="mooncake")
        hiradix = HiRadixCache.attach_storage_backend(
            make_hiradix_stub(storage_backend="file"),
            storage_backend="mooncake",
        )

        self.assertEqual(unified, hiradix)
        self.assertFalse(unified[0])
        self.assertIn("Detach first", unified[1])
        self.assertTrue(cache.enable_storage)
        self.assertEqual(cache.cache_controller.storage_backend_type, "file")

    def test_attach_reports_controller_failure(self):
        cache = make_cache()

        def boom(**kwargs):
            raise RuntimeError("backend unavailable")

        cache.cache_controller.attach_storage_backend = boom
        ok, msg = cache.attach_storage_backend(storage_backend="file")
        self.assertFalse(ok)
        self.assertIn("backend unavailable", msg)
        self.assertFalse(cache.enable_storage)

    def test_attach_reports_bad_extra_config_json(self):
        cache = make_cache()
        ok, msg = cache.attach_storage_backend(
            storage_backend="file", storage_backend_extra_config_json="{not json"
        )
        self.assertFalse(ok)
        self.assertIn("Failed to parse storage_backend_extra_config_json", msg)
        self.assertFalse(cache.enable_storage)


class TestUnifiedRadixStorageDetach(unittest.TestCase):
    def test_detach_succeeds_and_leaves_tree_usable(self):
        cache = make_cache(storage_backend="file")
        cache.enable_storage_metrics = True
        cache.storage_metrics_collector = object()

        ok, msg = cache.detach_storage_backend()

        self.assertTrue(ok, msg)
        self.assertEqual(cache.cache_controller.detach_calls, 1)
        self.assertFalse(cache.enable_storage)
        self.assertFalse(cache.enable_storage_metrics)
        self.assertIsNone(cache.storage_metrics_collector)
        # The tree stays usable without storage: attach works again afterwards.
        ok, msg = cache.attach_storage_backend(storage_backend="file")
        self.assertTrue(ok, msg)
        self.assertTrue(cache.enable_storage)

    def test_detach_releases_pending_storage_ops(self):
        cache = make_cache(storage_backend="file")
        host_indices = torch.arange(4)
        cache.ongoing_prefetch["req-0"] = _OngoingPrefetch(
            anchor_node_id=7,
            prefetch_key=[1, 2, 3, 4],
            host_indices=host_indices,
            operation=SimpleNamespace(),
            anchor_lock_params=None,
            comp_xfers={},
        )
        cache.ongoing_backup[11] = (9, None)
        cache.prefetch_loaded_tokens_by_reqid["req-0"] = 4
        cache.cache_controller.prefetch_tokens_occupied = 4

        ok, msg = cache.detach_storage_backend()

        self.assertTrue(ok, msg)
        self.assertEqual(cache.ongoing_prefetch, {})
        self.assertEqual(cache.ongoing_backup, {})
        self.assertEqual(cache.prefetch_loaded_tokens_by_reqid, {})
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)
        self.assertEqual(len(cache.cache_controller.mem_pool_host.freed), 1)
        self.assertTrue(
            torch.equal(cache.cache_controller.mem_pool_host.freed[0], host_indices)
        )
        # Host locks of the anchor node and the pending backup were dropped.
        self.assertIn((7, None), cache.tree_core.dec_host_lock_ref_calls)
        self.assertIn((9, None), cache.tree_core.dec_host_lock_ref_calls)

    def test_detach_without_storage_matches_hiradix(self):
        cache = make_cache()
        unified = cache.detach_storage_backend()
        hiradix = HiRadixCache.detach_storage_backend(make_hiradix_stub())

        self.assertEqual(unified, hiradix)
        self.assertTrue(unified[0])
        self.assertFalse(cache.enable_storage)
        # Idempotent: the controller is asked to clean up regardless.
        self.assertEqual(cache.cache_controller.detach_calls, 1)

    def test_detach_reports_controller_failure(self):
        cache = make_cache(storage_backend="file")

        def boom():
            raise RuntimeError("threads still alive")

        cache.cache_controller.detach_storage_backend = boom
        ok, msg = cache.detach_storage_backend()
        self.assertFalse(ok)
        self.assertIn("threads still alive", msg)
        # Storage stays marked enabled so the caller can retry the detach.
        self.assertTrue(cache.enable_storage)


class TestUnifiedRadixStorageStubsRemoved(unittest.TestCase):
    def test_no_unsupported_stub_messages(self):
        for method in (
            UnifiedRadixCache.attach_storage_backend,
            UnifiedRadixCache.detach_storage_backend,
        ):
            source = inspect.getsource(method)
            self.assertNotIn("does not support runtime HiCache storage", source)


if __name__ == "__main__":
    unittest.main()
