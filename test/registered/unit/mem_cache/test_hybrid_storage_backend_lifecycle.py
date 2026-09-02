"""Unit tests for hybrid HiCache storage-backend registration order."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _RecordingBackend:
    def __init__(self, events, *, fail_finalize=False):
        self.events = events
        self.fail_finalize = fail_finalize

    def register_mem_pool_host(self, host_pool):
        self.events.append(("anchor", host_pool))

    def register_mem_host_pool_v2(self, host_pool, host_pool_name):
        self.events.append(("pool", host_pool_name, host_pool))

    def finalize_mem_pool_registration(self):
        self.events.append(("finalize",))
        if self.fail_finalize:
            raise RuntimeError("finalize failed")

    def close(self):
        self.events.append(("close",))


class TestHybridStorageBackendLifecycle(unittest.TestCase):
    def _controller(self, events):
        anchor = object()
        sidecar = object()
        entries = [
            SimpleNamespace(name=PoolName.KV, host_pool=anchor),
            SimpleNamespace(name=PoolName.INDEXER, host_pool=sidecar),
        ]

        controller = object.__new__(HybridCacheController)
        controller.enable_storage = False
        controller.storage_backend = None
        controller.storage_backend_type = None
        controller.prefetch_hits_sync_groups = []
        controller.prefetch_completion_sync_groups = []
        controller.storage_stop_event = threading.Event()
        controller.storage_host_pool = anchor
        controller.mem_pool_host = SimpleNamespace(size=32, entries=entries)
        controller.host_memory_mode = "cache"
        controller.page_size = 64
        controller._generate_storage_config = Mock(
            return_value=SimpleNamespace(
                is_mla_model=False,
                tp_rank=0,
                extra_config={},
            )
        )
        controller._create_sync_groups = Mock(return_value=[])
        controller._destroy_sync_groups = Mock()
        controller._stop_storage_threads = Mock(
            side_effect=lambda: events.append(("stop",))
        )
        controller._start_storage_threads = Mock(
            side_effect=lambda: events.append(("start",))
        )
        return controller, anchor, sidecar, entries

    def test_registration_and_finalize_happen_before_thread_start(self):
        events = []
        controller, anchor, sidecar, entries = self._controller(events)
        backend = _RecordingBackend(events)

        with patch.object(
            StorageBackendFactory, "create_backend", return_value=backend
        ):
            controller.attach_storage_backend(
                storage_backend="recording",
                host_pools=entries,
            )

        self.assertEqual(
            events,
            [
                ("stop",),
                ("anchor", anchor),
                ("pool", PoolName.KV, anchor),
                ("pool", PoolName.INDEXER, sidecar),
                ("finalize",),
                ("start",),
            ],
        )

    def test_finalize_failure_rolls_back_without_starting_threads(self):
        events = []
        controller, _anchor, _sidecar, entries = self._controller(events)
        backend = _RecordingBackend(events, fail_finalize=True)

        with (
            patch.object(StorageBackendFactory, "create_backend", return_value=backend),
            self.assertRaisesRegex(RuntimeError, "finalize failed"),
        ):
            controller.attach_storage_backend(
                storage_backend="recording",
                host_pools=entries,
            )

        self.assertNotIn(("start",), events)
        self.assertEqual(events[-2:], [("stop",), ("close",)])
        self.assertIsNone(controller.storage_backend)
        self.assertIsNone(controller.storage_backend_type)
        self.assertFalse(controller.enable_storage)


if __name__ == "__main__":
    unittest.main()
