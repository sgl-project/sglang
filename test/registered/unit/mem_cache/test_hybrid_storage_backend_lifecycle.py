"""Unit tests for hybrid HiCache storage-backend registration order."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.managers.cache_controller import StorageLifecycleConsensusError
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _RecordingBackend:
    def __init__(self, events, *, fail_finalize=False, fail_close=False):
        self.events = events
        self.fail_finalize = fail_finalize
        self.fail_close = fail_close

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
        if self.fail_close:
            raise RuntimeError("close failed")


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
        controller.storage_lifecycle_failed = False
        controller.prefetch_hits_sync_groups = []
        controller.prefetch_completion_sync_groups = []
        controller.storage_data_sync_groups_initialized = True
        controller.storage_lifecycle_sync_groups = []
        controller.storage_lifecycle_groups_initialized = True
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
        controller._create_lifecycle_sync_groups = Mock(return_value=[])
        controller._destroy_sync_groups = Mock()
        controller._storage_detach_consensus = Mock(side_effect=lambda local: local)
        controller._stop_storage_threads = Mock(
            side_effect=lambda: events.append(("stop",))
        )
        controller._start_storage_threads = Mock(
            side_effect=lambda: events.append(("start",))
        )
        controller._generic_page_get = object()
        controller._generic_page_set = object()
        controller.page_get_func = object()
        controller.page_set_func = object()
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
                coordinated_lifecycle=True,
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

    def test_lifecycle_groups_copy_replica_topology(self):
        events = []
        controller, _anchor, _sidecar, _entries = self._controller(events)
        replica_groups = [object(), object(), object()]
        controller._create_sync_groups = Mock(return_value=replica_groups)

        groups = HybridCacheController._create_lifecycle_sync_groups(controller)

        self.assertEqual(groups, replica_groups)
        controller._create_sync_groups.assert_called_once_with()

    def test_data_sync_groups_are_precreated_once(self):
        events = []
        controller, _anchor, _sidecar, _entries = self._controller(events)
        hit_group = object()
        completion_group = object()
        controller.storage_data_sync_groups_initialized = False
        controller._create_sync_groups = Mock(
            side_effect=[[hit_group], [completion_group]]
        )

        controller.initialize_storage_data_sync_groups()
        controller.initialize_storage_data_sync_groups()

        self.assertEqual(controller.prefetch_hits_sync_groups, [hit_group])
        self.assertEqual(controller.prefetch_completion_sync_groups, [completion_group])
        self.assertEqual(controller._create_sync_groups.call_count, 2)

    def test_attach_preparation_failure_participates_in_consensus(self):
        events = []
        controller, _anchor, _sidecar, entries = self._controller(events)
        lifecycle_group = object()
        controller.storage_lifecycle_sync_groups = [lifecycle_group]
        controller._stop_storage_threads = Mock(
            side_effect=RuntimeError("thread stop failed")
        )
        controller._storage_detach_consensus = Mock(return_value=False)

        with self.assertRaisesRegex(RuntimeError, "preparation failed.*restart"):
            controller.attach_storage_backend(
                storage_backend="recording",
                host_pools=entries,
                coordinated_lifecycle=True,
            )

        self.assertEqual(controller.storage_lifecycle_sync_groups, [lifecycle_group])
        controller._storage_detach_consensus.assert_called_once_with(False)
        self.assertTrue(controller.storage_lifecycle_failed)

    def test_finalize_failure_retains_possibly_registered_memory(self):
        events = []
        controller, _anchor, _sidecar, entries = self._controller(events)
        backend = _RecordingBackend(events, fail_finalize=True)

        with (
            patch.object(StorageBackendFactory, "create_backend", return_value=backend),
            self.assertRaisesRegex(RuntimeError, "registration failed.*restart"),
        ):
            controller.attach_storage_backend(
                storage_backend="recording",
                host_pools=entries,
            )

        self.assertNotIn(("start",), events)
        self.assertNotIn(("close",), events)
        self.assertIs(controller.storage_backend, backend)
        self.assertEqual(controller.storage_backend_type, "recording")
        self.assertFalse(controller.enable_storage)
        self.assertTrue(controller.storage_lifecycle_failed)

    def test_post_registration_activation_failure_retains_backend_and_groups(self):
        events = []
        controller, _anchor, _sidecar, entries = self._controller(events)
        backend = _RecordingBackend(events, fail_close=True)
        hit_group = object()
        completion_group = object()
        lifecycle_group = object()
        controller.prefetch_hits_sync_groups = [hit_group]
        controller.prefetch_completion_sync_groups = [completion_group]
        controller.storage_lifecycle_sync_groups = [lifecycle_group]

        def fail_start():
            events.append(("start",))
            raise RuntimeError("thread start failed")

        controller._start_storage_threads = Mock(side_effect=fail_start)

        with (
            patch.object(StorageBackendFactory, "create_backend", return_value=backend),
            self.assertRaisesRegex(RuntimeError, "activation failed.*restart"),
        ):
            controller.attach_storage_backend(
                storage_backend="recording",
                host_pools=entries,
            )

        self.assertNotIn(("close",), events)
        self.assertIs(controller.storage_backend, backend)
        self.assertEqual(controller.storage_backend_type, "recording")
        self.assertEqual(controller.prefetch_hits_sync_groups, [hit_group])
        self.assertEqual(controller.prefetch_completion_sync_groups, [completion_group])
        self.assertEqual(controller.storage_lifecycle_sync_groups, [lifecycle_group])
        self.assertFalse(controller.enable_storage)
        self.assertTrue(controller.storage_lifecycle_failed)
        self.assertIs(controller.page_get_func, controller._generic_page_get)
        self.assertIs(controller.page_set_func, controller._generic_page_set)

        with self.assertRaisesRegex(RuntimeError, "restart.*required"):
            controller.attach_storage_backend(
                storage_backend="recording",
                host_pools=entries,
            )

    def test_close_failure_keeps_backend_and_groups_mapped_for_process_reclaim(self):
        events = []
        controller, _anchor, _sidecar, _entries = self._controller(events)
        backend = _RecordingBackend(events, fail_close=True)
        hit_group = object()
        completion_group = object()
        controller.storage_backend = backend
        controller.storage_backend_type = "recording"
        controller.enable_storage = True
        controller.prefetch_hits_sync_groups = [hit_group]
        controller.prefetch_completion_sync_groups = [completion_group]
        controller.storage_lifecycle_sync_groups = [object()]

        with self.assertRaisesRegex(RuntimeError, "close failed"):
            controller.detach_storage_backend(coordinated_lifecycle=True)

        self.assertEqual(events, [("stop",), ("close",)])
        controller._destroy_sync_groups.assert_not_called()
        self.assertIs(controller.storage_backend, backend)
        self.assertEqual(controller.storage_backend_type, "recording")
        self.assertFalse(controller.enable_storage)
        self.assertIs(controller.page_get_func, controller._generic_page_get)
        self.assertIs(controller.page_set_func, controller._generic_page_set)
        self.assertEqual(controller.prefetch_hits_sync_groups, [hit_group])
        self.assertEqual(controller.prefetch_completion_sync_groups, [completion_group])
        self.assertEqual(len(controller.storage_lifecycle_sync_groups), 1)

    def test_reset_refuses_to_wake_threads_after_lifecycle_failure(self):
        events = []
        controller, _anchor, _sidecar, _entries = self._controller(events)
        controller.storage_lifecycle_failed = True
        controller.storage_stop_event.set()
        controller.write_queue = [object()]
        controller.load_queue = [object()]
        controller.ack_write_queue = [object()]
        controller.ack_load_queue = [object()]

        with self.assertRaisesRegex(RuntimeError, "restart.*required"):
            controller.reset()

        self.assertTrue(controller.storage_stop_event.is_set())
        self.assertEqual(len(controller.write_queue), 1)
        self.assertEqual(len(controller.load_queue), 1)

    def test_successful_detach_retains_startup_created_sync_groups(self):
        events = []
        controller, _anchor, _sidecar, _entries = self._controller(events)
        backend = _RecordingBackend(events)
        hit_group = object()
        completion_group = object()
        lifecycle_group = object()
        controller.storage_backend = backend
        controller.storage_backend_type = "recording"
        controller.enable_storage = True
        controller.prefetch_hits_sync_groups = [hit_group]
        controller.prefetch_completion_sync_groups = [completion_group]
        controller.storage_lifecycle_sync_groups = [lifecycle_group]

        controller.detach_storage_backend()

        self.assertEqual(events, [("stop",), ("close",)])
        controller._destroy_sync_groups.assert_not_called()
        self.assertIsNone(controller.storage_backend)
        self.assertEqual(controller.prefetch_hits_sync_groups, [hit_group])
        self.assertEqual(controller.prefetch_completion_sync_groups, [completion_group])
        self.assertEqual(controller.storage_lifecycle_sync_groups, [lifecycle_group])

    def test_publication_peer_failure_stops_and_retains_backend(self):
        events = []
        controller, _anchor, _sidecar, _entries = self._controller(events)
        backend = _RecordingBackend(events)
        controller.storage_backend = backend
        controller.storage_backend_type = "recording"
        controller.enable_storage = True
        controller._storage_detach_consensus = Mock(side_effect=[False, True])

        published = controller.finalize_storage_attach(local_success=True)

        self.assertFalse(published)
        self.assertEqual(events, [("stop",)])
        self.assertIs(controller.storage_backend, backend)
        self.assertTrue(controller.storage_lifecycle_failed)
        self.assertFalse(controller.enable_storage)

    def test_peer_close_failure_keeps_every_rank_from_committing_detach(self):
        """A locally closed TP rank must retain shared lifecycle state."""
        events = []
        controller, _anchor, _sidecar, _entries = self._controller(events)
        backend = _RecordingBackend(events)
        hit_group = object()
        completion_group = object()
        controller.storage_backend = backend
        controller.storage_backend_type = "recording"
        controller.enable_storage = True
        controller.prefetch_hits_sync_groups = [hit_group]
        controller.prefetch_completion_sync_groups = [completion_group]
        controller.storage_lifecycle_sync_groups = [object()]
        controller._storage_detach_consensus = Mock(side_effect=[True, False])

        with self.assertRaisesRegex(RuntimeError, "peer rank.*restart.*required"):
            controller.detach_storage_backend(coordinated_lifecycle=True)

        self.assertEqual(events, [("stop",), ("close",)])
        self.assertEqual(
            controller._storage_detach_consensus.call_args_list,
            [
                unittest.mock.call(True),
                unittest.mock.call(True),
            ],
        )
        controller._destroy_sync_groups.assert_not_called()
        self.assertIs(controller.storage_backend, backend)
        self.assertEqual(controller.storage_backend_type, "recording")
        self.assertFalse(controller.enable_storage)
        self.assertEqual(controller.prefetch_hits_sync_groups, [hit_group])
        self.assertEqual(controller.prefetch_completion_sync_groups, [completion_group])
        self.assertEqual(len(controller.storage_lifecycle_sync_groups), 1)

    def test_detach_consensus_uses_the_dedicated_lifecycle_groups(self):
        events = []
        controller, _anchor, _sidecar, _entries = self._controller(events)
        data_group = object()
        lifecycle_group = object()
        controller.prefetch_hits_sync_groups = [data_group]
        controller.storage_lifecycle_sync_groups = [lifecycle_group]

        def report_peer_failure(tensor, op, groups):
            self.assertIs(op, torch.distributed.ReduceOp.MIN)
            self.assertEqual(groups, [lifecycle_group])
            tensor.zero_()

        controller._all_reduce = Mock(side_effect=report_peer_failure)

        agreed = HybridCacheController._storage_detach_consensus(controller, True)

        self.assertFalse(agreed)
        controller._all_reduce.assert_called_once()

    def test_detach_consensus_fault_requires_worker_termination(self):
        events = []
        controller, _anchor, _sidecar, _entries = self._controller(events)
        controller.enable_storage = True
        controller.storage_lifecycle_sync_groups = [object()]
        controller._all_reduce = Mock(side_effect=RuntimeError("gloo failed"))

        with self.assertRaisesRegex(
            StorageLifecycleConsensusError, "termination.*required"
        ):
            HybridCacheController._storage_detach_consensus(controller, True)

        self.assertFalse(controller.enable_storage)
        self.assertIs(controller.page_get_func, controller._generic_page_get)
        self.assertIs(controller.page_set_func, controller._generic_page_set)

    def test_attach_refuses_a_backend_retained_after_close_failure(self):
        events = []
        controller, _anchor, _sidecar, entries = self._controller(events)
        retained = _RecordingBackend(events)
        controller.storage_backend = retained
        controller.storage_backend_type = "recording"

        with (
            patch.object(StorageBackendFactory, "create_backend") as create_backend,
            self.assertRaisesRegex(RuntimeError, "restart.*required"),
        ):
            controller.attach_storage_backend(
                storage_backend="recording",
                host_pools=entries,
            )

        create_backend.assert_not_called()
        self.assertIs(controller.storage_backend, retained)

    def test_detach_failure_publishes_a_disabled_data_path(self):
        events = []
        controller = SimpleNamespace(
            enable_storage=True,
            prepare_storage_lifecycle=Mock(return_value=True),
        )

        def fail_after_disable(**_kwargs):
            controller.enable_storage = False
            raise RuntimeError("backend close failed; restart required")

        controller.detach_storage_backend = fail_after_disable
        cache = SimpleNamespace(
            cache_controller=controller,
            enable_storage=True,
            enable_storage_metrics=True,
            drain_storage_control_queues_local=lambda: events.append("drain"),
        )
        attachment = StorageAttachment(cache)

        detached, message = attachment.detach()

        self.assertFalse(detached)
        self.assertIn("restart required", message)
        self.assertFalse(cache.enable_storage)
        self.assertFalse(cache.enable_storage_metrics)

    def test_attachment_does_not_downgrade_consensus_failure(self):
        controller = SimpleNamespace(
            enable_storage=False,
            prepare_storage_lifecycle=Mock(return_value=True),
            detach_storage_backend=Mock(
                side_effect=StorageLifecycleConsensusError("gloo failed")
            ),
        )
        cache = SimpleNamespace(
            cache_controller=controller,
            enable_storage=True,
            enable_storage_metrics=True,
            drain_storage_control_queues_local=Mock(),
        )
        attachment = StorageAttachment(cache)

        with self.assertRaisesRegex(StorageLifecycleConsensusError, "gloo failed"):
            attachment.detach()

    def test_detach_drain_failure_is_consensed_before_backend_close(self):
        controller = SimpleNamespace(
            enable_storage=True,
            prepare_storage_lifecycle=Mock(side_effect=[True, False]),
            detach_storage_backend=Mock(),
        )
        cache = SimpleNamespace(
            cache_controller=controller,
            enable_storage=True,
            enable_storage_metrics=True,
            drain_storage_control_queues_local=Mock(
                side_effect=RuntimeError("drain failed")
            ),
        )
        attachment = StorageAttachment(cache)

        detached, message = attachment.detach()

        self.assertFalse(detached)
        self.assertIn("drain failed", message)
        self.assertEqual(
            controller.prepare_storage_lifecycle.call_args_list,
            [unittest.mock.call(True), unittest.mock.call(False)],
        )
        controller.detach_storage_backend.assert_not_called()

    def test_process_exit_detach_is_local_and_skips_collectives(self):
        controller = SimpleNamespace(
            enable_storage=True,
            prepare_storage_lifecycle=Mock(
                side_effect=AssertionError("must not enter a collective")
            ),
            detach_storage_backend=Mock(),
        )
        cache = SimpleNamespace(
            cache_controller=controller,
            enable_storage=True,
            enable_storage_metrics=True,
            drain_storage_control_queues_local=Mock(),
        )
        attachment = StorageAttachment(cache)
        attachment._release_pending_storage_ops = Mock()

        detached, _message = attachment.detach(coordinated_lifecycle=False)

        self.assertTrue(detached)
        controller.prepare_storage_lifecycle.assert_not_called()
        controller.detach_storage_backend.assert_called_once_with(
            coordinated_lifecycle=False
        )

    def test_publication_failure_is_consensed_and_disables_the_tree(self):
        controller = SimpleNamespace(
            enable_storage=False,
            storage_backend_type=None,
            mem_pool_host=SimpleNamespace(entries=[]),
            prepare_storage_lifecycle=Mock(return_value=True),
            attach_storage_backend=Mock(),
            finalize_storage_attach=Mock(return_value=False),
        )
        cache = SimpleNamespace(
            cache_controller=controller,
            enable_storage=False,
            enable_storage_metrics=False,
            _enable_metrics_flag=False,
            extra_metric_labels=None,
        )
        attachment = StorageAttachment(cache)
        attachment.apply_runtime_config = Mock(
            side_effect=RuntimeError("hash backfill failed")
        )

        attached, message = attachment.attach(
            storage_backend="recording",
            local_ready=True,
        )

        self.assertFalse(attached)
        self.assertIn("publication failed", message)
        self.assertEqual(
            controller.prepare_storage_lifecycle.call_args_list,
            [unittest.mock.call(True), unittest.mock.call(True)],
        )
        controller.finalize_storage_attach.assert_called_once_with(False)
        self.assertFalse(cache.enable_storage)
        self.assertFalse(cache.enable_storage_metrics)

    def test_attach_parse_failure_participates_in_consensus(self):
        controller = SimpleNamespace(
            enable_storage=False,
            storage_backend_type=None,
            mem_pool_host=SimpleNamespace(entries=[]),
            prepare_storage_lifecycle=Mock(side_effect=[True, False]),
            attach_storage_backend=Mock(),
        )
        cache = SimpleNamespace(
            cache_controller=controller,
            enable_storage=False,
            enable_storage_metrics=False,
        )
        attachment = StorageAttachment(cache)

        with patch.object(
            HybridCacheController,
            "parse_storage_backend_extra_config",
            side_effect=RuntimeError("missing @file"),
        ):
            attached, message = attachment.attach(
                storage_backend="recording",
                storage_backend_extra_config_json="@/rank-local/config.json",
            )

        self.assertFalse(attached)
        self.assertIn("missing @file", message)
        self.assertEqual(
            controller.prepare_storage_lifecycle.call_args_list,
            [unittest.mock.call(True), unittest.mock.call(False)],
        )
        controller.attach_storage_backend.assert_not_called()

    def test_host_pools_are_not_destroyed_when_storage_detach_refuses(self):
        cache = SimpleNamespace(
            enable_storage=True,
            linker=Mock(),
            host_pool_group=Mock(),
        )
        attachment = StorageAttachment(cache)
        attachment.detach = Mock(
            return_value=(
                False,
                "KVCR still owns registered memory",
            )
        )

        with self.assertRaisesRegex(RuntimeError, "still owns registered memory"):
            attachment.release_host_resources()

        attachment.detach.assert_called_once_with(coordinated_lifecycle=False)
        cache.linker.close.assert_not_called()
        cache.host_pool_group.destroy.assert_not_called()

    def test_host_resources_are_released_only_after_successful_detach(self):
        events = []
        cache = SimpleNamespace(
            enable_storage=True,
            linker=Mock(side_effect=lambda: events.append("linker")),
            host_pool_group=Mock(side_effect=lambda: events.append("pool")),
        )
        cache.linker.close.side_effect = lambda: events.append("linker")
        cache.host_pool_group.destroy.side_effect = lambda: events.append("pool")
        attachment = StorageAttachment(cache)
        attachment.detach = Mock(
            side_effect=lambda **_kwargs: (
                events.append("detach")
                or (
                    True,
                    "detached",
                )
            )
        )

        attachment.release_host_resources()

        self.assertEqual(events, ["detach", "linker", "pool"])
        attachment.detach.assert_called_once_with(coordinated_lifecycle=False)


if __name__ == "__main__":
    unittest.main()
