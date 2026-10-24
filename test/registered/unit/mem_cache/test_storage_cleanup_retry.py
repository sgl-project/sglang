"""A storage backend whose close() fails must stay attached to the controller
until a retried detach succeeds, and that failure must reach the admin API."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.managers.io_struct import DetachHiCacheStorageReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RetryCloseBackend:
    requires_explicit_memory_cleanup = True

    def __init__(self):
        self.fail_close = True
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("toy memory unregister failed")

    def register_mem_pool_host(self, pool):
        pass

    def register_mem_host_pool_v2(self, pool, name):
        pass

    def on_host_pools_registered(self):
        pass


def _controller(backend):
    pool = SimpleNamespace(size=16, logical_size=16)
    controller = SimpleNamespace(
        storage_backend=backend,
        storage_backend_type="dynamic",
        enable_storage=True,
        prefetch_hits_sync_groups=[],
        prefetch_completion_sync_groups=[],
        _stop_storage_threads=Mock(),
        _destroy_sync_groups=Mock(),
        _create_sync_groups=Mock(return_value=[]),
        _start_storage_threads=Mock(),
        storage_stop_event=threading.Event(),
        storage_host_pool=pool,
        mem_pool_host=pool,
        host_memory_mode="cache",
        page_size=1,
        write_policy="write_through",
        _generate_storage_config=Mock(return_value=SimpleNamespace(extra_config={})),
        _generic_page_get=object(),
        _generic_page_set=object(),
    )
    for name in (
        "attach_storage_backend",
        "detach_storage_backend",
        "_close_storage_backend",
        "_storage_host_pools",
    ):
        method = getattr(HiCacheController, name)
        setattr(controller, name, lambda *a, _m=method, **kw: _m(controller, *a, **kw))
    return controller


def _fixture():
    backend = _RetryCloseBackend()
    controller = _controller(backend)
    cache = SimpleNamespace(
        cache_controller=controller,
        enable_storage=True,
        enable_storage_metrics=True,
        drain_storage_control_queues_local=Mock(),
        is_write_back=False,
        host_memory_mode="cache",
        sliding_window_size=0,
        prefetch_stop_policy="best_effort",
        write_through_threshold=1,
        _enable_metrics_flag=False,
        extra_metric_labels=None,
        page_size=1,
        tree_core=SimpleNamespace(backfill_missing_hash_values=Mock(return_value=0)),
    )
    attachment = StorageAttachment(cache)
    attachment._release_pending_storage_ops = Mock()
    return attachment, cache, controller, backend


class TestStorageCleanupRetry(CustomTestCase):
    def setUp(self):
        super().setUp()
        parallel = patch(
            "sglang.srt.managers.cache_controller.get_parallel",
            return_value=SimpleNamespace(attn_dcp_size=1),
        )
        parallel.start()
        self.addCleanup(parallel.stop)

    def test_close_failure_blocks_reattach_until_a_retry_succeeds(self):
        attachment, cache, controller, backend = _fixture()
        self.assertFalse(attachment.detach()[0])
        self.assertFalse(cache.enable_storage)
        self.assertFalse(controller.enable_storage)
        self.assertIs(controller.storage_backend, backend)
        attachment._release_pending_storage_ops.assert_not_called()
        self.assertIn("retry detach", attachment.attach("dynamic")[1])
        # The controller itself refuses too, so a new attachment object cannot bypass it.
        self.assertFalse(StorageAttachment(cache).attach("dynamic")[0])
        controller._start_storage_threads.assert_not_called()

        backend.fail_close = False
        self.assertTrue(attachment.detach()[0])
        self.assertIsNone(controller.storage_backend)
        replacement = _RetryCloseBackend()
        with patch.object(
            StorageBackendFactory, "create_backend", return_value=replacement
        ):
            ok, message = attachment.attach("dynamic")
        self.assertTrue(ok, message)
        self.assertTrue(controller.enable_storage)
        self.assertIs(controller.storage_backend, replacement)
        controller._start_storage_threads.assert_called_once()

    def test_thread_stop_failure_keeps_storage_enabled(self):
        attachment, cache, controller, backend = _fixture()
        controller._stop_storage_threads.side_effect = RuntimeError("worker alive")
        self.assertFalse(attachment.detach()[0])
        self.assertTrue(cache.enable_storage)
        self.assertTrue(controller.enable_storage)
        self.assertEqual(backend.close_calls, 0)
        attachment._release_pending_storage_ops.assert_not_called()

    def test_stop_failure_rejects_same_backend_policy_update_until_detach(self):
        attachment, cache, controller, backend = _fixture()
        worker = Mock(name="storage-worker")
        worker.is_alive.return_value = True
        controller.prefetch_thread = worker
        controller._stop_storage_threads = lambda: (
            HiCacheController._stop_storage_threads(controller)
        )
        self.assertFalse(attachment.detach()[0])
        self.assertTrue(controller.storage_stop_event.is_set())
        ok, message = attachment.attach(
            "dynamic", hicache_storage_prefetch_policy="wait_complete"
        )
        self.assertFalse(ok)
        self.assertIn("retry detach", message)
        self.assertEqual(cache.prefetch_stop_policy, "best_effort")
        worker.is_alive.return_value = False
        backend.fail_close = False
        self.assertTrue(attachment.detach()[0])
        self.assertFalse(controller.storage_stop_event.is_set())

    def test_failed_attach_cleanup_is_retried_on_shutdown(self):
        attachment, cache, controller, backend = _fixture()
        cache.enable_storage = False
        controller.enable_storage = False
        controller.storage_backend = None
        backend.register_mem_pool_host = Mock(side_effect=RuntimeError("registration"))
        with patch.object(
            StorageBackendFactory, "create_backend", return_value=backend
        ):
            self.assertFalse(attachment.attach("dynamic")[0])
        self.assertIs(controller.storage_backend, backend)
        self.assertEqual(backend.close_calls, 1)
        self.assertFalse(cache.enable_storage)
        self.assertIn("retry detach", attachment.attach("dynamic")[1])

        backend.fail_close = False
        attachment.shutdown()
        self.assertEqual(backend.close_calls, 2)
        self.assertIsNone(controller.storage_backend)

    def test_attach_rollback_keeps_backend_and_groups_until_threads_stop(self):
        attachment, cache, controller, backend = _fixture()
        cache.enable_storage = False
        controller.enable_storage = False
        controller.storage_backend = None
        backend.fail_close = False
        controller._create_sync_groups.side_effect = [["hits"], ["completion"]]
        controller._start_storage_threads.side_effect = RuntimeError("partial startup")
        controller._stop_storage_threads.side_effect = [
            None,  # pre-attach check
            RuntimeError("worker alive"),  # rollback must retain resources
            None,  # shutdown retries after the worker has stopped
        ]
        with patch.object(
            StorageBackendFactory, "create_backend", return_value=backend
        ):
            ok, message = attachment.attach("dynamic")
        self.assertFalse(ok)
        self.assertIn("attach rollback", message)
        self.assertIs(controller.storage_backend, backend)
        self.assertEqual(backend.close_calls, 0)
        controller._destroy_sync_groups.assert_not_called()
        self.assertEqual(controller.prefetch_hits_sync_groups, ["hits"])
        self.assertEqual(controller.prefetch_completion_sync_groups, ["completion"])
        self.assertTrue(controller.enable_storage)
        self.assertFalse(cache.enable_storage)
        self.assertIn("retry detach", attachment.attach("dynamic")[1])

        attachment.shutdown()
        controller._destroy_sync_groups.assert_called_once_with(["hits", "completion"])
        self.assertEqual(backend.close_calls, 1)
        self.assertIsNone(controller.storage_backend)

    def test_hybrid_rollback_retains_pp_group_until_worker_stop_retry(self):
        backend = _RetryCloseBackend()
        backend.fail_close = False
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.__dict__.update(_controller(None).__dict__)
        # Bind the base helpers to the real Hybrid instance, not the fixture.
        for name in ("_close_storage_backend", "_storage_host_pools"):
            controller.__dict__.pop(name)
        controller.enable_storage = False
        controller.pp_group = None
        pp_group = controller.pp_prefetch_command_group = object()
        controller.pp_prefetch_state_lock = threading.Lock()
        controller.pp_prefetch_states = {}
        controller.pp_prefetch_decisions = {}
        controller._start_storage_threads.side_effect = RuntimeError("partial startup")
        controller._stop_storage_threads.side_effect = [
            None,
            RuntimeError("worker alive"),
            None,
        ]
        with (
            patch.object(StorageBackendFactory, "create_backend", return_value=backend),
            patch("torch.distributed.destroy_process_group") as destroy,
        ):
            with self.assertRaisesRegex(RuntimeError, "attach rollback"):
                HybridCacheController.attach_storage_backend(controller, "dynamic")
            destroy.assert_not_called()
            self.assertIs(controller.pp_prefetch_command_group, pp_group)
            self.assertIs(controller.storage_backend, backend)
            self.assertEqual(backend.close_calls, 0)
            HybridCacheController.detach_storage_backend(controller)
            destroy.assert_called_once_with(pp_group)
        self.assertIsNone(controller.pp_prefetch_command_group)
        self.assertIsNone(controller.storage_backend)
        self.assertEqual(backend.close_calls, 1)

    def test_bookkeeping_failure_requires_restart_even_on_shutdown(self):
        attachment, cache, controller, backend = _fixture()
        backend.fail_close = False
        attachment._release_pending_storage_ops.side_effect = [
            RuntimeError("pending"),
            None,
        ]
        self.assertFalse(attachment.detach()[0])
        self.assertIsNone(controller.storage_backend)
        self.assertFalse(attachment.attach("dynamic")[0])
        attachment.shutdown()
        self.assertEqual(attachment._release_pending_storage_ops.call_count, 1)
        ok, message = attachment.detach()
        self.assertFalse(ok)
        self.assertIn("restart the model process", message)

    def test_failed_backup_unlock_keeps_its_record_without_replaying(self):
        attachment, cache, _, _ = _fixture()
        cache.ongoing_prefetch = {}
        cache.ongoing_backup = {7: (11, "lock")}
        cache.dec_host_lock_ref = Mock(side_effect=[RuntimeError("unlock"), None])
        cache._storage_prefetch_hit_remaining_by_reqid = {}
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.prefetch_loaded_storage_start_by_reqid = {}
        cache._storage_prefetch_wait = {}
        cache.storage_prefetch_retries = Mock()
        with self.assertRaises(RuntimeError):
            StorageAttachment._release_pending_storage_ops(attachment)
        self.assertEqual(cache.ongoing_backup, {7: (11, "lock")})
        with self.assertRaisesRegex(RuntimeError, "restart the model process"):
            StorageAttachment._release_pending_storage_ops(attachment)
        cache.dec_host_lock_ref.assert_called_once()
        self.assertEqual(cache.ongoing_backup, {7: (11, "lock")})

    def test_partial_initial_drain_is_not_replayed(self):
        attachment, cache, _, backend = _fixture()
        cache.ongoing_prefetch = {"retained": object()}
        cache.drain_storage_control_queues_local.side_effect = RuntimeError(
            "partial drain failure"
        )
        self.assertFalse(attachment.detach()[0])
        self.assertFalse(attachment.detach()[0])
        self.assertFalse(attachment.attach("dynamic")[0])
        attachment.shutdown()
        cache.drain_storage_control_queues_local.assert_called_once()
        self.assertEqual(backend.close_calls, 0)

    def test_disabled_scheduler_does_not_report_a_failed_detach_as_success(self):
        attachment, _, _, _ = _fixture()
        scheduler = SimpleNamespace(
            enable_hierarchical_cache=True,
            enable_hicache_storage=False,
            is_fully_idle=lambda: True,
            tree_cache=SimpleNamespace(detach_storage_backend=attachment.detach),
        )
        result = Scheduler.detach_hicache_storage_wrapped(
            scheduler, DetachHiCacheStorageReqInput()
        )
        self.assertFalse(result.success)
        self.assertIn("cleanup failed", result.message)


if __name__ == "__main__":
    unittest.main()
