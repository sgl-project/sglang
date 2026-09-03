"""Runtime HiCache attach/detach lands on the config bags.

The attach RPC used to mutate the scheduler's ServerArgs so the readback would
show the change; the namespace readers never saw it. Both now go through
get_context().override, so get_memory() and the resolved-config readback agree
and the published instance stays as the launcher left it.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.managers.io_struct import (
    AttachHiCacheStorageReqInput,
    DetachHiCacheStorageReqInput,
)
from sglang.srt.managers.cache_controller import StorageLifecycleConsensusError
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, get_memory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerHiCacheAttach(CustomTestCase):
    def _scheduler(self, **fields):
        override = get_context().override_server_args(
            enable_hierarchical_cache=True, **fields
        )
        self.server_args = override.install()
        self.addCleanup(override.restore)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = self.server_args
        scheduler.enable_hierarchical_cache = True
        scheduler.enable_hicache_storage = False
        scheduler.is_fully_idle = lambda: True
        scheduler.tree_cache = SimpleNamespace(
            supports_coordinated_storage_lifecycle=True,
            attach_storage_backend=Mock(return_value=(True, "attached")),
            detach_storage_backend=Mock(return_value=(True, "detached")),
        )
        return scheduler

    def test_attach_reaches_the_namespace_readers(self):
        scheduler = self._scheduler(hicache_storage_backend=None)
        out = scheduler.attach_hicache_storage_wrapped(
            AttachHiCacheStorageReqInput(
                hicache_storage_backend="file",
                hicache_write_policy="write_through",
            )
        )

        self.assertTrue(out.success)
        self.assertEqual(get_memory().hicache_storage_backend, "file")
        self.assertEqual(get_memory().hicache_write_policy, "write_through")
        self.assertEqual(
            get_context().resolved_server_args_dict()["hicache_storage_backend"],
            "file",
        )
        self.assertIsNone(self.server_args.hicache_storage_backend)

    def test_detach_clears_the_backend_for_the_same_readers(self):
        scheduler = self._scheduler(hicache_storage_backend="file")
        scheduler.enable_hicache_storage = True

        out = scheduler.detach_hicache_storage_wrapped(DetachHiCacheStorageReqInput())

        self.assertTrue(out.success)
        self.assertIsNone(get_memory().hicache_storage_backend)
        self.assertIsNone(
            get_context().resolved_server_args_dict()["hicache_storage_backend"]
        )
        # The record is not written any more: the attach is a declaration on
        # it and the detach is a bag override (asserted above), so the two are
        # meant to differ here.
        self.assertEqual(
            resolution_result(self.server_args, "hicache_storage_backend"), "file"
        )

    def test_detach_refusal_still_disables_a_stopped_tree_data_path(self):
        scheduler = self._scheduler(hicache_storage_backend="kvcr")
        scheduler.enable_hicache_storage = True
        scheduler.tree_cache.enable_storage = False
        scheduler.tree_cache.detach_storage_backend = lambda **_kwargs: (
            False,
            "KVCR close failed; worker restart required",
        )

        out = scheduler.detach_hicache_storage_wrapped(DetachHiCacheStorageReqInput())

        self.assertFalse(out.success)
        self.assertFalse(scheduler.enable_hicache_storage)
        self.assertIsNone(get_memory().hicache_storage_backend)

    def test_detach_consensus_failure_is_fatal_to_the_scheduler(self):
        scheduler = self._scheduler(hicache_storage_backend="kvcr")
        scheduler.enable_hicache_storage = True

        def fail_consensus(**_kwargs):
            raise StorageLifecycleConsensusError("gloo failed")

        scheduler.tree_cache.detach_storage_backend = fail_consensus

        with self.assertRaisesRegex(StorageLifecycleConsensusError, "gloo failed"):
            scheduler.detach_hicache_storage_wrapped(DetachHiCacheStorageReqInput())

    def test_attach_busy_state_is_voted_inside_the_storage_lifecycle(self):
        scheduler = self._scheduler(hicache_storage_backend=None)
        scheduler.is_fully_idle = lambda: False
        scheduler.waiting_queue = [object()]
        scheduler.running_batch = SimpleNamespace(reqs=[])
        scheduler.tree_cache.attach_storage_backend.return_value = (
            False,
            "scheduler is not idle on this or a peer rank",
        )

        out = scheduler.attach_hicache_storage_wrapped(
            AttachHiCacheStorageReqInput(hicache_storage_backend="file")
        )

        self.assertFalse(out.success)
        self.assertEqual(
            scheduler.tree_cache.attach_storage_backend.call_args.kwargs["local_ready"],
            False,
        )

    def test_detach_busy_state_is_voted_inside_the_storage_lifecycle(self):
        scheduler = self._scheduler(hicache_storage_backend="file")
        scheduler.enable_hicache_storage = True
        scheduler.is_fully_idle = lambda: False
        scheduler.waiting_queue = [object()]
        scheduler.running_batch = SimpleNamespace(reqs=[])
        scheduler.tree_cache.detach_storage_backend.return_value = (
            False,
            "scheduler is not idle on this or a peer rank",
        )

        out = scheduler.detach_hicache_storage_wrapped(DetachHiCacheStorageReqInput())

        self.assertFalse(out.success)
        scheduler.tree_cache.detach_storage_backend.assert_called_once_with(
            local_ready=False
        )

    def test_legacy_tree_keeps_the_original_attach_signature(self):
        scheduler = self._scheduler(hicache_storage_backend=None)
        scheduler.tree_cache.supports_coordinated_storage_lifecycle = False

        out = scheduler.attach_hicache_storage_wrapped(
            AttachHiCacheStorageReqInput(hicache_storage_backend="file")
        )

        self.assertTrue(out.success)
        self.assertNotIn(
            "local_ready",
            scheduler.tree_cache.attach_storage_backend.call_args.kwargs,
        )


if __name__ == "__main__":
    unittest.main()
