"""Runtime HiCache attach/detach lands on the config bags.

The attach RPC used to mutate the scheduler's ServerArgs so the readback would
show the change; the namespace readers never saw it. Both now go through
get_context().override, so get_memory() and the resolved-config readback agree
and the published instance stays as the launcher left it.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import (
    AttachHiCacheStorageReqInput,
    DetachHiCacheStorageReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, get_memory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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
            attach_storage_backend=lambda **kwargs: (True, "attached"),
            detach_storage_backend=lambda: (True, "detached"),
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
        self.assertEqual(self.server_args.hicache_storage_backend, "file")


if __name__ == "__main__":
    unittest.main()
