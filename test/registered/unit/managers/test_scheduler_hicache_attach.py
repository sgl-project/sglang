"""Runtime HiCache attach/detach lands on the config bags.

The attach RPC used to mutate the scheduler's ServerArgs so the readback would
show the change; the namespace readers never saw it. Both now go through
get_context().override, so get_memory() and the resolved-config readback agree
and the published instance stays as the launcher left it.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.managers.io_struct import (
    AttachHiCacheStorageReqInput,
    ClearHiCacheReqOutput,
    DetachHiCacheStorageReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_memory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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
        # The record is not written any more: the attach is a declaration on
        # it and the detach is a bag override (asserted above), so the two are
        # meant to differ here.
        self.assertEqual(
            resolution_result(self.server_args, "hicache_storage_backend"), "file"
        )


class TestHiCacheStorageClear(CustomTestCase):
    """A backend failure must reach the admin response, including fan-out replies."""

    def test_mooncake_clear_result_reaches_http_response(self):
        from sglang.srt.entrypoints import http_server

        for ret_code in (-1, 0, 7):
            with self.subTest(ret_code=ret_code):
                backend = MooncakeStore.__new__(MooncakeStore)
                backend.store = SimpleNamespace(remove_all=lambda force=False: ret_code)
                controller = HybridCacheController.__new__(HybridCacheController)
                controller.enable_storage = True
                controller.storage_backend = backend
                cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
                cache.cache_controller = controller
                cache.storage_existence_cache = {"page": True}
                cache._storage_attachment = StorageAttachment(cache)
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.enable_hierarchical_cache = True
                scheduler.enable_lmcache = False
                scheduler.tree_cache = cache

                async def communicate(request):
                    return [scheduler.clear_hicache_storage_wrapped(request)]

                manager = TokenizerControlMixin()
                manager.auto_create_handle_loop = lambda: None
                manager.clear_hicache_storage_communicator = communicate
                with patch.object(
                    http_server,
                    "_global_state",
                    SimpleNamespace(tokenizer_manager=manager),
                ):
                    response = asyncio.run(http_server.clear_hicache_storage_backend())

                self.assertEqual(response.status_code, 400 if ret_code < 0 else 200)
                if ret_code < 0:
                    self.assertIn(b"Failed to clear", response.body)
                    self.assertEqual(cache.storage_existence_cache, {"page": True})
                else:
                    self.assertIn(b"backend cleared", response.body)
                    self.assertEqual(cache.storage_existence_cache, {})

    def test_one_failed_replica_prevents_success(self):
        for statuses in ((True, False), (False, True), (True, True)):
            with self.subTest(statuses=statuses):
                manager = TokenizerControlMixin()
                manager.auto_create_handle_loop = lambda: None
                manager.clear_hicache_storage_communicator = AsyncMock(
                    return_value=[ClearHiCacheReqOutput(success=s) for s in statuses]
                )

                result = asyncio.run(manager.clear_hicache_storage())

                self.assertEqual(result.success, statuses == (True, True))


if __name__ == "__main__":
    unittest.main()
