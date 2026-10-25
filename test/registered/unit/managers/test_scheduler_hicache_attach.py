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
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, get_memory
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestSchedulerHiCacheAttach(CustomTestCase):
    def test_linker_detach_failure_escapes_admin_handler(self):
        scheduler = self._scheduler(enable_linker_mla_dedup=False)
        scheduler.enable_unified_cache_external_linker = True
        scheduler.tree_cache.detach_cache_linker = Mock(
            side_effect=RuntimeError("scheduler must stop")
        )
        with self.assertRaisesRegex(RuntimeError, "scheduler must stop"):
            scheduler.detach_hicache_storage_wrapped(DetachHiCacheStorageReqInput())

    def test_linker_attach_and_detach_use_only_linker_config(self):
        scheduler = self._scheduler(
            hicache_storage_backend_extra_config='{"legacy": true}',
            enable_linker_mla_dedup=False,
        )
        scheduler.enable_unified_cache_external_linker = True
        scheduler.tree_cache.attach_cache_linker = Mock(return_value="attached")
        scheduler.tree_cache.detach_cache_linker = Mock(return_value="detached")
        config = {"linker": "Plugin", "linker_module_path": "my_plugin"}
        out = scheduler.attach_hicache_storage_wrapped(
            AttachHiCacheStorageReqInput(
                unified_cache_external_linker_backend="dynamic",
                unified_cache_external_linker_config=config,
            )
        )
        self.assertTrue(out.success)
        scheduler.tree_cache.attach_cache_linker.assert_called_once_with(
            "dynamic", config
        )
        self.assertEqual(get_memory().unified_cache_external_linker_backend, "dynamic")
        self.assertEqual(get_memory().unified_cache_external_linker_config, config)
        self.assertEqual(
            get_memory().hicache_storage_backend_extra_config, '{"legacy": true}'
        )
        out = scheduler.detach_hicache_storage_wrapped(DetachHiCacheStorageReqInput())
        self.assertTrue(out.success)
        self.assertIsNone(get_memory().unified_cache_external_linker_config)
        self.assertEqual(
            get_memory().hicache_storage_backend_extra_config, '{"legacy": true}'
        )

    def test_attach_rejects_fields_for_the_other_mode(self):
        scheduler = self._scheduler(enable_linker_mla_dedup=False)
        scheduler.tree_cache.attach_cache_linker = Mock()
        out = scheduler.attach_hicache_storage_wrapped(
            AttachHiCacheStorageReqInput(
                unified_cache_external_linker_backend="mooncake"
            )
        )
        self.assertFalse(out.success)
        scheduler.enable_unified_cache_external_linker = True
        for req in (
            AttachHiCacheStorageReqInput(hicache_storage_backend="mooncake"),
            AttachHiCacheStorageReqInput(
                unified_cache_external_linker_backend="mooncake",
                hicache_storage_backend_extra_config_json="{}",
            ),
            AttachHiCacheStorageReqInput(
                unified_cache_external_linker_backend="mooncake",
                hicache_write_policy="write_back",
            ),
        ):
            self.assertFalse(scheduler.attach_hicache_storage_wrapped(req).success)
        scheduler.tree_cache.attach_cache_linker.assert_not_called()

    def _scheduler(self, **fields):
        override = get_context().override_server_args(
            enable_hierarchical_cache=True, **fields
        )
        self.server_args = override.install()
        self.addCleanup(override.restore)

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = self.server_args
        scheduler.enable_hierarchical_cache = True
        scheduler.enable_unified_cache_external_linker = False
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


if __name__ == "__main__":
    unittest.main()
