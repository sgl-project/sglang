import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDraftL3GracefulDegradation(CustomTestCase):
    """When register_mem_host_pool_v2 fails for the draft pool (e.g.
    standalone Mooncake mode where the draft buffer is outside the shared
    memory segment), the controller must keep draft_page_get_func /
    draft_page_set_func as None instead of crashing.
    """

    def _make_controller(self, backend="mooncake"):
        from sglang.srt.managers.cache_controller import HiCacheController

        ctrl = HiCacheController.__new__(HiCacheController)
        ctrl.has_draft = True
        ctrl.enable_storage = True
        ctrl.storage_backend_type = backend
        ctrl.storage_config = MagicMock()
        ctrl.storage_config.should_split_heads = False
        ctrl.storage_backend = MagicMock()
        ctrl.mem_pool_host_draft = MagicMock()
        ctrl.draft_page_get_func = None
        ctrl.draft_page_set_func = None
        return ctrl

    def test_mooncake_register_failure_keeps_funcs_none(self):
        """register_mem_host_pool_v2 raising RuntimeError must not crash."""
        ctrl = self._make_controller("mooncake")
        ctrl.storage_backend.register_mem_host_pool_v2.side_effect = RuntimeError(
            "Failed to register buffer to Mooncake Store, error code: -1"
        )

        ctrl._maybe_register_draft_with_storage()

        self.assertIsNone(ctrl.draft_page_get_func)
        self.assertIsNone(ctrl.draft_page_set_func)

    def test_mooncake_register_failure_any_exception(self):
        """Any exception type (not just RuntimeError) must be caught."""
        ctrl = self._make_controller("mooncake")
        ctrl.storage_backend.register_mem_host_pool_v2.side_effect = OSError("boom")

        ctrl._maybe_register_draft_with_storage()

        self.assertIsNone(ctrl.draft_page_get_func)
        self.assertIsNone(ctrl.draft_page_set_func)

    def test_mooncake_register_success_sets_funcs(self):
        """When registration succeeds, v2 funcs must be set."""
        ctrl = self._make_controller("mooncake")

        ctrl._maybe_register_draft_with_storage()

        self.assertIsNotNone(ctrl.draft_page_get_func)
        self.assertIsNotNone(ctrl.draft_page_set_func)

    def test_non_mooncake_backend_unaffected(self):
        """Generic backends must still set generic funcs."""
        ctrl = self._make_controller("file")

        ctrl._maybe_register_draft_with_storage()

        self.assertIsNotNone(ctrl.draft_page_get_func)
        self.assertIsNotNone(ctrl.draft_page_set_func)


if __name__ == "__main__":
    unittest.main(verbosity=3)
