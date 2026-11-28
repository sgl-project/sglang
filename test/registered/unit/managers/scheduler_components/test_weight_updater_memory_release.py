import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput
from sglang.srt.managers.scheduler_components import weight_updater

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestWeightUpdaterMemoryRelease(CustomTestCase):
    def setUp(self):
        self.enterContext(patch.object(weight_updater, "get_parallel"))
        self.enterContext(
            patch.object(
                weight_updater,
                "get_model",
                return_value=SimpleNamespace(weight_cache_mode="client"),
            )
        )
        self.enterContext(
            patch.object(
                weight_updater, "control_request_all_gather_object", return_value=[True]
            )
        )
        self.barrier = self.enterContext(
            patch.object(weight_updater, "control_request_barrier")
        )
        self.updater = weight_updater.SchedulerWeightUpdaterManager(
            tp_worker=None,
            draft_worker=None,
            memory_saver_adapter=Mock(),
            flush_cache=Mock(),
            is_fully_idle=lambda **_: True,
        )

    def test_rejects_default_release_before_pausing_kv_cache(self):
        result = self.updater.release_memory_occupation(
            ReleaseMemoryOccupationReqInput()
        )

        self.assertFalse(result.success)
        self.assertIn("weight cache is active", result.message)
        self.updater.memory_saver_adapter.pause.assert_not_called()
        self.updater.flush_cache.assert_not_called()
        self.barrier.assert_not_called()
        self.assertEqual(self.updater.offload_tags, set())

    def test_allows_kv_only_release_with_shared_weights(self):
        with patch.object(weight_updater.torch, "get_device_module"):
            result = self.updater.release_memory_occupation(
                ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            )

        self.assertTrue(result.success)
        self.updater.memory_saver_adapter.pause.assert_called_once_with(
            GPU_MEMORY_TYPE_KV_CACHE
        )
        self.updater.flush_cache.assert_called_once()
        self.assertEqual(self.updater.offload_tags, {GPU_MEMORY_TYPE_KV_CACHE})


if __name__ == "__main__":
    unittest.main()
