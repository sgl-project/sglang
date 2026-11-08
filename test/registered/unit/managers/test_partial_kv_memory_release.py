"""A paused request must retain its cache ownership across a memory handoff."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput
from sglang.srt.managers.scheduler_components.pause_state import PauseState
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPartialKvMemoryRelease(CustomTestCase):
    def setUp(self):
        publish(
            ServerArgs(
                model_path="dummy",
                device="cuda",
                enable_memory_saver=True,
                enable_weights_cpu_backup=True,
                enable_kv_cache_cpu_backup=True,
            ),
            role="test",
        )
        self.addCleanup(reset_context)
        self.request_slots = {"unfinished-request": [3, 4, 9]}
        self.pause = PauseState()
        self.saver = Mock(enabled=True)
        scheduler = SimpleNamespace(
            pause_state=self.pause, disaggregation_mode=DisaggregationMode.NULL
        )
        with patch(
            "sglang.srt.managers.scheduler_components.weight_updater.get_parallel",
            return_value=SimpleNamespace(tp_group=SimpleNamespace(cpu_group=None)),
        ):
            self.manager = SchedulerWeightUpdaterManager(
                tp_worker=None,
                draft_worker=None,
                memory_saver_adapter=self.saver,
                flush_cache=self.request_slots.clear,
                is_fully_idle=lambda **kwargs: False,
                scheduler=scheduler,
            )
        self.device = patch("torch.get_device_module", return_value=Mock())
        self.device.start()
        self.addCleanup(self.device.stop)

    def test_paused_release_keeps_request_slot_ownership(self):
        """The ordinary release rejects live work or clears its KV slot mapping."""
        slots = self.request_slots["unfinished-request"]
        self.pause.pause()
        result = self.manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
        )
        self.assertTrue(result.success, result.message)
        self.assertIs(self.request_slots["unfinished-request"], slots)
        self.assertEqual(slots, [3, 4, 9])
        self.assertEqual(self.manager.offload_tags, {"kv_cache"})
        self.assertTrue(self.pause.paused)

    def test_release_requires_pause_and_rejects_repeated_release(self):
        request = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
        result = self.manager.release_memory_occupation(request)
        self.assertFalse(result.success)
        self.assertEqual(self.manager.offload_tags, set())
        self.pause.pause()
        self.assertTrue(self.manager.release_memory_occupation(request).success)
        self.assertFalse(self.manager.release_memory_occupation(request).success)
        self.assertEqual(self.request_slots, {"unfinished-request": [3, 4, 9]})

    def test_backup_rejects_unsupported_lifecycles_at_startup(self):
        common = dict(
            model_path="dummy",
            device="cuda",
            enable_memory_saver=True,
            enable_weights_cpu_backup=True,
            enable_kv_cache_cpu_backup=True,
        )
        for override in (
            dict(enable_memory_saver=False),
            dict(enable_weights_cpu_backup=False),
            dict(device="cpu"),
            dict(pp_size=2),
            dict(disaggregation_mode="decode"),
            dict(enable_hierarchical_cache=True),
            dict(enable_lmcache=True),
            dict(enable_unified_memory=True),
            dict(speculative_algorithm="EAGLE"),
            dict(weight_cache_mode="daemon"),
        ):
            with (
                self.subTest(override=override),
                self.assertRaisesRegex(ValueError, "enable-kv-cache-cpu-backup"),
            ):
                ServerArgs(**(common | override)).resolve_once()

    def test_default_release_still_requires_idle_and_flushes_cache(self):
        publish(ServerArgs(model_path="dummy"), role="test")
        request = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
        self.pause.pause()
        self.assertFalse(self.manager.release_memory_occupation(request).success)
        self.manager.is_fully_idle = lambda **kwargs: True
        self.assertTrue(self.manager.release_memory_occupation(request).success)
        self.assertEqual(self.request_slots, {})


if __name__ == "__main__":
    unittest.main()
