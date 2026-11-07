"""profile_by_stage without num_steps must not crash the scheduler."""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _prefill_batch():
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_prefill=lambda: True,
            is_decode=lambda: False,
            is_idle=lambda: False,
            is_prebuilt=lambda: False,
        )
    )


class TestProfileByStageWithoutNumSteps(CustomTestCase):
    def test_omitted_num_steps_counts_prefill_without_raising(self):
        manager = SchedulerProfilerManager.__new__(SchedulerProfilerManager)
        manager.profile_in_progress = False
        manager.profile_by_stage = False
        manager.get_forward_ct = lambda: 0
        manager.profiler_start_forward_ct = None
        manager.profiler_target_forward_ct = None
        manager.profiler_prefill_ct = None
        manager.profiler_decode_ct = None
        manager.profiler_target_prefill_ct = None
        manager.profiler_target_decode_ct = None
        manager.merge_profiles = False
        manager.detailed_annotations = False
        manager.torch_profiler = None
        started = []
        manager._start_profile = lambda stage=None: started.append(stage)
        manager._stop_profile = lambda stage=None: None

        result = SchedulerProfilerManager._init_profile(
            manager,
            "/tmp",
            None,
            None,
            ["CPU"],
            False,
            False,
            True,
            "id",
        )
        self.assertTrue(result.success)
        self.assertEqual(manager.profiler_prefill_ct, 0)
        self.assertIsNone(manager.profiler_target_prefill_ct)

        SchedulerProfilerManager._profile_batch_predicate(manager, _prefill_batch())
        self.assertEqual(manager.profiler_prefill_ct, 1)
        self.assertEqual(len(started), 1)


if __name__ == "__main__":
    unittest.main()
