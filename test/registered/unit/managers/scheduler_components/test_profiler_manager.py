import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSchedulerProfilerManager(unittest.TestCase):
    def test_target_verify_counts_as_decode_stage(self):
        with patch.object(envs.SGLANG_PROFILE_V2, "get", return_value=False):
            manager = SchedulerProfilerManager(
                ps=SimpleNamespace(),
                dp_tp_cpu_group=None,
                get_forward_ct=lambda: 0,
            )

        manager.profile_by_stage = True
        manager.profiler_prefill_ct = 0
        manager.profiler_decode_ct = 0
        manager.profiler_target_prefill_ct = 1
        manager.profiler_target_decode_ct = 1

        def start_profile(_stage):
            manager.profile_in_progress = True

        def stop_profile(*, stage):
            manager.profile_in_progress = False

        manager._start_profile = MagicMock(side_effect=start_profile)
        manager._stop_profile = MagicMock(side_effect=stop_profile)

        for mode in (
            ForwardMode.EXTEND,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.TARGET_VERIFY,
        ):
            manager._profile_batch_predicate(SimpleNamespace(forward_mode=mode))

        self.assertEqual(manager.profiler_prefill_ct, 1)
        self.assertEqual(manager.profiler_decode_ct, 2)
        self.assertEqual(
            manager._start_profile.call_args_list,
            [call(ForwardMode.EXTEND), call(ForwardMode.TARGET_VERIFY)],
        )
        self.assertEqual(
            manager._stop_profile.call_args_list,
            [call(stage=ForwardMode.EXTEND), call(stage=ForwardMode.DECODE)],
        )


if __name__ == "__main__":
    unittest.main()
