"""CPU-only regression tests for SchedulerProfilerManager.

These tests do not start a server and do not require a GPU.  They exercise
the counter initialization and the first-batch predicate directly, which is
where profile_by_stage without num_steps used to crash.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)


class _FakeForwardMode:
    def __init__(self, *, prefill: bool):
        self._prefill = prefill

    def is_prefill(self) -> bool:
        return self._prefill

    def is_decode(self) -> bool:
        return not self._prefill

    def is_idle(self) -> bool:
        return False

    def is_prebuilt(self) -> bool:
        return False


class TestProfileByStageWithoutNumSteps(unittest.TestCase):
    def _manager(self) -> SchedulerProfilerManager:
        # __post_init__ needs a distributed group; this unit test only needs
        # the manager's profile state machine, so bypass construction.
        manager = object.__new__(SchedulerProfilerManager)
        manager.profile_in_progress = False
        # Mirror SchedulerProfilerManager.__init__; object.__new__ skips it.
        manager.profiler_prefill_ct = None
        manager.profiler_decode_ct = None
        manager.profiler_target_prefill_ct = None
        manager.profiler_target_decode_ct = None
        manager.started = []
        manager.stopped = []
        manager._start_profile = lambda stage=None: manager.started.append(stage)
        manager._stop_profile = lambda stage=None: manager.stopped.append(stage)
        return manager

    def _init(self, manager: SchedulerProfilerManager, stage: str):
        with patch.object(envs.SGLANG_PROFILE_V2, "get", return_value=False):
            result = manager._init_profile(
                output_dir="/tmp/sglang-profiler-unit",
                start_step=None,
                num_steps=None,
                activities=["CPU", "GPU"],
                with_stack=True,
                record_shapes=True,
                profile_by_stage=True,
                profile_id="unit",
                merge_profiles=False,
                profile_prefix="",
                detailed_annotations=False,
                profile_stages=[stage],
            )
        self.assertTrue(result.success)
        self.assertIsNone(manager.profiler_target_prefill_ct)
        self.assertIsNone(manager.profiler_target_decode_ct)

    def _predicate(self, manager: SchedulerProfilerManager, *, prefill: bool):
        batch = SimpleNamespace(forward_mode=_FakeForwardMode(prefill=prefill))
        with patch.object(envs.SGLANG_PROFILE_V2, "get", return_value=False):
            manager._profile_batch_predicate(batch)

    def test_prefill_starts_without_num_steps(self):
        manager = self._manager()
        self._init(manager, "prefill")
        self._predicate(manager, prefill=True)
        self.assertEqual(manager.profiler_prefill_ct, 1)
        self.assertEqual(len(manager.started), 1)
        self.assertEqual(manager.stopped, [])

    def test_decode_starts_without_num_steps(self):
        manager = self._manager()
        self._init(manager, "decode")
        self._predicate(manager, prefill=False)
        self.assertEqual(manager.profiler_decode_ct, 1)
        self.assertEqual(len(manager.started), 1)
        self.assertEqual(manager.stopped, [])


if __name__ == "__main__":
    unittest.main()
