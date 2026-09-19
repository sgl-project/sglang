"""Profiler request transitions must preserve ownership and disarm old triggers."""

import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestProfilerLifecycle(CustomTestCase):
    def setUp(self):
        super().setUp()
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(envs.SGLANG_PROFILE_V2.override(False))
        contexts.enter_context(envs.SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS.override(0))
        self.directory = contexts.enter_context(tempfile.TemporaryDirectory())
        self.factory = contexts.enter_context(patch("torch.profiler.profile"))
        contexts.enter_context(patch("torch.distributed.barrier"))
        contexts.enter_context(
            patch(
                "sglang.srt.managers.scheduler_components.profiler_manager."
                "set_detailed_annotations_enabled"
            )
        )
        self.new_manager()

    def new_manager(self):
        self.forward = 0
        self.factory.reset_mock()
        self.factory.side_effect = lambda **kwargs: Mock()
        self.manager = SchedulerProfilerManager(
            ps=SimpleNamespace(tp_rank=0, dp_size=1, pp_size=1, moe_ep_size=1),
            dp_tp_cpu_group=None,
            get_forward_ct=lambda: self.forward,
        )

    def start(self, **kwargs):
        return self.manager._profile(
            ProfileReq(
                output_dir=self.directory,
                activities=["CPU"],
                profile_id=kwargs.pop("profile_id", "first"),
                profile_prefix="",
                **kwargs,
            )
        )

    def stop(self):
        return self.manager._profile(ProfileReq(req_type=ProfileReqType.STOP_PROFILE))

    def step(self, mode):
        self.forward += 1
        self.manager._profile_batch_predicate(
            SimpleNamespace(forward_mode=mode, batch_size=lambda: 4)
        )

    def test_successful_manual_stop_disarms_remaining_stage(self):
        for first, later in (
            (ForwardMode.EXTEND, ForwardMode.DECODE),
            (ForwardMode.DECODE, ForwardMode.EXTEND),
        ):
            with self.subTest(first=first):
                self.new_manager()
                self.assertTrue(self.start(profile_by_stage=True, num_steps=3).success)
                self.step(first)
                recorder = self.manager.torch_profiler
                self.assertTrue(self.stop().success)
                recorder.stop.assert_called_once_with()
                self.step(later)
                self.step(first)
                self.assertFalse(self.manager.profile_in_progress)
                self.assertIsNone(self.manager.torch_profiler)
                self.assertEqual(self.factory.call_count, 1)

    def test_immediate_start_replaces_pending_deferred_trigger(self):
        for num_steps in (None, 4):
            with self.subTest(num_steps=num_steps):
                self.new_manager()
                self.assertTrue(self.start(start_step=3, num_steps=2).success)
                self.assertTrue(self.start(num_steps=num_steps).success)
                recorder = self.manager.torch_profiler
                for _ in range(3):
                    self.step(ForwardMode.EXTEND)
                self.assertIs(self.manager.torch_profiler, recorder)
                self.assertEqual(self.factory.call_count, 1)
                self.assertTrue(self.stop().success)
                recorder.stop.assert_called_once_with()

    def test_automatic_stage_transition_still_starts_decode(self):
        self.assertTrue(self.start(profile_by_stage=True, num_steps=1).success)
        self.step(ForwardMode.EXTEND)
        prefill = self.manager.torch_profiler
        self.step(ForwardMode.EXTEND)
        self.assertFalse(self.manager.profile_in_progress)
        self.step(ForwardMode.DECODE)
        decode = self.manager.torch_profiler
        self.assertTrue(self.manager.profile_in_progress)
        self.assertIsNot(prefill, decode)
        self.step(ForwardMode.DECODE)
        prefill.stop.assert_called_once_with()
        decode.stop.assert_called_once_with()
        self.assertFalse(self.manager.profile_in_progress)
        self.assertEqual(self.factory.call_count, 2)

    def test_reconfigured_deferred_start_uses_new_step(self):
        self.assertTrue(self.start(start_step=3, num_steps=2).success)
        self.assertTrue(self.start(start_step=5, num_steps=1).success)
        for _ in range(4):
            self.step(ForwardMode.EXTEND)
        self.factory.assert_not_called()
        self.step(ForwardMode.EXTEND)
        self.assertTrue(self.manager.profile_in_progress)
        self.step(ForwardMode.EXTEND)
        self.assertFalse(self.manager.profile_in_progress)
        self.assertEqual(self.factory.call_count, 1)

    def test_new_stage_session_after_manual_stop_resets_counters(self):
        self.assertTrue(self.start(profile_by_stage=True, num_steps=3).success)
        self.step(ForwardMode.EXTEND)
        self.assertTrue(self.stop().success)
        self.assertTrue(self.start(profile_by_stage=True, num_steps=1).success)
        for mode in (ForwardMode.EXTEND, ForwardMode.DECODE):
            self.step(mode)
            self.assertTrue(self.manager.profile_in_progress)
            self.step(mode)
            self.assertFalse(self.manager.profile_in_progress)
        self.assertEqual(self.factory.call_count, 3)

    def test_stopped_stage_then_replaced_deferred_session_keeps_one_owner(self):
        self.assertTrue(self.start(profile_by_stage=True, num_steps=3).success)
        self.step(ForwardMode.EXTEND)
        stage = self.manager.torch_profiler
        self.assertTrue(self.stop().success)
        self.step(ForwardMode.DECODE)
        self.assertFalse(self.manager.profile_in_progress)

        self.assertTrue(self.start(start_step=self.forward + 2, num_steps=2).success)
        self.assertTrue(self.start().success)
        immediate = self.manager.torch_profiler
        self.assertFalse(self.start().success)
        self.step(ForwardMode.EXTEND)
        self.step(ForwardMode.DECODE)
        self.assertIs(self.manager.torch_profiler, immediate)
        self.assertTrue(self.stop().success)
        self.step(ForwardMode.EXTEND)
        self.step(ForwardMode.DECODE)
        self.assertFalse(self.manager.profile_in_progress)
        self.assertEqual(self.factory.call_count, 2)
        stage.stop.assert_called_once_with()
        immediate.stop.assert_called_once_with()

    def test_duplicate_start_preserves_session_until_stop(self):
        first, second = Mock(), Mock()
        self.factory.side_effect = [first, second]
        self.assertTrue(self.start().success)

        result = self.start(profile_id="rejected")

        self.assertFalse(result.success)
        self.assertIn("already in progress", result.message)
        self.assertIs(self.manager.torch_profiler, first)
        self.assertTrue(self.manager.profile_in_progress)
        self.assertEqual(self.manager.profile_id, "first")
        self.assertEqual(self.factory.call_count, 1)
        first.start.assert_called_once_with()
        first.stop.assert_not_called()

        self.assertTrue(self.stop().success)
        first.stop.assert_called_once_with()
        first.export_chrome_trace.assert_called_once()
        self.assertFalse(self.manager.profile_in_progress)
        self.assertIsNone(self.manager.torch_profiler)

        self.assertTrue(self.start(profile_id="next").success)
        self.assertIs(self.manager.torch_profiler, second)
        second.start.assert_called_once_with()

    def test_deferred_starts_only_configure(self):
        for kwargs in ({"start_step": 5}, {"profile_by_stage": True, "num_steps": 2}):
            with self.subTest(kwargs=kwargs):
                self.assertTrue(self.start(**kwargs).success)
                self.factory.assert_not_called()
                self.assertFalse(self.manager.profile_in_progress)

    def test_start_failure_allows_retry(self):
        failed, retry = Mock(), Mock()
        failed.start.side_effect = RuntimeError("profiler unavailable")
        self.factory.side_effect = [failed, retry]
        result = self.start()
        self.assertFalse(result.success)
        self.assertIn("profiler unavailable", result.message)
        self.assertFalse(self.manager.profile_in_progress)
        self.assertIsNone(self.manager.torch_profiler)

        self.assertTrue(self.start().success)
        self.assertIs(self.manager.torch_profiler, retry)
        retry.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
