"""A rejected start request must leave the active profiler session intact."""

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

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestProfilerStartRejection(CustomTestCase):
    def setUp(self):
        super().setUp()
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(envs.SGLANG_PROFILE_V2.override(False))
        self.output_dir = contexts.enter_context(tempfile.TemporaryDirectory())
        self.factory = contexts.enter_context(patch("torch.profiler.profile"))
        contexts.enter_context(patch("torch.distributed.barrier"))
        contexts.enter_context(
            patch(
                "sglang.srt.managers.scheduler_components.profiler_manager."
                "set_detailed_annotations_enabled"
            )
        )
        self.manager = SchedulerProfilerManager(
            ps=SimpleNamespace(tp_rank=0, dp_size=1, pp_size=1, moe_ep_size=1),
            dp_tp_cpu_group=None,
            get_forward_ct=lambda: 0,
        )

    def request(self, **kwargs):
        return ProfileReq(
            output_dir=self.output_dir,
            activities=["CPU"],
            profile_id=kwargs.pop("profile_id", "first"),
            profile_prefix="",
            **kwargs,
        )

    def test_duplicate_start_preserves_session_until_stop(self):
        first, second = Mock(), Mock()
        self.factory.side_effect = [first, second]
        self.assertTrue(self.manager._profile(self.request()).success)

        result = self.manager._profile(self.request(profile_id="rejected"))

        self.assertFalse(result.success)
        self.assertIn("already in progress", result.message)
        self.assertIs(self.manager.torch_profiler, first)
        self.assertTrue(self.manager.profile_in_progress)
        self.assertEqual(self.manager.profile_id, "first")
        self.assertEqual(self.factory.call_count, 1)
        first.start.assert_called_once_with()
        first.stop.assert_not_called()

        self.assertTrue(
            self.manager._profile(
                self.request(req_type=ProfileReqType.STOP_PROFILE)
            ).success
        )
        first.stop.assert_called_once_with()
        first.export_chrome_trace.assert_called_once()
        self.assertFalse(self.manager.profile_in_progress)
        self.assertIsNone(self.manager.torch_profiler)

        self.assertTrue(self.manager._profile(self.request(profile_id="next")).success)
        self.assertIs(self.manager.torch_profiler, second)
        second.start.assert_called_once_with()

    def test_deferred_starts_only_configure(self):
        for kwargs in ({"start_step": 5}, {"profile_by_stage": True, "num_steps": 2}):
            with self.subTest(kwargs=kwargs):
                self.assertTrue(self.manager._profile(self.request(**kwargs)).success)
                self.factory.assert_not_called()
                self.assertFalse(self.manager.profile_in_progress)

    def test_start_failure_allows_retry(self):
        failed, retry = Mock(), Mock()
        failed.start.side_effect = RuntimeError("profiler unavailable")
        self.factory.side_effect = [failed, retry]
        result = self.manager._profile(self.request())
        self.assertFalse(result.success)
        self.assertIn("profiler unavailable", result.message)
        self.assertFalse(self.manager.profile_in_progress)
        self.assertIsNone(self.manager.torch_profiler)

        self.assertTrue(self.manager._profile(self.request()).success)
        self.assertIs(self.manager.torch_profiler, retry)
        retry.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
