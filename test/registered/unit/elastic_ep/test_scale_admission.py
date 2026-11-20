"""Scheduler admission of a /scale_elastic_ep request: the resend of a pending target
and the joiner widths a recover grow must match."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.elastic_ep import elastic_ep as eep
from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
from sglang.srt.managers.io_struct import ScaleElasticEPReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

LAUNCH_EP = 4


def _scheduler():
    return SimpleNamespace(
        server_args=SimpleNamespace(
            elastic_ep_backend="mooncake",
            elastic_ep_initial_size=LAUNCH_EP,
            tp_size=LAUNCH_EP,
        ),
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(eplb_manager=None)),
    )


def _admit(new_ep_size, *, effective, scaling=False, pending=None, widths=None):
    """Run the scheduler's admission at ``effective``; slots past it are retired."""
    parallel = SimpleNamespace(max_world_size=8, attn_tp_size=1, attn_cp_size=1)
    active = [1] * effective + [0] * (LAUNCH_EP - effective)
    with ExitStack() as stack:
        for target in (
            "sglang.srt.managers.scheduler.get_parallel",
            "sglang.srt.elastic_ep.topology.get_parallel",
        ):
            stack.enter_context(patch(target, return_value=parallel))
        for name, value in {
            "get_effective_ep_size": effective,
            "is_scaling": scaling,
            "get_inactive_ranks": (),
            "get_pending_ep_size": pending,
            "get_scale_phase": "pending" if scaling else "idle",
            "instance": SimpleNamespace(active_ranks_cpu=active),
        }.items():
            stack.enter_context(
                patch.object(ElasticEPStateManager, name, return_value=value)
            )
        request_scale = stack.enter_context(
            patch.object(ElasticEPStateManager, "request_scale", return_value=True)
        )
        stack.enter_context(
            patch.object(eep, "announced_recover_widths", return_value=widths)
        )
        result = Scheduler.handle_scale_elastic_ep(
            _scheduler(), ScaleElasticEPReqInput(new_ep_size=new_ep_size)
        )
    return result, request_scale


class TestPendingResend(CustomTestCase):
    def test_resend_of_the_pending_target_is_accepted_without_requeueing(self):
        """A rejection here would fail the call while peers wait on this rank."""
        result, request_scale = _admit(4, effective=2, scaling=True, pending=4)
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.pending_ep_size, 4)
        request_scale.assert_not_called()

    def test_a_different_target_while_pending_is_still_rejected(self):
        result, request_scale = _admit(3, effective=2, scaling=True, pending=4)
        self.assertFalse(result.success)
        self.assertIn("has not completed yet", result.message)
        request_scale.assert_not_called()


class TestRecoverWidths(CustomTestCase):
    def test_a_target_matching_the_announced_width_is_admitted(self):
        result, request_scale = _admit(4, effective=2, widths={4})
        self.assertTrue(result.success, result.message)
        request_scale.assert_called_once_with(4, [2, 3])

    def test_a_target_past_the_announced_width_is_rejected(self):
        """The joiner sizes its map to 3, so a grow to 4 would kill it."""
        result, request_scale = _admit(4, effective=2, widths={3})
        self.assertFalse(result.success)
        self.assertIn("Grow to 3 first", result.message)
        request_scale.assert_not_called()

    def test_a_target_below_the_announced_width_names_it(self):
        result, request_scale = _admit(3, effective=2, widths={4})
        self.assertFalse(result.success)
        self.assertEqual(result.required_ep_size, 4)
        request_scale.assert_not_called()

    def test_joiners_announcing_different_widths_are_rejected(self):
        result, request_scale = _admit(4, effective=2, widths={3, 4})
        self.assertFalse(result.success)
        self.assertIn("Grow to 3 first", result.message)
        request_scale.assert_not_called()

    def test_no_announcement_asks_for_a_retry(self):
        result, request_scale = _admit(4, effective=2, widths=None)
        self.assertFalse(result.success)
        self.assertTrue(result.retry_when_joiner_announces)
        request_scale.assert_not_called()


if __name__ == "__main__":
    unittest.main()
