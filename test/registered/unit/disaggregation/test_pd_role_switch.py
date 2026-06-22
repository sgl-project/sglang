import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    PdRoleSwitchReqInput,
    PdRoleSwitchReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestPdRoleSwitchServerArg(unittest.TestCase):
    def test_flag_defaults_to_false(self):
        field = ServerArgs.__dataclass_fields__["enable_pd_role_switch"]
        self.assertIs(field.default, False)

    def test_cli_flag_parses(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        off = parser.parse_args(["--model-path", "dummy"])
        self.assertFalse(off.enable_pd_role_switch)

        on = parser.parse_args(["--model-path", "dummy", "--enable-pd-role-switch"])
        self.assertTrue(on.enable_pd_role_switch)


class TestHandlePdRoleSwitch(unittest.TestCase):
    """Cover the control-plane contract of Scheduler.handle_pd_role_switch.

    Only the role-flip *decision* logic is exercised here (no GPU): the heavy
    teardown/rebuild is mocked, so this asserts the guard branches and the
    orchestration order without standing up a model.
    """

    def _scheduler(self, mode, *, enable=True, idle=True):
        s = Scheduler.__new__(Scheduler)
        s.disaggregation_mode = mode
        s.server_args = SimpleNamespace(
            enable_pd_role_switch=enable,
            disaggregation_mode=mode.value,
        )
        s.is_fully_idle = MagicMock(return_value=idle)
        s._teardown_disaggregation = MagicMock()
        s.init_disaggregation = MagicMock()
        s._event_loop_should_restart = False
        return s

    def test_rejected_when_flag_disabled(self):
        s = self._scheduler(DisaggregationMode.PREFILL, enable=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertIsInstance(out, PdRoleSwitchReqOutput)
        self.assertFalse(out.success)
        self.assertIn("enable-pd-role-switch", out.message)
        s._teardown_disaggregation.assert_not_called()

    def test_rejected_on_invalid_role(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(s, PdRoleSwitchReqInput(new_role="both"))
        self.assertFalse(out.success)
        self.assertIn("invalid new_role", out.message)
        s._teardown_disaggregation.assert_not_called()

    def test_rejected_when_not_in_pd_mode(self):
        s = self._scheduler(DisaggregationMode.NULL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("not running in PD", out.message)
        s._teardown_disaggregation.assert_not_called()

    def test_same_role_is_noop(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="prefill")
        )
        self.assertTrue(out.success)
        self.assertEqual(out.message, "already in target role")
        s._teardown_disaggregation.assert_not_called()
        s.init_disaggregation.assert_not_called()
        self.assertFalse(s._event_loop_should_restart)

    def test_rejected_when_not_idle(self):
        s = self._scheduler(DisaggregationMode.PREFILL, idle=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("not idle", out.message)
        s._teardown_disaggregation.assert_not_called()

    def test_successful_flip_orchestration(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        self.assertTrue(out.success)
        self.assertEqual(out.old_role, "prefill")
        self.assertEqual(out.new_role, "decode")
        # Orchestration: drain -> teardown -> flip server arg -> rebuild -> signal.
        s._teardown_disaggregation.assert_called_once()
        self.assertEqual(s.server_args.disaggregation_mode, "decode")
        s.init_disaggregation.assert_called_once()
        self.assertTrue(s._event_loop_should_restart)


if __name__ == "__main__":
    unittest.main()
