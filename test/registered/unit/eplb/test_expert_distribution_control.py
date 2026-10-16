import asyncio
import unittest
from unittest.mock import AsyncMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.eplb.expert_distribution import (
    _ExpertDistributionRecorderNoop,
    set_global_expert_distribution_recorder,
)
from sglang.srt.managers.io_struct import (
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestExpertDistributionControl(CustomTestCase):
    def setUp(self):
        set_global_expert_distribution_recorder(_ExpertDistributionRecorderNoop())
        self.scheduler = Scheduler.__new__(Scheduler)

    def test_disabled_recorder_actions_return_failure(self):
        for action in (
            ExpertDistributionReqType.START_RECORD,
            ExpertDistributionReqType.STOP_RECORD,
            ExpertDistributionReqType.DUMP_RECORD,
        ):
            with self.subTest(action=action):
                output = self.scheduler.expert_distribution_handle(
                    ExpertDistributionReq(action=action)
                )

                self.assertFalse(output.success)
                self.assertIn(
                    "expert distribution",
                    output.message.lower(),
                )

    def test_tokenizer_manager_propagates_failure(self):
        manager = Mock(spec=TokenizerControlMixin)
        manager.auto_create_handle_loop = Mock()
        manager.expert_distribution_communicator = AsyncMock(
            return_value=[
                ExpertDistributionReqOutput(
                    success=False,
                    message="Expert distribution recording is disabled.",
                )
            ]
        )

        async def run():
            return await TokenizerControlMixin.stop_expert_distribution_record(manager)

        output = asyncio.run(run())

        self.assertIsInstance(output, ExpertDistributionReqOutput)
        self.assertFalse(output.success)
        self.assertIn("disabled", output.message.lower())

    def test_enabled_recorder_actions_return_success(self):
        recorder = Mock()
        set_global_expert_distribution_recorder(recorder)

        for action, method_name in (
            (ExpertDistributionReqType.START_RECORD, "start_record"),
            (ExpertDistributionReqType.STOP_RECORD, "stop_record"),
            (ExpertDistributionReqType.DUMP_RECORD, "dump_record"),
        ):
            with self.subTest(action=action):
                output = self.scheduler.expert_distribution_handle(
                    ExpertDistributionReq(action=action)
                )

                self.assertTrue(output.success)
                getattr(recorder, method_name).assert_called_once()
                getattr(recorder, method_name).reset_mock()


if __name__ == "__main__":
    unittest.main()
