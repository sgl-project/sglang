"""Control replies that need every pipeline stage come from the last one."""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
)
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

MODULE = "sglang.srt.managers.scheduler_components.ipc_channels"


class TestSendControlOutput(CustomTestCase):
    def setUp(self):
        self.first_stage = MagicMock()
        self.last_stage = MagicMock()
        self.channels = SchedulerIpcChannels(
            recv_from_tokenizer=None,
            recv_from_rpc=None,
            send_to_tokenizer=self.first_stage,
            send_to_tokenizer_from_last_stage=self.last_stage,
            send_to_detokenizer=MagicMock(),
            send_metrics_from_scheduler=None,
        )

    def test_the_release_memory_ack_is_sent_by_the_last_stage(self):
        request = ReleaseMemoryOccupationReqInput()
        output = ReleaseMemoryOccupationReqOutput(success=True, message="Success")
        self.channels.send_control_output(output, request)
        self.last_stage.send_output.assert_called_once_with(output, request)
        self.first_stage.send_output.assert_not_called()

    def test_other_control_replies_are_sent_by_the_first_stage(self):
        request = ResumeMemoryOccupationReqInput()
        output = ResumeMemoryOccupationReqOutput()
        self.channels.send_control_output(output, request)
        self.first_stage.send_output.assert_called_once_with(output, request)
        self.last_stage.send_output.assert_not_called()


class TestCreate(CustomTestCase):
    def _create(self, *, is_rank_zero, is_last_stage_io_rank):
        with patch(f"{MODULE}.zmq.Context"), patch(f"{MODULE}.get_zmq_socket"):
            return SchedulerIpcChannels.create(
                port_args=MagicMock(),
                is_rank_zero=is_rank_zero,
                is_last_stage_io_rank=is_last_stage_io_rank,
                skip_tokenizer_init=False,
                metrics_enabled=False,
                enable_scripted_runtime=False,
            )

    def test_a_later_last_stage_only_has_the_last_stage_socket(self):
        channels = self._create(is_rank_zero=False, is_last_stage_io_rank=True)
        self.assertIsNone(channels.send_to_tokenizer.socket)
        self.assertIsNotNone(channels.send_to_tokenizer_from_last_stage.socket)

    def test_the_first_stage_of_a_pipeline_has_no_last_stage_socket(self):
        channels = self._create(is_rank_zero=True, is_last_stage_io_rank=False)
        self.assertIsNotNone(channels.send_to_tokenizer.socket)
        self.assertIsNone(channels.send_to_tokenizer_from_last_stage.socket)


if __name__ == "__main__":
    unittest.main()
