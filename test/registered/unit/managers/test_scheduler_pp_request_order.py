import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    InitWeightsUpdateGroupReqInput,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerPPRequestOrder(CustomTestCase):
    def test_forwards_weight_update_group_init_before_local_processing(self):
        scheduler = SchedulerPPMixin()
        scheduler.pp_group = SimpleNamespace(is_last_rank=False)
        previous_send_work = ["previous-send"]
        current_send_work = ["current-send"]
        recv_reqs = [
            InitWeightsUpdateGroupReqInput(
                master_address="127.0.0.1",
                master_port=12345,
                rank_offset=0,
                world_size=2,
            )
        ]
        scheduler.send_req_work = previous_send_work

        calls = Mock()
        scheduler._pp_commit_comm_work = calls.commit
        scheduler._pp_send_pyobj_to_next_stage = calls.forward
        scheduler.process_input_requests = calls.process
        calls.forward.return_value = current_send_work

        scheduler._pp_forward_and_process_input_requests(recv_reqs)

        self.assertEqual(
            calls.mock_calls,
            [
                call.commit(previous_send_work),
                call.forward(recv_reqs, async_send=True),
                call.process(recv_reqs),
            ],
        )
        self.assertIs(scheduler.send_req_work, current_send_work)

    def test_last_stage_only_processes_requests(self):
        scheduler = SchedulerPPMixin()
        scheduler.pp_group = SimpleNamespace(is_last_rank=True)
        scheduler.send_req_work = []
        recv_reqs = [
            InitWeightsUpdateGroupReqInput(
                master_address="127.0.0.1",
                master_port=12345,
                rank_offset=0,
                world_size=2,
            )
        ]

        calls = Mock()
        scheduler._pp_commit_comm_work = calls.commit
        scheduler._pp_send_pyobj_to_next_stage = calls.forward
        scheduler.process_input_requests = calls.process

        scheduler._pp_forward_and_process_input_requests(recv_reqs)

        self.assertEqual(calls.mock_calls, [call.process(recv_reqs)])


if __name__ == "__main__":
    unittest.main()
