"""PP must agree on metadata readiness before any stage consumes a transfer."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPPTransferReadiness(CustomTestCase):
    def stage(self, rank, *, room, status=KVPoll.Success, fake=False):
        receiver = SimpleNamespace(poll=lambda: status)
        req = SimpleNamespace(
            rid="delayed-metadata",
            bootstrap_host=FAKE_BOOTSTRAP_HOST if fake else "127.0.0.1",
        )
        queue = SimpleNamespace(
            queue=[
                SimpleNamespace(req=req, kv_receiver=receiver, metadata_buffer_index=0)
            ],
            metadata_buffers=SimpleNamespace(bootstrap_room=torch.tensor([[room]])),
        )
        scheduler = SimpleNamespace(
            pp_group=SimpleNamespace(is_first_rank=rank == 0),
            attn_cp_cpu_group="cp",
            attn_tp_cpu_group="tp",
            server_args=SimpleNamespace(disaggregation_transfer_backend="mooncake"),
            disagg_decode_transfer_queue=queue,
        )
        scheduler.get_rids = SchedulerPPMixin.get_rids.__get__(scheduler)
        return scheduler

    @patch("sglang.srt.disaggregation.utils.dist.all_reduce")
    def test_late_metadata_holds_all_pp_stages_until_next_consensus(self, reduce):
        first, last = self.stage(0, room=19), self.stage(1, room=0)
        get_ready = SchedulerPPMixin._pp_pd_get_decode_transferred_ids
        last._pp_recv_pyobj_from_prev_stage = lambda: get_ready(first)
        self.assertEqual(get_ready(last), [])
        self.assertEqual(len(first.disagg_decode_transfer_queue.queue), 1)
        self.assertEqual(len(last.disagg_decode_transfer_queue.queue), 1)
        last.disagg_decode_transfer_queue.metadata_buffers.bootstrap_room[0, 0] = 19
        self.assertEqual(get_ready(last), ["delayed-metadata"])
        self.assertEqual(
            [call.kwargs["group"] for call in reduce.call_args_list],
            ["tp", "cp"] * 4,
        )

    @patch("sglang.srt.disaggregation.utils.dist.all_reduce")
    def test_tp_metadata_gate_precedes_cp_and_pp_agreement(self, reduce):
        stage = self.stage(0, room=0)
        observed = []
        reduce.side_effect = lambda value, **kwargs: observed.append(
            (kwargs["group"], value.tolist())
        )
        self.assertEqual(SchedulerPPMixin._pp_pd_get_decode_transferred_ids(stage), [])
        self.assertEqual(
            observed,
            [("tp", [int(KVPoll.Transferring)]), ("cp", [int(KVPoll.Transferring)])],
        )

    @patch("sglang.srt.disaggregation.utils.dist.all_reduce")
    def test_failed_and_fake_transfers_do_not_wait_for_metadata(self, reduce):
        for status, fake in ((KVPoll.Failed, False), (KVPoll.Success, True)):
            with self.subTest(status=status, fake=fake):
                stage = self.stage(1, room=0, status=status, fake=fake)
                stage._pp_recv_pyobj_from_prev_stage = Mock(
                    return_value=["delayed-metadata"]
                )
                self.assertEqual(
                    SchedulerPPMixin._pp_pd_get_decode_transferred_ids(stage),
                    ["delayed-metadata"],
                )


if __name__ == "__main__":
    unittest.main()
