"""PP transport must preserve rank-local boundary tensors in both directions."""

import unittest
from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPPProxyTransport(CustomTestCase):
    def make_scheduler(self, *, send_group, recv_group):
        scheduler = SimpleNamespace(
            attn_tp_group=object(),
            pp_send_all_gather_group=send_group,
            pp_recv_all_gather_group=recv_group,
            pp_comm_stream_ctx=nullcontext(),
            pp_group=SimpleNamespace(
                send_tensor_dict=Mock(return_value=[]),
                recv_tensor_dict=Mock(
                    return_value={"__msg_type__": "proxy", "hidden_states": [1, 2]}
                ),
                is_first_rank=False,
            ),
            _pp_tensor_dict_inbox=defaultdict(deque),
            _pp_record_comm_event=lambda: None,
        )
        scheduler._pp_recv_typed_dict = lambda **kwargs: (
            SchedulerPPMixin._pp_recv_typed_dict(scheduler, **kwargs)
        )
        return scheduler

    def test_sender_preserves_rank_local_proxy(self):
        scheduler = self.make_scheduler(send_group=None, recv_group=object())
        SchedulerPPMixin._pp_send_dict_to_next_stage(
            scheduler, {"hidden_states": [1, 2]}, msg_type="proxy"
        )
        self.assertIsNone(
            scheduler.pp_group.send_tensor_dict.call_args.kwargs["all_gather_group"]
        )

    def test_receiver_preserves_rank_local_proxy(self):
        scheduler = self.make_scheduler(send_group=object(), recv_group=None)
        SchedulerPPMixin._pp_recv_proxy_tensors(scheduler)
        self.assertIsNone(
            scheduler.pp_group.recv_tensor_dict.call_args.kwargs["all_gather_group"]
        )

    def test_replicated_boundary_uses_the_directional_group(self):
        send_group, recv_group = object(), object()
        scheduler = self.make_scheduler(send_group=send_group, recv_group=recv_group)
        SchedulerPPMixin._pp_send_dict_to_next_stage(
            scheduler, {"hidden_states": [1, 2]}, msg_type="proxy"
        )
        SchedulerPPMixin._pp_recv_proxy_tensors(scheduler)
        self.assertIs(
            scheduler.pp_group.send_tensor_dict.call_args.kwargs["all_gather_group"],
            send_group,
        )
        self.assertIs(
            scheduler.pp_group.recv_tensor_dict.call_args.kwargs["all_gather_group"],
            recv_group,
        )

    def test_stashed_output_uses_the_same_link_policy_as_proxy(self):
        scheduler = self.make_scheduler(send_group=None, recv_group=None)
        scheduler.pp_group.recv_tensor_dict.side_effect = [
            {"__msg_type__": "output", "next_token_ids": [7]},
            {"__msg_type__": "proxy", "hidden_states": [1, 2]},
        ]
        SchedulerPPMixin._pp_recv_proxy_tensors(scheduler)
        output, _ = SchedulerPPMixin._pp_recv_dict_from_prev_stage(scheduler)
        self.assertEqual(output["next_token_ids"], [7])
        self.assertEqual(scheduler.pp_group.recv_tensor_dict.call_count, 2)
        for call in scheduler.pp_group.recv_tensor_dict.call_args_list:
            self.assertIsNone(call.kwargs["all_gather_group"])

    def test_npu_paired_exchange_keeps_asymmetric_link_groups(self):
        recv_group = object()
        scheduler = self.make_scheduler(send_group=None, recv_group=recv_group)
        scheduler.pp_group.is_last_rank = False
        scheduler.pp_group.send_recv_tensor_dict = Mock(
            return_value={"__msg_type__": "output", "__skip__": True}
        )
        scheduler._pp_make_skip_output_result = lambda *args: (None, "skipped", None)
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_prebuilt=lambda: False))
        result = SchedulerPPMixin._pp2_only_send_recv_output_tensors_npu(
            scheduler,
            0,
            0,
            [batch],
            [None],
            deque(),
            SimpleNamespace(tensors={"__skip__": True}),
        )
        self.assertEqual(result[1], "skipped")
        kwargs = scheduler.pp_group.send_recv_tensor_dict.call_args.kwargs
        self.assertIsNone(kwargs["send_all_gather_group"])
        self.assertIs(kwargs["recv_all_gather_group"], recv_group)


if __name__ == "__main__":
    unittest.main()
