"""PP transport and graph buffers consume attention rows after handoff."""

import unittest
from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.model_executor.runner_utils.buffers import allocate_pp_proxy_tensors
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _TensorDictLink:
    """CPU model of the external send-slice / receive-gather boundary."""

    def __init__(self, messages, rank, world_size):
        self.messages = messages
        self.rank = rank
        self.world_size = world_size
        self.is_first_rank = False
        self.send_index = 0
        self.recv_index = 0

    def send_tensor_dict(self, tensor_dict, all_gather_group, async_send):
        message = {}
        for name, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                value = (
                    value.chunk(self.world_size)[self.rank]
                    if all_gather_group is not None
                    else value
                ).clone()
            message[name] = value
        self.messages[self.send_index][self.rank] = message
        self.send_index += 1
        return []

    def recv_tensor_dict(self, all_gather_group):
        pieces = self.messages[self.recv_index]
        self.recv_index += 1
        return {
            name: (
                torch.cat([pieces[rank][name] for rank in range(self.world_size)])
                if isinstance(value, torch.Tensor) and all_gather_group is not None
                else value
            )
            for name, value in pieces[self.rank].items()
        }


class TestPPProxyTransport(CustomTestCase):
    def make_scheduler(self, link):
        scheduler = SimpleNamespace(
            attn_tp_group=object(),
            pp_comm_stream_ctx=nullcontext(),
            pp_group=link,
            _pp_tensor_dict_inbox=defaultdict(deque),
            _pp_record_comm_event=lambda: None,
        )
        scheduler._pp_recv_typed_dict = lambda **kwargs: (
            SchedulerPPMixin._pp_recv_typed_dict(scheduler, **kwargs)
        )
        return scheduler

    def test_proxy_and_stashed_output_restore_attention_rows(self):
        for world_size in (2, 4):
            with self.subTest(attn_tp=world_size):
                messages = defaultdict(dict)
                schedulers = [
                    self.make_scheduler(_TensorDictLink(messages, rank, world_size))
                    for rank in range(world_size)
                ]
                hidden = torch.arange(32).reshape(8, 4)
                output = torch.arange(8)
                for scheduler in schedulers:
                    SchedulerPPMixin._pp_send_dict_to_next_stage(
                        scheduler, {"next_token_ids": output}, msg_type="output"
                    )
                    SchedulerPPMixin._pp_send_dict_to_next_stage(
                        scheduler, {"hidden_states": hidden}, msg_type="proxy"
                    )
                for scheduler in schedulers:
                    proxy = SchedulerPPMixin._pp_recv_proxy_tensors(scheduler)
                    torch.testing.assert_close(proxy.tensors["hidden_states"], hidden)
                    received, _ = SchedulerPPMixin._pp_recv_dict_from_prev_stage(
                        scheduler
                    )
                    torch.testing.assert_close(received["next_token_ids"], output)
                    self.assertEqual(scheduler.pp_group.recv_index, 2)

    def test_proxy_buffers_preserve_cp_and_mhc_dimensions(self):
        common = dict(
            max_num_tokens=16,
            max_hidden_tokens=8,
            hidden_size=4,
            dtype=torch.float32,
            pp_proxy_topk_size=3,
        )
        plain = allocate_pp_proxy_tensors(**common)
        self.assertEqual(plain["hidden_states"].shape, (8, 4))
        self.assertEqual(plain["residual"].shape, (8, 4))
        self.assertEqual(plain["topk_indices"].shape, (16, 3))
        mhc = allocate_pp_proxy_tensors(**common, hc_hidden_size=12)
        self.assertEqual(mhc["hidden_states"].shape, (8, 12))
        self.assertNotIn("residual", mhc)
        bank = allocate_pp_proxy_tensors(**common, pp_proxy_residual_num_blocks=5)
        self.assertEqual(bank["attn_res_bank"].shape, (16, 5, 4))
        self.assertNotIn("residual", bank)

    def test_npu_paired_exchange_preserves_skip_marker(self):
        link = SimpleNamespace(
            is_last_rank=False,
            send_recv_tensor_dict=Mock(
                return_value={"__msg_type__": "output", "__skip__": True}
            ),
        )
        scheduler = self.make_scheduler(link)
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
        self.assertTrue(result[0].tensors["__skip__"])
        kwargs = link.send_recv_tensor_dict.call_args.kwargs
        self.assertIs(kwargs["send_all_gather_group"], scheduler.attn_tp_group)
        self.assertIs(kwargs["recv_all_gather_group"], scheduler.attn_tp_group)


if __name__ == "__main__":
    unittest.main()
