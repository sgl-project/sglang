import pickle
import unittest
from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator, _split_tensor_dict
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers import scheduler_pp_mixin as pp_mod
from sglang.srt.managers import utils as managers_utils
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class FakeStream:
    def __init__(self, stream_id):
        self.cuda_stream = stream_id


class FakeEvent:
    def __init__(self):
        self.recorded_stream = None

    def record(self, stream):
        self.recorded_stream = stream


def _make_scheduler(**attrs):
    scheduler = object.__new__(SchedulerPPMixin)
    scheduler.__dict__.update(attrs)
    return scheduler


def _holds_tensor(value):
    if torch.is_tensor(value):
        return True
    if isinstance(value, (list, tuple)):
        return any(_holds_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_holds_tensor(item) for item in value.values())
    return False


class TestPPCommOverlap(CustomTestCase):
    def test_graph_proxy_send_records_forward_reuse_fence(self):
        comm_stream = FakeStream(4)
        work = Mock()
        works = [SimpleNamespace(work=work)]
        scheduler = _make_scheduler(
            pp_comm_stream=comm_stream,
            pp_comm_stream_ctx=nullcontext(),
            pp_send_done_event=None,
            device_module=SimpleNamespace(Event=FakeEvent),
        )

        scheduler._pp_commit_comm_work(works, fence_next_forward=True)

        work.wait.assert_called_once_with()
        self.assertEqual(works, [])
        self.assertIs(scheduler.pp_send_done_event.recorded_stream, comm_stream)

    def test_no_fence_event_without_comm_stream(self):
        scheduler = _make_scheduler(
            pp_comm_stream=None,
            pp_comm_stream_ctx=nullcontext(),
            pp_send_done_event=None,
        )

        scheduler._pp_commit_comm_work([SimpleNamespace(work=Mock())], True)

        self.assertIsNone(scheduler.pp_send_done_event)

    def test_forward_waits_for_graph_send_and_proxy_receive(self):
        schedule_stream = FakeStream(1)
        send_done_event = object()
        recv_event = object()
        forward_stream = Mock()
        scheduler = _make_scheduler(
            schedule_stream=schedule_stream,
            forward_stream=forward_stream,
            pp_send_done_event=send_done_event,
            pp_proxy_recv_event=recv_event,
        )

        scheduler._pp_wait_forward_dependencies()

        forward_stream.wait_stream.assert_called_once_with(schedule_stream)
        self.assertEqual(
            forward_stream.wait_event.call_args_list,
            [call(send_done_event), call(recv_event)],
        )
        self.assertIsNone(scheduler.pp_send_done_event)
        self.assertIsNone(scheduler.pp_proxy_recv_event)

    def test_inbox_returns_original_receive_event(self):
        recv_event = object()
        tensor_dict = {"__msg_type__": "output", "value": torch.arange(2)}
        scheduler = _make_scheduler(
            _pp_tensor_dict_inbox=defaultdict(
                deque, {"output": deque([(tensor_dict, recv_event)])}
            ),
        )

        received, event = scheduler._pp_recv_typed_dict("output")

        self.assertIs(received, tensor_dict)
        self.assertIs(event, recv_event)

    def _send_last_stage_output(self, tensor_dict):
        """Send one queued output message from the last stage; returns the message as
        sent and the order of host waits and sends."""
        ops, sent = [], []

        def send_tensor_dict(tensor_dict, all_gather_group, async_send):
            ops.append("send")
            sent.append(dict(tensor_dict))
            return []

        scheduler = _make_scheduler(
            pp_group=SimpleNamespace(
                is_last_rank=True, send_tensor_dict=send_tensor_dict
            ),
            pp_comm_stream_ctx=nullcontext(),
            attn_tp_group=None,
            device_module=SimpleNamespace(
                current_stream=lambda: SimpleNamespace(wait_event=lambda event: None)
            ),
        )
        rows_ready = SimpleNamespace(synchronize=lambda: ops.append("synchronize"))
        target = SimpleNamespace(forward_mode=pp_mod.ForwardMode.EXTEND)
        scheduler._pp_send_output_to_next_stage(
            0,
            [target],
            deque([(rows_ready, pp_mod.PPProxyTensors(tensor_dict))]),
            None,
        )
        return sent[0], ops

    def test_output_rows_reach_the_pickled_metadata_as_lists(self):
        """The output message pickles its per-request logprob rows: a device tensor
        there is rebuilt on the sender's device index by the receiver, and a host view
        pickles the whole buffer it shares with the other rows."""
        num_reqs, k = 256, 20
        message = {
            "next_token_ids": torch.zeros(num_reqs, dtype=torch.long),
            "next_token_top_logprobs_val": [
                torch.empty(k, device="meta") for _ in range(num_reqs)
            ],
            "next_token_top_logprobs_idx": [
                torch.empty(k, dtype=torch.long, device="meta") for _ in range(num_reqs)
            ],
            "next_token_token_ids_logprobs_idx": [[1, 2]] * num_reqs,
            "input_top_logprobs_val": None,
        }

        def to_host(tensor):
            return torch.arange(tensor.numel(), dtype=tensor.dtype)

        with patch.object(managers_utils, "_async_d2h", side_effect=to_host):
            self.assertTrue(pp_mod.pp_output_rows_to_host(message))
        sent, ops = self._send_last_stage_output(message)

        self.assertEqual(ops, ["synchronize", "send"])
        metadata, tensors = _split_tensor_dict(sent)
        self.assertEqual(len(tensors), 1)
        self.assertFalse(_holds_tensor([value for _, value in metadata]))
        self.assertEqual(sent["next_token_top_logprobs_idx"][1], list(range(k, 2 * k)))
        payload = num_reqs * k * (4 + 8)
        self.assertLess(len(pickle.dumps(metadata)), 3 * payload)

    def test_output_without_logprob_rows_sends_without_a_host_wait(self):
        token_ids = [[1, 2]]
        message = {
            "next_token_ids": torch.zeros(1, dtype=torch.long),
            "next_token_token_ids_logprobs_idx": token_ids,
        }
        self.assertFalse(pp_mod.pp_output_rows_to_host(message))
        sent, ops = self._send_last_stage_output(message)
        self.assertEqual(ops, ["send"])
        self.assertIs(sent["next_token_token_ids_logprobs_idx"], token_ids)

    def test_first_stage_processes_logprob_rows_received_as_lists(self):
        """The first stage builds its LogitsProcessorOutput from the received
        message, whose logprob rows are lists; prefill and decode result
        processing must take them as they take host tensor rows."""
        top_val = [torch.tensor([-0.5, -1.5]), torch.tensor([-0.25, -2.0])]
        top_idx = [torch.tensor([7, 3]), torch.tensor([1, 9])]
        ids_val = [torch.tensor([-3.0, -4.0]), torch.tensor([-5.0, -6.0])]
        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(
                next_token_logits=None,
                next_token_logprobs=torch.tensor([-0.5, -0.25]),
                next_token_top_logprobs_val=top_val,
                next_token_top_logprobs_idx=top_idx,
                next_token_token_ids_logprobs_val=ids_val,
                next_token_token_ids_logprobs_idx=[[5, 6], [5, 6]],
            )
        )
        message = managers_utils.get_logprob_dict_from_result(result)
        pp_mod.pp_output_rows_to_lists(
            message, SimpleNamespace(synchronize=lambda: None)
        )
        received = pickle.loads(pickle.dumps(message))
        processor = object.__new__(SchedulerBatchResultProcessor)
        object.__setattr__(
            processor, "logprob_result_processor", SchedulerLogprobResultProcessor
        )
        batch = SimpleNamespace(
            return_logprob=True,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )

        prefill_output, _, _ = managers_utils.get_logprob_from_pp_outputs(
            pp_mod.PPProxyTensors(dict(received))
        )
        processor.move_logprobs_to_cpu(batch=batch, logits_output=prefill_output)
        self.assertEqual(prefill_output.next_token_top_logprobs_idx, [[7, 3], [1, 9]])
        self.assertEqual(
            prefill_output.next_token_token_ids_logprobs_val,
            [[-3.0, -4.0], [-5.0, -6.0]],
        )

        decode_output, _, _ = managers_utils.get_logprob_from_pp_outputs(
            pp_mod.PPProxyTensors(dict(received))
        )
        next_token_ids, next_token_logprobs = processor._normalize_decode_outputs(
            batch=batch,
            result=result,
            still_in_batch_rows=[True, True],
            logits_output=decode_output,
            next_token_ids=torch.tensor([7, 1]),
        )
        logprob = SimpleNamespace(
            top_logprobs_num=2,
            token_ids_logprob=[5, 6],
            output_token_logprobs_val=[],
            output_token_logprobs_idx=[],
            output_top_logprobs_val=[],
            output_top_logprobs_idx=[],
            output_token_ids_logprobs_val=[],
            output_token_ids_logprobs_idx=[],
        )
        processor._apply_decode_logprobs(
            req=SimpleNamespace(logprob=logprob),
            i=1,
            batch=batch,
            next_token_id=next_token_ids[1],
            next_token_logprobs=next_token_logprobs,
            logits_output=decode_output,
        )
        self.assertEqual(logprob.output_token_logprobs_val, [-0.25])
        self.assertEqual(logprob.output_top_logprobs_val, [[-0.25, -2.0]])
        self.assertEqual(logprob.output_top_logprobs_idx, [[1, 9]])
        self.assertEqual(logprob.output_token_ids_logprobs_val, [[-5.0, -6.0]])

    def test_npu_pp2_outputs_match_across_paired_and_one_sided_transfers(self):
        """Output receives during graph replay must avoid model collectives,
        while paired and one-sided peers agree on every transmitted tensor.
        """
        for tp_size in (2, 4):
            for tp_rank in range(tp_size):
                for paired_stage in (0, 1):
                    with self.subTest(
                        tp_size=tp_size, tp_rank=tp_rank, paired_stage=paired_stage
                    ):
                        self._check_npu_pp2_transfers(tp_size, tp_rank, paired_stage)

    def _check_npu_pp2_transfers(self, tp_size, tp_rank, paired_stage):
        # Only the distributed boundary is replaced: real GroupCoordinator
        # methods serialize metadata, slice sends, allocate receives and gather.
        mailboxes = defaultdict(deque)
        active_stage = 0
        work = SimpleNamespace(wait=lambda: None)

        def send(tensor, dst, group=None, tag=0):
            mailboxes[active_stage, dst].append(tensor.clone())
            return work

        def recv(tensor, src, group=None, tag=0):
            sent = mailboxes[src, active_stage].popleft()
            self.assertEqual(tensor.shape, sent.shape)
            self.assertEqual(tensor.dtype, sent.dtype)
            tensor.copy_(sent)
            return work

        def p2p_op(op, tensor, peer, group=None):
            return SimpleNamespace(op=op, tensor=tensor, peer=peer, group=group)

        def batch_p2p(ops):
            return [op.op(op.tensor, op.peer, group=op.group) for op in ops]

        outputs = [
            {"next_token_ids": torch.arange(8) + offset, "empty": torch.empty(0)}
            for offset in (100, 200)
        ]
        expected_gather = outputs[0]["next_token_ids"]

        def all_gather(tensor, dim=0):
            self.assertEqual(active_stage, 1, "Model graph owns stage 0's HCCL group")
            self.assertEqual(dim, 0)
            torch.testing.assert_close(
                tensor, expected_gather.reshape(tp_size, -1)[tp_rank]
            )
            return expected_gather.flatten().clone()

        stream = SimpleNamespace(
            wait_event=lambda event: None, wait_stream=lambda s: None
        )
        schedulers = []
        for stage in (0, 1):
            group = object.__new__(GroupCoordinator)
            group.__dict__.update(
                rank=stage,
                rank_in_group=stage,
                ranks=[0, 1],
                world_size=2,
                cpu_group=object(),
                device_group=object(),
            )
            schedulers.append(
                _make_scheduler(
                    pp_group=group,
                    attn_tp_group=SimpleNamespace(
                        world_size=tp_size, rank_in_group=tp_rank, all_gather=all_gather
                    ),
                    pp_comm_stream=None,
                    pp_comm_stream_ctx=nullcontext(),
                    _pp_tensor_dict_inbox=defaultdict(deque),
                    copy_stream=stream,
                    copy_stream_ctx=nullcontext(),
                    schedule_stream=stream,
                    device_module=SimpleNamespace(
                        current_stream=lambda: stream, Event=FakeEvent
                    ),
                    _pp_prep_batch_result=lambda *args: None,
                )
            )

        with (
            published_topology(role="scheduler", pp_size=2, tp_size=tp_size),
            patch.object(pp_mod, "_is_npu", True),
            pp_mod.envs.SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM.override(False),
            patch.multiple(
                torch.distributed,
                is_initialized=lambda: True,
                isend=send,
                irecv=recv,
                P2POp=p2p_op,
                batch_isend_irecv=batch_p2p,
            ),
        ):
            peer = 1 - paired_stage
            active_stage = peer
            schedulers[peer]._pp_send_dict_to_next_stage(
                dict(outputs[peer]), msg_type="output"
            )
            self.assertEqual(
                mailboxes[peer, paired_stage][-1].numel(),
                8 if peer == 1 else 8 // tp_size,
            )
            active_stage = paired_stage
            payload = pp_mod.PPProxyTensors(outputs[paired_stage])
            target = SimpleNamespace(forward_mode=pp_mod.ForwardMode.DECODE)
            received, _, _, _ = schedulers[
                paired_stage
            ]._pp2_only_send_recv_output_tensors_npu(
                0, 1, [target, target], [None, None], deque([(None, payload)]), payload
            )
            self.assertEqual(
                mailboxes[paired_stage, peer][-1].numel(),
                8 if paired_stage == 1 else 8 // tp_size,
            )
            torch.testing.assert_close(
                received.tensors["next_token_ids"], outputs[peer]["next_token_ids"]
            )
            self.assertEqual(received.tensors["empty"].numel(), 0)
            active_stage = peer
            received, _ = schedulers[peer]._pp_recv_dict_from_prev_stage()
            torch.testing.assert_close(
                received["next_token_ids"], outputs[paired_stage]["next_token_ids"]
            )
            self.assertEqual(received["empty"].numel(), 0)
            self.assertTrue(all(not queue for queue in mailboxes.values()))

            # The opposite direction also carries proxies: preserve their
            # sharding and reconstruct the original hidden-state shape/values.
            expected_gather = torch.arange(8).reshape(2, 4)
            active_stage = 0
            schedulers[0]._pp_send_dict_to_next_stage(
                {"hidden_states": expected_gather}, msg_type="proxy"
            )
            self.assertEqual(mailboxes[0, 1][-1].numel(), 8 // tp_size)
            active_stage = 1
            proxy = schedulers[1]._pp_recv_proxy_tensors()
            torch.testing.assert_close(proxy.tensors["hidden_states"], expected_gather)
            self.assertTrue(all(not queue for queue in mailboxes.values()))


if __name__ == "__main__":
    unittest.main()
