import unittest
from collections import defaultdict, deque
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.distributed.vpp_activation import (
    VPPActivationReceiver,
    VPPReceivedActivation,
    VPPSendWindow,
)
from sglang.srt.managers import scheduler_pp_mixin as pp_mod
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
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


class TestVPPActivationTransport(CustomTestCase):
    """Protect wire compatibility and cross-stream activation storage lifetime."""

    def test_async_metadata_progress_does_not_wait_for_payload(self):
        # The actual GroupCoordinator sender and receiver share a CPU mailbox.
        # Work completion is controlled explicitly, without timing-based sleeps.
        mailbox = deque()
        metadata_gate = Event()
        payload_gate = Event()

        class ReceiveWork:
            def __init__(self, tensor, sent):
                self.tensor = tensor
                self.sent = sent

            def wait(self):
                if self.tensor.dtype == torch.long:
                    if not metadata_gate.wait(timeout=5):
                        raise RuntimeError("Metadata was never released")
                if self.tensor.dtype == torch.float32:
                    if not payload_gate.wait(timeout=5):
                        raise RuntimeError("Unexpected blocking payload wait")
                self.tensor.copy_(self.sent)

        def send(tensor, dst, group=None, tag=0):
            mailbox.append(tensor.clone())
            return SimpleNamespace(wait=lambda: None)

        def recv(tensor, src, group=None, tag=0):
            return ReceiveWork(tensor, mailbox.popleft())

        group = object.__new__(GroupCoordinator)
        group.__dict__.update(
            rank_in_group=0,
            ranks=[0, 1],
            world_size=2,
            cpu_group=object(),
            device_group=object(),
            device=torch.device("cpu"),
        )
        expected = {
            "batch_seq": 7,
            "stage_id": 2,
            "hidden_states": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "hc_pre": torch.empty(0),
        }
        with patch.multiple(
            torch.distributed,
            is_initialized=lambda: True,
            isend=send,
            irecv=recv,
        ):
            group.send_tensor_dict(expected, dst=1, async_send=True)
            handle = group.recv_tensor_dict_async(src=1)
            self.assertFalse(handle.poll_metadata_ready())
            self.assertEqual(len(mailbox), 2)
            metadata_gate.set()
            self.assertTrue(handle._metadata_recv_waiter._done.wait(timeout=5))
            handle.poll_metadata_ready()
            self.assertTrue(handle._metadata_recv_waiter._done.wait(timeout=5))
            self.assertTrue(handle.poll_metadata_ready())
            self.assertFalse(payload_gate.is_set())
            payload_gate.set()
            actual = handle.wait()
            self.assertEqual(actual["batch_seq"], 7)
            self.assertEqual(actual["stage_id"], 2)
            torch.testing.assert_close(
                actual["hidden_states"], expected["hidden_states"]
            )
            self.assertEqual(actual["hc_pre"].numel(), 0)
            self.assertIs(handle.wait(), actual)
            self.assertFalse(mailbox)

    def test_async_metadata_wait_propagates_distributed_error(self):
        def fail():
            raise RuntimeError("peer disconnected")

        group = object.__new__(GroupCoordinator)
        group.__dict__.update(
            rank_in_group=0, ranks=[0, 1], world_size=2, cpu_group=object()
        )
        with patch.multiple(
            torch.distributed,
            is_initialized=lambda: True,
            irecv=lambda *args, **kwargs: SimpleNamespace(wait=fail),
        ):
            with self.assertRaisesRegex(RuntimeError, "peer disconnected"):
                group.recv_tensor_dict_async(src=1).wait()

    def _device_module(self):
        stream = Mock()
        module = SimpleNamespace(
            Stream=Mock(return_value=stream),
            stream=lambda _: nullcontext(),
            Event=Mock(side_effect=lambda: Mock()),
        )
        return module, stream

    def test_receiver_keeps_event_local_and_protects_forward_storage(self):
        module, comm_stream = self._device_module()
        tensor = Mock(spec=torch.Tensor)
        tensor.is_cpu = False
        tensors = {"hidden_states": tensor, "batch_seq": 0}
        handle = Mock()
        handle.wait.return_value = tensors
        handle.poll_metadata_ready.return_value = False
        group = SimpleNamespace(
            device="cuda",
            recv_tensor_dict_async=Mock(return_value=handle),
        )
        with patch.object(torch, "get_device_module", return_value=module):
            receiver = VPPActivationReceiver(group)
            receiver.start()
            self.assertFalse(receiver.progress())
            handle.wait.assert_not_called()
            received = receiver.take()
        self.assertIsInstance(received, VPPReceivedActivation)
        self.assertIs(received.tensors, tensors)
        self.assertEqual(set(tensors), {"hidden_states", "batch_seq"})
        received.recv_event.record.assert_called_once_with(comm_stream)
        forward_stream = Mock()
        received.wait_on(forward_stream)
        forward_stream.wait_event.assert_called_once_with(received.recv_event)
        tensor.record_stream.assert_called_once_with(forward_stream)
        self.assertEqual(group.recv_tensor_dict_async.call_count, 2)

    def test_send_credit_and_graph_reuse_fence_all_unfinished_readers(self):
        module, comm_stream = self._device_module()
        operations = []
        items = []
        for index in range(3):
            tensor = Mock(spec=torch.Tensor)
            tensor.is_cpu = False
            work = Mock()
            work.wait.side_effect = lambda i=index: operations.append(("wait", i))
            tensor.record_stream.side_effect = lambda stream, i=index: (
                operations.append(("retain", i))
            )
            items.append(SimpleNamespace(wait=work.wait, payload=tensor))
        group = SimpleNamespace(
            device="cuda",
            send_tensor_dict=Mock(side_effect=[[item] for item in items]),
        )
        forward_event = object()
        with patch.object(torch, "get_device_module", return_value=module):
            window = VPPSendWindow(group, max_inflight=2)
            with self.assertRaisesRegex(ValueError, "requires a forward event"):
                window.send({})
            window.send({}, forward_event)
            window.send({}, forward_event)
            self.assertEqual(operations, [])
            with self.assertRaisesRegex(RuntimeError, "Acquire"):
                window.send({}, forward_event)
            fence = window.acquire()
            fence.record.assert_called_once_with(comm_stream)
            self.assertEqual(operations, [("wait", 0), ("retain", 0)])
            window.send({}, forward_event)
            # Both remaining sends are inside the credit window, but their
            # storage is about to be overwritten by the next graph replay.
            fence = window.wait_for_reuse()
            fence.record.assert_called_once_with(comm_stream)
            self.assertEqual(
                operations,
                [
                    ("wait", 0),
                    ("retain", 0),
                    ("wait", 1),
                    ("retain", 1),
                    ("wait", 2),
                    ("retain", 2),
                ],
            )
            self.assertEqual(len(window), 0)
            self.assertIsNone(window.wait_for_reuse())
        self.assertEqual(
            comm_stream.wait_event.call_args_list, [call(forward_event)] * 3
        )
        for item in items:
            item.payload.record_stream.assert_called_once_with(comm_stream)


if __name__ == "__main__":
    unittest.main()
