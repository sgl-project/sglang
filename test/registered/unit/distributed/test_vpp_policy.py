"""VPP scheduling lifetimes and FIFO transport without a model or accelerator."""

import json
import unittest
from collections import deque
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import torch

from sglang.srt.distributed.vpp_policy import (
    PipelineControlEnvelope,
    PipelineControlKind,
    PipelinePrefixRegistry,
    PipelineRankSchedule,
)
from sglang.srt.distributed.vpp_transport import (
    P2PWorkGroup,
    VPPSchedulerActivationReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestVPPPolicy(CustomTestCase):
    def test_control_forwarding_preserves_batch_and_runtime_identity(self):
        """Forwarded controls must still refer to the originating batch."""
        for kind in PipelineControlKind:
            with self.subTest(kind=kind):
                message = PipelineControlEnvelope(
                    protocol_version=1,
                    runtime_epoch=7,
                    layout_digest="layout",
                    kind=kind,
                    source_rank=1,
                    batch_seq=5,
                    generation=3,
                    slot_id=1,
                    payload={"rids": ["request"]},
                )
                wire = json.loads(json.dumps(message.to_dict()))
                self.assertEqual(
                    PipelineControlEnvelope.from_dict(wire).forwarded(),
                    replace(message, hops=1),
                )
                self.assertEqual(message.hops, 0)
        for invalid in (
            {"__msg_type__": "other"},
            {**wire, "kind": "unknown"},
            {**wire, "generation": None},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    PipelineControlEnvelope.from_dict(invalid)

    def test_slot_survives_cancellation_and_first_pass_until_retirement(self):
        """Cancelled work must drain before a modulo slot can be reused."""
        ranks = [PipelineRankSchedule(rank, 2, 2, 2) for rank in range(2)]
        for rank in ranks:
            rank.admit(0)
            rank.admit(1)
            rank.cancel(0)
            with self.assertRaises(RuntimeError):
                rank.retire(0)
        for tick, stage in enumerate(range(4)):
            rank = ranks[stage % 2]
            if stage:
                rank.mark_ready(0, stage)
            action = rank.next_action(tick)
            self.assertEqual((action.batch_seq, action.stage_id), (0, stage))
            self.assertIsNone(rank.next_action(tick))
            with self.assertRaises(RuntimeError):
                rank.mark_ready(0, stage)
            transition = rank.complete(action)
            self.assertTrue(transition.cancelled)
            self.assertEqual(transition.first_pass_done, stage < 2)
            self.assertEqual(transition.batch_complete, stage == 3)
            self.assertEqual(
                transition.successor_stage_id, stage + 1 if stage < 3 else None
            )
            self.assertFalse(rank.can_admit(2))
        # Rank zero learns global completion through a control message.
        with self.assertRaises(RuntimeError):
            ranks[0].retire(0)
        ranks[0].mark_completed(0)
        for rank in ranks:
            rank.retire(0)
            self.assertTrue(rank.can_admit(2))
            rank.admit(2)
            self.assertEqual(rank.slot_batch_seqs, (2, 1))
        with self.assertRaises(RuntimeError):
            ranks[0].mark_ready(0, 2)
        with self.assertRaises(RuntimeError):
            ranks[0].mark_ready(2, 1)

    def test_second_pass_priority_respects_input_readiness(self):
        """A missing second-pass activation must not stall ready first passes."""
        rank = PipelineRankSchedule(0, 2, 2, 2)
        rank.admit(0)
        rank.admit(1)
        rank.complete(rank.next_action(0))
        rank.mark_ready(0, 2)
        first_pass = rank.next_action(1, lambda batch, stage: stage == 0)
        self.assertEqual((first_pass.batch_seq, first_pass.stage_id), (1, 0))
        rank.complete(first_pass)
        rank.mark_ready(1, 2)
        second_pass = rank.next_action(2)
        self.assertEqual((second_pass.batch_seq, second_pass.stage_id), (0, 2))

    def test_prefix_commit_waits_for_every_stage_and_residency_reset(self):
        """A fast stage cannot publish KV that another stage has not produced."""
        registry = PipelinePrefixRegistry()
        entry = registry.plan("request", 1, 3, 4, 12, range(4))
        for stage in range(3):
            self.assertEqual(registry.mark_materialized("request", 1, stage, 12), 4)
        with self.assertRaises(RuntimeError):
            registry.commit("request", 1, 12)
        registry.mark_materialized("request", 1, 3, 8)
        registry.commit("request", 1, 8)
        self.assertTrue(entry.locked)
        registry.invalidate_residency("request", 1, 4)
        self.assertEqual(
            (entry.planned_end, entry.committed_end, entry.materialized_end), (8, 8, 8)
        )
        self.assertFalse(entry.locked)
        with self.assertRaises(RuntimeError):
            registry.plan("request", 1, 3, 8, 12, range(4))
        with self.assertRaises(RuntimeError):
            registry.plan("request", 1, 4, 12, 16, range(4))
        registry.plan("request", 1, 4, 8, 12, range(4))
        for stage in range(4):
            registry.mark_materialized("request", 1, stage, 12)
        registry.commit("request", 1, 12)
        self.assertFalse(entry.locked)
        new_request = registry.plan("request", 2, 0, 0, 4, range(4))
        registry.release("request", 1)
        self.assertIs(registry.get("request", 2), new_request)


class _CPUWork:
    def __init__(self, payload, error=None):
        self.payload = payload
        self.work = self
        self.entered = Event()
        self.release = Event()
        self.error = error

    def wait(self):
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("CPU work was not released")
        if self.error is not None:
            raise self.error

    def is_completed(self):
        raise AssertionError("CPU P2P progress requires wait(), not is_completed()")


class TestVPPTransport(CustomTestCase):
    def test_send_credit_retains_payload_until_cpu_and_device_work_finish(self):
        """Host metadata completion alone must not release device send credit."""
        cpu = _CPUWork(torch.ones(4))
        device = SimpleNamespace(
            payload=None,
            work=SimpleNamespace(is_completed=lambda: device_done.is_set()),
            wait=lambda: None,
        )
        device_done = Event()
        items = [cpu, device]
        group = P2PWorkGroup(items)
        try:
            self.assertTrue(cpu.entered.wait(timeout=5))
            self.assertFalse(group.poll())
            self.assertEqual(group.payload_bytes(), 16)
            cpu.release.set()
            self.assertTrue(group._done.wait(timeout=5))
            self.assertFalse(group.poll())
            self.assertEqual(group.payload_bytes(), 16)
            device_done.set()
            self.assertTrue(group.poll())
            self.assertEqual(items, [])
            self.assertEqual(group.payload_bytes(), 0)
        finally:
            cpu.release.set()
            group._done.wait(timeout=5)

    def test_cpu_send_failure_propagates_without_releasing_credit(self):
        cpu = _CPUWork(torch.ones(1), error=RuntimeError("send failed"))
        items = [cpu]
        group = P2PWorkGroup(items)
        cpu.release.set()
        self.assertTrue(group._done.wait(timeout=5))
        with self.assertRaisesRegex(RuntimeError, "send failed"):
            group.poll()
        self.assertEqual(items, [cpu])
        self.assertEqual(group.payload_bytes(), 4)

    def test_receiver_consumes_only_the_ready_fifo_head(self):
        """A ready later receive cannot overtake the pending FIFO head."""
        ready = [Event(), Event()]
        ready[1].set()
        handles = deque(
            SimpleNamespace(
                poll_metadata_ready=event.is_set,
                wait=lambda index=index: {"batch_seq": index},
            )
            for index, event in enumerate(ready)
        )
        # The continuous scheduler posts the next receive after every take.
        handles.append(SimpleNamespace(poll_metadata_ready=lambda: False))
        group = SimpleNamespace(
            device="cpu",
            recv_tensor_dict_async=lambda *, tag, all_gather_group, batch_p2p: (
                handles.popleft()
            ),
        )
        receiver = VPPSchedulerActivationReceiver(group, None, batch_p2p=False, tag=17)
        receiver.start()
        receiver.progress()
        self.assertEqual(receiver.receive_count(), 0)
        ready[0].set()
        receiver.progress()
        self.assertEqual(receiver.receive_count(), 1)
        self.assertEqual(receiver.take(), {"batch_seq": 0})
        self.assertEqual(receiver.receive_count(), 0)
        receiver.progress()
        self.assertEqual(receiver.receive_count(), 1)
        self.assertEqual(receiver.take(), {"batch_seq": 1})
        receiver.progress()
        self.assertEqual(receiver.receive_count(), 0)


if __name__ == "__main__":
    unittest.main()
