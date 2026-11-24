"""Finite VPP waves against queue transport and CPU model boundaries."""

import unittest
from array import array
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.distributed.vpp_activation import VPPReceivedActivation
from sglang.srt.distributed.vpp_batch import VPPBatchLedger, VPPBatchState
from sglang.srt.distributed.vpp_layout import VPPLayout
from sglang.srt.distributed.vpp_policy import (
    PipelineControlEnvelope,
    PipelineControlKind,
    PipelinePrefixRegistry,
    PipelineRankSchedule,
    PipelineReplicaIdentity,
    PipelineWavefrontAction,
)
from sglang.srt.distributed.vpp_schedule import VPPFixedSchedule
from sglang.srt.distributed.vpp_transport import P2PWorkGroup
from sglang.srt.managers import scheduler_vpp_wave_mixin as vpp_mod
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.scheduler_vpp_core import SchedulingVPPCore
from sglang.srt.managers.scheduler_vpp_wave_mixin import SchedulerVPPMixin
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner.vpp_stage import execute_vpp_stage
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Stream:
    def wait_stream(self, stream):
        pass

    def wait_event(self, event):
        pass


class _Event:
    def __init__(self):
        self.recorded = False

    def record(self, stream=None):
        self.recorded = True


class _Receiver:
    def __init__(self, queue):
        self.queue = queue
        self.posted = False

    def start(self):
        if self.posted:
            raise RuntimeError("Receive leaked from the preceding wave")
        self.posted = True

    def take(self, *, post_next=True):
        if not self.posted:
            raise RuntimeError("No receive posted")
        result = self.queue.get(timeout=10)
        self.posted = post_next
        return VPPReceivedActivation(result)


class _Sends:
    def __init__(self, queue):
        self.queue = queue

    def acquire(self):
        return None

    def send(self, tensors, forward_event):
        if not forward_event.recorded:
            raise RuntimeError("Activation published before the forward fence")
        self.queue.put(tensors)

    def wait_for_reuse(self):
        return None


class _Worker:
    def __init__(self, rank, calls):
        self.rank = rank
        self.calls = calls
        self.model_runner = object()
        # Deliberately reused output storage; the scheduler must snapshot views.
        self.workspace = torch.empty(1)

    def set_hicache_consumer(self, index):
        pass

    def forward_batch_generation(
        self, batch, *, forward_batch, pp_proxy_tensors, vpp_stage_id
    ):
        self.calls.append((forward_batch.batch_seq, vpp_stage_id))
        value = (
            forward_batch.batch_seq
            if pp_proxy_tensors is None
            else pp_proxy_tensors["hidden_states"].item()
        )
        self.workspace.fill_(value + 1)
        if vpp_stage_id == forward_batch.final_stage:
            return SimpleNamespace(
                next_token_ids=self.workspace,
                pp_hidden_states_proxy_tensors=None,
            )
        return SimpleNamespace(
            pp_hidden_states_proxy_tensors=PPProxyTensors(
                {"hidden_states": self.workspace}
            )
        )


class _SchedulingCore(SchedulingVPPCore):
    def _pp_vpp_start_receiver(self):
        # The native fixed-TP adapter keeps its continuous receiver posted.
        pass


class _AdmissionCore(_SchedulingCore):
    """Supply CPU batch construction and prefix storage at the adapter boundary."""

    def _pp_vpp_prepare_wavefront_batch(
        self, batch_seq, slot_batch_seqs, expected_manifest=None
    ):
        slot_id = batch_seq % len(slot_batch_seqs)
        self.mbs[slot_id] = self.candidate_batch
        slot_batch_seqs[slot_id] = batch_seq
        return self.candidate_batch

    def _pp_vpp_register_prefix_batch(self, batch):
        for req, start, length in zip(batch.reqs, batch.prefix_lens, batch.extend_lens):
            generation = int(req.session_generation or 0)
            self._pp_vpp_prefix_registry.plan(
                req.rid, generation, generation, start, start + length, range(4)
            )


class _ControlCore(_AdmissionCore):
    """Keep communication and request queues on CPU for control-loop tests."""

    def _init_scheduling_native_loop_state(self):
        self.mbs = [None, None]
        self.last_mbs = [None, None]
        self.mb_metadata = [None, None]
        self.bootstrap_outcomes = {}

    def _pp_vpp_max_inflight(self):
        return 2

    def _pp_vpp_layout_digest(self):
        return "layout"

    def _pp_vpp_activation_group(self, source_rank):
        self.activation_source = source_rank
        return self.activation_group

    def _pp_vpp_start_control_receiver(self):
        self.receive_tags.append(2)

    def process_input_requests(self, requests):
        self.waiting_queue.extend(requests)

    def _pp_vpp_try_apply_bootstrap_status(self, boundaries, good, bad):
        good.difference_update(
            rid
            for rid, status in self.bootstrap_outcomes.items()
            if status == KVPoll.WaitingForInput
        )
        bad.difference_update(
            rid
            for rid, status in self.bootstrap_outcomes.items()
            if status == KVPoll.Failed
        )
        return not good and not bad


class TestSchedulingVPPCore(CustomTestCase):
    def _scheduler(self, *, rank=0, owner=0, retained=True, core_type=_SchedulingCore):
        scheduler = core_type()
        scheduler.__dict__.update(
            _pp_vpp_runtime_epoch=3,
            _pp_vpp_control_layout_digest="layout",
            _pp_vpp_slot_batch_seqs=[owner, None],
            _pp_vpp_ready_proxies={},
            _pp_vpp_early_activations={},
            _pp_vpp_fixed_received=deque(),
            _pp_vpp_arrivals=deque(),
            _pp_vpp_fixed_batch=True,
            _pp_vpp_pending_recv=object(),
            pp_group=SimpleNamespace(
                world_size=2,
                rank_in_group=rank,
                is_first_rank=rank == 0,
                is_last_rank=rank == 1,
            ),
            mbs=[object() if retained else None, None],
        )
        return scheduler

    def _wire(self, *, batch_seq=0, stage=2):
        return {
            "__msg_type__": "vpp_proxy",
            "vpp_protocol_version": 1,
            "vpp_runtime_epoch": 3,
            "vpp_layout_digest": "layout",
            "vpp_batch_seq": batch_seq,
            "vpp_generation": batch_seq // 2,
            "vpp_src_stage_id": stage - 1,
            "vpp_stage_id": stage,
            "hidden_states": torch.tensor([batch_seq]),
        }

    def _fixed_scheduler(self, *, rank=0):
        scheduler = self._scheduler(rank=rank)
        scheduler._vpp_layout = SimpleNamespace(virtual_stages=2)
        scheduler._pp_vpp_overlap_chunks = False
        scheduler._pp_vpp_init_fixed_batch(2)
        return scheduler

    def _remote_admission(self, *, core_type=_SchedulingCore):
        scheduler = self._scheduler(core_type=core_type)
        scheduler.attn_tp_group = SimpleNamespace(
            world_size=1, rank_in_group=0, cpu_group=None
        )
        req = SimpleNamespace(
            rid="request",
            session_generation=1,
            bootstrap_room=3,
            origin_input_ids=list(range(16)),
            kv=SimpleNamespace(holds_kv=False),
            finished=lambda: False,
        )
        scheduler.waiting_queue = [req]
        scheduler.chunked_req = None
        scheduler._pp_vpp_prefix_registry = PipelinePrefixRegistry()
        scheduler.page_size = 4
        scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
        scheduler._pp_vpp_available_prefill_tokens = lambda: 9
        manifest = scheduler._pp_vpp_batch_manifest(
            0, 0, SimpleNamespace(reqs=[req], prefix_lens=[4], extend_lens=[5])
        )
        envelope = scheduler._pp_vpp_new_control(
            PipelineControlKind.ADMIT,
            batch_seq=0,
            slot_id=0,
            payload={"manifest": manifest},
        )
        return scheduler, req, envelope

    def _downstream_admission(self, *, batch_seq=0, owner=None):
        scheduler, req, _ = self._remote_admission(core_type=_AdmissionCore)
        scheduler.pp_group.rank_in_group = 1
        scheduler.pp_group.is_first_rank = False
        scheduler.pp_group.is_last_rank = True
        scheduler._vpp_layout = SimpleNamespace(virtual_stages=2)
        scheduler._pp_vpp_overlap_chunks = False
        scheduler._pp_vpp_init_fixed_batch(2)
        scheduler._pp_vpp_slot_batch_seqs = [owner, None]
        scheduler.mbs = [None, None]
        scheduler.candidate_batch = SimpleNamespace(
            reqs=[req], prefix_lens=[4], extend_lens=[5]
        )
        wire = self._wire(batch_seq=batch_seq, stage=1)
        wire["vpp_batch_manifest"] = scheduler._pp_vpp_batch_manifest(
            batch_seq, batch_seq % 2, scheduler.candidate_batch
        )
        return scheduler, PipelineRankSchedule(1, 2, 2, 2), wire

    def _control_loop(self, *, rank=1):
        scheduler, _, _ = self._remote_admission(core_type=_ControlCore)
        scheduler.pp_group.rank_in_group = rank
        scheduler.pp_group.is_first_rank = rank == 0
        scheduler.pp_group.is_last_rank = rank == 1
        scheduler._vpp_layout = SimpleNamespace(virtual_stages=2)
        scheduler.model_config = SimpleNamespace(
            hidden_size=8, hf_config=SimpleNamespace(hc_mult=4)
        )
        scheduler.receive_tags = []
        scheduler.activation_group = SimpleNamespace(
            device="cpu",
            recv_tensor_dict_async=lambda tag: scheduler.receive_tags.append(tag),
        )
        scheduler.world_group = SimpleNamespace(
            rank_in_group=rank,
            broadcast_object=lambda value, src: 3,
        )
        with get_context().override_server_args(
            chunked_prefill_size=16, enable_dynamic_chunking=False
        ):
            context = scheduler._event_loop_pp_disagg_prefill_vpp_rank_local()
        return scheduler, context

    def _control_message(
        self, scheduler, kind, *, payload=None, seq=-1, returned=False
    ):
        wire = scheduler._pp_vpp_new_control(
            kind, batch_seq=seq, slot_id=seq % 2, payload=payload
        ).to_dict()
        wire["source_rank"] = (
            scheduler.pp_group.rank_in_group
            if returned
            else 1 - scheduler.pp_group.rank_in_group
        )
        wire["hops"] = 2 if returned else 1
        return PipelineControlEnvelope.from_dict(wire), wire

    def test_control_loop_initializes_identity_receivers_and_resource_bounds(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                scheduler, context = self._control_loop(rank=rank)
                self.assertEqual(scheduler._pp_vpp_runtime_epoch, 3)
                self.assertEqual(scheduler.activation_source, 1 - rank)
                self.assertEqual(scheduler.receive_tags, [1, 2])
                self.assertEqual(scheduler._pp_vpp_slot_batch_seqs, [None, None])
                self.assertIs(
                    context["slot_batch_seqs"], scheduler._pp_vpp_slot_batch_seqs
                )
                self.assertEqual(context["rank_schedule"].inflight_count, 0)
                self.assertEqual(context["max_control_sends"], 64)
                self.assertEqual(
                    context["resource_gate"].activation_high_watermark, 2048
                )
                self.assertEqual(
                    context["resource_gate"].activation_low_watermark, 1024
                )
                self.assertFalse(scheduler._pp_vpp_resource_control_enabled)
                self.assertEqual(scheduler._pp_vpp_pending_batch_controls, deque())

    def test_control_publication_waits_for_fifo_events_and_keeps_shared_fences(self):
        scheduler, context = self._control_loop()
        registry = scheduler._pp_vpp_prefix_registry
        entry = registry.plan("request", 1, 1, 0, 8, range(4))
        identity = PipelineReplicaIdentity("content", 0, 1, 1)
        ready = [False]
        event = SimpleNamespace(query=lambda: ready[0])
        scheduler._pp_vpp_stage_events = {
            (0, 0): event,
            (0, 1): SimpleNamespace(query=lambda: True),
        }
        scheduler._pp_vpp_pending_materialized.extend(
            [(0, 0, "request", 1, 8), (0, 1, "request", 1, 8)]
        )
        scheduler._pp_vpp_pending_replica_updates.append((0, 0, identity, 1, 0, 8))
        context["flush_materialized"]()
        context["flush_replica_updates"]()
        self.assertEqual(entry.materialized_by_stage[1], 0)
        self.assertEqual(len(scheduler._pp_vpp_pending_materialized), 2)
        self.assertEqual(len(scheduler._pp_vpp_pending_replica_updates), 1)
        self.assertEqual(list(scheduler._pp_vpp_control_outbox), [])
        ready[0] = True
        context["flush_materialized"]()
        self.assertIs(scheduler._pp_vpp_stage_events[(0, 0)], event)
        self.assertNotIn((0, 1), scheduler._pp_vpp_stage_events)
        self.assertEqual(entry.materialized_by_stage, {0: 8, 1: 8, 2: 0, 3: 0})
        self.assertEqual(entry.committed_end, 0)
        context["flush_replica_updates"]()
        replica = scheduler._pp_vpp_replica_registry.get(identity)
        self.assertEqual((replica.valid_end, replica.locked_until), (8, 8))
        self.assertEqual(replica.lock_owners, {0: 8})
        self.assertEqual(scheduler._pp_vpp_batch_replicas[0][identity], (1, 8))
        self.assertEqual(scheduler._pp_vpp_stage_events, {})
        messages = [
            PipelineControlEnvelope.from_dict(wire)
            for wire in scheduler._pp_vpp_control_outbox
        ]
        self.assertEqual(
            [message.kind for message in messages],
            [
                PipelineControlKind.PREFIX_MATERIALIZED,
                PipelineControlKind.PREFIX_MATERIALIZED,
                PipelineControlKind.CACHE_UPDATE,
            ],
        )
        self.assertEqual([message.hops for message in messages], [1, 1, 1])

    def test_replica_updates_merge_overlap_and_reject_gaps_and_stale_generations(self):
        scheduler, context = self._control_loop()
        payload = {
            "content_id": "content",
            "source_id": 0,
            "format_version": 1,
            "consumer_rank": 1,
            "residency_generation": 1,
            "start": 0,
            "end": 4,
        }
        identity = context["replica_identity"](payload)
        context["apply_cache_update"](payload)
        context["apply_cache_update"](payload)
        context["apply_cache_update"]({**payload, "start": 2, "end": 8})
        registry = scheduler._pp_vpp_replica_registry
        self.assertEqual(registry.get(identity).valid_end, 8)
        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            context["apply_cache_update"]({**payload, "start": 10, "end": 12})
        with self.assertRaisesRegex(RuntimeError, "stale replica"):
            context["apply_cache_update"]({**payload, "residency_generation": 2})
        registry.lock(identity, 8, owner_id=0)
        envelope, wire = self._control_message(
            scheduler, PipelineControlKind.EVICT_ACK, payload=payload
        )
        context["handle_control"](envelope, wire)
        self.assertEqual(registry.get(identity).residency_generation, 1)
        registry.unlock(identity, 8, owner_id=0)
        context["handle_control"](envelope, wire)
        self.assertEqual(registry.get(identity).residency_generation, 2)
        self.assertEqual(registry.get(identity).valid_end, 0)

    def test_replica_eviction_waits_for_all_batch_owners_and_last_chunks(self):
        scheduler, context = self._control_loop()
        identity = PipelineReplicaIdentity("content", 0, 1, 1)
        registry = scheduler._pp_vpp_replica_registry
        registry.install(identity, 1, 0, 8)
        for seq in (0, 1):
            registry.lock(identity, 8, owner_id=seq)
            scheduler._pp_vpp_batch_replicas[seq][identity] = (1, 8)
        scheduler._pp_vpp_batch_replicas[2][identity] = (1, 8)
        context["evict_batch_replicas"](
            0, SimpleNamespace(contains_last_prefill_chunk=False)
        )
        self.assertEqual(registry.get(identity).lock_owners, {0: 8, 1: 8})
        final_batch = SimpleNamespace(contains_last_prefill_chunk=True)
        context["evict_batch_replicas"](0, final_batch)
        self.assertEqual(registry.get(identity).lock_owners, {1: 8})
        self.assertEqual(list(scheduler._pp_vpp_control_outbox), [])
        context["evict_batch_replicas"](1, final_batch)
        self.assertEqual(registry.get(identity).residency_generation, 2)
        self.assertEqual(registry.get(identity).valid_end, 0)
        self.assertEqual(scheduler._pp_vpp_batch_replicas[2], {})
        ack = PipelineControlEnvelope.from_dict(
            scheduler._pp_vpp_control_outbox.popleft()
        )
        self.assertEqual(ack.kind, PipelineControlKind.EVICT_ACK)
        self.assertEqual(ack.payload["residency_generation"], 1)
        self.assertEqual(ack.hops, 1)

    def test_batch_controls_wait_for_activation_admission(self):
        scheduler, context = self._control_loop()
        kinds = (
            PipelineControlKind.CANCEL,
            PipelineControlKind.COMPLETION,
            PipelineControlKind.PREFIX_MATERIALIZED,
            PipelineControlKind.PREFIX_COMMIT,
            PipelineControlKind.CACHE_UPDATE,
            PipelineControlKind.EVICT_ACK,
        )
        for kind in kinds:
            envelope, wire = self._control_message(scheduler, kind, seq=0)
            context["handle_control"](envelope, wire)
        self.assertEqual(
            [envelope.kind for envelope, _ in scheduler._pp_vpp_pending_batch_controls],
            list(kinds),
        )
        self.assertEqual(list(scheduler._pp_vpp_control_outbox), [])
        self.assertEqual(context["rank_schedule"].inflight_count, 0)

    def test_control_admission_retries_missing_batch_then_promotes_its_activation(self):
        scheduler, context = self._control_loop()
        req = scheduler.waiting_queue[0]
        batch = SimpleNamespace(reqs=[req], prefix_lens=[4], extend_lens=[5])
        manifest = scheduler._pp_vpp_batch_manifest(0, 0, batch)
        activation = self._wire(stage=1)
        activation["vpp_batch_manifest"] = manifest
        scheduler._pp_vpp_accept_received_tensors(activation)
        envelope, wire = self._control_message(
            scheduler,
            PipelineControlKind.ADMIT,
            payload={"manifest": manifest},
            seq=0,
        )
        scheduler.candidate_batch = None
        context["handle_control"](envelope, wire)
        self.assertEqual(scheduler._pp_vpp_slot_batch_seqs, [None, None])
        self.assertEqual(scheduler.mb_metadata, [None, None])
        self.assertEqual(context["rank_schedule"].inflight_count, 0)
        scheduler.candidate_batch = batch
        pending = context["pending_remote_admits"].popleft()
        context["handle_control"](*pending)
        self.assertEqual(context["rank_schedule"].slot_batch_seqs, (0, None))
        self.assertEqual(scheduler._pp_vpp_highest_admitted_batch_seq, 0)
        self.assertEqual(scheduler._pp_vpp_early_activations, {})
        self.assertIn((0, 1), scheduler._pp_vpp_ready_proxies)
        forwarded = PipelineControlEnvelope.from_dict(
            scheduler._pp_vpp_control_outbox.popleft()
        )
        self.assertEqual(forwarded.hops, 2)

    def test_control_cancel_preserves_newer_incarnations_and_marks_the_batch(self):
        scheduler, context = self._control_loop()
        req = scheduler.waiting_queue[0]
        newer = SimpleNamespace(
            rid=req.rid, session_generation=2, bootstrap_room=3, to_finish=None
        )
        scheduler.waiting_queue = [newer]
        scheduler.mbs[0] = SimpleNamespace(reqs=[req])
        scheduler._pp_vpp_slot_batch_seqs[0] = 0
        scheduler._pp_vpp_highest_admitted_batch_seq = 0
        schedule = context["rank_schedule"]
        schedule.admit(0)
        schedule.mark_ready(0, 1)
        envelope, wire = self._control_message(
            scheduler,
            PipelineControlKind.CANCEL,
            payload={
                "rid": req.rid,
                "request_generation": 1,
                "bootstrap_room": 3,
            },
            seq=0,
        )
        context["handle_control"](envelope, wire)
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertIsNone(newer.to_finish)
        action = schedule.dispatch(0, 0, 1)
        self.assertTrue(schedule.complete(action).cancelled)

    def test_returned_completion_retires_only_its_owned_slot(self):
        scheduler, context = self._control_loop()
        schedule = context["rank_schedule"]
        schedule.admit(0)
        schedule.mark_completed(0)
        scheduler._pp_vpp_slot_batch_seqs[0] = 0
        scheduler._pp_vpp_highest_admitted_batch_seq = 0
        context["pending_chunk_batches"].add(0)
        envelope, wire = self._control_message(
            scheduler, PipelineControlKind.COMPLETION, seq=0, returned=True
        )
        context["handle_control"](envelope, wire)
        self.assertEqual(schedule.inflight_count, 0)
        self.assertEqual(scheduler._pp_vpp_slot_batch_seqs, [None, None])
        self.assertEqual(context["pending_chunk_batches"], set())
        self.assertEqual(list(scheduler._pp_vpp_control_outbox), [])
        with self.assertRaisesRegex(RuntimeError, "does not own its slot"):
            context["handle_control"](envelope, wire)

    def test_request_ring_preserves_payload_and_stops_at_its_source(self):
        scheduler, context = self._control_loop()
        req = SimpleNamespace(rid="new")
        envelope, wire = self._control_message(
            scheduler, PipelineControlKind.REQUEST, payload={"requests": (req,)}
        )
        metadata = torch.tensor([9])
        wire["metadata"] = metadata
        context["handle_control"](envelope, wire)
        self.assertIs(scheduler.waiting_queue[-1], req)
        forwarded = scheduler._pp_vpp_control_outbox.popleft()
        self.assertIs(forwarded["metadata"], metadata)
        self.assertEqual(forwarded["hops"], 2)
        returned, returned_wire = self._control_message(
            scheduler,
            PipelineControlKind.REQUEST,
            payload={"requests": (req,)},
            returned=True,
        )
        context["handle_control"](returned, returned_wire)
        self.assertEqual(sum(item is req for item in scheduler.waiting_queue), 1)
        self.assertEqual(list(scheduler._pp_vpp_control_outbox), [])

    def test_bootstrap_apply_stays_pending_until_local_outcomes_are_available(self):
        scheduler, context = self._control_loop()
        envelope, wire = self._control_message(
            scheduler,
            PipelineControlKind.BOOTSTRAP_STATUS,
            payload={
                "phase": "apply",
                "good": ["ready"],
                "bad": ["failed"],
                "prefix_boundaries": {"ready": 4},
            },
        )
        context["handle_control"](envelope, wire)
        pending = context["pending_bootstrap_applies"].popleft()
        self.assertEqual(list(scheduler._pp_vpp_control_outbox), [])
        scheduler.bootstrap_outcomes["ready"] = KVPoll.WaitingForInput
        self.assertFalse(context["try_apply_bootstrap"](*pending))
        self.assertEqual(pending[2], set())
        self.assertEqual(pending[3], {"failed"})
        scheduler.bootstrap_outcomes["failed"] = KVPoll.Failed
        self.assertTrue(context["try_apply_bootstrap"](*pending))
        self.assertEqual(pending[3], set())
        forwarded = scheduler._pp_vpp_control_outbox.popleft()
        self.assertEqual(forwarded["hops"], 2)
        self.assertEqual(forwarded["payload"]["prefix_boundaries"], {"ready": 4})
        for kind in (
            PipelineControlKind.BOOTSTRAP_STATUS,
            PipelineControlKind.TRANSFER_STATUS,
        ):
            with self.subTest(kind=kind):
                invalid, invalid_wire = self._control_message(
                    scheduler, kind, payload={"phase": "unknown"}
                )
                with self.assertRaisesRegex(RuntimeError, "invalid .* phase"):
                    context["handle_control"](invalid, invalid_wire)

    def test_cpu_broadcast_reuses_buffers_and_validates_source_values(self):
        scheduler = self._scheduler()
        scheduler.attn_tp_group = SimpleNamespace(world_size=1, rank_in_group=0)
        first = scheduler._pp_vpp_broadcast_cpu_state((3, 7), width=2)
        self.assertEqual(first.tolist(), [3, 7])
        self.assertEqual(first.device.type, "cpu")
        self.assertEqual(first.dtype, torch.int64)
        second = scheduler._pp_vpp_broadcast_cpu_state((4, 8), width=2)
        self.assertIs(second, first)
        self.assertEqual(first.tolist(), [4, 8])
        narrow = scheduler._pp_vpp_broadcast_cpu_state((9,), width=1)
        self.assertIsNot(narrow, first)
        self.assertEqual(narrow.tolist(), [9])
        for values in (None, (), (1,), (1, 2, 3)):
            with self.subTest(values=values):
                with self.assertRaisesRegex(RuntimeError, "invalid VPP TP broadcast"):
                    scheduler._pp_vpp_broadcast_cpu_state(values, width=2)

    def test_cpu_broadcast_maps_tp_source_to_its_global_rank(self):
        cpu_group = object()
        transmitted = []
        for tp_rank in (1, 0):
            scheduler = self._scheduler()
            scheduler.attn_tp_group = SimpleNamespace(
                world_size=2,
                rank_in_group=tp_rank,
                ranks=[4, 6],
                cpu_group=cpu_group,
            )

            def broadcast(state, *, src, group):
                self.assertEqual(src, 6)
                self.assertIs(group, cpu_group)
                if tp_rank == 1:
                    transmitted.append(state.clone())
                else:
                    state.copy_(transmitted[0])

            with patch("torch.distributed.broadcast", side_effect=broadcast):
                state = scheduler._pp_vpp_broadcast_cpu_state(
                    (5, 3) if tp_rank == 1 else None, width=2, src=1
                )
            self.assertEqual(state.tolist(), [5, 3])

    def test_cancel_waits_for_admission_and_never_targets_a_reused_slot(self):
        scheduler = self._fixed_scheduler()
        req = SimpleNamespace(rid="request", session_generation=1, bootstrap_room=3)
        scheduler.mbs[0] = SimpleNamespace(reqs=[req])
        envelope = scheduler._pp_vpp_new_control(
            PipelineControlKind.CANCEL,
            batch_seq=0,
            slot_id=0,
            payload={
                "rid": req.rid,
                "request_generation": 1,
                "bootstrap_room": 3,
            },
        )
        self.assertEqual(scheduler._pp_vpp_cancel_target(envelope), (False, None))
        scheduler._pp_vpp_highest_admitted_batch_seq = 0
        handled, target = scheduler._pp_vpp_cancel_target(envelope)
        self.assertTrue(handled)
        self.assertIs(target, req)
        for generation, room in ((2, 3), (1, 4)):
            with self.subTest(generation=generation, room=room):
                req.session_generation = generation
                req.bootstrap_room = room
                self.assertEqual(
                    scheduler._pp_vpp_cancel_target(envelope), (True, None)
                )
        req.session_generation, req.bootstrap_room = 1, 3
        for owner in (None, 2):
            with self.subTest(owner=owner):
                scheduler._pp_vpp_slot_batch_seqs[0] = owner
                self.assertEqual(
                    scheduler._pp_vpp_cancel_target(envelope), (True, None)
                )
        scheduler._pp_vpp_slot_batch_seqs[0] = 0
        scheduler.mbs[0] = None
        self.assertEqual(scheduler._pp_vpp_cancel_target(envelope), (True, None))

    def test_unbatched_cancel_searches_all_live_request_locations(self):
        scheduler = self._scheduler()
        stale = SimpleNamespace(rid="request", session_generation=1, bootstrap_room=3)
        current = SimpleNamespace(rid="request", session_generation=2, bootstrap_room=4)
        envelope = scheduler._pp_vpp_new_control(
            PipelineControlKind.CANCEL,
            payload={
                "rid": "request",
                "request_generation": 2,
                "bootstrap_room": 4,
            },
        )
        for location in ("waiting", "bootstrap", "batch", "chunked"):
            with self.subTest(location=location):
                scheduler.waiting_queue = [stale]
                scheduler.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
                scheduler.mbs = [None, None]
                scheduler.chunked_req = None
                if location == "waiting":
                    scheduler.waiting_queue.append(current)
                elif location == "bootstrap":
                    scheduler.disagg_prefill_bootstrap_queue.queue.append(current)
                elif location == "batch":
                    scheduler.mbs[1] = SimpleNamespace(reqs=[current])
                else:
                    scheduler.chunked_req = current
                handled, target = scheduler._pp_vpp_cancel_target(envelope)
                self.assertTrue(handled)
                self.assertIs(target, current)
        scheduler.chunked_req = None
        self.assertEqual(scheduler._pp_vpp_cancel_target(envelope), (True, None))

    def test_find_request_prefers_current_batch_then_retained_batch_then_chunk(self):
        scheduler = self._scheduler()
        current, retained, chunked = (SimpleNamespace(rid="request") for _ in range(3))
        scheduler.mbs = [None, SimpleNamespace(reqs=[current])]
        scheduler.last_mbs = [SimpleNamespace(reqs=[retained]), None]
        scheduler.chunked_req = chunked
        self.assertIs(scheduler._pp_vpp_find_req("request"), current)
        scheduler.mbs[1] = None
        self.assertIs(scheduler._pp_vpp_find_req("request"), retained)
        scheduler.last_mbs[0] = None
        self.assertIs(scheduler._pp_vpp_find_req("request"), chunked)
        self.assertIsNone(scheduler._pp_vpp_find_req("missing"))

    def test_fixed_remote_admission_waits_for_capacity_then_promotes_once(self):
        scheduler, schedule, wire = self._downstream_admission()
        scheduler._pp_vpp_accept_received_tensors(wire)
        proxy = scheduler._pp_vpp_early_activations[(0, 1)]
        scheduler.req_to_token_pool.available_size = lambda: 0
        self.assertFalse(scheduler._pp_vpp_fixed_remote_admit(0, schedule))
        self.assertEqual(schedule.inflight_count, 0)
        self.assertEqual(scheduler._pp_vpp_slot_batch_seqs, [None, None])
        self.assertIsNone(scheduler._pp_vpp_prefix_registry.get("request", 1))
        scheduler.req_to_token_pool.available_size = lambda: 1
        self.assertTrue(scheduler._pp_vpp_fixed_remote_admit(0, schedule))
        self.assertEqual(schedule.slot_batch_seqs, (0, None))
        self.assertEqual(scheduler._pp_vpp_highest_admitted_batch_seq, 0)
        self.assertEqual(scheduler._pp_vpp_early_activations, {})
        self.assertIs(scheduler._pp_vpp_ready_proxies[(0, 1)], proxy)
        self.assertEqual(list(scheduler._pp_vpp_arrivals), [(0, 1)])
        entry = scheduler._pp_vpp_prefix_registry.get("request", 1)
        self.assertEqual((entry.committed_end, entry.planned_end), (4, 9))
        self.assertTrue(entry.locked)
        self.assertTrue(scheduler._pp_vpp_fixed_remote_admit(0, schedule))
        self.assertIs(scheduler._pp_vpp_prefix_registry.get("request", 1), entry)
        self.assertEqual(list(scheduler._pp_vpp_arrivals), [(0, 1)])
        self.assertEqual(schedule.inflight_count, 1)

    def test_fixed_remote_admission_requires_first_activation_and_retired_slot(self):
        scheduler, schedule, _ = self._downstream_admission()
        with self.assertRaisesRegex(RuntimeError, "missing its first activation"):
            scheduler._pp_vpp_fixed_remote_admit(0, schedule)
        scheduler, schedule, wire = self._downstream_admission(batch_seq=2, owner=0)
        scheduler._pp_vpp_accept_received_tensors(wire)
        self.assertFalse(scheduler._pp_vpp_fixed_remote_admit(2, schedule))
        self.assertEqual(schedule.inflight_count, 0)
        self.assertEqual(scheduler._pp_vpp_slot_batch_seqs, [0, None])
        scheduler._pp_vpp_slot_batch_seqs[0] = None
        self.assertTrue(scheduler._pp_vpp_fixed_remote_admit(2, schedule))
        self.assertEqual(schedule.slot_batch_seqs, (2, None))
        self.assertEqual(scheduler._pp_vpp_highest_admitted_batch_seq, 2)

    def test_fixed_remote_admission_rejects_a_different_constructed_batch(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                scheduler, schedule, wire = self._downstream_admission()
                scheduler._pp_vpp_accept_received_tensors(wire)
                if missing:
                    scheduler.candidate_batch = None
                else:
                    scheduler.candidate_batch.extend_lens = [4]
                with self.assertRaisesRegex(RuntimeError, "batch manifest mismatch"):
                    scheduler._pp_vpp_fixed_remote_admit(0, schedule)
                self.assertEqual(schedule.inflight_count, 0)
                self.assertEqual(scheduler._pp_vpp_highest_admitted_batch_seq, -1)
                self.assertIsNone(scheduler._pp_vpp_prefix_registry.get("request", 1))
                self.assertIn((0, 1), scheduler._pp_vpp_early_activations)

    def test_fixed_task_selection_waits_for_send_credit_and_local_readiness(self):
        scheduler = self._fixed_scheduler()
        scheduler.pp_comm_stream_ctx = nullcontext()
        schedule = PipelineRankSchedule(0, 2, 2, 2)
        schedule.admit(0)
        completed = []
        pending = deque(
            [
                [
                    SimpleNamespace(
                        payload=torch.ones(1),
                        work=SimpleNamespace(wait=lambda: completed.append("oldest")),
                    )
                ],
                [],
            ]
        )

        def blocked(batch_seq, stage_id):
            self.assertEqual((batch_seq, stage_id), (0, 0))
            self.assertEqual(completed, ["oldest"])
            self.assertEqual(len(pending), 1)
            return False

        self.assertIsNone(
            scheduler._pp_vpp_fixed_select_tp_action(schedule, 5, blocked, pending)
        )
        self.assertIsNone(schedule.running)
        self.assertEqual(scheduler._pp_vpp_fixed_wave, [])
        action = scheduler._pp_vpp_fixed_select_tp_action(
            schedule, 6, lambda batch, stage: True, pending
        )
        self.assertEqual(action, PipelineWavefrontAction(6, 0, 0, 0, 0))
        self.assertIs(schedule.running, action)
        self.assertEqual(scheduler._pp_vpp_fixed_wave, [0])

    def test_fixed_task_selection_admits_downstream_input_and_preserves_receive_fifo(
        self,
    ):
        scheduler, schedule, wire = self._downstream_admission()
        scheduler._pp_vpp_accept_received_tensors(wire)
        scheduler.req_to_token_pool.available_size = lambda: 0
        self.assertIsNone(
            scheduler._pp_vpp_fixed_select_tp_action(
                schedule, 2, lambda batch, stage: True
            )
        )
        self.assertEqual(schedule.inflight_count, 0)
        self.assertEqual(list(scheduler._pp_vpp_fixed_received), [(0, 1)])
        scheduler.req_to_token_pool.available_size = lambda: 1
        self.assertIsNone(
            scheduler._pp_vpp_fixed_select_tp_action(
                schedule, 3, lambda batch, stage: False
            )
        )
        self.assertEqual(schedule.ready_tasks, ((0, 1),))
        self.assertEqual(list(scheduler._pp_vpp_fixed_received), [(0, 1)])
        self.assertEqual(list(scheduler._pp_vpp_arrivals), [])
        action = scheduler._pp_vpp_fixed_select_tp_action(
            schedule, 4, lambda batch, stage: True
        )
        self.assertEqual(action, PipelineWavefrontAction(4, 0, 0, 1, 1))
        self.assertEqual(list(scheduler._pp_vpp_fixed_received), [])
        schedule.complete(action)
        scheduler._pp_vpp_accept_received_tensors(self._wire(stage=3))
        pending = deque([object(), object()])
        final = scheduler._pp_vpp_fixed_select_tp_action(
            schedule, 5, lambda batch, stage: True, pending
        )
        self.assertEqual(final, PipelineWavefrontAction(5, 0, 0, 3, 1))
        self.assertEqual(len(pending), 2)
        self.assertEqual(list(scheduler._pp_vpp_fixed_received), [])

    def test_collective_task_selection_requires_every_tp_lane_to_permit_dispatch(self):
        scheduler = self._fixed_scheduler()
        scheduler._pp_vpp_fixed_tp = False
        cpu_group = object()
        scheduler.attn_tp_group = SimpleNamespace(
            world_size=2, rank_in_group=1, ranks=[4, 6], cpu_group=cpu_group
        )
        schedule = PipelineRankSchedule(0, 2, 2, 2)
        schedule.admit(0)

        def broadcast(state, *, src, group):
            self.assertEqual(src, 4)
            self.assertIs(group, cpu_group)
            state.copy_(torch.tensor([0, 0]))

        for remote_ready in (0, 1):

            def gather(output, local, *, group):
                self.assertIs(group, cpu_group)
                self.assertEqual(local.tolist(), [1])
                output.copy_(torch.cat((torch.tensor([remote_ready]), local)))

            with (
                patch("torch.distributed.broadcast", side_effect=broadcast),
                patch("torch.distributed.all_gather_into_tensor", side_effect=gather),
            ):
                action = scheduler._pp_vpp_fixed_select_tp_action(
                    schedule, 7, lambda batch, stage: True
                )
            if remote_ready:
                self.assertEqual(action, PipelineWavefrontAction(7, 0, 0, 0, 0))
                self.assertIs(schedule.running, action)
            else:
                self.assertIsNone(action)
                self.assertIsNone(schedule.running)
                self.assertEqual(scheduler._pp_vpp_fixed_wave, [])

    def test_collective_task_selection_waits_for_every_lanes_input(self):
        scheduler, schedule, wire = self._downstream_admission()
        scheduler._pp_vpp_accept_received_tensors(wire)
        scheduler._pp_vpp_fixed_tp = False
        cpu_group = object()
        scheduler.attn_tp_group = SimpleNamespace(
            world_size=2, rank_in_group=0, ranks=[4, 6], cpu_group=cpu_group
        )

        def broadcast(state, *, src, group):
            self.assertIs(group, cpu_group)
            self.assertEqual(src, 4)
            self.assertEqual(state.tolist(), [0, 1])

        for remote_input in (0, 1):
            gathers = []

            def gather(output, local, *, group):
                self.assertIs(group, cpu_group)
                self.assertEqual(local.tolist(), [1])
                other = remote_input if not gathers else 1
                gathers.append(other)
                output.copy_(torch.cat((local, torch.tensor([other]))))

            with (
                patch("torch.distributed.broadcast", side_effect=broadcast),
                patch("torch.distributed.all_gather_into_tensor", side_effect=gather),
            ):
                action = scheduler._pp_vpp_fixed_select_tp_action(
                    schedule, 8, lambda batch, stage: True
                )
            if remote_input:
                self.assertEqual(action, PipelineWavefrontAction(8, 0, 0, 1, 1))
                # Input presence, remote admission readiness, dispatch permission.
                self.assertEqual(gathers, [1, 1, 1])
            else:
                self.assertIsNone(action)
                self.assertEqual(gathers, [0])
                self.assertEqual(schedule.inflight_count, 0)
                self.assertEqual(scheduler._pp_vpp_slot_batch_seqs, [None, None])
                self.assertIn((0, 1), scheduler._pp_vpp_early_activations)

    def test_control_queue_has_one_tp_sender_and_rejects_device_payloads(self):
        scheduler = self._scheduler(rank=1)
        scheduler.attn_tp_group = SimpleNamespace(rank_in_group=0)
        scheduler._pp_vpp_control_outbox = deque()
        envelope = scheduler._pp_vpp_new_control(
            PipelineControlKind.ADMIT, batch_seq=5, slot_id=1, payload={}
        )
        self.assertEqual(
            (
                envelope.protocol_version,
                envelope.runtime_epoch,
                envelope.layout_digest,
                envelope.source_rank,
                envelope.generation,
            ),
            (1, 3, "layout", 1, 2),
        )
        payload = torch.ones(1)
        scheduler._pp_vpp_queue_control(envelope, {"metadata": payload})
        queued = scheduler._pp_vpp_control_outbox.popleft()
        self.assertEqual(PipelineControlEnvelope.from_dict(queued), envelope)
        self.assertIs(queued["metadata"], payload)
        with self.assertRaisesRegex(RuntimeError, "CPU-only"):
            scheduler._pp_vpp_queue_control(
                envelope, {"device_payload": torch.empty(1, device="meta")}
            )
        self.assertEqual(list(scheduler._pp_vpp_control_outbox), [])
        scheduler.attn_tp_group.rank_in_group = 1
        scheduler._pp_vpp_queue_control(envelope, {"metadata": payload})
        self.assertEqual(list(scheduler._pp_vpp_control_outbox), [])

    def test_remote_manifest_identity_includes_all_request_incarnations(self):
        scheduler, _, envelope = self._remote_admission()
        manifest = envelope.payload["manifest"]
        self.assertIs(scheduler._pp_vpp_validate_admit_manifest(envelope), manifest)
        for changes in (
            {"protocol_version": 2},
            {"batch_seq": 2},
            {"slot_id": 1},
            {"generation": 1},
            {"requests": ()},
            {"requests": (("request", 4, 9), ("request", 9, 12))},
            {"request_generations": {}},
            {"bootstrap_rooms": {"other": 3}},
        ):
            with self.subTest(changes=changes):
                bad = scheduler._pp_vpp_new_control(
                    PipelineControlKind.ADMIT,
                    batch_seq=0,
                    slot_id=0,
                    payload={"manifest": {**manifest, **changes}},
                )
                with self.assertRaisesRegex(RuntimeError, "invalid VPP admission"):
                    scheduler._pp_vpp_validate_admit_manifest(bad)

    def test_remote_admission_requires_matching_request_and_prefix_order(self):
        scheduler, req, envelope = self._remote_admission()
        self.assertTrue(scheduler._pp_vpp_remote_manifest_ready(envelope))
        req.session_generation = 2
        self.assertFalse(scheduler._pp_vpp_remote_manifest_ready(envelope))
        req.session_generation = 1
        req.bootstrap_room = 4
        self.assertFalse(scheduler._pp_vpp_remote_manifest_ready(envelope))
        req.bootstrap_room = 3
        req.finished = lambda: True
        self.assertFalse(scheduler._pp_vpp_remote_manifest_ready(envelope))
        req.finished = lambda: False
        scheduler.waiting_queue = []
        self.assertFalse(scheduler._pp_vpp_remote_manifest_ready(envelope))
        scheduler.waiting_queue = [req]
        scheduler.chunked_req = SimpleNamespace(rid="other")
        self.assertFalse(scheduler._pp_vpp_remote_manifest_ready(envelope))
        scheduler.chunked_req = req
        scheduler.waiting_queue = []
        self.assertTrue(scheduler._pp_vpp_remote_manifest_ready(envelope))
        registry = scheduler._pp_vpp_prefix_registry
        registry.plan("request", 1, 1, 0, 4, range(4))
        self.assertTrue(scheduler._pp_vpp_remote_manifest_ready(envelope))
        registry.plan("request", 1, 1, 4, 8, range(4))
        with self.assertRaisesRegex(RuntimeError, "prefix order mismatch"):
            scheduler._pp_vpp_remote_manifest_ready(envelope)

    def test_remote_admission_prices_new_rows_and_requires_valid_spans(self):
        scheduler, req, envelope = self._remote_admission()
        scheduler.req_to_token_pool.available_size = lambda: 0
        self.assertFalse(scheduler._pp_vpp_remote_manifest_ready(envelope))
        req.kv.holds_kv = True
        self.assertTrue(scheduler._pp_vpp_remote_manifest_ready(envelope))
        # Five extension tokens plus a four-token page reservation.
        scheduler._pp_vpp_available_prefill_tokens = lambda: 8
        self.assertFalse(scheduler._pp_vpp_remote_manifest_ready(envelope))
        scheduler._pp_vpp_available_prefill_tokens = lambda: 9
        self.assertTrue(scheduler._pp_vpp_remote_manifest_ready(envelope))
        manifest = envelope.payload["manifest"]
        for start, end in ((4, 4), (-1, 9), (4, 17)):
            with self.subTest(start=start, end=end):
                bad = scheduler._pp_vpp_new_control(
                    PipelineControlKind.ADMIT,
                    batch_seq=0,
                    slot_id=0,
                    payload={
                        "manifest": {
                            **manifest,
                            "requests": (("request", start, end),),
                        }
                    },
                )
                with self.assertRaises(RuntimeError):
                    scheduler._pp_vpp_remote_manifest_ready(bad)

    def test_remote_admission_waits_for_every_tp_lane(self):
        """Local capacity cannot admit a batch when another TP lane is blocked."""
        scheduler, _, envelope = self._remote_admission()
        cpu_group = object()
        scheduler.attn_tp_group = SimpleNamespace(
            world_size=2, rank_in_group=0, cpu_group=cpu_group
        )
        for other_ready in (0, 1):
            with self.subTest(other_ready=other_ready):

                def gather(output, local, *, group):
                    self.assertIs(group, cpu_group)
                    output.copy_(torch.cat((local, torch.tensor([other_ready]))))

                with patch(
                    "torch.distributed.all_gather_into_tensor", side_effect=gather
                ):
                    self.assertEqual(
                        scheduler._pp_vpp_remote_manifest_ready(envelope),
                        bool(other_ready),
                    )

    def test_control_round_consumes_only_the_tp_leaders_fifo_prefix(self):
        """A follower with more received messages must retain its extra message."""
        broadcast_round = []
        for tp_rank in (0, 1):
            scheduler = self._fixed_scheduler()
            scheduler._pp_vpp_slot_batch_seqs[1] = 1
            scheduler._pp_vpp_round_candidate = False
            queued = deque([self._wire(batch_seq=0)])
            if tp_rank == 1:
                queued.append(self._wire(batch_seq=1))
            progressed = []
            scheduler._pp_vpp_activation_receiver = SimpleNamespace(
                progress=lambda: progressed.append(True),
                receive_count=lambda: len(queued),
                take=queued.popleft,
            )
            envelope = scheduler._pp_vpp_new_control(
                PipelineControlKind.RESOURCE, payload={}
            )
            wire = envelope.to_dict()
            if tp_rank == 0:
                scheduler._pp_vpp_poll_control_receiver = lambda: (envelope, wire)

            def broadcast(value, *, src):
                self.assertEqual(src, 0)
                self.assertEqual(progressed, [True])
                if tp_rank == 0:
                    broadcast_round.append(value)
                else:
                    self.assertIsNone(value)
                return broadcast_round[0]

            scheduler.attn_tp_group = SimpleNamespace(
                rank_in_group=tp_rank, broadcast_object=broadcast
            )
            result = scheduler._pp_vpp_poll_control_receiver_tp_broadcast()
            self.assertEqual(result, (envelope, wire))
            self.assertEqual(list(scheduler._pp_vpp_ready_proxies), [(0, 2)])
            self.assertEqual(list(scheduler._pp_vpp_arrivals), [(0, 2)])
            self.assertEqual(len(queued), tp_rank)
            self.assertTrue(scheduler._pp_vpp_fixed_activation_received)
            self.assertFalse(scheduler._pp_vpp_round_check)

    def test_control_round_progresses_activations_without_a_control_message(self):
        scheduler = self._fixed_scheduler()
        queued = deque([self._wire()])
        scheduler._pp_vpp_activation_receiver = SimpleNamespace(
            progress=lambda: None,
            receive_count=lambda: len(queued),
            take=queued.popleft,
        )
        scheduler._pp_vpp_poll_control_receiver = lambda: None
        scheduler._pp_vpp_round_candidate = True
        scheduler.attn_tp_group = SimpleNamespace(
            rank_in_group=0, broadcast_object=lambda value, src: value
        )
        self.assertIsNone(scheduler._pp_vpp_poll_control_receiver_tp_broadcast())
        self.assertEqual(list(scheduler._pp_vpp_ready_proxies), [(0, 2)])
        self.assertTrue(scheduler._pp_vpp_round_check)
        self.assertEqual(list(queued), [])

    def test_control_broadcast_rejects_foreign_protocol_epoch_and_layout(self):
        scheduler = self._scheduler()
        scheduler._pp_vpp_fixed_tp = False
        scheduler.attn_tp_group = SimpleNamespace(
            rank_in_group=0, broadcast_object=lambda value, src: value
        )
        envelope = scheduler._pp_vpp_new_control(PipelineControlKind.RESOURCE)
        for changes, error in (
            ({"protocol_version": 2}, "protocol version mismatch"),
            ({"runtime_epoch": 2}, "stale VPP control runtime epoch"),
            ({"layout_digest": "other"}, "layout digest mismatch"),
        ):
            with self.subTest(changes=changes):
                wire = {**envelope.to_dict(), **changes}
                scheduler._pp_vpp_poll_control_receiver = lambda: (envelope, wire)
                with self.assertRaisesRegex(RuntimeError, error):
                    scheduler._pp_vpp_poll_control_receiver_tp_broadcast()
        scheduler._pp_vpp_poll_control_receiver = lambda: None
        self.assertIsNone(scheduler._pp_vpp_poll_control_receiver_tp_broadcast())

    def test_bootstrap_merge_uses_consensus_and_the_smallest_resource_frontier(self):
        scheduler = self._scheduler()
        cpu_group = object()
        scheduler.attn_tp_group = SimpleNamespace(world_size=2, cpu_group=cpu_group)
        scheduler.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 4
        )
        # This boundary supplies the prefix lengths from the bootstrap queue.
        boundaries = {"a": 4, "b": 12}
        scheduler._pp_vpp_bootstrap_prefix_boundaries = lambda rids: {
            rid: boundaries[rid] for rid in rids
        }

        def gather(output, local, *, group):
            self.assertIs(group, cpu_group)
            self.assertEqual(local.tolist(), [4])
            output.copy_(torch.cat((local, torch.tensor([1]))))

        for carried_slots, expected_slots in ((3, 1), (0, 0)):
            with self.subTest(carried_slots=carried_slots):
                payload = {
                    "good": ["c", "b", "a", "a"],
                    "bad": ["x"],
                    "metadata_slots": carried_slots,
                    "prefix_boundaries": {"a": 8, "b": 6, "c": 10},
                }
                with patch(
                    "torch.distributed.all_gather_into_tensor", side_effect=gather
                ):
                    scheduler._pp_vpp_merge_bootstrap_status(
                        payload, ["b", "a"], ["c", "x"]
                    )
                self.assertEqual(
                    payload,
                    {
                        "good": ["a", "b"],
                        "bad": ["c", "x"],
                        "metadata_slots": expected_slots,
                        "prefix_boundaries": {"a": 4, "b": 6},
                    },
                )
                self.assertEqual(
                    scheduler._pp_vpp_bootstrap_apply_status(payload),
                    [["a"][:expected_slots], ["c", "x"]],
                )

    def test_bootstrap_application_deduplicates_sorts_and_caps_admissions(self):
        scheduler = self._scheduler()
        for slots, expected in ((-1, []), (0, []), (1, ["a"]), (8, ["a", "b"])):
            with self.subTest(slots=slots):
                self.assertEqual(
                    scheduler._pp_vpp_bootstrap_apply_status(
                        {
                            "good": ["b", "a", "b"],
                            "bad": ["z", "c", "c"],
                            "metadata_slots": slots,
                        }
                    ),
                    [expected, ["c", "z"]],
                )

    def test_prefix_notifications_commit_only_the_shared_stage_frontier(self):
        scheduler = self._scheduler()
        registry = scheduler._pp_vpp_prefix_registry = PipelinePrefixRegistry()
        entry = registry.plan("request", 1, 1, 0, 8, range(4))

        def notify(stage, end, generation=1):
            scheduler._pp_vpp_apply_prefix_materialized(
                scheduler._pp_vpp_new_control(
                    PipelineControlKind.PREFIX_MATERIALIZED,
                    payload={
                        "rid": "request",
                        "request_generation": generation,
                        "stage_id": stage,
                        "end": end,
                    },
                )
            )

        notify(0, 8, generation=2)
        self.assertEqual(entry.materialized_end, 0)
        for stage in (2, 0, 3):
            notify(stage, 8)
            self.assertEqual(entry.committed_end, 0)
            self.assertTrue(entry.locked)
        notify(1, 4)
        self.assertEqual(entry.committed_end, 4)
        self.assertTrue(entry.locked)
        notify(1, 8)
        self.assertEqual(entry.committed_end, 8)
        self.assertFalse(entry.locked)
        notify(1, 8)
        self.assertEqual(entry.committed_end, 8)
        with self.assertRaisesRegex(RuntimeError, "invalid materialized"):
            notify(1, 4)
        with self.assertRaisesRegex(RuntimeError, "not a prefix dependency"):
            notify(4, 8)

    def test_local_completion_waits_before_reading_and_checks_batch_identity(self):
        scheduler = self._scheduler()
        for output_seq in (0, 1, None):
            with self.subTest(output_seq=output_seq):
                synchronized = []
                tensors = {"next_token_ids": torch.tensor([7]), "metadata": "done"}
                if output_seq is not None:
                    tensors["vpp_batch_seq"] = output_seq

                class OutputProxy:
                    @property
                    def tensors(proxy):
                        self.assertEqual(synchronized, [True])
                        return tensors

                queued = deque(
                    [
                        (
                            SimpleNamespace(
                                synchronize=lambda: synchronized.append(True)
                            ),
                            OutputProxy(),
                        )
                    ]
                )
                if output_seq == 0:
                    result = scheduler._pp_vpp_take_local_completion(0, queued)
                    self.assertEqual(result["next_token_ids"].tolist(), [7])
                    self.assertEqual(result["next_token_ids"].device.type, "cpu")
                    self.assertEqual(result["metadata"], "done")
                    self.assertEqual(result["vpp_batch_seq"], 0)
                else:
                    with self.assertRaisesRegex(RuntimeError, "completion mismatch"):
                        scheduler._pp_vpp_take_local_completion(0, queued)
                self.assertEqual(list(queued), [])
                self.assertEqual(synchronized, [True])

    def test_fixed_wave_requires_the_supported_layout_and_capacity(self):
        for pp_size, virtual_stages, inflight, overlap in (
            (3, 2, 2, False),
            (2, 3, 2, False),
            (2, 2, 1, False),
            (2, 2, 2, True),
        ):
            with self.subTest(
                pp=pp_size, vpp=virtual_stages, inflight=inflight, overlap=overlap
            ):
                scheduler = self._scheduler()
                scheduler.pp_group.world_size = pp_size
                scheduler._vpp_layout = SimpleNamespace(virtual_stages=virtual_stages)
                scheduler._pp_vpp_overlap_chunks = overlap
                with self.assertRaises(ValueError):
                    scheduler._pp_vpp_init_fixed_batch(inflight)

    def test_full_fixed_wave_keeps_second_passes_in_dispatch_order(self):
        """A later ready activation cannot overtake the wave's first batch."""
        scheduler = self._fixed_scheduler()
        scheduler._pp_vpp_slot_batch_seqs[1] = 1
        schedule = PipelineRankSchedule(0, 2, 2, 2)
        for seq in (0, 1):
            schedule.admit(seq)
            target = scheduler._pp_vpp_fixed_next_task(schedule)
            self.assertEqual(target, (seq, 0))
            action = schedule.dispatch(seq, *target)
            scheduler._pp_vpp_fixed_record_dispatch(action, schedule)
            schedule.complete(action)
        scheduler._pp_vpp_accept_received_tensors(self._wire(batch_seq=1))
        self.assertIsNone(scheduler._pp_vpp_fixed_next_task(schedule))
        scheduler._pp_vpp_accept_received_tensors(self._wire(batch_seq=0))
        while scheduler._pp_vpp_arrivals:
            schedule.mark_ready(*scheduler._pp_vpp_arrivals.popleft())
        for seq in (0, 1):
            target = scheduler._pp_vpp_fixed_next_task(schedule)
            self.assertEqual(target, (seq, 2))
            action = schedule.dispatch(seq + 2, *target)
            scheduler._pp_vpp_fixed_record_dispatch(action, schedule)
            schedule.complete(action)
        self.assertEqual(scheduler._pp_vpp_fixed_wave, [])
        self.assertFalse(scheduler._pp_vpp_fixed_second_pass)
        self.assertEqual(list(scheduler._pp_vpp_fixed_received), [])
        self.assertIsNone(scheduler._pp_vpp_fixed_next_task(schedule))

    def test_partial_fixed_wave_drains_when_its_activation_arrives(self):
        """One admitted batch must finish even when a second cannot be admitted."""
        scheduler = self._fixed_scheduler()
        schedule = PipelineRankSchedule(0, 2, 2, 2)
        schedule.admit(0)
        action = schedule.dispatch(0, *scheduler._pp_vpp_fixed_next_task(schedule))
        scheduler._pp_vpp_fixed_record_dispatch(action, schedule)
        schedule.complete(action)
        self.assertIsNone(scheduler._pp_vpp_fixed_next_task(schedule))
        scheduler._pp_vpp_accept_received_tensors(self._wire())
        schedule.mark_ready(*scheduler._pp_vpp_arrivals.popleft())
        target = scheduler._pp_vpp_fixed_next_task(schedule)
        self.assertEqual(target, (0, 2))
        action = schedule.dispatch(1, *target)
        scheduler._pp_vpp_fixed_record_dispatch(action, schedule)
        schedule.complete(action)
        self.assertFalse(scheduler._pp_vpp_fixed_second_pass)
        self.assertEqual(scheduler._pp_vpp_fixed_wave, [])

    def test_downstream_fixed_tasks_follow_receive_fifo(self):
        scheduler = self._fixed_scheduler(rank=1)
        scheduler._pp_vpp_slot_batch_seqs[1] = 1
        schedule = PipelineRankSchedule(1, 2, 2, 2)
        for seq in (0, 1):
            schedule.admit(seq)
        for seq in (1, 0):
            scheduler._pp_vpp_accept_received_tensors(
                self._wire(batch_seq=seq, stage=1)
            )
        while scheduler._pp_vpp_arrivals:
            schedule.mark_ready(*scheduler._pp_vpp_arrivals.popleft())
        for seq in (1, 0):
            target = scheduler._pp_vpp_fixed_next_task(schedule)
            self.assertEqual(target, (seq, 1))
            action = schedule.dispatch(seq, *target)
            scheduler._pp_vpp_fixed_record_dispatch(action, schedule)
            schedule.complete(action)
        self.assertIsNone(scheduler._pp_vpp_fixed_next_task(schedule))

    def test_send_credit_waits_for_oldest_payload_before_releasing_capacity(self):
        scheduler = self._scheduler()
        scheduler.pp_comm_stream_ctx = nullcontext()
        completed = []
        oldest = [
            SimpleNamespace(
                payload=torch.ones(1),
                work=SimpleNamespace(wait=lambda: completed.append("oldest")),
            )
        ]
        newest = [
            SimpleNamespace(
                payload=torch.ones(1),
                work=SimpleNamespace(wait=lambda: completed.append("newest")),
            )
        ]
        pending = deque([oldest, newest])
        scheduler._pp_vpp_fixed_wait_send_credit(pending, 2)
        self.assertEqual(completed, ["oldest"])
        self.assertEqual(oldest, [])
        self.assertIs(pending[0], newest)
        self.assertEqual(len(newest), 1)
        scheduler._pp_vpp_fixed_wait_send_credit(pending, 2)
        self.assertEqual(completed, ["oldest"])
        scheduler._pp_vpp_fixed_wait_send_credit(pending, 1)
        self.assertEqual(completed, ["oldest", "newest"])
        self.assertEqual(newest, [])
        self.assertEqual(list(pending), [])

    def test_send_reaping_stops_at_the_first_unfinished_group(self):
        """A later completion must not release credit ahead of an older send."""
        from threading import Event

        release = Event()
        entered = Event()

        def wait():
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("Send was not released")

        items = [SimpleNamespace(payload=torch.ones(1), work=None, wait=wait)]
        later = SimpleNamespace(poll=lambda: True)
        pending = deque([items, later])
        scheduler = self._scheduler()
        try:
            scheduler._pp_vpp_reap_send_work(pending)
            self.assertTrue(entered.wait(timeout=5))
            self.assertEqual(len(pending), 2)
            self.assertIsInstance(pending[0], P2PWorkGroup)
            self.assertIs(pending[0].items, items)
            self.assertIs(pending[1], later)
            group = pending[0]
            release.set()
            self.assertTrue(group._done.wait(timeout=5))
            scheduler._pp_vpp_reap_send_work(pending)
            self.assertEqual(list(pending), [])
            self.assertEqual(items, [])
        finally:
            release.set()

    def test_manifest_uses_retained_batch_spans_and_request_incarnations(self):
        """A later request update must not change the retained batch's manifest."""
        scheduler = self._scheduler()
        batch = SimpleNamespace(
            reqs=[
                SimpleNamespace(
                    rid="first", prefix_len=99, session_generation=7, bootstrap_room=8
                ),
                SimpleNamespace(rid="second", prefix_len=99, session_generation=None),
            ],
            prefix_lens=[4, 8],
            extend_lens=[5, 3],
        )
        self.assertEqual(
            scheduler._pp_vpp_batch_manifest(5, 1, batch),
            {
                "protocol_version": 1,
                "batch_seq": 5,
                "generation": 2,
                "slot_id": 1,
                "requests": (("first", 4, 9), ("second", 8, 11)),
                "request_generations": {"first": 7, "second": 0},
                "bootstrap_rooms": {"first": 8, "second": None},
            },
        )

    def test_admission_waits_for_slot_and_rank_schedule_retirement(self):
        scheduler = self._scheduler()
        envelope = PipelineControlEnvelope(
            protocol_version=1,
            runtime_epoch=3,
            layout_digest="layout",
            kind=PipelineControlKind.ADMIT,
            source_rank=0,
            batch_seq=2,
            slot_id=0,
        )
        self.assertTrue(
            scheduler._pp_vpp_remote_admit_slot_available(envelope, [None, None])
        )
        self.assertFalse(
            scheduler._pp_vpp_remote_admit_slot_available(envelope, [0, None])
        )
        for owner in (2, 4):
            with self.subTest(owner=owner):
                with self.assertRaisesRegex(RuntimeError, "still owned"):
                    scheduler._pp_vpp_remote_admit_slot_available(
                        envelope, [owner, None]
                    )
        invalid = PipelineControlEnvelope(
            protocol_version=1,
            runtime_epoch=3,
            layout_digest="layout",
            kind=PipelineControlKind.ADMIT,
            source_rank=0,
            batch_seq=2,
            slot_id=1,
        )
        with self.assertRaisesRegex(RuntimeError, "slot identity"):
            scheduler._pp_vpp_remote_admit_slot_available(invalid, [None, None])
        schedule = PipelineRankSchedule(0, 2, 2, 2)
        self.assertTrue(
            scheduler._pp_vpp_can_queue_admit(2, set(), [None, None], schedule)
        )
        self.assertFalse(
            scheduler._pp_vpp_can_queue_admit(2, {0, 1}, [None, None], schedule)
        )
        self.assertFalse(
            scheduler._pp_vpp_can_queue_admit(2, set(), [0, None], schedule)
        )
        schedule.admit(0)
        self.assertFalse(
            scheduler._pp_vpp_can_queue_admit(2, set(), [None, None], schedule)
        )

    def test_activation_rejects_wrong_identity_stale_owner_and_duplicates(self):
        """Bad activations must not enter either the ready or temporary store."""
        for changes in (
            {"__msg_type__": "vpp_control"},
            {"vpp_protocol_version": 2},
            {"vpp_runtime_epoch": 2},
            {"vpp_layout_digest": "other"},
            {"vpp_generation": 1},
            {"vpp_src_stage_id": 0},
            {"vpp_batch_seq": -1, "vpp_generation": -1},
            {"vpp_stage_id": 1, "vpp_src_stage_id": 0},
            {"vpp_batch_seq": 2, "vpp_generation": 1},
        ):
            with self.subTest(changes=changes):
                scheduler = self._scheduler()
                with self.assertRaises(RuntimeError):
                    scheduler._pp_vpp_accept_received_tensors(
                        {**self._wire(), **changes}
                    )
                self.assertEqual(scheduler._pp_vpp_ready_proxies, {})
                self.assertEqual(scheduler._pp_vpp_early_activations, {})
                self.assertEqual(list(scheduler._pp_vpp_fixed_received), [])
        scheduler = self._scheduler()
        scheduler._pp_vpp_accept_received_tensors(self._wire())
        self.assertEqual(list(scheduler._pp_vpp_arrivals), [(0, 2)])
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            scheduler._pp_vpp_accept_received_tensors(self._wire())

    def test_early_activation_waits_for_admission_and_is_consumed_once(self):
        scheduler = self._scheduler(owner=None, retained=False)
        wire = self._wire()
        scheduler._pp_vpp_accept_received_tensors(wire)
        self.assertEqual(list(scheduler._pp_vpp_fixed_received), [(0, 2)])
        self.assertEqual(scheduler._pp_vpp_ready_proxies, {})
        self.assertEqual(list(scheduler._pp_vpp_arrivals), [])
        self.assertIsNone(scheduler._pp_vpp_pending_recv)
        with self.assertRaisesRegex(RuntimeError, "before admitting"):
            scheduler._pp_vpp_promote_early_activations(0)
        scheduler._pp_vpp_slot_batch_seqs[0] = 0
        scheduler._pp_vpp_promote_early_activations(0)
        self.assertEqual(scheduler._pp_vpp_early_activations, {})
        self.assertEqual(list(scheduler._pp_vpp_arrivals), [(0, 2)])
        action = PipelineWavefrontAction(0, 0, 0, 2, 0)
        proxy = scheduler._pp_vpp_take_ready_proxy(action)
        self.assertIs(proxy.tensors, wire)
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            scheduler._pp_vpp_take_ready_proxy(action)

    def test_last_rank_quarantines_next_generation_only_after_local_completion(self):
        """The returning completion control may lag behind the next activation."""
        wire = self._wire(batch_seq=2, stage=1)
        scheduler = self._scheduler(rank=1, retained=True)
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            scheduler._pp_vpp_accept_received_tensors(wire)
        scheduler.mbs[0] = None
        scheduler._pp_vpp_accept_received_tensors(wire)
        self.assertEqual(scheduler._pp_vpp_slot_batch_seqs, [0, None])
        self.assertEqual(scheduler._pp_vpp_ready_proxies, {})
        self.assertEqual(list(scheduler._pp_vpp_early_activations), [(2, 1)])
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            scheduler._pp_vpp_accept_received_tensors(wire)
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            scheduler._pp_vpp_accept_received_tensors(self._wire(batch_seq=4, stage=1))
        scheduler._pp_vpp_slot_batch_seqs[0] = 2
        scheduler._pp_vpp_promote_early_activations(2)
        self.assertEqual(list(scheduler._pp_vpp_arrivals), [(2, 1)])
        self.assertEqual(scheduler._pp_vpp_early_activations, {})

    def test_early_activation_capacity_and_generation_conflicts_are_bounded(self):
        scheduler = self._scheduler(rank=1, owner=None, retained=False)
        for batch_seq in (0, 1):
            scheduler._pp_vpp_accept_received_tensors(
                self._wire(batch_seq=batch_seq, stage=1)
            )
        with self.assertRaisesRegex(RuntimeError, "quarantine is full"):
            scheduler._pp_vpp_accept_received_tensors(self._wire(batch_seq=2, stage=1))
        self.assertEqual(len(scheduler._pp_vpp_early_activations), 2)
        scheduler._pp_vpp_slot_batch_seqs[0] = 2
        with self.assertRaisesRegex(RuntimeError, "Stale VPP early activation"):
            scheduler._pp_vpp_promote_early_activations(2)


class TestSchedulerVPP(CustomTestCase):
    def test_prefix_match_respects_vpp_frontier_and_existing_caps(self):
        """A local cache hit cannot advance past the agreed pipeline frontier."""
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req(
            rid="prefix",
            origin_input_text="",
            origin_input_ids=array("q", range(16)),
            sampling_params=SamplingParams(max_new_tokens=2),
        )
        self.assertEqual(req._compute_max_prefix_len(16), 15)
        for limit, expected in ((8, 8), (0, 0), (32, 15)):
            with self.subTest(limit=limit):
                req.vpp_prefix_limit = limit
                self.assertEqual(req._compute_max_prefix_len(16), expected)
        req.vpp_prefix_limit = 8
        req.return_logprob = True
        req.logprob_start_len = 4
        self.assertEqual(req._compute_max_prefix_len(16), 4)
        req.vpp_prefix_limit = None
        self.assertEqual(req._compute_max_prefix_len(16), 4)

    def test_bootstrap_consensus_holds_unconfirmed_requests_and_failure_wins(self):
        """Consensus success still needs local readiness; missing RIDs stay queued."""
        from sglang.srt.disaggregation.base.conn import KVPoll
        from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.managers.scheduler_components.request_list import (
            IndexedReqList,
            ListKind,
            move,
        )
        from sglang.srt.sampling.sampling_params import SamplingParams

        for outcomes in (
            {},
            {"ready": False, "pending": False, "failed": True},
        ):
            with self.subTest(outcomes=outcomes):
                queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
                queue.scheduler_stage_metrics = None
                queue.pp_size = 2
                failed = []
                queue.scheduler = SimpleNamespace(
                    attn_cp_cpu_group=None,
                    attn_tp_cpu_group=None,
                    processed_tokens_counter=7,
                    handle_bootstrap_failure=failed.append,
                    token_to_kv_pool_allocator=SimpleNamespace(page_size=4),
                )
                queue.ensure_metadata_buffer = lambda req: True
                reqs = [
                    Req(
                        rid=rid,
                        origin_input_text="",
                        origin_input_ids=array("q", range(12)),
                        sampling_params=SamplingParams(max_new_tokens=2),
                    )
                    for rid in ("ready", "unconfirmed", "pending", "failed")
                ]
                for req in reqs:
                    req.pending_bootstrap = True
                    req.metadata_buffer_index = 0
                    req.disagg_kv_sender = SimpleNamespace(
                        pop_decode_prefix_len=lambda: 4,
                        init=lambda num_pages, metadata_index: None,
                    )
                queue.queue = IndexedReqList(ListKind.PD_BOOTSTRAP)
                move(reqs, queue.queue)
                dst = IndexedReqList(ListKind.WAITING)
                with (
                    patch(
                        "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                        return_value=[
                            KVPoll.WaitingForInput,
                            KVPoll.WaitingForInput,
                            KVPoll.Bootstrapping,
                            KVPoll.WaitingForInput,
                        ],
                    ),
                    patch(
                        "sglang.srt.disaggregation.prefill.poll_and_all_reduce_prefill_pp",
                        side_effect=AssertionError("Consensus must not poll PP again"),
                    ),
                    patch(
                        "sglang.srt.disaggregation.prefill.should_force_retry",
                        return_value=False,
                    ),
                ):
                    admitted = queue.move_bootstrapped(dst, outcomes=outcomes)
                expected = [reqs[0]] if outcomes else []
                self.assertEqual(admitted, expected)
                self.assertEqual(list(dst), expected)
                self.assertEqual(failed, [reqs[3]] if outcomes else [])
                self.assertEqual(list(queue.queue), reqs[1:] if outcomes else reqs)
                self.assertTrue(all(req.pending_bootstrap for req in reqs[1:]))
                if outcomes:
                    self.assertFalse(reqs[0].pending_bootstrap)
                    self.assertEqual(reqs[0].start_send_idx, 4)
                    self.assertEqual(reqs[0].arrival_processed_tokens, 7)

    def test_manifest_tracks_real_request_chunk_boundaries(self):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req(
            rid="chunked",
            origin_input_text="",
            origin_input_ids=array("q", range(12)),
            sampling_params=SamplingParams(max_new_tokens=2),
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(value=1),
            reqs=[req],
            seq_lens_cpu=torch.tensor([8]),
            extend_lens=[4],
        )
        self.assertEqual(
            vpp_mod._vpp_manifest(batch)[1], (("chunked", None, None, None),)
        )
        req.prefix_len, req.extend_end = 4, 8
        first = vpp_mod._vpp_manifest(batch)
        self.assertEqual(first[1], (("chunked", None, 4, 8),))
        req.prefix_len, req.extend_end = 8, 12
        self.assertNotEqual(first, vpp_mod._vpp_manifest(batch))

    def test_pd_terminal_consensus_waits_for_all_layers_and_failure_wins(self):
        """A stage that finished early cannot admit or free a partial KV request."""
        from threading import Barrier

        from sglang.srt.disaggregation.base.conn import KVPoll
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        class TransferQueue(dict):
            # IndexedReqList.items() exposes queue entries, rather than RID pairs.
            def items(self):
                return list(self.values())

        size = 3
        channels = [Queue() for _ in range(size - 1)]
        barrier = Barrier(size)
        broadcast = [None]
        schedulers = []
        for rank in range(size):
            scheduler = object.__new__(SchedulerVPPMixin)

            def recv(rank=rank):
                return channels[rank - 1].get(timeout=10)

            def send(value, async_send=True, rank=rank):
                channels[rank].put(value)
                return []

            def bcast(value, src, rank=rank):
                if rank == src:
                    broadcast[0] = value
                barrier.wait(timeout=10)
                result = broadcast[0]
                barrier.wait(timeout=10)
                return result

            polls = {
                "request": (KVPoll.Success, KVPoll.Success, KVPoll.Transferring)[rank]
            }
            queue = TransferQueue(
                request=SimpleNamespace(req=SimpleNamespace(rid="request"))
            )
            retired = {}

            def get_rids(reqs, is_send, *groups, polls=polls):
                return tuple(
                    [req.req.rid for req in reqs if KVPoll.Success in group]
                    for group in groups
                )

            def transfer(waiting, outcomes, queue=queue, retired=retired):
                for rid, failed in outcomes.items():
                    queue.pop(rid)
                    retired[rid] = failed
                return [], outcomes

            scheduler.__dict__.update(
                disaggregation_mode=vpp_mod.DisaggregationMode.DECODE,
                _vpp_next_slot=0,
                pp_group=SimpleNamespace(
                    is_first_rank=rank == 0,
                    is_last_rank=rank == size - 1,
                    world_size=size,
                    broadcast_object=bcast,
                ),
                _pp_recv_pyobj_from_prev_stage=recv,
                _pp_send_pyobj_to_next_stage=send,
                _pp_commit_comm_work=lambda work: None,
                pause_state=SimpleNamespace(may_launch=lambda: True),
                waiting_queue=[],
                disagg_decode_prealloc_queue=SimpleNamespace(
                    queue={},
                    retracted_queue=[],
                    publish_pending_destinations=lambda: None,
                ),
                disagg_decode_transfer_queue=SimpleNamespace(
                    queue=queue,
                    resolve_deferred_releases=lambda: None,
                    publish_pending_metadata=lambda: None,
                    enable_staging=True,
                    _poll_with_staging=lambda polls=polls: list(polls.values()),
                    move_transferred_with_outcomes=transfer,
                ),
                get_rids=get_rids,
                enable_hisparse=False,
                process_retract_queue=lambda rids: rids,
                process_prealloc_queue=lambda rids: rids,
                retired=retired,
                polls=polls,
            )
            for name in (
                "_pp_pd_get_retract_ids",
                "_pp_pd_get_prealloc_ids",
                "_route_aborts_to_bad",
                "process_decode_transfer_queue",
            ):
                method = getattr(SchedulerPPMixin, name)
                setattr(
                    scheduler,
                    name,
                    method
                    if name == "_route_aborts_to_bad"
                    else method.__get__(scheduler),
                )
            schedulers.append(scheduler)

        def advance():
            with ThreadPoolExecutor(max_workers=size) as pool:
                results = [pool.submit(s._vpp_pd_advance) for s in schedulers]
                for result in results:
                    result.result(timeout=10)

        advance()
        self.assertTrue(all(s.disagg_decode_transfer_queue.queue for s in schedulers))
        self.assertTrue(all(not s.retired for s in schedulers))
        schedulers[1].polls["request"] = KVPoll.Failed
        schedulers[2].polls["request"] = KVPoll.Success
        advance()
        self.assertTrue(
            all(not s.disagg_decode_transfer_queue.queue for s in schedulers)
        )
        self.assertEqual([s.retired for s in schedulers], [{"request": True}] * size)
        self.assertTrue(all(channel.empty() for channel in channels))

    def test_abort_inputs_are_not_dispatched_between_retained_admissions(self):
        """A later input poll could release the preceding unlaunched batch's KV."""
        layout = VPPLayout(12, 2, virtual_stages=3)
        scheduler = object.__new__(SchedulerVPPMixin)
        polls = []
        batches = [
            SimpleNamespace(
                reqs=[
                    SimpleNamespace(
                        rid=f"req-{i}",
                        session_generation=0,
                        prefix_len=0,
                        extend_end=None,
                        beam_group=None,
                        finished=False,
                    )
                ],
                seq_lens_cpu=torch.tensor([10]),
                extend_lens=[10],
                forward_mode=SimpleNamespace(value=1),
                return_hidden_states=False,
                chunked_req=None,
            )
            for i in range(2)
        ]

        def ingest():
            polls.append(True)
            if len(polls) > 1:
                batches[0].reqs[0].finished = True
            return []

        pending = iter(batches)
        scheduler.__dict__.update(
            disaggregation_mode=vpp_mod.DisaggregationMode.NULL,
            pp_loop_size=2,
            _vpp_next_slot=0,
            _vpp_next_batch_seq=0,
            running_mbs=[object(), object()],
            last_mbs=[None, None],
            mbs=[None, None],
            chunked_req=None,
            vpp_batches=VPPBatchLedger(layout, 0, 2),
            vpp_schedule=VPPFixedSchedule(layout, 0, 2),
            pp_group=SimpleNamespace(
                is_first_rank=True,
                is_last_rank=False,
                all_gather_object=lambda manifest: [manifest, manifest],
            ),
            send_req_work=[],
            increment_scheduler_iterations=lambda: None,
            _pp_mark_capacity_digest=lambda: None,
            _pp_check_capacity_digest=lambda: None,
            _pp_ingest_requests=ingest,
            _pp_commit_comm_work=lambda works: None,
            _pp_send_reqs_to_next_stage=lambda requests: [],
            _pp_send_capacity_digest=lambda: None,
            pause_state=SimpleNamespace(may_launch=lambda: True),
            get_next_batch_to_run=lambda **kwargs: SimpleNamespace(
                running_batch=kwargs["running_batch"], batch_to_run=next(pending)
            ),
        )
        wave = scheduler._vpp_admit_wave()
        self.assertEqual(
            [state.batch.reqs[0].rid for state in wave], ["req-0", "req-1"]
        )
        self.assertFalse(batches[0].reqs[0].finished)
        self.assertEqual([state.slot_id for state in wave], [0, 1])

    def test_physical_last_rank_samples_only_at_global_final_stage(self):
        """The last physical rank's earlier passes must return activations."""
        samples = []
        layout = VPPLayout(12, 2, virtual_stages=3)

        def forward(batch, *, pp_proxy_tensors, vpp_stage_id):
            return SimpleNamespace(
                logits_output=(
                    SimpleNamespace(next_token_logits=torch.ones(1, 4))
                    if vpp_stage_id == 5
                    else PPProxyTensors({"hidden_states": torch.ones(1, 2)})
                ),
                can_run_graph=False,
                expert_distribution_metrics=None,
                routed_experts_output=None,
                indexer_topk_output=None,
                kv_weight_version_record=None,
            )

        worker = object.__new__(TpModelWorker)
        worker.__dict__.update(
            pp_group=SimpleNamespace(is_last_rank=True),
            enable_overlap=True,  # VPP must also bypass delayed sampling.
            enable_spec=False,
            is_dllm=lambda: False,
            _maybe_finalize_elastic_cuda_graph_scale=lambda: None,
            _model_runner=SimpleNamespace(
                model=SimpleNamespace(vpp_layout=layout),
                forward=forward,
                sample=lambda logits, batch: samples.append(batch) or torch.tensor([2]),
            ),
        )
        batch = SimpleNamespace(
            apply_deprecated_skip_attn_backend_init=lambda value: None,
            is_prefill_only=False,
            sampling_info=SimpleNamespace(grammars=[object()]),
        )
        for stage_id in (1, 3, 5):
            result = worker.forward_batch_generation(
                None, forward_batch=batch, vpp_stage_id=stage_id
            )
            if stage_id < 5:
                self.assertIsNotNone(result.pp_hidden_states_proxy_tensors)
                self.assertIsNone(result.next_token_ids)
                self.assertEqual(samples, [])
            else:
                self.assertEqual(result.next_token_ids.tolist(), [2])
                self.assertIsNone(result.delay_sample_func)
                self.assertEqual(samples, [batch])

    def test_stage_replans_metadata_without_replacing_retained_inputs(self):
        """Each pass must rebuild shared metadata after another batch used it."""
        seen = []
        backend = SimpleNamespace(current=None)

        def plan(batch):
            backend.current = batch.input_ids.clone()

        def forward(stage_id, input_ids, positions, batch, **kwargs):
            torch.testing.assert_close(backend.current, input_ids)
            seen.append((stage_id, input_ids.data_ptr(), input_ids.tolist()))
            return input_ids + stage_id

        backend.init_forward_metadata = plan
        layout = VPPLayout(12, 2, virtual_stages=3)
        model = SimpleNamespace(
            vpp_layout=layout,
            vpp_stages=layout.stages_for_rank(0),
            forward_vpp_stage=forward,
        )
        batches = [
            SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_decode=lambda: True,
                    is_extend=lambda: False,
                    is_mixed=lambda: False,
                ),
                global_num_tokens_cpu=None,
                input_ids=torch.tensor([i, i + 1]),
                positions=torch.tensor([7, 8]),
                input_embeds=None,
                replace_embeds=None,
                attn_tp_sequence_sharded=False,
            )
            for i in (10, 20)
        ]
        for stage_id in (0, 2, 4):
            for batch in batches:
                execute_vpp_stage(
                    model=model,
                    attn_backend=backend,
                    forward_batch=batch,
                    stage_id=stage_id,
                    pp_proxy_tensors=None,
                    device_timer=None,
                )
        self.assertEqual(
            seen,
            [
                (stage_id, batch.input_ids.data_ptr(), batch.input_ids.tolist())
                for stage_id in (0, 2, 4)
                for batch in batches
            ],
        )

    def test_multiple_waves_keep_inputs_outputs_and_completion_independent(self):
        """A partial wave must not leave receives or recycle workspace views."""
        from threading import local

        topology = local()
        layout = VPPLayout(12, physical_size=2, virtual_stages=3)
        queues = [Queue(), Queue()]
        schedulers = []
        calls = [[], []]
        launches = [[], []]
        completions = [[], []]
        for rank in range(2):
            scheduler = object.__new__(SchedulerVPPMixin)
            scheduler.__dict__.update(
                pp_group=SimpleNamespace(is_first_rank=rank == 0),
                vpp_schedule=VPPFixedSchedule(layout, rank, 2),
                vpp_batches=VPPBatchLedger(layout, rank, 2),
                vpp_receiver=_Receiver(queues[rank]),
                vpp_sends=_Sends(queues[1 - rank]),
                forward_stream=_Stream(),
                schedule_stream=_Stream(),
                forward_stream_ctx=nullcontext(),
                device_module=SimpleNamespace(Event=_Event),
                tp_worker=_Worker(rank, calls[rank]),
                future_map=None,
                enable_unified_memory=False,
                resolve_forward_inputs=lambda batch, future_map: None,
                _pp_prepare_tensor_dict=lambda result, batch: {
                    "next_token_ids": result.next_token_ids
                },
                _on_step_launch=lambda batch, r=rank: (
                    launches[r].append(batch.batch_seq) or batch.batch_seq
                ),
                _on_step_forward_recorded=lambda event, r=rank: completions[r].append(
                    event
                ),
            )
            schedulers.append(scheduler)

        def init_forward_batch(batch, runner, *, return_hidden_states_before_norm):
            self.assertFalse(return_hidden_states_before_norm)
            return SimpleNamespace(
                batch_seq=batch.batch_seq, final_stage=layout.logical_size - 1
            )

        def execute(rank, wave):
            topology.rank = rank
            schedulers[rank]._vpp_execute_wave(wave)

        with (
            patch.object(
                vpp_mod,
                "get_parallel",
                lambda: SimpleNamespace(pp_rank=topology.rank),
            ),
            patch.object(vpp_mod.ForwardBatch, "init_new", init_forward_batch),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            for sequences in ((0, 1), (2,), (3, 4)):
                waves = [[], []]
                for rank, scheduler in enumerate(schedulers):
                    for slot_id, seq in enumerate(sequences):
                        batch = SimpleNamespace(
                            batch_seq=seq,
                            hicache_consumer_index=-1,
                            return_logprob=False,
                            extend_lens=[1],
                            forward_mode=SimpleNamespace(is_extend=lambda: True),
                        )
                        state = VPPBatchState(seq, slot_id, ("req", seq), batch)
                        scheduler.vpp_batches.admit(state)
                        waves[rank].append(state)
                        if rank == 0:
                            scheduler.vpp_schedule.enqueue(vpp_mod.VPPTask(seq, 0))
                futures = [
                    executor.submit(execute, rank, waves[rank]) for rank in range(2)
                ]
                for future in futures:
                    future.result(timeout=15)
                for rank, scheduler in enumerate(schedulers):
                    self.assertFalse(scheduler.vpp_receiver.posted)
                    for state in waves[rank]:
                        self.assertTrue(state.metadata.forward_done.recorded)
                        if rank == 1:
                            self.assertEqual(
                                state.output_tensors["next_token_ids"].item(),
                                state.batch_seq + layout.logical_size,
                            )
                        scheduler.vpp_batches.complete(state.batch_seq)
        for rank in range(2):
            self.assertEqual(launches[rank], list(range(5)))
            self.assertEqual(len(completions[rank]), 5)
            self.assertEqual(
                calls[rank],
                [
                    (seq, stage_id)
                    for sequences in ((0, 1), (2,), (3, 4))
                    for stage_id in range(rank, 6, 2)
                    for seq in sequences
                ],
            )


if __name__ == "__main__":
    unittest.main()
