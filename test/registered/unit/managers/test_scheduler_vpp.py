"""Finite VPP waves against queue transport and CPU model boundaries."""

import ast
import inspect
import sys
import textwrap
import unittest
from array import array
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.distributed.vpp_layout import VPPLayout
from sglang.srt.distributed.vpp_policy import (
    PipelineControlEnvelope,
    PipelineControlKind,
    PipelinePrefixRegistry,
    PipelineRankSchedule,
    PipelineWavefrontAction,
)
from sglang.srt.distributed.vpp_transport import P2PWorkGroup
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.scheduler_vpp_core import SchedulingVPPCore
from sglang.srt.managers.scheduler_vpp_mixin import SchedulerVPPMixin, _snapshot_tensors
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.forward_mode import ForwardMode
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


class _AdmissionCore(SchedulingVPPCore):
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


class _LoopStopped(Exception):
    pass


def _run_native_ticks(scheduler, ticks, before_tick=None):
    """Bound the real loop at its tick boundary without changing its body."""
    method = SchedulingVPPCore._event_loop_vpp_rank_local
    lines, first_line = inspect.getsourcelines(method)
    definition = ast.parse(textwrap.dedent("".join(lines))).body[0]
    loop = next(node for node in definition.body if isinstance(node, ast.While))
    boundary = first_line + loop.body[0].lineno - 1
    context = {}

    def trace(frame, event, arg):
        if frame.f_code is method.__code__ and event == "line":
            if frame.f_lineno == boundary:
                context.update(frame.f_locals)
                if context["tick"] >= ticks:
                    raise _LoopStopped
                scheduler.loop_tick = context["tick"]
                if before_tick is not None:
                    before_tick(scheduler, context)
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        scheduler._event_loop_vpp_rank_local()
    except _LoopStopped:
        pass
    finally:
        sys.settrace(previous)
    return context


class _ActivationFIFO:
    """CPU communication boundary with the native receiver's FIFO contract."""

    def __init__(self):
        self.queue = deque()

    def progress(self):
        pass

    def receive_count(self):
        return len(self.queue)

    def take(self):
        return self.queue.popleft()


class _PendingSend:
    def __init__(self, action, waits):
        self.action = action
        self.waits = waits

    def poll(self):
        return False

    def wait(self):
        self.waits.append((self.action.batch_seq, self.action.stage_id))


class _TickCore(_ControlCore):
    """Native core with CPU request, model, and communication boundaries."""

    def _init_scheduling_native_loop_state(self):
        super()._init_scheduling_native_loop_state()
        self.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
        self.disagg_prefill_transfer_queue = []
        self.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 2
        )
        self._pending_chunked_abort_req = None
        self.last_rank_comm_queue = deque()
        self.controls = deque()
        self.published = []
        self.actions = []
        self.send_waits = []
        self.output_waits = []
        self.finalized = []
        self.idle_ticks = []
        self.pp_comm_stream_ctx = nullcontext()
        self.attn_tp_group.broadcast_object = lambda value, src: value
        self._pp_vpp_available_prefill_tokens = lambda: 32

    def ingest_requests(self):
        return []

    def process_pending_chunked_abort(self):
        self._pending_chunked_abort_req = None

    def _pp_vpp_poll_control_receiver(self):
        return self.controls.popleft() if self.controls else None

    def _pp_vpp_flush_control_outbox(self, pending_work, max_pending):
        while self._pp_vpp_control_outbox:
            self.published.append(self._pp_vpp_control_outbox.popleft())

    def _pp_vpp_prepare_wavefront_batch(
        self, batch_seq, slot_batch_seqs, expected_manifest=None
    ):
        batch = self.batches.get(batch_seq)
        if batch is None:
            return None
        if not all(req in self.waiting_queue for req in batch.reqs):
            return None
        for req in batch.reqs:
            self.waiting_queue.remove(req)
        slot_id = batch_seq % len(slot_batch_seqs)
        self.mbs[slot_id] = batch
        slot_batch_seqs[slot_id] = batch_seq
        return batch

    def _pp_vpp_execute_wavefront_action(self, action, slot_batch_seqs, proxy):
        self.actions.append((action.batch_seq, action.stage_id))
        value = torch.tensor([float(action.batch_seq)])
        if action.stage_id:
            value = proxy.tensors["hidden_states"]
        value = value + 1
        event = SimpleNamespace(
            query=lambda: True,
            synchronize=lambda: self.output_waits.append(action.batch_seq),
        )
        self._pp_vpp_stage_events[(action.batch_seq, action.stage_id)] = event
        if action.stage_id == 3:
            self.last_rank_comm_queue.append(
                (
                    event,
                    PPProxyTensors(
                        {"vpp_batch_seq": action.batch_seq, "next_token_ids": value}
                    ),
                )
            )
            return True, None
        return True, _PendingSend(action, self.send_waits)

    def _pp_vpp_finalize_rank_local_batch(
        self, batch_seq, slot_batch_seqs, output_tensors, release_slot=True
    ):
        slot_id = batch_seq % len(slot_batch_seqs)
        if slot_batch_seqs[slot_id] != batch_seq or self.mbs[slot_id] is None:
            raise RuntimeError("CPU finalization does not own its batch")
        self.finalized.append((batch_seq, release_slot, output_tensors))
        self.last_mbs[slot_id] = self.mbs[slot_id]
        self.mbs[slot_id] = None
        self.mb_metadata[slot_id] = None
        if release_slot:
            slot_batch_seqs[slot_id] = None

    def on_idle(self):
        self.idle_ticks.append(self.loop_tick)


class TestSchedulingVPPCore(CustomTestCase):
    def _scheduler(
        self, *, rank=0, owner=0, retained=True, core_type=SchedulingVPPCore
    ):
        scheduler = core_type()
        scheduler.__dict__.update(
            disaggregation_mode=DisaggregationMode.PREFILL,
            pause_state=SimpleNamespace(may_launch=lambda: True),
            _pp_vpp_runtime_epoch=3,
            _pp_vpp_control_layout_digest="layout",
            _pp_vpp_slot_batch_seqs=[owner, None],
            _pp_vpp_ready_proxies={},
            _pp_vpp_early_activations={},
            _pp_vpp_fixed_received=deque(),
            _pp_vpp_arrivals=deque(),
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
        scheduler._pp_vpp_init_fixed_batch(2)
        return scheduler

    def _remote_admission(self, *, rank=0, core_type=SchedulingVPPCore):
        scheduler = self._scheduler(rank=rank, core_type=core_type)
        scheduler.attn_tp_group = SimpleNamespace(
            world_size=1, rank_in_group=0, cpu_group=None
        )
        req = SimpleNamespace(
            rid="request",
            session_generation=1,
            bootstrap_room=3,
            origin_input_ids=list(range(16)),
            extend_end=None,
            prefix_len=0,
            kv=SimpleNamespace(holds_kv=False),
            finished=lambda: False,
        )
        scheduler.waiting_queue = [req]
        scheduler.chunked_req = None
        scheduler.mbs = [None, None]
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

    def _retained_remote_admission(self, *, committed):
        scheduler, req, envelope = self._remote_admission(rank=1)
        req.extend_end = 4
        req.kv.holds_kv = True
        req.skip_radix_cache_insert = True
        req._vpp_original_skip_radix_cache_insert = False
        scheduler.chunked_req = req
        scheduler.waiting_queue = []
        registry = scheduler._pp_vpp_prefix_registry
        registry.plan(req.rid, 1, 1, 0, 4, range(4))
        if committed:
            for stage in range(4):
                registry.mark_materialized(req.rid, 1, stage, 4)
            registry.commit(req.rid, 1, 4)
        cached_ends = []
        cache = SimpleNamespace(
            checkpoint=lambda row, *, up_to: cached_ends.append(up_to)
        )
        scheduler.stash_chunked_request = lambda row: checkpoint_kv_cache(row, cache)
        return scheduler, req, envelope, cached_ends

    def _downstream_admission(self, *, batch_seq=0, owner=None):
        scheduler, req, _ = self._remote_admission(rank=1, core_type=_AdmissionCore)
        scheduler._vpp_layout = SimpleNamespace(virtual_stages=2)
        scheduler._pp_vpp_init_fixed_batch(2)
        scheduler._pp_vpp_slot_batch_seqs = [owner, None]
        scheduler.candidate_batch = SimpleNamespace(
            reqs=[req], prefix_lens=[4], extend_lens=[5]
        )
        wire = self._wire(batch_seq=batch_seq, stage=1)
        wire["vpp_batch_manifest"] = scheduler._pp_vpp_batch_manifest(
            batch_seq, batch_seq % 2, scheduler.candidate_batch
        )
        return scheduler, PipelineRankSchedule(1, 2, 2, 2), wire

    def _control_loop(
        self,
        *,
        rank=1,
        core_type=_ControlCore,
        ticks=0,
        before_tick=None,
        mode=DisaggregationMode.PREFILL,
    ):
        scheduler, _, _ = self._remote_admission(rank=rank, core_type=core_type)
        scheduler.disaggregation_mode = mode
        scheduler._vpp_layout = SimpleNamespace(virtual_stages=2)
        scheduler.receive_tags = []
        scheduler.pp_group.vpp_activation_tp_group = scheduler.attn_tp_group
        scheduler.activation_group = SimpleNamespace(
            device="cpu",
            recv_tensor_dict_async=lambda tag, **kwargs: scheduler.receive_tags.append(
                tag
            ),
        )
        scheduler.world_group = SimpleNamespace(
            rank_in_group=rank,
            broadcast_object=lambda value, src: 3,
        )
        return scheduler, _run_native_ticks(scheduler, ticks, before_tick)

    def _tick_batches(self, scheduler, count):
        scheduler.batches = {}
        scheduler.waiting_queue = []
        for seq in range(count):
            req = SimpleNamespace(
                rid=f"request-{seq}",
                session_generation=1,
                bootstrap_room=seq,
                origin_input_ids=list(range(8)),
                extend_end=8,
                kv=SimpleNamespace(holds_kv=False),
                finished=lambda: False,
            )
            scheduler.waiting_queue.append(req)
            scheduler.batches[seq] = SimpleNamespace(
                reqs=[req],
                prefix_lens=[0],
                extend_lens=[8],
                chunked_req=None,
                disagg_prefill_chunk_end_by_rid={req.rid: 8},
            )
        scheduler._pp_vpp_activation_receiver = _ActivationFIFO()

    def _tick_activation(self, scheduler, seq, stage):
        wire = self._wire(batch_seq=seq, stage=stage)
        wire["hidden_states"] = torch.tensor([float(seq + stage)])
        if stage == 1:
            wire["vpp_batch_manifest"] = scheduler._pp_vpp_batch_manifest(
                seq, seq % 2, scheduler.batches[seq]
            )
        scheduler._pp_vpp_activation_receiver.queue.append(wire)

    def test_native_ticks_run_two_batch_waves_and_reuse_retired_slots(self):
        def before_tick(scheduler, context):
            tick = context["tick"]
            if tick == 0:
                self._tick_batches(scheduler, 4)
            if tick in (2, 3, 6, 7):
                seq = {2: 0, 3: 1, 6: 2, 7: 3}[tick]
                self._tick_activation(scheduler, seq, 2)
            if tick in (4, 5, 8, 9):
                seq = {4: 0, 5: 1, 8: 2, 9: 3}[tick]
                envelope, wire = self._control_message(
                    scheduler, PipelineControlKind.COMPLETION, seq=seq
                )
                wire["next_token_ids"] = torch.tensor([seq + 4])
                scheduler.controls.append((envelope, wire))

        scheduler, context = self._control_loop(
            rank=0, core_type=_TickCore, ticks=10, before_tick=before_tick
        )
        self.assertEqual(
            scheduler.actions,
            [(0, 0), (1, 0), (0, 2), (1, 2), (2, 0), (3, 0), (2, 2), (3, 2)],
        )
        self.assertEqual(context["rank_schedule"].inflight_count, 0)
        self.assertEqual(scheduler._pp_vpp_slot_batch_seqs, [None, None])
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(scheduler._pp_vpp_ready_proxies, {})
        self.assertEqual(
            [(seq, released) for seq, released, _ in scheduler.finalized],
            [(0, True), (1, True), (2, True), (3, True)],
        )
        self.assertEqual(
            [output["next_token_ids"].item() for _, _, output in scheduler.finalized],
            [4, 5, 6, 7],
        )
        self.assertEqual(scheduler.send_waits, scheduler.actions[:-2])
        self.assertLessEqual(len(context["activation_send_work"]), 2)
        self.assertFalse(scheduler._pp_vpp_fixed_wave)
        self.assertFalse(scheduler._pp_vpp_fixed_second_pass)
        materialized = [
            wire
            for wire in scheduler.published
            if wire["kind"] == PipelineControlKind.PREFIX_MATERIALIZED.value
        ]
        self.assertEqual(len(materialized), 8)
        self.assertEqual(
            {(wire["batch_seq"], wire["payload"]["stage_id"]) for wire in materialized},
            set(scheduler.actions),
        )
        self.assertTrue(all(wire["payload"]["end"] == 8 for wire in materialized))

    def test_native_ticks_admit_unrelated_batch_while_chunk_is_pending(self):
        def before_tick(scheduler, context):
            if context["tick"] == 0:
                self._tick_batches(scheduler, 2)
                batch = scheduler.batches[0]
                batch.chunked_req = batch.reqs[0]

        scheduler, context = self._control_loop(
            rank=0, core_type=_TickCore, ticks=2, before_tick=before_tick
        )
        self.assertEqual(scheduler.actions, [(0, 0), (1, 0)])
        self.assertEqual(context["pending_chunk_batches"], {0})
        self.assertEqual(scheduler.finalized, [])

    def test_native_ticks_quarantine_next_generation_until_completion_returns(self):
        snapshots = {}

        def before_tick(scheduler, context):
            tick = context["tick"]
            if tick == 0:
                self._tick_batches(scheduler, 3)
            snapshots[tick] = (
                tuple(scheduler._pp_vpp_slot_batch_seqs),
                tuple(scheduler.mbs),
            )
            if tick in (0, 1, 2, 3, 4):
                seq, stage = [(0, 1), (1, 1), (0, 3), (1, 3), (2, 1)][tick]
                self._tick_activation(scheduler, seq, stage)
            if tick in (4, 5):
                scheduler.controls.append(
                    self._control_message(
                        scheduler,
                        PipelineControlKind.COMPLETION,
                        seq=tick - 4,
                        returned=True,
                    )
                )

        scheduler, context = self._control_loop(
            rank=1, core_type=_TickCore, ticks=6, before_tick=before_tick
        )
        self.assertEqual(scheduler.actions, [(0, 1), (1, 1), (0, 3), (1, 3), (2, 1)])
        self.assertEqual(snapshots[4], ((0, 1), (None, None)))
        self.assertEqual(context["rank_schedule"].slot_batch_seqs, (2, None))
        self.assertEqual(scheduler._pp_vpp_slot_batch_seqs, [2, None])
        self.assertEqual(scheduler._pp_vpp_early_activations, {})
        self.assertEqual(scheduler._pp_vpp_ready_proxies, {})
        self.assertEqual(scheduler.output_waits, [0, 1])
        self.assertEqual(
            [(seq, released) for seq, released, _ in scheduler.finalized],
            [(0, False), (1, False)],
        )
        completions = [
            wire
            for wire in scheduler.published
            if wire["kind"] == PipelineControlKind.COMPLETION.value
        ]
        self.assertEqual([wire["batch_seq"] for wire in completions], [0, 1])
        self.assertEqual(
            [wire["next_token_ids"].item() for wire in completions], [4, 5]
        )

    def test_native_ticks_call_idle_only_without_inflight_batches_or_sends(self):
        def before_tick(scheduler, context):
            if context["tick"] == 0:
                self._tick_batches(scheduler, 0)

        scheduler, context = self._control_loop(
            rank=0, core_type=_TickCore, ticks=3, before_tick=before_tick
        )
        self.assertEqual(scheduler.idle_ticks, [0, 1, 2])
        self.assertEqual(scheduler.actions, [])
        self.assertEqual(scheduler.published, [])
        self.assertEqual(context["rank_schedule"].inflight_count, 0)

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

    def test_control_loop_initializes_identity_receivers_and_send_bounds(self):
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
                self.assertEqual(scheduler._pp_vpp_pending_batch_controls, deque())

    def test_control_publication_waits_for_fifo_events(self):
        scheduler, context = self._control_loop()
        registry = scheduler._pp_vpp_prefix_registry
        entry = registry.plan("request", 1, 1, 0, 8, range(4))
        ready = [False]
        event = SimpleNamespace(query=lambda: ready[0])
        scheduler._pp_vpp_stage_events = {
            (0, 0): event,
            (0, 1): SimpleNamespace(query=lambda: True),
        }
        scheduler._pp_vpp_pending_materialized.extend(
            [(0, 0, "request", 1, 8), (0, 1, "request", 1, 8)]
        )
        context["flush_materialized"]()
        self.assertEqual(entry.materialized_by_stage[1], 0)
        self.assertEqual(len(scheduler._pp_vpp_pending_materialized), 2)
        self.assertEqual(list(scheduler._pp_vpp_control_outbox), [])
        ready[0] = True
        context["flush_materialized"]()
        self.assertNotIn((0, 0), scheduler._pp_vpp_stage_events)
        self.assertNotIn((0, 1), scheduler._pp_vpp_stage_events)
        self.assertEqual(entry.materialized_by_stage, {0: 8, 1: 8, 2: 0, 3: 0})
        self.assertEqual(entry.committed_end, 0)
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
            ],
        )
        self.assertEqual([message.hops for message in messages], [1, 1])

    def test_batch_controls_wait_for_activation_admission(self):
        scheduler, context = self._control_loop()
        kinds = (
            PipelineControlKind.CANCEL,
            PipelineControlKind.COMPLETION,
            PipelineControlKind.PREFIX_MATERIALIZED,
            PipelineControlKind.PREFIX_COMMIT,
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
        self.assertTrue(scheduler._pp_vpp_remote_manifest_ready(envelope))
        scheduler.chunked_req = req
        scheduler.waiting_queue = []
        self.assertTrue(scheduler._pp_vpp_remote_manifest_ready(envelope))
        registry = scheduler._pp_vpp_prefix_registry
        registry.plan("request", 1, 1, 0, 4, range(4))
        self.assertTrue(scheduler._pp_vpp_remote_manifest_ready(envelope))
        registry.plan("request", 1, 1, 4, 8, range(4))
        with self.assertRaisesRegex(RuntimeError, "prefix order mismatch"):
            scheduler._pp_vpp_remote_manifest_ready(envelope)

    def test_remote_manifest_can_select_a_different_chunked_request(self):
        scheduler, req, envelope = self._remote_admission()
        first = SimpleNamespace(rid="first")
        scheduler.chunked_req = first
        scheduler.chunked_reqs = [first, req]
        scheduler.waiting_queue = []
        scheduler.mbs = [None, SimpleNamespace(reqs=[first])]
        self.assertTrue(scheduler._pp_vpp_remote_manifest_ready(envelope))
        scheduler.mbs[0] = SimpleNamespace(reqs=[req])
        self.assertFalse(scheduler._pp_vpp_remote_manifest_ready(envelope))
        scheduler.mbs[0] = None
        self.assertTrue(scheduler._pp_vpp_remote_manifest_ready(envelope))

    def test_remote_admission_checkpoints_completed_chunk_before_capacity_check(self):
        scheduler, req, envelope, cached_ends = self._retained_remote_admission(
            committed=True
        )
        scheduler._pp_vpp_available_prefill_tokens = lambda: 9 if cached_ends else 8

        self.assertTrue(scheduler._pp_vpp_remote_manifest_ready(envelope))
        self.assertEqual(cached_ends, [4])
        self.assertEqual(req.prefix_len, 4)
        self.assertFalse(req.skip_radix_cache_insert)
        self.assertTrue(scheduler._pp_vpp_remote_manifest_ready(envelope))
        self.assertEqual(cached_ends, [4])

    def test_remote_admission_waits_for_retained_chunk_commit(self):
        for missing_entry in (False, True):
            with self.subTest(missing_entry=missing_entry):
                scheduler, req, envelope, cached_ends = self._retained_remote_admission(
                    committed=False
                )
                if missing_entry:
                    scheduler._pp_vpp_prefix_registry.release(req.rid, 1)

                self.assertFalse(scheduler._pp_vpp_remote_manifest_ready(envelope))
                self.assertEqual(cached_ends, [])
                self.assertEqual(req.prefix_len, 0)
                self.assertTrue(req.skip_radix_cache_insert)

    def test_remote_chunk_checkpoint_requires_tp_readiness_and_matching_counts(self):
        for peer_state in ((0, 1), (1, 0), (1, 1)):
            with self.subTest(peer_state=peer_state):
                scheduler, req, envelope, cached_ends = self._retained_remote_admission(
                    committed=True
                )

                def gather(local):
                    peer = peer_state if len(local) == 2 else (1,)
                    return torch.tensor([local, peer], dtype=torch.int64)

                scheduler._pp_vpp_all_gather_cpu_state = gather
                ready = peer_state == (1, 1)
                self.assertEqual(
                    scheduler._pp_vpp_remote_manifest_ready(envelope), ready
                )
                self.assertEqual(cached_ends, [4] if ready else [])
                self.assertEqual(req.prefix_len, 4 if ready else 0)
                self.assertEqual(req.skip_radix_cache_insert, not ready)

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
                    peer = local.clone()
                    peer[0] = other_ready
                    output.copy_(torch.cat((local, peer)))

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
                PipelineControlKind.REQUEST, payload={}
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
        scheduler._pp_vpp_activation_receiver = _ActivationFIFO()
        scheduler._pp_vpp_round_candidate = False
        scheduler.attn_tp_group = SimpleNamespace(
            rank_in_group=0, broadcast_object=lambda value, src: value
        )
        envelope = scheduler._pp_vpp_new_control(PipelineControlKind.REQUEST)
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
        for pp_size, virtual_stages, inflight in (
            (3, 2, 2),
            (2, 3, 2),
            (2, 2, 1),
        ):
            with self.subTest(pp=pp_size, vpp=virtual_stages, inflight=inflight):
                scheduler = self._scheduler()
                scheduler.pp_group.world_size = pp_size
                scheduler._vpp_layout = SimpleNamespace(virtual_stages=virtual_stages)
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
    def test_wavefront_batch_does_not_publish_a_parked_chunk(self):
        for chunk_runs in (False, True):
            with self.subTest(chunk_runs=chunk_runs):
                running = SimpleNamespace(
                    rid="running", origin_input_ids=list(range(16)), beam_group=None
                )
                parked = SimpleNamespace(rid="parked")
                chunk = running if chunk_runs else parked
                batch = SimpleNamespace(
                    reqs=[running],
                    chunked_req=chunk,
                    chunked_req_next_prompt_token=7,
                    prefix_lens=[4],
                    extend_lens=[4],
                    return_hidden_states=False,
                )
                scheduler = SchedulerVPPMixin()
                scheduler.mbs = [None, None]
                scheduler.running_mbs = [object(), object()]
                scheduler.last_mbs = [None, None]
                scheduler.mb_metadata = [None, None]
                scheduler.chunked_reqs = [chunk]
                scheduler.process_pending_chunked_abort = lambda: None
                scheduler.process_prefill_chunk = lambda **kwargs: None
                scheduler.get_new_batch_prefill = lambda running_batch: SimpleNamespace(
                    batch_to_run=batch, running_batch=running_batch
                )
                scheduler.dp_attn_adapter = SimpleNamespace(
                    maybe_prepare_mlp_sync_batch=lambda value: value
                )
                seen = []

                def prepare(value, chunked_req):
                    seen.append(chunked_req)
                    return value

                scheduler.ngram_embedding_manager = SimpleNamespace(
                    prepare_for_forward=prepare
                )
                owners = [None, None]
                result = scheduler._pp_vpp_prepare_wavefront_batch(0, owners)
                expected = running if chunk_runs else None
                self.assertIs(result.chunked_req, expected)
                self.assertEqual(seen, [expected])
                self.assertEqual(result.disagg_prefill_chunk_end_by_rid, {"running": 8})
                self.assertEqual(scheduler.chunked_reqs, [chunk])
                self.assertEqual(owners, [0, None])
                self.assertEqual(
                    result.chunked_req_next_prompt_token, 7 if chunk_runs else None
                )

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


class TestNativeVPPAdapter(CustomTestCase):
    def test_completion_fences_copy_before_retiring_batch(self):
        order = []
        event = SimpleNamespace(
            record=lambda stream: order.append("record"),
            synchronize=lambda: order.append("copy_ready"),
        )
        batch = SimpleNamespace(reqs=[], contains_last_prefill_chunk=False)
        launch = object()
        metadata = SimpleNamespace(forward_launch=launch, forward_done=object())
        result = SimpleNamespace(copy_done=None, forward_launch=launch)
        scheduler = object.__new__(SchedulerVPPMixin)
        scheduler.__dict__.update(
            mbs=[batch, None],
            mb_metadata=[metadata, None],
            last_mbs=[None, None],
            _vpp_forward_batches={0: object()},
            copy_stream_ctx=nullcontext(),
            copy_stream=SimpleNamespace(
                wait_stream=lambda stream: order.append("wait_schedule"),
                wait_event=lambda event: order.append("wait_forward"),
            ),
            schedule_stream=object(),
            device="cpu",
            device_module=SimpleNamespace(Event=lambda: event),
            disaggregation_mode=DisaggregationMode.PREFILL,
            _pp_prep_batch_result=lambda *args: result,
            _pp_process_batch_result=lambda batch, result: order.append("processed"),
        )
        slots = [0, None]
        scheduler._pp_vpp_finalize_rank_local_batch(
            0, slots, {"next_token_ids": torch.tensor([3])}
        )
        self.assertEqual(
            order,
            ["wait_schedule", "wait_forward", "record", "copy_ready", "processed"],
        )
        self.assertIsNone(result.copy_done)
        self.assertIs(result.forward_launch, launch)
        self.assertIs(scheduler.last_mbs[0], batch)
        self.assertIsNone(scheduler.mbs[0])
        self.assertEqual(slots, [None, None])
        self.assertFalse(scheduler._vpp_forward_batches)

    def test_completion_restores_cache_policy_after_fence_without_prefix_wait(self):
        for committed in (False, True):
            for original_skip in (False, True):
                with self.subTest(committed=committed, original_skip=original_skip):
                    req = SimpleNamespace(
                        rid="request",
                        session_generation=1,
                        prefix_len=4,
                        extend_end=8,
                        finished=lambda: False,
                        skip_radix_cache_insert=True,
                        _vpp_original_skip_radix_cache_insert=original_skip,
                    )
                    registry = PipelinePrefixRegistry()
                    registry.plan(req.rid, 1, 1, 0, 8, range(4))
                    if committed:
                        for stage in range(4):
                            registry.mark_materialized(req.rid, 1, stage, 8)
                        registry.commit(req.rid, 1, 8)
                    order = []
                    cached_ends = []
                    cache = SimpleNamespace(
                        checkpoint=lambda row, *, up_to: cached_ends.append(up_to)
                    )

                    def process(batch, result):
                        order.append(("processed", req.skip_radix_cache_insert))
                        checkpoint_kv_cache(req, cache)

                    event = SimpleNamespace(
                        record=lambda stream: None,
                        synchronize=lambda: order.append(
                            ("copy_ready", req.skip_radix_cache_insert)
                        ),
                    )
                    batch = SimpleNamespace(
                        reqs=[req], contains_last_prefill_chunk=False
                    )
                    scheduler = object.__new__(SchedulerVPPMixin)
                    scheduler.__dict__.update(
                        mbs=[batch, None],
                        mb_metadata=[SimpleNamespace(forward_done=object()), None],
                        last_mbs=[None, None],
                        _vpp_forward_batches={0: object()},
                        _pp_vpp_prefix_registry=registry,
                        copy_stream_ctx=nullcontext(),
                        copy_stream=_Stream(),
                        schedule_stream=object(),
                        device="cpu",
                        device_module=SimpleNamespace(Event=lambda: event),
                        disaggregation_mode=DisaggregationMode.PREFILL,
                        _pp_prep_batch_result=lambda *args: SimpleNamespace(),
                        _pp_process_batch_result=process,
                    )
                    slots = [0, None]
                    scheduler._pp_vpp_finalize_rank_local_batch(0, slots, {})

                    self.assertEqual(
                        order, [("copy_ready", True), ("processed", original_skip)]
                    )
                    self.assertEqual(cached_ends, [] if original_skip else [8])
                    self.assertEqual(req.prefix_len, 8)
                    self.assertEqual(req.skip_radix_cache_insert, original_skip)
                    self.assertIs(scheduler.last_mbs[0], batch)
                    self.assertIsNone(scheduler.mbs[0])
                    self.assertEqual(slots, [None, None])

    def test_snapshot_preserves_owned_outputs_and_freezes_reused_metadata(self):
        hidden = torch.arange(6).reshape(2, 3)
        source = torch.arange(8)
        index = torch.tensor([1, 2])
        candidate = torch.tensor([3, 4])
        snapshot = _snapshot_tensors(
            {
                "hidden_states": hidden,
                "vpp_source_kv": source,
                "vpp_index_positions": index,
                "vpp_candidate_blocks": candidate,
                "stage": 2,
            }
        )
        index.fill_(99)
        candidate.fill_(99)
        torch.testing.assert_close(
            snapshot["vpp_index_positions"], torch.tensor([1, 2])
        )
        torch.testing.assert_close(
            snapshot["vpp_candidate_blocks"], torch.tensor([3, 4])
        )
        self.assertIs(snapshot["hidden_states"], hidden)
        self.assertIs(snapshot["vpp_source_kv"], source)
        self.assertEqual(snapshot["stage"], 2)

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
                forward_mode=ForwardMode.EXTEND,
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

    def test_stage_rejects_decode_and_mixed_batches_before_metadata(self):
        layout = VPPLayout(12, 2, virtual_stages=2)
        model = SimpleNamespace(vpp_layout=layout, vpp_stages=layout.stages_for_rank(0))
        for mode in (ForwardMode.DECODE, ForwardMode.MIXED):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(NotImplementedError, "prefill batches only"),
            ):
                execute_vpp_stage(
                    model=model,
                    attn_backend=None,
                    forward_batch=SimpleNamespace(forward_mode=mode),
                    stage_id=0,
                    pp_proxy_tensors=None,
                    device_timer=None,
                )

    def test_control_identity_is_checked_before_reposting_receive(self):
        """An old epoch/layout must not enter a newly reused control channel."""
        for changed in (
            {},
            {"runtime_epoch": 2},
            {"layout_digest": "old"},
            {"protocol_version": 2},
        ):
            with self.subTest(changed=changed):
                wire = PipelineControlEnvelope(
                    1, 3, "layout", PipelineControlKind.REQUEST, 1
                ).to_dict()
                wire.update(changed)
                ready = [False]
                received = []
                handle = SimpleNamespace(
                    poll_metadata_ready=lambda: ready[0],
                    wait=lambda: wire,
                )

                def receive(*, tag):
                    received.append(tag)
                    return handle

                scheduler = object.__new__(SchedulerVPPMixin)
                scheduler.__dict__.update(
                    attn_tp_group=SimpleNamespace(rank_in_group=0),
                    pp_group=SimpleNamespace(recv_tensor_dict_async=receive),
                    _pp_vpp_pending_control_recv=None,
                    _pp_vpp_runtime_epoch=3,
                    _pp_vpp_control_layout_digest="layout",
                )
                self.assertIsNone(scheduler._pp_vpp_poll_control_receiver())
                self.assertEqual(received, [2])
                ready[0] = True
                if changed:
                    with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                        scheduler._pp_vpp_poll_control_receiver()
                    self.assertEqual(received, [2])
                else:
                    envelope, actual = scheduler._pp_vpp_poll_control_receiver()
                    self.assertEqual(envelope.kind, PipelineControlKind.REQUEST)
                    self.assertIs(actual, wire)
                    self.assertEqual(received, [2, 2])

    def test_control_send_window_preserves_fifo_until_oldest_send_completes(self):
        """A later completed send cannot free an earlier live FIFO owner."""
        completed = [False, False, True]
        sent = []
        works = [SimpleNamespace(poll=lambda i=i: completed[i]) for i in range(3)]

        def send(wire, *, async_send, tag):
            self.assertTrue(async_send)
            self.assertEqual(tag, 2)
            sent.append(wire)
            return works[wire]

        scheduler = object.__new__(SchedulerVPPMixin)
        scheduler.__dict__.update(
            attn_tp_group=SimpleNamespace(rank_in_group=0),
            pp_group=SimpleNamespace(send_tensor_dict=send),
            pp_comm_stream_ctx=nullcontext(),
            _pp_vpp_control_outbox=deque(range(3)),
        )
        pending = deque()
        scheduler._pp_vpp_flush_control_outbox(pending, 2)
        self.assertEqual(sent, [0, 1])
        completed[1] = True
        scheduler._pp_vpp_flush_control_outbox(pending, 2)
        self.assertEqual(sent, [0, 1])
        completed[0] = True
        scheduler._pp_vpp_flush_control_outbox(pending, 2)
        self.assertEqual(sent, [0, 1, 2])
        scheduler._pp_vpp_flush_control_outbox(pending, 2)
        self.assertFalse(pending)

    def test_virtual_passes_retain_sampling_rows_and_snapshot_shared_outputs(self):
        """Pool-row and backend metadata reuse must not alter an earlier batch."""
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.penalties import PenaltyStatusPool
        from sglang.srt.sampling.sampler_params import SamplerParamsPool
        from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
        from sglang.srt.sampling.sampling_params import SamplingParams

        class Event(_Event):
            def synchronize(self):
                if not self.recorded:
                    raise AssertionError("Unrecorded forward fence")

        class Stream(_Stream):
            def wait_event(self, event):
                if not event.recorded:
                    raise AssertionError("Consumer preceded producer")

        stream = Stream()
        workspace = torch.empty(1)
        sent, calls = [], []
        batches = []
        params = SamplerParamsPool(8, "cpu")
        penalties = PenaltyStatusPool(10, "cpu")
        for seq in range(2):
            req = Req(
                rid=str(seq),
                origin_input_text="",
                origin_input_ids=array("q", [1]),
                sampling_params=SamplingParams(
                    max_new_tokens=2,
                    temperature=1.0 if seq == 0 else 0.7,
                    top_k=1 if seq == 0 else 7,
                    sampling_seed=123 + seq,
                    logit_bias={str(seq): 0.5},
                ),
            )
            req.kv.req_pool_idx = seq + 2
            params.write([req])
            batches.append(
                SimpleNamespace(
                    reqs=[req],
                    seq=seq,
                    device="cpu",
                    hicache_consumer_index=0,
                    extend_lens=[1],
                    return_logprob=False,
                    _vpp_fixed_manifest={"batch_seq": seq},
                    req_pool_indices=torch.tensor([seq + 2]),
                    req_pool_indices_cpu=torch.tensor([seq + 2]),
                    sampling_info=SamplingBatchInfo(vocab_size=10, device="cpu"),
                )
            )

        def forward(_, *, forward_batch, pp_proxy_tensors, vpp_stage_id):
            seq = forward_batch.seq
            calls.append((seq, vpp_stage_id, id(forward_batch)))
            info = forward_batch.sampling_info
            self.assertEqual(info.top_ks.tolist(), [1 if seq == 0 else 7])
            self.assertEqual(info.is_all_greedy, seq == 0)
            self.assertAlmostEqual(info.temperatures.item(), 1.0 if seq == 0 else 0.7)
            if seq == 0:
                self.assertIsNone(info.sampling_seed)
            else:
                self.assertEqual(info.sampling_seed.tolist(), [123 + seq])
            self.assertEqual(info.logit_bias[0, seq].item(), 0.5)
            workspace.fill_(seq + vpp_stage_id)
            return SimpleNamespace(
                pp_hidden_states_proxy_tensors=PPProxyTensors(
                    {
                        "hidden_states": workspace.clone(),
                        "vpp_index_positions": workspace,
                    }
                )
            )

        def send(tensors, *, async_send, tag, all_gather_group, batch_p2p):
            self.assertTrue(async_send)
            self.assertEqual(tag, 1)
            self.assertTrue(batch_p2p)
            sent.append(tensors)
            return []

        group = SimpleNamespace(send_tensor_dict=send)
        scheduler = object.__new__(SchedulerVPPMixin)
        scheduler.__dict__.update(
            _vpp_layout=VPPLayout(8, 2),
            _vpp_forward_batches={},
            running_mbs=[None, None],
            last_mbs=[None, None],
            mbs=batches,
            mb_metadata=[None, None],
            _pp_vpp_stage_events={},
            pp_group=SimpleNamespace(rank_in_group=0, send_tensor_dict=send),
            _pp_vpp_runtime_epoch=3,
            _pp_vpp_control_layout_digest="layout",
            forward_stream_ctx=nullcontext(),
            pp_comm_stream_ctx=nullcontext(),
            forward_stream=stream,
            schedule_stream=stream,
            future_map=None,
            attn_tp_group=SimpleNamespace(rank_in_group=0, world_size=4),
            device_module=SimpleNamespace(Event=Event, current_stream=lambda: stream),
            tp_worker=SimpleNamespace(
                model_runner=object(),
                set_hicache_consumer=lambda index: None,
                forward_batch_generation=forward,
            ),
            _on_step_launch=lambda batch: None,
            resolve_forward_inputs=lambda batch, future_map: None,
            _on_step_forward_recorded=lambda event: None,
            sampler_params_pool=params,
            penalty_status_pool=penalties,
        )
        scheduler.pp_group.vpp_reverse_group = group
        # Replace the model-input boundary; sampling, dispatch and tensor
        # lifetime logic all execute their production implementations.
        with (
            get_context().override_server_args(enable_watermark=False),
            patch(
                "sglang.srt.managers.scheduler_vpp_mixin.ForwardBatch.init_new",
                side_effect=lambda batch, runner, **kw: SimpleNamespace(
                    seq=batch.seq, sampling_info=batch.sampling_info
                ),
            ) as build,
        ):
            for seq, stage in ((0, 0), (1, 0), (0, 2), (1, 2)):
                proxy = None
                if stage:
                    proxy = PPProxyTensors(
                        {
                            "hidden_states": torch.ones(1),
                            "vpp_batch_seq": seq,
                            "vpp_stage_id": stage,
                        }
                    )
                action = PipelineWavefrontAction(0, seq, seq, stage, 0)
                self.assertEqual(
                    scheduler._pp_vpp_execute_wavefront_action(action, [0, 1], proxy),
                    (True, []),
                )
                if stage == 0:
                    self.assertFalse(scheduler.mb_metadata[seq].forward_done.recorded)
                    params.table[SamplerParamsPool.TOP_K, seq + 2] = 99
            self.assertEqual(build.call_count, 2)
        self.assertTrue(all(m.forward_done.recorded for m in scheduler.mb_metadata))
        self.assertEqual([t["hidden_states"].item() for t in sent], [0, 1, 2, 3])
        self.assertEqual([t["vpp_index_positions"].item() for t in sent], [0, 1, 2, 3])
        self.assertEqual(calls[0][2], calls[2][2])
        self.assertEqual(calls[1][2], calls[3][2])
        self.assertNotEqual(calls[0][2], calls[1][2])

    def test_retained_chunk_only_blocks_its_own_continuation(self):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = object.__new__(Scheduler)
        first = SimpleNamespace(rid="first")
        second = SimpleNamespace(rid="second")
        scheduler.chunked_reqs = [first]
        scheduler.mbs = [SimpleNamespace(reqs=[first]), None]
        schedule = PipelineRankSchedule(0, 2, 2, 2)
        self.assertTrue(
            scheduler._pp_vpp_can_queue_admit(1, set(), [0, None], schedule)
        )
        with scheduler._pp_vpp_select_chunk():
            self.assertIsNone(scheduler.chunked_req)
            scheduler.chunked_reqs.append(second)
            self.assertIs(scheduler.chunked_req, second)
        self.assertIs(scheduler.chunked_req, first)
        scheduler.mbs[1] = SimpleNamespace(reqs=[second])
        with scheduler._pp_vpp_select_chunk():
            self.assertIsNone(scheduler.chunked_req)
        with scheduler._pp_vpp_select_chunk({"second"}, include_retained=True):
            self.assertIs(scheduler.chunked_req, second)
        scheduler.mbs[0] = None
        with scheduler._pp_vpp_select_chunk():
            self.assertIs(scheduler.chunked_req, first)
        with scheduler._pp_vpp_select_chunk({"second"}):
            self.assertIsNone(scheduler.chunked_req)
        scheduler.mbs[1] = None
        with scheduler._pp_vpp_select_chunk({"second"}):
            self.assertIs(scheduler.chunked_req, second)
        self.assertEqual(scheduler.chunked_reqs, [first, second])

    def test_chunk_selection_restores_after_planning_failure(self):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = object.__new__(Scheduler)
        req = SimpleNamespace(rid="request")
        scheduler.chunked_reqs = [req]
        scheduler.mbs = [SimpleNamespace(reqs=[req]), None]
        with self.assertRaisesRegex(RuntimeError, "planning failed"):
            with scheduler._pp_vpp_select_chunk():
                self.assertIsNone(scheduler.chunked_req)
                raise RuntimeError("planning failed")
        self.assertIs(scheduler.chunked_req, req)

    def test_prefix_publication_preserves_span_until_native_result_processing(self):
        """Publishing remote completion cannot erase the span just executed."""
        scheduler = object.__new__(SchedulerVPPMixin)
        req = SimpleNamespace(
            prefix_len=4,
            extend_end=8,
            session_generation=0,
            kv=SimpleNamespace(holds_kv=True, kv_allocated_len=8),
        )
        scheduler._pp_vpp_find_req = lambda rid: req
        scheduler._pp_vpp_advance_prefix_mapping("request", 0, 8)
        self.assertEqual(req.extend_end - req.prefix_len, 4)
        with self.assertRaisesRegex(RuntimeError, "allocated request row"):
            scheduler._pp_vpp_advance_prefix_mapping("request", 0, 9)

    def test_scheduler_mro_resolves_native_adapter_hooks(self):
        """An earlier PP mixin must not shadow the VPP runtime entry."""
        from sglang.srt.managers.scheduler import Scheduler

        self.assertIs(Scheduler.event_loop_vpp, SchedulerVPPMixin.event_loop_vpp)
        self.assertIs(
            Scheduler._init_scheduling_native_loop_state,
            SchedulerVPPMixin._init_scheduling_native_loop_state,
        )


if __name__ == "__main__":
    unittest.main()
