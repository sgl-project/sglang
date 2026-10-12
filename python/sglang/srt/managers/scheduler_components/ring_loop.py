"""The pipeline-parallel ring loop and the slot stages of each mode that runs it.

Each ring step walks the microbatch slots in order, and each slot runs its
mode's stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import torch

from sglang.srt.runtime_context import get_disagg, get_parallel

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


def _pp_exchange_outputs_before_forward(
    cur_batch: Optional[ScheduleBatch],
    spec_relay: bool,
    is_last_rank: bool,
    async_batch_depth: int,
) -> bool:
    """Extend microbatches launch first: they need nothing from the relay, and
    exchanging first caps every stage at (pp_size - 1) / pp_size. A verify round
    must exchange first or the ring deadlocks on its tree rebuild."""
    if async_batch_depth > 0:
        return True
    if not spec_relay or is_last_rank or cur_batch is None:
        return False
    return not (cur_batch.forward_mode.is_extend() or cur_batch.is_extend_in_batch)


@dataclass
class _RingSlot:
    """The values one slot's stages hand each other within a ring step."""

    mb_id: int
    next_mb_id: int
    next_first_rank_mb_id: int
    recv_reqs: Optional[List] = None
    cur_batch: Optional[ScheduleBatch] = None
    pp_proxy_tensors: Optional[PPProxyTensors] = None
    result: Optional[GenerationBatchResult] = None
    exchange_before: bool = False
    next_pp_outputs: Optional[PPProxyTensors] = None
    next_batch_result: Optional[GenerationBatchResult] = None
    d2h_event: Optional[torch.Event] = None
    next_batch: Optional[ScheduleBatch] = None
    # PD: the request ids this slot hands to the PP consensus.
    transferred_rids: Optional[List[str]] = None
    retract_rids: Optional[List[str]] = None
    prealloc_rids: Optional[List[str]] = None
    # PD: the consensus ids the next slot's ring step receives.
    next_release_rids: Optional[List[str]] = None
    next_consensus_retract_rids: Optional[List[str]] = None
    next_consensus_prealloc_rids: Optional[List[str]] = None


class _RingChannels:
    """The PD consensus state carried from slot to slot, set up on loop entry."""

    def __init__(self, loop_size: int):
        # Per slot: the ids it sent to the consensus, or None.
        self.tmbs = [None] * loop_size
        self.rmbs = [None] * loop_size
        self.pmbs = [None] * loop_size
        self.release_rids: Optional[List[str]] = None  # consensus transferred rids
        self.consensus_retract_rids: Optional[List[str]] = None
        self.consensus_prealloc_rids: Optional[List[str]] = None
        self.send_transfer_work = []
        self.send_retract_work = []
        self.send_prealloc_work = []
        self.send_consensus_retract_work = []
        self.send_consensus_prealloc_work = []
        self.send_release_work = []


RingStage = Callable[["Scheduler", _RingChannels, _RingSlot], None]


@dataclass(frozen=True)
class RingSpec:
    # Writes the mode's flags and installs the ring.
    enter: Callable[[Scheduler], None]
    # What each slot runs, in order. Every rank runs the same order, so the
    # sends and receives on each relay pair up first in, first out.
    slot_stages: Tuple[RingStage, ...]
    # Whether an idle round may run on_idle.
    may_idle: Callable[[Scheduler], bool]


def run_ring_loop(self: Scheduler, spec: RingSpec) -> None:
    spec.enter(self)
    ring = _RingChannels(self.pp_loop_size)
    while True:
        server_is_idle = True
        for mb_id in range(self.pp_loop_size):
            slot = _bind_slot(self, mb_id)
            for stage in spec.slot_stages:
                stage(self, ring, slot)
            if slot.cur_batch:
                server_is_idle = False

        # When the server is idle, self-check and re-init some states
        if server_is_idle:
            self._sched_idled = True
            if spec.may_idle(self):
                self.on_idle()


def _bind_slot(self: Scheduler, mb_id: int) -> _RingSlot:
    self.decoding_batch = self.decoding_batches[mb_id]
    self.last_batch = self._pp_slot_last_batch(mb_id)
    next_first_rank_mb_id = (mb_id + get_parallel().pp_size) % self.pp_loop_size
    next_mb_id = (mb_id + 1) % self.pp_loop_size
    return _RingSlot(mb_id, next_mb_id, next_first_rank_mb_id)


# Entry and idle gates.


def _enter_pp(self: Scheduler) -> None:
    self.init_pp_loop_state()
    self.rows_join_before_plan = True


def _enter_pp_disagg(self: Scheduler) -> None:
    self.init_pp_loop_state()


def _always_idle(self: Scheduler) -> bool:
    return True


def _no_prefill_inflight(self: Scheduler) -> bool:
    return len(self.disagg_prefill_inflight_queue) == 0


def _no_decode_queued(self: Scheduler) -> bool:
    queue_size = (
        len(self.waiting_queue)
        + len(self.disagg_decode_transfer_queue.queue)
        + len(self.disagg_decode_prealloc_queue.queue)
    )
    if get_disagg().disaggregation_decode_enable_offload_kvcache:
        queue_size += len(self.decode_offload_manager.ongoing_offload)
    return queue_size == 0


# Intake and the request relay.


def _ingest_labeled(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    with torch.profiler.record_function("recv_requests"):
        slot.recv_reqs = self.ingest_requests()


def _ingest(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    slot.recv_reqs = self.ingest_requests()


def _commit_req_send(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    if not self.pp_group.is_last_rank:
        self._pp_commit_comm_work(self.send_req_work)


def _send_reqs_labeled(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    if not self.pp_group.is_last_rank:
        with torch.profiler.record_function("send_reqs_to_next_stage"):
            self.send_req_work = self._pp_send_pyobj_to_next_stage(
                slot.recv_reqs,
                async_send=True,
            )


def _send_reqs(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    if not self.pp_group.is_last_rank:
        self.send_req_work = self._pp_send_pyobj_to_next_stage(
            slot.recv_reqs, async_send=True
        )


# PD consensus ids a slot takes before its plan.


def _process_bootstrapped(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    self.process_bootstrapped_queue()


def _take_prefill_transferred_ids(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    slot.transferred_rids = self._pp_pd_get_prefill_transferred_ids()
    self._pp_commit_comm_work(ring.send_transfer_work)
    ring.tmbs[slot.mb_id] = slot.transferred_rids


def _take_retract_ids(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    # reaching consensus through PP ranks
    slot.retract_rids = self._pp_pd_get_retract_ids(slot.mb_id)
    ring.rmbs[slot.mb_id] = slot.retract_rids
    self._pp_commit_comm_work(ring.send_retract_work)


def _take_prealloc_ids(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    slot.prealloc_rids = self._pp_pd_get_prealloc_ids()
    ring.pmbs[slot.mb_id] = slot.prealloc_rids
    self._pp_commit_comm_work(ring.send_prealloc_work)


def _take_decode_transferred_ids(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    slot.transferred_rids = self._pp_pd_get_decode_transferred_ids()
    ring.tmbs[slot.mb_id] = slot.transferred_rids
    self._pp_commit_comm_work(ring.send_transfer_work)


# Plans: each mode's schedule, then the slot's ring record.


def _plan(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    with torch.profiler.record_function("get_next_batch_to_run"):
        plan = self.get_next_batch_to_run(
            decoding_batch=self.decoding_batch, last_batch=self.last_batch
        )
        self._pp_debug_check_slot(slot.mb_id)
        self.inflight_steps.push(plan.batch_to_run, None)
    slot.cur_batch = plan.batch_to_run
    self.cur_batch_for_debug = slot.cur_batch


def _plan_pd_prefill(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    self.process_prefill_chunk(last_batch=self.last_batch)
    self._process_hicache_events()
    prefill_plan = self.get_new_batch_prefill(self.decoding_batch)
    batch = prefill_plan.batch_to_run
    batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)
    self._pp_debug_check_slot(slot.mb_id)
    self.inflight_steps.push(batch, None)

    slot.cur_batch = batch
    self.cur_batch_for_debug = slot.cur_batch


def _plan_pd_decode(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    # get batch to run and proxy tensors if needed
    plan = self.get_next_disagg_decode_batch_to_run(decoding_batch=self.decoding_batch)
    batch = plan.batch_to_run
    self._pp_debug_check_slot(slot.mb_id)
    self.inflight_steps.push(batch, None)

    slot.cur_batch = batch
    self.cur_batch_for_debug = slot.cur_batch


# Launch and the output relay.


def _recv_proxy(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    if slot.cur_batch:
        slot.pp_proxy_tensors = self._pp_recv_proxy_tensors()


def _exchange_outputs(self: Scheduler, slot: _RingSlot) -> None:
    slot.next_pp_outputs, slot.next_batch_result, slot.d2h_event = (
        self._pp_commit_send_output_work_and_preprocess_output_tensors(
            slot.next_first_rank_mb_id,
            slot.next_mb_id,
        )
    )


def _exchange_outputs_early(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    slot.exchange_before = _pp_exchange_outputs_before_forward(
        cur_batch=slot.cur_batch,
        spec_relay=self._pp_spec_relay,
        is_last_rank=self.pp_group.is_last_rank,
        async_batch_depth=get_parallel().pp_async_batch_depth,
    )
    if slot.exchange_before:
        _exchange_outputs(self, slot)


def _exchange_outputs_early_if_async(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    # early send output if possible
    slot.exchange_before = get_parallel().pp_async_batch_depth > 0
    if slot.exchange_before:
        _exchange_outputs(self, slot)


def _exchange_outputs_late(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    if not slot.exchange_before:
        _exchange_outputs(self, slot)


def _commit_proxy_send(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    self._pp_commit_proxy_send_work()


def _prefetch_staging(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    if slot.cur_batch:
        if self.enable_staging:
            self.maybe_prefetch_staging_for_batch(slot.cur_batch)


def _launch(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    if slot.cur_batch:
        slot.result, self.launch_event = self._pp_launch_batch(
            slot.mb_id,
            slot.cur_batch,
            slot.pp_proxy_tensors,
            self.mb_metadata,
            self.last_rank_comm_queue,
        )


def _send_proxy(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    if not self.pp_group.is_last_rank:
        if slot.cur_batch:
            self._pp_send_proxy_to_next_stage(slot.result)


# PD consensus sends and receives after the launch.


def _send_retract_consensus(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    # reach consensus on last rank and send to PP=0
    # otherwise, just pass along previous consensus
    ring.send_consensus_retract_work, ring.consensus_retract_rids = (
        self._pp_pd_send_consensus_bootstrapped_ids(
            ring.rmbs,
            slot.next_first_rank_mb_id,
            ring.consensus_retract_rids,
            slot.retract_rids,
        )
    )


def _send_prealloc_consensus(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    ring.send_consensus_prealloc_work, ring.consensus_prealloc_rids = (
        self._pp_pd_send_consensus_bootstrapped_ids(
            ring.pmbs,
            slot.next_first_rank_mb_id,
            ring.consensus_prealloc_rids,
            slot.prealloc_rids,
        )
    )


def _send_release_consensus(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    ring.send_release_work, ring.release_rids = self._pp_pd_send_consensus_release_ids(
        ring.tmbs, slot.next_first_rank_mb_id, ring.release_rids, slot.transferred_rids
    )


def _check_offload_progress(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    if get_disagg().disaggregation_decode_enable_offload_kvcache:
        self.decode_offload_manager.check_offload_progress()


def _recv_release(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    if ring.tmbs[slot.next_mb_id] is not None:
        slot.next_release_rids = self._pp_recv_pyobj_from_prev_stage()
    self._pp_commit_comm_work(ring.send_release_work)


def _recv_retract(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    if ring.rmbs[slot.next_mb_id] is not None:
        slot.next_consensus_retract_rids = self._pp_recv_pyobj_from_prev_stage()
        slot.next_consensus_retract_rids = self.process_retract_queue(
            slot.next_consensus_retract_rids
        )
    self._pp_commit_comm_work(ring.send_consensus_retract_work)


def _recv_prealloc(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    if ring.pmbs[slot.next_mb_id] is not None:
        slot.next_consensus_prealloc_rids = self._pp_recv_pyobj_from_prev_stage()
        slot.next_consensus_prealloc_rids = self.process_prealloc_queue(
            slot.next_consensus_prealloc_rids
        )
    self._pp_commit_comm_work(ring.send_consensus_prealloc_work)


def _recv_decode_transfer(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    if ring.tmbs[slot.next_mb_id] is not None:
        slot.next_release_rids = self._pp_recv_pyobj_from_prev_stage()
        slot.next_release_rids = self.process_decode_transfer_queue(
            slot.next_release_rids
        )
    self._pp_commit_comm_work(ring.send_release_work)


# Commit: the next slot's step, launched L - 1 ring steps ago.


def _commit_due_slot_spec_relay(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    # The commit of the non-PD ring: it may process against the step's
    # forward-time snapshot, and labels the processing for the profiler.
    slot.next_batch = self._pp_pop_due_batch()
    if slot.next_batch is not None:
        slot.d2h_event.synchronize()
        process_target = slot.next_batch
        next_md = self.mb_metadata[slot.next_mb_id]
        if next_md is not None and next_md.fwd_batch is not None:
            # PP+spec: process against the forward-time snapshot;
            # the live mb may have been recomposed since launch.
            process_target = next_md.fwd_batch
        with torch.profiler.record_function("process_batch_result"):
            self._pp_process_batch_result(
                process_target,
                slot.next_batch_result,
            )


def _commit_due_slot(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    # post-process the coming microbatch
    slot.next_batch = self._pp_pop_due_batch()
    if slot.next_batch is not None:
        slot.d2h_event.synchronize()
        self._pp_process_batch_result(
            slot.next_batch,
            slot.next_batch_result,
        )


def _update_rows_at_commit(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    if slot.next_batch is not None:
        self._pp_update_rows_at_commit(slot.next_mb_id, slot.next_batch)


def _release_transfers(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    if ring.tmbs[slot.next_mb_id] is not None:
        self.process_disagg_prefill_inflight_queue(slot.next_release_rids)


# PD id relays after the commit, in the order the next rank receives them.


def _send_retract_ids(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    if not self.pp_group.is_last_rank:
        ring.send_retract_work = self._pp_send_pyobj_to_next_stage(
            slot.retract_rids, async_send=True
        )


def _send_prealloc_ids(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    if not self.pp_group.is_last_rank:
        ring.send_prealloc_work = self._pp_send_pyobj_to_next_stage(
            slot.prealloc_rids, async_send=True
        )


def _send_transferred_ids(
    self: Scheduler, ring: _RingChannels, slot: _RingSlot
) -> None:
    if not self.pp_group.is_last_rank:
        ring.send_transfer_work = self._pp_send_pyobj_to_next_stage(
            slot.transferred_rids, async_send=True
        )


def _end_slot(self: Scheduler, ring: _RingChannels, slot: _RingSlot) -> None:
    self.pp_outputs = slot.next_pp_outputs
    self.inflight_steps.end_step(slot.cur_batch)
    ring.release_rids = slot.next_release_rids
    ring.consensus_retract_rids = slot.next_consensus_retract_rids
    ring.consensus_prealloc_rids = slot.next_consensus_prealloc_rids


PP_RING = RingSpec(
    enter=_enter_pp,
    slot_stages=(
        _ingest_labeled,
        _commit_req_send,
        _send_reqs_labeled,
        _plan,
        _recv_proxy,
        _exchange_outputs_early,
        _commit_proxy_send,
        _launch,
        _exchange_outputs_late,
        _commit_due_slot_spec_relay,
        _update_rows_at_commit,
        _send_proxy,
        _end_slot,
    ),
    may_idle=_always_idle,
)

PP_DISAGG_PREFILL_RING = RingSpec(
    enter=_enter_pp_disagg,
    slot_stages=(
        _ingest,
        _commit_req_send,
        _process_bootstrapped,
        _take_prefill_transferred_ids,
        _plan_pd_prefill,
        _recv_proxy,
        _exchange_outputs_early_if_async,
        _commit_proxy_send,
        _prefetch_staging,
        _launch,
        _exchange_outputs_late,
        _send_release_consensus,
        _recv_release,
        _commit_due_slot,
        _release_transfers,
        _send_reqs,
        _send_transferred_ids,
        _send_proxy,
        _end_slot,
    ),
    may_idle=_no_prefill_inflight,
)

PP_DISAGG_DECODE_RING = RingSpec(
    enter=_enter_pp_disagg,
    slot_stages=(
        _ingest,
        _commit_req_send,
        _take_retract_ids,
        _take_prealloc_ids,
        _take_decode_transferred_ids,
        _plan_pd_decode,
        _recv_proxy,
        _exchange_outputs_early_if_async,
        _commit_proxy_send,
        _launch,
        _exchange_outputs_late,
        _send_retract_consensus,
        _send_prealloc_consensus,
        _send_release_consensus,
        _check_offload_progress,
        _recv_retract,
        _recv_prealloc,
        _recv_decode_transfer,
        _commit_due_slot,
        _update_rows_at_commit,
        _send_reqs,
        _send_retract_ids,
        _send_prealloc_ids,
        _send_transferred_ids,
        _send_proxy,
        _end_slot,
    ),
    may_idle=_no_decode_queued,
)
