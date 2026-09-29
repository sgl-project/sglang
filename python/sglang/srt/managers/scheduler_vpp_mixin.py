from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.distributed

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.distributed.parallel_state import (
    P2PWork,
    P2PWorkGroup,
    TensorDictRecvHandle,
    get_vpp_pp_reverse_group,
)
from sglang.srt.distributed.pipeline_layout import (
    PipelineControlEnvelope,
    PipelineControlKind,
    PipelinePrefixRegistry,
    PipelineRankSchedule,
    PipelineReplicaIdentity,
    PipelineReplicaRegistry,
    PipelineResourceGate,
    PipelineResourceSnapshot,
    PipelineWavefrontAction,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req, ScheduleBatch
from sglang.srt.managers.scheduler_pp_mixin import PPBatchMetadata
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.runtime_context import get_parallel, max_prefill_buffer_tokens
from sglang.srt.utils import DynamicGradMode

logger = logging.getLogger(__name__)

_VPP_ACTIVATION_TAG = 1
_VPP_CONTROL_TAG = 2
_VPP_PROTOCOL_VERSION = 1

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class SchedulerVPPMixin:
    @DynamicGradMode()
    def event_loop_pp_disagg_prefill(self: Scheduler):
        if self._pp_vpp_enabled():
            return self._event_loop_pp_disagg_prefill_vpp()
        return super().event_loop_pp_disagg_prefill()

    def _pp_vpp_enabled(self: Scheduler) -> bool:
        return get_parallel().pp_virtual_stages > 1

    def _pp_vpp_max_inflight(self: Scheduler) -> int:
        parallel = get_parallel()
        configured = parallel.pp_vpp_max_inflight
        if configured is not None:
            return configured
        burst_size = parallel.pp_vpp_prefill_burst_size
        if burst_size == 1:
            return self.pp_group.world_size
        return burst_size + self.pp_group.world_size

    def _pp_vpp_can_queue_admit(
        self: Scheduler,
        batch_seq: int,
        pending_admits: set[int],
        slot_batch_seqs: List[Optional[int]],
        rank_schedule: PipelineRankSchedule,
    ) -> bool:
        slot_id = batch_seq % len(slot_batch_seqs)
        return (
            len(pending_admits) < len(slot_batch_seqs)
            and slot_batch_seqs[slot_id] is None
            and rank_schedule.can_admit(batch_seq)
        )

    def _pp_vpp_all_gather_cpu_state(
        self: Scheduler,
        values: Tuple[int, ...],
    ) -> torch.Tensor:
        width = len(values)
        buffers = getattr(self, "_pp_vpp_cpu_state_buffers", None)
        if buffers is None:
            buffers = self._pp_vpp_cpu_state_buffers = {}
        if width not in buffers:
            local = torch.empty(width, dtype=torch.int64, device="cpu")
            gathered = torch.empty(
                (self.attn_tp_group.world_size, width),
                dtype=torch.int64,
                device="cpu",
            )
            buffers[width] = (local, gathered)
        local, gathered = buffers[width]
        for index, value in enumerate(values):
            local[index] = int(value)
        if self.attn_tp_group.world_size == 1:
            gathered[0].copy_(local)
        else:
            torch.distributed.all_gather_into_tensor(
                gathered.flatten(),
                local,
                group=self.attn_tp_group.cpu_group,
            )
        return gathered

    def _pp_vpp_broadcast_cpu_state(
        self: Scheduler,
        values: Optional[Tuple[int, ...]],
        width: int,
        src: int = 0,
    ) -> torch.Tensor:
        buffers = getattr(self, "_pp_vpp_cpu_broadcast_buffers", None)
        if buffers is None:
            buffers = self._pp_vpp_cpu_broadcast_buffers = {}
        if width not in buffers:
            buffers[width] = torch.empty(width, dtype=torch.int64, device="cpu")
        state = buffers[width]
        if self.attn_tp_group.rank_in_group == src:
            if values is None or len(values) != width:
                raise RuntimeError("invalid VPP TP broadcast state")
            for index, value in enumerate(values):
                state[index] = int(value)
        if self.attn_tp_group.world_size > 1:
            torch.distributed.broadcast(
                state,
                src=self.attn_tp_group.ranks[src],
                group=self.attn_tp_group.cpu_group,
            )
        return state

    def _pp_prewarm_vpp_device_group(self: Scheduler) -> None:
        if not self._pp_vpp_enabled():
            return
        warmup_tensor = torch.zeros(
            1,
            dtype=torch.int32,
            device=self.pp_group.device,
        )
        torch.distributed.all_reduce(
            warmup_tensor,
            group=self.pp_group.device_group,
        )
        self.pp_group.device_module.synchronize()
        logger.info("VPP pipeline device group prewarm completed")

    def _pp_vpp_batch_manifest(
        self: Scheduler,
        batch_seq: int,
        slot_id: int,
        batch: ScheduleBatch,
    ) -> Dict[str, object]:
        requests = []
        for req in batch.reqs:
            extend_range = req.extend_range
            requests.append(
                (
                    req.rid,
                    -1 if extend_range is None else int(extend_range.start),
                    -1 if extend_range is None else int(extend_range.end),
                )
            )
        return {
            "protocol_version": _VPP_PROTOCOL_VERSION,
            "batch_seq": batch_seq,
            "generation": batch_seq // len(self.mbs),
            "slot_id": slot_id,
            "requests": tuple(requests),
        }

    def _pp_vpp_activation_group(self: Scheduler, source_rank: int):
        if self.pp_group.world_size == 2 and source_rank == 1:
            return get_vpp_pp_reverse_group()
        return self.pp_group

    def _pp_vpp_start_receiver(self: Scheduler) -> None:
        if self._pp_vpp_pending_recv is not None:
            return
        source_rank = (self.pp_group.rank_in_group - 1) % self.pp_group.world_size
        activation_group = self._pp_vpp_activation_group(source_rank)
        self._pp_vpp_pending_recv = activation_group.recv_tensor_dict_async(
            all_gather_group=self.attn_tp_group,
            batch_p2p=True,
            tag=_VPP_ACTIVATION_TAG,
        )
        self._pp_vpp_receiver_gather_started = False

    def _pp_vpp_accept_received_tensors(
        self: Scheduler,
        tensors: Dict[str, object],
    ) -> None:
        message_kind = tensors.get("__msg_type__", "default")
        protocol_version = int(tensors.get("vpp_protocol_version", -1))
        runtime_epoch = int(tensors.get("vpp_runtime_epoch", -1))
        layout_digest = str(tensors.get("vpp_layout_digest", ""))
        batch_seq = int(tensors.get("vpp_batch_seq", -1))
        generation = int(tensors.get("vpp_generation", -1))
        source_stage_id = int(tensors.get("vpp_src_stage_id", -1))
        stage_id = int(tensors.get("vpp_stage_id", -1))
        if message_kind != "vpp_proxy":
            raise RuntimeError(
                f"VPP activation receiver got unexpected message kind {message_kind}"
            )
        if (
            protocol_version != _VPP_PROTOCOL_VERSION
            or runtime_epoch != self._pp_vpp_runtime_epoch
            or layout_digest != self._pp_vpp_control_layout_digest
            or batch_seq < 0
            or generation != batch_seq // len(self._pp_vpp_slot_batch_seqs)
            or stage_id <= 0
            or source_stage_id != stage_id - 1
        ):
            raise RuntimeError(
                "VPP activation receiver got invalid identity: "
                f"protocol={protocol_version}, epoch={runtime_epoch}, "
                f"layout={layout_digest}, batch={batch_seq}, "
                f"generation={generation}, "
                f"source_stage={source_stage_id}, stage={stage_id}"
            )
        if stage_id % self.pp_group.world_size != self.pp_group.rank_in_group:
            raise RuntimeError(
                f"VPP activation for stage {stage_id} arrived on PP rank "
                f"{self.pp_group.rank_in_group}"
            )

        slot_id = batch_seq % len(self._pp_vpp_slot_batch_seqs)
        slot_batch_seq = self._pp_vpp_slot_batch_seqs[slot_id]
        # The last rank retains slot ownership until COMPLETION returns, even
        # after local finalization clears mbs. PP0 can reuse its retired slot
        # and send the next generation before that control round trip finishes.
        retiring_previous_generation = (
            slot_batch_seq is not None
            and batch_seq == slot_batch_seq + len(self._pp_vpp_slot_batch_seqs)
            and self.pp_group.is_last_rank
            and stage_id == self.pp_group.rank_in_group
            and self.mbs[slot_id] is None
        )
        if slot_batch_seq not in (None, batch_seq) and not retiring_previous_generation:
            raise RuntimeError(
                f"Stale VPP activation for batch {batch_seq} in slot {slot_id}"
            )
        key = (batch_seq, stage_id)
        if key in self._pp_vpp_ready_proxies or key in self._pp_vpp_early_activations:
            raise RuntimeError(f"Duplicate VPP activation for batch/stage {key}")
        proxy = PPProxyTensors(tensors)
        if slot_batch_seq is None or retiring_previous_generation:
            if len(self._pp_vpp_early_activations) >= len(self._pp_vpp_slot_batch_seqs):
                raise RuntimeError("VPP early-activation quarantine is full")
            self._pp_vpp_early_activations[key] = proxy
        else:
            self._pp_vpp_ready_proxies[key] = proxy
            if hasattr(self, "_pp_vpp_arrivals"):
                self._pp_vpp_arrivals.append(key)
        self._pp_vpp_pending_recv = None
        self._pp_vpp_start_receiver()

    def _pp_vpp_promote_early_activations(self: Scheduler, batch_seq: int) -> None:
        slot_id = batch_seq % len(self._pp_vpp_slot_batch_seqs)
        if self._pp_vpp_slot_batch_seqs[slot_id] != batch_seq:
            raise RuntimeError(
                f"Cannot promote VPP activation before admitting batch {batch_seq}"
            )
        stale = [
            key
            for key in self._pp_vpp_early_activations
            if key[0] % len(self._pp_vpp_slot_batch_seqs) == slot_id
            and key[0] != batch_seq
        ]
        if stale:
            raise RuntimeError(f"Stale VPP early activation in slot {slot_id}: {stale}")
        keys = sorted(
            key for key in self._pp_vpp_early_activations if key[0] == batch_seq
        )
        for key in keys:
            if key in self._pp_vpp_ready_proxies:
                raise RuntimeError(f"Duplicate VPP activation for batch/stage {key}")
            self._pp_vpp_ready_proxies[key] = self._pp_vpp_early_activations.pop(key)
            self._pp_vpp_arrivals.append(key)

    def _pp_vpp_gather_tick_consensus(
        self: Scheduler,
        rank_schedule: PipelineRankSchedule,
        is_ready,
    ) -> Tuple[bool, torch.Tensor]:
        handle: Optional[TensorDictRecvHandle] = self._pp_vpp_pending_recv
        if handle is None:
            self._pp_vpp_start_receiver()
            handle = self._pp_vpp_pending_recv
        gather_started = getattr(self, "_pp_vpp_receiver_gather_started", False)
        received = gather_started and handle.poll_all_gather()
        if received:
            self._pp_vpp_accept_received_tensors(handle.result())
            handle = self._pp_vpp_pending_recv
            gather_started = False
        payload_ready = handle.poll_payload_ready()

        local_ready = tuple(
            task for task in rank_schedule.ready_tasks if is_ready(task[0], task[1])
        )
        if len(local_ready) > rank_schedule.max_inflight:
            raise RuntimeError("VPP ready-task count exceeds the inflight window")
        encoded_ready = [len(local_ready)]
        for batch_seq, stage_id in local_ready:
            encoded_ready.extend((batch_seq, stage_id))
        encoded_ready.extend(
            [-1] * (2 * rank_schedule.max_inflight - 2 * len(local_ready))
        )
        states = self._pp_vpp_all_gather_cpu_state((payload_ready, *encoded_ready))

        if not gather_started and bool(states[:, 0].all()):
            handle.start_all_gather()
            self._pp_vpp_receiver_gather_started = True
        return received, states[:, 1:]

    def _pp_vpp_select_tp_action(
        self: Scheduler,
        rank_schedule: PipelineRankSchedule,
        tick: int,
        is_ready,
        ready_states: Optional[torch.Tensor] = None,
    ) -> Optional[PipelineWavefrontAction]:
        if ready_states is None:
            local_ready = tuple(
                task for task in rank_schedule.ready_tasks if is_ready(task[0], task[1])
            )
            if len(local_ready) > rank_schedule.max_inflight:
                raise RuntimeError("VPP ready-task count exceeds the inflight window")
            encoded_ready = [len(local_ready)]
            for batch_seq, stage_id in local_ready:
                encoded_ready.extend((batch_seq, stage_id))
            encoded_ready.extend(
                [-1] * (2 * rank_schedule.max_inflight - 2 * len(local_ready))
            )
            ready_states = self._pp_vpp_all_gather_cpu_state(tuple(encoded_ready))
        ready_by_lane = []
        for state in ready_states:
            count = int(state[0])
            if not 0 <= count <= rank_schedule.max_inflight:
                raise RuntimeError(f"invalid VPP TP ready-task count {count}")
            encoded_tasks = state[1 : 1 + 2 * count].tolist()
            ready_by_lane.append(set(zip(encoded_tasks[::2], encoded_tasks[1::2])))
        common_ready = set.intersection(*ready_by_lane) if ready_by_lane else set()
        action = (
            rank_schedule.next_action(
                tick,
                is_ready=lambda batch_seq, stage_id: (
                    (
                        batch_seq,
                        stage_id,
                    )
                    in common_ready
                ),
            )
            if self.attn_tp_group.rank_in_group == 0
            else None
        )
        identity = (
            (-1, -1, -1)
            if action is None
            else (action.tick, action.batch_seq, action.stage_id)
        )
        identity_state = self._pp_vpp_broadcast_cpu_state(
            identity if self.attn_tp_group.rank_in_group == 0 else None,
            width=3,
            src=0,
        )
        if int(identity_state[0]) < 0:
            return None
        action_tick, batch_seq, stage_id = map(int, identity_state)
        if self.attn_tp_group.rank_in_group == 0:
            return action
        if not is_ready(batch_seq, stage_id):
            raise RuntimeError(
                f"TP leader selected unavailable VPP task {(batch_seq, stage_id)}"
            )
        return rank_schedule.dispatch(action_tick, batch_seq, stage_id)

    def _pp_vpp_take_ready_proxy(
        self: Scheduler,
        action: PipelineWavefrontAction,
    ) -> PPProxyTensors:
        key = (action.batch_seq, action.stage_id)
        try:
            return self._pp_vpp_ready_proxies.pop(key)
        except KeyError:
            raise RuntimeError(
                f"VPP activation for batch/stage {key} is not ready"
            ) from None

    def _pp_vpp_reap_send_work(self: Scheduler, pending_work: deque) -> None:
        while pending_work:
            work_group = pending_work[0]
            if isinstance(work_group, list):
                work_group = P2PWorkGroup(work_group)
                pending_work[0] = work_group
            if not work_group.poll():
                return
            pending_work.popleft()

    def _pp_vpp_layout_digest(self: Scheduler) -> str:
        # The control plane must describe the partition actually loaded by the
        # model, including nonuniform logical stages.
        return self.tp_worker.model_runner.model.model.pipeline_layout.digest

    def _pp_vpp_start_control_receiver(self: Scheduler) -> None:
        if self.attn_tp_group.rank_in_group != 0:
            return
        if self._pp_vpp_pending_control_recv is not None:
            return
        self._pp_vpp_pending_control_recv = self.pp_group.recv_tensor_dict_async(
            batch_p2p=True,
            tag=_VPP_CONTROL_TAG,
        )

    def _pp_vpp_poll_control_receiver(
        self: Scheduler,
    ) -> Optional[Tuple[PipelineControlEnvelope, Dict[str, object]]]:
        if self.attn_tp_group.rank_in_group != 0:
            raise RuntimeError("only TP0 may poll the VPP control ring")
        handle: Optional[TensorDictRecvHandle] = self._pp_vpp_pending_control_recv
        if handle is None:
            self._pp_vpp_start_control_receiver()
            handle = self._pp_vpp_pending_control_recv
        wire = handle.poll()
        if wire is None:
            return None
        envelope = PipelineControlEnvelope.from_dict(wire)
        if envelope.protocol_version != _VPP_PROTOCOL_VERSION:
            raise RuntimeError("VPP control protocol version mismatch")
        if envelope.runtime_epoch != self._pp_vpp_runtime_epoch:
            raise RuntimeError("stale VPP control runtime epoch")
        if envelope.layout_digest != self._pp_vpp_control_layout_digest:
            raise RuntimeError("VPP control layout digest mismatch")
        self._pp_vpp_pending_control_recv = None
        self._pp_vpp_start_control_receiver()
        return envelope, wire

    def _pp_vpp_poll_control_receiver_tp_broadcast(
        self: Scheduler,
    ) -> Optional[Tuple[PipelineControlEnvelope, Dict[str, object]]]:
        wire = None
        if self.attn_tp_group.rank_in_group == 0:
            message = self._pp_vpp_poll_control_receiver()
            if message is not None:
                wire = message[1]
        wire = self.attn_tp_group.broadcast_object(wire, src=0)
        if wire is None:
            return None
        envelope = PipelineControlEnvelope.from_dict(wire)
        if envelope.protocol_version != _VPP_PROTOCOL_VERSION:
            raise RuntimeError("VPP control protocol version mismatch")
        if envelope.runtime_epoch != self._pp_vpp_runtime_epoch:
            raise RuntimeError("stale VPP control runtime epoch")
        if envelope.layout_digest != self._pp_vpp_control_layout_digest:
            raise RuntimeError("VPP control layout digest mismatch")
        return envelope, wire

    def _pp_vpp_queue_control(
        self: Scheduler,
        envelope: PipelineControlEnvelope,
        tensors: Optional[Dict[str, object]] = None,
    ) -> None:
        if self.attn_tp_group.rank_in_group != 0:
            return
        wire = envelope.to_dict()
        if tensors:
            cuda_keys = [
                key
                for key, value in tensors.items()
                if isinstance(value, torch.Tensor) and not value.is_cpu
            ]
            if cuda_keys:
                raise RuntimeError(f"VPP control payload must be CPU-only: {cuda_keys}")
            wire.update(tensors)
        self._pp_vpp_control_outbox.append(wire)

    def _pp_vpp_flush_control_outbox(
        self: Scheduler,
        pending_work: deque,
        max_pending: int,
    ) -> None:
        if self.attn_tp_group.rank_in_group != 0:
            return
        self._pp_vpp_reap_send_work(pending_work)
        while self._pp_vpp_control_outbox and len(pending_work) < max_pending:
            wire = self._pp_vpp_control_outbox.popleft()
            work = self.pp_group.send_tensor_dict(
                wire,
                async_send=True,
                batch_p2p=True,
                tag=_VPP_CONTROL_TAG,
            )
            if work:
                pending_work.append(work)

    def _pp_vpp_new_control(
        self: Scheduler,
        kind: PipelineControlKind,
        *,
        batch_seq: int = -1,
        slot_id: int = -1,
        payload: Optional[Dict[str, object]] = None,
    ) -> PipelineControlEnvelope:
        generation = (
            -1 if batch_seq < 0 else batch_seq // len(self._pp_vpp_slot_batch_seqs)
        )
        return PipelineControlEnvelope(
            protocol_version=_VPP_PROTOCOL_VERSION,
            runtime_epoch=self._pp_vpp_runtime_epoch,
            layout_digest=self._pp_vpp_control_layout_digest,
            kind=kind,
            source_rank=self.pp_group.rank_in_group,
            batch_seq=batch_seq,
            generation=generation,
            slot_id=slot_id,
            payload=payload,
        )

    def _pp_vpp_resource_snapshot(
        self: Scheduler,
        pending_send_work,
    ) -> PipelineResourceSnapshot:
        allocator = self.token_to_kv_pool_allocator
        if hasattr(allocator, "full_available_size") and hasattr(
            allocator, "swa_available_size"
        ):
            kv_tokens = min(
                int(allocator.full_available_size())
                + int(self.tree_cache.full_evictable_size()),
                int(allocator.swa_available_size())
                + int(self.tree_cache.swa_evictable_size()),
            )
        else:
            kv_tokens = int(allocator.available_size()) + int(
                self.tree_cache.evictable_size()
            )
        activation_bytes = 0
        activation_proxies = [
            *self._pp_vpp_ready_proxies.values(),
            *self._pp_vpp_early_activations.values(),
        ]
        for proxy in activation_proxies:
            activation_bytes += sum(
                value.numel() * value.element_size()
                for value in proxy.tensors.values()
                if isinstance(value, torch.Tensor)
            )
        pending_recv = getattr(self, "_pp_vpp_pending_recv", None)
        if pending_recv is not None and hasattr(pending_recv, "buffered_tensor_bytes"):
            activation_bytes += pending_recv.buffered_tensor_bytes()
        if isinstance(pending_send_work, int):
            pending_send_count = pending_send_work
        else:
            pending_send_count = len(pending_send_work)
            for work_group in pending_send_work:
                if isinstance(work_group, P2PWorkGroup):
                    activation_bytes += work_group.payload_bytes()
                else:
                    activation_bytes += sum(
                        item.payload.numel() * item.payload.element_size()
                        for item in work_group
                        if isinstance(item.payload, torch.Tensor)
                    )
        return PipelineResourceSnapshot(
            request_slots=int(self.req_to_token_pool.available_size()),
            kv_tokens=kv_tokens,
            activation_bytes=activation_bytes,
            pending_sends=pending_send_count,
            metadata_slots=int(
                self.req_to_metadata_buffer_idx_allocator.available_size()
            ),
        )

    def _pp_vpp_find_req(self: Scheduler, rid: str) -> Optional[Req]:
        for batch in [*self.mbs, *self.last_mbs]:
            if batch is None:
                continue
            for req in batch.reqs:
                if req.rid == rid:
                    return req
        if self.chunked_req is not None and self.chunked_req.rid == rid:
            return self.chunked_req
        return None

    def _pp_vpp_bootstrap_prefix_boundaries(
        self: Scheduler,
        rids: List[str],
    ) -> Dict[str, int]:
        wanted = set(rids)
        return {
            req.rid: len(req.prefix_indices)
            for req in self.disagg_prefill_bootstrap_queue.queue
            if req.rid in wanted
        }

    def _pp_vpp_merge_bootstrap_status(
        self: Scheduler,
        payload: Dict[str, object],
        local_good: List[str],
        local_bad: List[str],
    ) -> None:
        payload["good"] = sorted(set(payload["good"]).intersection(local_good))
        payload["bad"] = sorted(set(payload["bad"]).union(local_bad))
        metadata_states = self._pp_vpp_all_gather_cpu_state(
            (self.req_to_metadata_buffer_idx_allocator.available_size(),)
        )
        local_metadata_slots = int(metadata_states[:, 0].min())
        payload["metadata_slots"] = min(
            int(payload["metadata_slots"]),
            local_metadata_slots,
        )
        local_boundaries = self._pp_vpp_bootstrap_prefix_boundaries(payload["good"])
        payload["prefix_boundaries"] = {
            rid: min(
                int(payload["prefix_boundaries"][rid]),
                local_boundaries[rid],
            )
            for rid in payload["good"]
        }

    def _pp_vpp_bootstrap_apply_status(
        self: Scheduler,
        payload: Dict[str, object],
    ) -> List[List[str]]:
        good = sorted(set(payload["good"]))
        metadata_slots = max(0, int(payload["metadata_slots"]))
        return [
            good[:metadata_slots],
            sorted(set(payload["bad"])),
        ]

    def _pp_vpp_apply_bootstrap_prefix_boundaries(
        self: Scheduler,
        boundaries: Dict[str, int],
        rids: set[str],
    ) -> None:
        for req in self.disagg_prefill_bootstrap_queue.queue:
            if req.rid not in rids:
                continue
            boundary = int(boundaries[req.rid])
            already_applied = (
                req.vpp_prefix_limit == boundary and len(req.prefix_indices) <= boundary
            )
            req.vpp_prefix_limit = boundary
            if not already_applied and len(req.prefix_indices) > boundary:
                req.init_next_round_input(self.tree_cache)
            if len(req.prefix_indices) > boundary:
                raise RuntimeError(
                    f"VPP prefix cap failed for {req.rid}: "
                    f"expected <= {boundary}, got {len(req.prefix_indices)}"
                )

    def _pp_vpp_try_apply_bootstrap_status(
        self: Scheduler,
        boundaries: Dict[str, int],
        remaining_good: set[str],
        remaining_bad: set[str],
    ) -> bool:
        self._pp_vpp_apply_bootstrap_prefix_boundaries(boundaries, remaining_good)
        applied_good, applied_bad = self.process_bootstrapped_queue(
            [sorted(remaining_good), sorted(remaining_bad)]
        )
        remaining_good.difference_update(applied_good)
        remaining_bad.difference_update(applied_bad)

        # Applying the status is idempotent. A target that disappeared from the
        # bootstrap queue was already moved to its next local state by an earlier
        # attempt and must not prevent the ring ACK from advancing.
        queued_rids = {req.rid for req in self.disagg_prefill_bootstrap_queue.queue}
        remaining_good.intersection_update(queued_rids)
        remaining_bad.intersection_update(queued_rids)
        return not remaining_good and not remaining_bad

    def _pp_vpp_register_prefix_batch(
        self: Scheduler,
        batch: ScheduleBatch,
    ) -> None:
        for req in batch.reqs:
            extend_range = req.extend_range
            if extend_range is None:
                continue
            request_generation = int(getattr(req, "session_generation", None) or 0)
            entry = self._pp_vpp_prefix_registry.get(req.rid, request_generation)
            start = extend_range.start if entry is None else entry.planned_end
            self._pp_vpp_prefix_registry.plan(
                rid=req.rid,
                request_generation=request_generation,
                residency_generation=request_generation,
                start=start,
                end=extend_range.end,
                required_stages=range(
                    self.pp_group.world_size * get_parallel().pp_virtual_stages
                ),
            )
            if not hasattr(req, "_vpp_original_skip_radix_cache_insert"):
                req._vpp_original_skip_radix_cache_insert = req.skip_radix_cache_insert
            req.skip_radix_cache_insert = True

    def _pp_vpp_apply_prefix_materialized(
        self: Scheduler,
        envelope: PipelineControlEnvelope,
    ) -> None:
        payload = envelope.payload or {}
        rid = str(payload["rid"])
        request_generation = int(payload["request_generation"])
        entry = self._pp_vpp_prefix_registry.get(rid, request_generation)
        if entry is None:
            return
        end = int(payload["end"])
        materialized_end = self._pp_vpp_prefix_registry.mark_materialized(
            rid,
            request_generation,
            int(payload["stage_id"]),
            end,
        )
        if materialized_end >= end and entry.committed_end < end:
            self._pp_vpp_prefix_registry.commit(
                rid,
                request_generation,
                end,
            )

    def _pp_vpp_advance_prefix_mapping(
        self: Scheduler,
        rid: str,
        request_generation: int,
        end: int,
    ) -> None:
        req = self._pp_vpp_find_req(rid)
        if req is None or not req.kv.holds_kv:
            return
        if end <= len(req.prefix_indices):
            return
        req.prefix_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :end
        ].to(dtype=torch.int64, copy=True)

    def _pp_vpp_prepare_wavefront_batch(
        self: Scheduler,
        batch_seq: int,
        slot_batch_seqs: List[Optional[int]],
    ) -> Optional[ScheduleBatch]:
        slot_id = batch_seq % len(self.mbs)
        if slot_batch_seqs[slot_id] is not None:
            raise RuntimeError(
                f"VPP wavefront slot {slot_id} is still owned by batch "
                f"{slot_batch_seqs[slot_id]}"
            )

        self.running_batch = self.running_mbs[slot_id]
        self.last_batch = self.last_mbs[slot_id]
        if getattr(self, "_pp_vpp_overlap_chunks", False):
            req = self.chunked_req
            if req is not None:
                generation = int(getattr(req, "session_generation", None) or 0)
                entry = self._pp_vpp_prefix_registry.get(req.rid, generation)
                if entry is not None:
                    self._pp_vpp_advance_prefix_mapping(
                        req.rid, generation, entry.planned_end
                    )
        self.process_prefill_chunk(
            last_batch=self.last_batch,
            running_batch=self.running_batch,
        )
        prefill_plan = self.get_new_batch_prefill(self.running_batch)
        batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
            prefill_plan.batch_to_run
        )
        batch = self.ngram_embedding_manager.prepare_for_forward(
            batch,
            chunked_req=self.chunked_req,
        )
        if batch is not None:
            reqs = getattr(batch, "reqs", None)
            if reqs is None:
                reqs = (
                    []
                    if getattr(batch, "chunked_req", None) is None
                    else [batch.chunked_req]
                )
            batch.disagg_prefill_chunk_end_by_rid = {
                req.rid: min(req.extend_range.end, len(req.origin_input_ids))
                for req in reqs
                if req.extend_range is not None
            }
            for req in reqs:
                req.vpp_prefix_limit = None
        self.running_batch = prefill_plan.running_batch
        self.running_mbs[slot_id] = self.running_batch
        self.mbs[slot_id] = batch
        self.mb_metadata[slot_id] = None
        slot_batch_seqs[slot_id] = batch_seq
        return batch

    def _pp_vpp_execute_wavefront_action(
        self: Scheduler,
        action: Optional[PipelineWavefrontAction],
        slot_batch_seqs: List[Optional[int]],
    ) -> Tuple[bool, List[P2PWork]]:
        if action is None:
            return False, []
        if slot_batch_seqs[action.slot_id] != action.batch_seq:
            raise RuntimeError(
                f"VPP wavefront batch {action.batch_seq} does not own "
                f"slot {action.slot_id}"
            )
        batch = self.mbs[action.slot_id]
        if batch is None:
            raise RuntimeError(
                f"VPP action for batch {action.batch_seq} found an empty "
                f"slot {action.slot_id}"
            )

        self.running_batch = self.running_mbs[action.slot_id]
        self.last_batch = self.last_mbs[action.slot_id]
        self.cur_batch_for_debug = batch
        pp_proxy_tensors = None
        if action.stage_id > 0:
            pp_proxy_tensors = self._pp_recv_vpp_proxy_tensors(
                first_visit=action.stage_id < self.pp_group.world_size,
                expected_batch_seq=action.batch_seq,
                expected_stage_id=action.stage_id,
            )
        if self.enable_staging and action.stage_id < self.pp_group.world_size:
            self.maybe_prefetch_staging_for_batch(batch)
        _, event, send_work = self._pp_launch_vpp_stage(
            action,
            batch,
            pp_proxy_tensors,
            self.mb_metadata,
            self.last_rank_comm_queue,
        )
        if not hasattr(self, "_pp_vpp_stage_events"):
            self._pp_vpp_stage_events = {}
        self._pp_vpp_stage_events[(action.batch_seq, action.stage_id)] = event
        return True, send_work

    def _pp_vpp_take_local_completion(
        self: Scheduler,
        batch_seq: int,
        last_rank_comm_queue: deque,
    ) -> Dict[str, object]:
        output_event, output_proxy = last_rank_comm_queue.popleft()
        output_event.synchronize()
        output_batch_seq = int(output_proxy.tensors.get("vpp_batch_seq", -1))
        if output_batch_seq != batch_seq:
            raise RuntimeError(
                f"VPP completion mismatch: expected {batch_seq}, got {output_batch_seq}"
            )
        return {
            key: value.to("cpu") if isinstance(value, torch.Tensor) else value
            for key, value in output_proxy.tensors.items()
        }

    def _pp_vpp_finalize_rank_local_batch(
        self: Scheduler,
        batch_seq: int,
        slot_batch_seqs: List[Optional[int]],
        output_tensors: Dict[str, object],
        release_slot: bool = True,
    ) -> bool:
        slot_id = batch_seq % len(self.mbs)
        if slot_batch_seqs[slot_id] != batch_seq:
            raise RuntimeError(
                f"VPP completion for batch {batch_seq} found slot "
                f"{slot_id} owned by {slot_batch_seqs[slot_id]}"
            )
        batch = self.mbs[slot_id]
        if batch is None:
            slot_batch_seqs[slot_id] = None
            return False
        metadata = self.mb_metadata[slot_id]
        if metadata is None:
            raise RuntimeError(
                f"VPP batch {batch_seq} completed without pipeline metadata"
            )
        for req in batch.reqs:
            request_generation = int(getattr(req, "session_generation", None) or 0)
            entry = self._pp_vpp_prefix_registry.get(
                req.rid,
                request_generation,
            )
            if (
                entry is not None
                and entry.committed_end >= entry.planned_end
                and not entry.locked
            ):
                req.skip_radix_cache_insert = getattr(
                    req,
                    "_vpp_original_skip_radix_cache_insert",
                    False,
                )
        with self.copy_stream_ctx:
            self.copy_stream.wait_stream(self.schedule_stream)
            output_tensors = {
                key: (
                    value.to(self.device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for key, value in output_tensors.items()
            }
            batch_result = self._pp_prep_batch_result(
                batch,
                metadata,
                PPProxyTensors(output_tensors),
            )
            d2h_event = self.device_module.Event()
            d2h_event.record(self.device_module.current_stream())
        d2h_event.synchronize()
        self._pp_process_batch_result(batch, batch_result)
        if getattr(batch, "contains_last_prefill_chunk", False):
            for req in batch.reqs:
                if req is batch.chunked_req:
                    continue
                if getattr(self, "_pp_vpp_overlap_chunks", False):
                    generation = int(getattr(req, "session_generation", None) or 0)
                    for stage_id in range(self.pp_group.world_size * 2):
                        self._pp_vpp_launched_prefix.pop(
                            (req.rid, generation, stage_id), None
                        )
                self._pp_vpp_prefix_registry.release(
                    req.rid,
                    int(getattr(req, "session_generation", None) or 0),
                )
        self.last_mbs[slot_id] = batch
        self.mbs[slot_id] = None
        self.mb_metadata[slot_id] = None
        if release_slot:
            slot_batch_seqs[slot_id] = None
        return True

    def _event_loop_pp_disagg_prefill_vpp(self: Scheduler):
        return self._event_loop_pp_disagg_prefill_vpp_rank_local()

    def _event_loop_pp_disagg_prefill_vpp_rank_local(self: Scheduler):
        self.init_pp_loop_state()
        self._pp_vpp_overlap_chunks = os.getenv("SGLANG_VPP_OVERLAP_CHUNKS") == "1"
        self._pp_vpp_launched_prefix = {}
        max_inflight = self._pp_vpp_max_inflight()
        rank_schedule = PipelineRankSchedule(
            physical_rank=self.pp_group.rank_in_group,
            physical_size=self.pp_group.world_size,
            virtual_stages=get_parallel().pp_virtual_stages,
            max_inflight=max_inflight,
            prefill_burst_size=get_parallel().pp_vpp_prefill_burst_size,
        )
        if self._pp_vpp_overlap_chunks and rank_schedule.prefill_burst_size != 1:
            raise RuntimeError(
                "SGLANG_VPP_OVERLAP_CHUNKS requires --pp-vpp-prefill-burst-size=1"
            )
        slot_batch_seqs: List[Optional[int]] = [None] * max_inflight
        self._pp_vpp_slot_batch_seqs = slot_batch_seqs
        self._pp_vpp_ready_proxies = {}
        self._pp_vpp_early_activations = {}
        self._pp_vpp_arrivals = deque()
        self._pp_vpp_pending_recv = None
        self._pp_vpp_pending_control_recv = None
        self._pp_vpp_control_outbox = deque()
        self._pp_vpp_stage_events = {}
        self._pp_vpp_pending_materialized = deque()
        self._pp_vpp_pending_replica_updates = deque()
        self._pp_vpp_batch_replicas = defaultdict(dict)
        self._pp_vpp_prefix_registry = PipelinePrefixRegistry()
        self._pp_vpp_replica_registry = PipelineReplicaRegistry()
        self._pp_vpp_cpu_state_buffers = {}
        self._pp_vpp_cpu_broadcast_buffers = {}
        self._pp_vpp_control_layout_digest = self._pp_vpp_layout_digest()
        is_epoch_source = self.world_group.rank_in_group == 0
        self._pp_vpp_runtime_epoch = self.world_group.broadcast_object(
            time.time_ns() if is_epoch_source else None,
            src=0,
        )
        self._pp_vpp_start_receiver()
        self._pp_vpp_start_control_receiver()

        activation_send_work = deque()
        control_send_work = deque()
        max_control_sends = max_inflight * rank_schedule.logical_size
        token_budget = max(max_prefill_buffer_tokens(), 1)
        hidden_size = int(self.model_config.hidden_size)
        hc_mult = int(getattr(self.model_config.hf_config, "hc_mult", 1))
        activation_high = max_inflight * token_budget * hidden_size * hc_mult * 2
        resource_gate = PipelineResourceGate(
            ranks=range(self.pp_group.world_size),
            activation_low_watermark=activation_high // 2,
            activation_high_watermark=activation_high,
            max_pending_sends=max_inflight,
        )
        last_resource_snapshot = None
        next_batch_seq = 0
        pending_admits: set[int] = set()
        pending_remote_admits = deque()
        pending_bootstrap_applies = deque()
        bootstrap_round_active = False
        transfer_round_active = False
        pending_first_pass: Dict[int, set[int]] = defaultdict(set)
        pending_chunk_batches = set()
        tick = 0
        stall_last_progress_at = time.monotonic()
        stall_last_progress_tick = 0
        stall_last_progress_event = "init"
        stall_last_log_at = 0.0

        def mark_progress(event: str) -> None:
            nonlocal stall_last_progress_at
            nonlocal stall_last_progress_tick
            nonlocal stall_last_progress_event
            stall_last_progress_at = time.monotonic()
            stall_last_progress_tick = tick
            stall_last_progress_event = event

        def control_extras(
            envelope: PipelineControlEnvelope,
            wire: Dict[str, object],
        ) -> Dict[str, object]:
            keys = envelope.to_dict().keys()
            return {key: value for key, value in wire.items() if key not in keys}

        def forward_control(
            envelope: PipelineControlEnvelope,
            wire: Dict[str, object],
        ) -> None:
            self._pp_vpp_queue_control(
                envelope.forwarded(),
                control_extras(envelope, wire),
            )

        def try_apply_bootstrap(
            envelope: PipelineControlEnvelope,
            wire: Dict[str, object],
            remaining_good: set[str],
            remaining_bad: set[str],
        ) -> bool:
            applied = self._pp_vpp_try_apply_bootstrap_status(
                envelope.payload["prefix_boundaries"],
                remaining_good,
                remaining_bad,
            )
            if not applied:
                return False
            forward_control(envelope, wire)
            return True

        def replica_identity(payload: Dict[str, object]) -> PipelineReplicaIdentity:
            return PipelineReplicaIdentity(
                content_id=str(payload["content_id"]),
                source_id=int(payload["source_id"]),
                format_version=int(payload["format_version"]),
                consumer_rank=int(payload["consumer_rank"]),
            )

        def apply_cache_update(payload: Dict[str, object]) -> None:
            identity = replica_identity(payload)
            generation = int(payload["residency_generation"])
            start = int(payload["start"])
            end = int(payload["end"])
            state = self._pp_vpp_replica_registry.get(identity)
            if (
                state is not None
                and state.residency_generation == generation
                and state.valid_end >= end
            ):
                return
            if (
                state is not None
                and state.residency_generation == generation
                and start <= state.valid_end
            ):
                start = state.valid_end
            self._pp_vpp_replica_registry.install(
                identity,
                generation,
                start,
                end,
            )

        def flush_replica_updates() -> None:
            while self._pp_vpp_pending_replica_updates:
                (
                    batch_seq,
                    stage_id,
                    identity,
                    generation,
                    start,
                    end,
                ) = self._pp_vpp_pending_replica_updates[0]
                event = self._pp_vpp_stage_events[(batch_seq, stage_id)]
                query = getattr(event, "query", None)
                if query is not None and not query():
                    return
                self._pp_vpp_pending_replica_updates.popleft()
                payload = {
                    "content_id": identity.content_id,
                    "source_id": identity.source_id,
                    "format_version": identity.format_version,
                    "consumer_rank": identity.consumer_rank,
                    "residency_generation": generation,
                    "start": start,
                    "end": end,
                }
                apply_cache_update(payload)
                self._pp_vpp_replica_registry.lock(
                    identity,
                    end,
                    owner_id=batch_seq,
                )
                self._pp_vpp_batch_replicas[batch_seq][identity] = (
                    generation,
                    end,
                )
                self._pp_vpp_queue_control(
                    self._pp_vpp_new_control(
                        PipelineControlKind.CACHE_UPDATE,
                        batch_seq=batch_seq,
                        payload=payload,
                    ).forwarded()
                )
                if not any(
                    pending[0] == batch_seq and pending[1] == stage_id
                    for pending in self._pp_vpp_pending_replica_updates
                ) and not any(
                    pending[0] == batch_seq and pending[1] == stage_id
                    for pending in self._pp_vpp_pending_materialized
                ):
                    self._pp_vpp_stage_events.pop((batch_seq, stage_id), None)

        def evict_batch_replicas(batch_seq: int, batch: ScheduleBatch) -> None:
            if not getattr(batch, "contains_last_prefill_chunk", False):
                return
            replicas = self._pp_vpp_batch_replicas.pop(batch_seq, {})
            for identity, (generation, end) in replicas.items():
                state = self._pp_vpp_replica_registry.get(identity)
                if state is None or state.residency_generation != generation:
                    continue
                if state.locked_until:
                    self._pp_vpp_replica_registry.unlock(
                        identity,
                        end,
                        owner_id=batch_seq,
                    )
                if state.locked_until:
                    continue
                self._pp_vpp_replica_registry.evict(identity, generation)
                for tracked in self._pp_vpp_batch_replicas.values():
                    tracked.pop(identity, None)
                self._pp_vpp_queue_control(
                    self._pp_vpp_new_control(
                        PipelineControlKind.EVICT_ACK,
                        batch_seq=batch_seq,
                        payload={
                            "content_id": identity.content_id,
                            "source_id": identity.source_id,
                            "format_version": identity.format_version,
                            "consumer_rank": identity.consumer_rank,
                            "residency_generation": generation,
                        },
                    ).forwarded()
                )

        def flush_materialized() -> None:
            while self._pp_vpp_pending_materialized:
                (
                    batch_seq,
                    stage_id,
                    rid,
                    request_generation,
                    end,
                ) = self._pp_vpp_pending_materialized[0]
                event = self._pp_vpp_stage_events[(batch_seq, stage_id)]
                query = getattr(event, "query", None)
                if query is not None and not query():
                    return
                self._pp_vpp_pending_materialized.popleft()
                materialized = self._pp_vpp_new_control(
                    PipelineControlKind.PREFIX_MATERIALIZED,
                    batch_seq=batch_seq,
                    payload={
                        "rid": rid,
                        "request_generation": request_generation,
                        "stage_id": stage_id,
                        "end": end,
                    },
                )
                self._pp_vpp_apply_prefix_materialized(materialized)
                self._pp_vpp_queue_control(materialized.forwarded())
                if not any(
                    pending[0] == batch_seq and pending[1] == stage_id
                    for pending in self._pp_vpp_pending_materialized
                ) and not any(
                    pending[0] == batch_seq and pending[1] == stage_id
                    for pending in self._pp_vpp_pending_replica_updates
                ):
                    self._pp_vpp_stage_events.pop((batch_seq, stage_id), None)

        def handle_control(
            envelope: PipelineControlEnvelope,
            wire: Dict[str, object],
        ) -> None:
            nonlocal bootstrap_round_active, transfer_round_active
            payload = envelope.payload or {}
            returned_to_source = (
                envelope.source_rank == self.pp_group.rank_in_group
                and envelope.hops >= self.pp_group.world_size
            )

            if envelope.kind == PipelineControlKind.RESOURCE:
                if self.pp_group.rank_in_group == 0:
                    resource_gate.update(
                        envelope.source_rank,
                        PipelineResourceSnapshot(**payload),
                    )
                else:
                    forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.REQUEST:
                if returned_to_source:
                    return
                requests = payload.get("requests") or ()
                if requests:
                    self.process_input_requests(list(requests))
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.BOOTSTRAP_STATUS:
                phase = str(payload["phase"])
                if phase == "collect":
                    if returned_to_source:
                        final_status = self._pp_vpp_bootstrap_apply_status(payload)
                        if not final_status[0] and not final_status[1]:
                            bootstrap_round_active = False
                            return
                        apply_envelope = self._pp_vpp_new_control(
                            PipelineControlKind.BOOTSTRAP_STATUS,
                            payload={
                                "phase": "apply",
                                "good": final_status[0],
                                "bad": final_status[1],
                                "prefix_boundaries": {
                                    rid: int(payload["prefix_boundaries"][rid])
                                    for rid in final_status[0]
                                },
                            },
                        )
                        apply_wire = apply_envelope.to_dict()
                        remaining_good = set(final_status[0])
                        remaining_bad = set(final_status[1])
                        if not try_apply_bootstrap(
                            apply_envelope,
                            apply_wire,
                            remaining_good,
                            remaining_bad,
                        ):
                            pending_bootstrap_applies.append(
                                (
                                    apply_envelope,
                                    apply_wire,
                                    remaining_good,
                                    remaining_bad,
                                )
                            )
                        return
                    local_good, local_bad = self.get_rids(
                        self.disagg_prefill_bootstrap_queue.queue,
                        True,
                        [KVPoll.WaitingForInput],
                        [KVPoll.Failed],
                    )
                    aborted = {
                        req.rid
                        for req in self.disagg_prefill_bootstrap_queue.queue
                        if isinstance(req.finished_reason, FINISH_ABORT)
                    }
                    local_good, local_bad = self._route_aborts_to_bad(
                        local_good,
                        local_bad,
                        aborted,
                    )
                    self._pp_vpp_merge_bootstrap_status(
                        payload,
                        local_good,
                        local_bad,
                    )
                    forward_control(envelope, wire)
                    return
                if phase != "apply":
                    raise RuntimeError(f"invalid bootstrap phase {phase}")
                if returned_to_source:
                    bootstrap_round_active = False
                    return
                remaining_good = set(payload["good"])
                remaining_bad = set(payload["bad"])
                if not try_apply_bootstrap(
                    envelope,
                    wire,
                    remaining_good,
                    remaining_bad,
                ):
                    pending_bootstrap_applies.append(
                        (
                            envelope,
                            wire,
                            remaining_good,
                            remaining_bad,
                        )
                    )
                return

            if envelope.kind == PipelineControlKind.TRANSFER_STATUS:
                phase = str(payload["phase"])
                if phase == "collect":
                    if returned_to_source:
                        final_rids = sorted(set(payload["rids"]))
                        self.process_disagg_prefill_inflight_queue(final_rids)
                        self._pp_vpp_queue_control(
                            self._pp_vpp_new_control(
                                PipelineControlKind.TRANSFER_STATUS,
                                payload={"phase": "apply", "rids": final_rids},
                            ).forwarded()
                        )
                        return
                    local_rids = self.get_rids(
                        self.disagg_prefill_inflight_queue,
                        True,
                        [KVPoll.Success, KVPoll.Failed],
                    )
                    payload["rids"] = sorted(
                        set(payload["rids"]).intersection(local_rids)
                    )
                    forward_control(envelope, wire)
                    return
                if phase != "apply":
                    raise RuntimeError(f"invalid transfer phase {phase}")
                if returned_to_source:
                    transfer_round_active = False
                    return
                self.process_disagg_prefill_inflight_queue(list(payload["rids"]))
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.ADMIT:
                if returned_to_source:
                    if envelope.batch_seq not in pending_admits:
                        raise RuntimeError(
                            f"unexpected VPP admission response {envelope.batch_seq}"
                        )
                    errors = payload.get("errors") or ()
                    if errors:
                        raise RuntimeError(f"VPP admission failed: {errors}")
                    pending_admits.remove(envelope.batch_seq)
                    return
                batch = self._pp_vpp_prepare_wavefront_batch(
                    envelope.batch_seq,
                    slot_batch_seqs,
                )
                local_manifest = (
                    None
                    if batch is None
                    else self._pp_vpp_batch_manifest(
                        envelope.batch_seq,
                        envelope.slot_id,
                        batch,
                    )
                )
                expected_manifest = payload.get("manifest")
                if batch is None:
                    self.mbs[envelope.slot_id] = None
                    self.mb_metadata[envelope.slot_id] = None
                    slot_batch_seqs[envelope.slot_id] = None
                    pending_remote_admits.appendleft((envelope, envelope.to_dict()))
                    return
                if local_manifest != expected_manifest:
                    payload.setdefault("errors", []).append(
                        (
                            self.pp_group.rank_in_group,
                            expected_manifest,
                            local_manifest,
                        )
                    )
                self._pp_vpp_register_prefix_batch(batch)
                rank_schedule.admit(envelope.batch_seq)
                if not payload.get("errors"):
                    self._pp_vpp_promote_early_activations(envelope.batch_seq)
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.FIRST_PASS_DONE:
                if self.pp_group.rank_in_group == 0:
                    pending_first_pass[envelope.batch_seq].add(envelope.source_rank)
                else:
                    forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.PREFIX_COMMIT:
                rid = str(payload["rid"])
                request_generation = int(payload["request_generation"])
                end = int(payload["end"])
                if returned_to_source:
                    pending_chunk_batches.discard(envelope.batch_seq)
                    return
                self._pp_vpp_advance_prefix_mapping(
                    rid,
                    request_generation,
                    end,
                )
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.PREFIX_MATERIALIZED:
                if returned_to_source:
                    return
                self._pp_vpp_apply_prefix_materialized(envelope)
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.CACHE_UPDATE:
                if returned_to_source:
                    return
                apply_cache_update(payload)
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.EVICT_ACK:
                if returned_to_source:
                    return
                identity = replica_identity(payload)
                state = self._pp_vpp_replica_registry.get(identity)
                generation = int(payload["residency_generation"])
                if (
                    state is not None
                    and state.residency_generation == generation
                    and not state.locked_until
                ):
                    self._pp_vpp_replica_registry.evict(identity, generation)
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.CANCEL:
                if returned_to_source:
                    return
                rid = str(payload["rid"])
                req = self._pp_vpp_find_req(rid)
                if req is not None:
                    self._pending_chunked_abort_req = req
                    self.process_pending_chunked_abort()
                rank_schedule.cancel(envelope.batch_seq)
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.COMPLETION:
                if returned_to_source:
                    slot_id = envelope.batch_seq % max_inflight
                    slot_batch_seqs[slot_id] = None
                    rank_schedule.retire(envelope.batch_seq)
                    pending_chunk_batches.discard(envelope.batch_seq)
                    return
                output_tensors = control_extras(envelope, wire)
                rank_schedule.mark_completed(envelope.batch_seq)
                batch = self.mbs[envelope.batch_seq % max_inflight]
                if batch is not None:
                    evict_batch_replicas(envelope.batch_seq, batch)
                self._pp_vpp_finalize_rank_local_batch(
                    envelope.batch_seq,
                    slot_batch_seqs,
                    output_tensors,
                )
                rank_schedule.retire(envelope.batch_seq)
                pending_chunk_batches.discard(envelope.batch_seq)
                forward_control(envelope, wire)
                return

            raise RuntimeError(f"unsupported VPP control kind {envelope.kind}")

        while True:
            server_is_idle = True
            while True:
                control_message = self._pp_vpp_poll_control_receiver_tp_broadcast()
                if control_message is None:
                    break
                envelope, wire = control_message
                returned_to_source = (
                    envelope.source_rank == self.pp_group.rank_in_group
                    and envelope.hops >= self.pp_group.world_size
                )
                if (
                    envelope.kind == PipelineControlKind.ADMIT
                    and pending_remote_admits
                    and not returned_to_source
                ):
                    pending_remote_admits.append((envelope, wire))
                else:
                    handle_control(envelope, wire)
                if envelope.kind not in (
                    PipelineControlKind.RESOURCE,
                    PipelineControlKind.BOOTSTRAP_STATUS,
                    PipelineControlKind.TRANSFER_STATUS,
                ):
                    mark_progress(f"control:{envelope.kind.value}")
            if pending_remote_admits:
                envelope, wire = pending_remote_admits.popleft()
                handle_control(envelope, wire)
            if pending_bootstrap_applies:
                apply_args = pending_bootstrap_applies.popleft()
                if not try_apply_bootstrap(*apply_args):
                    pending_bootstrap_applies.append(apply_args)

            self._pp_vpp_reap_send_work(activation_send_work)
            self._pp_vpp_flush_control_outbox(
                control_send_work,
                max_control_sends,
            )

            snapshot = self._pp_vpp_resource_snapshot(activation_send_work)
            if snapshot != last_resource_snapshot:
                last_resource_snapshot = snapshot
                if self.pp_group.rank_in_group == 0:
                    resource_gate.update(0, snapshot)
                else:
                    self._pp_vpp_queue_control(
                        self._pp_vpp_new_control(
                            PipelineControlKind.RESOURCE,
                            payload={
                                "request_slots": snapshot.request_slots,
                                "kv_tokens": snapshot.kv_tokens,
                                "activation_bytes": snapshot.activation_bytes,
                                "pending_sends": snapshot.pending_sends,
                                "metadata_slots": snapshot.metadata_slots,
                            },
                        )
                    )

            if self.pp_group.is_first_rank:
                recv_reqs = self.ingest_requests()
                if recv_reqs:
                    request_snapshot = pickle.loads(pickle.dumps(tuple(recv_reqs)))
                    self._pp_vpp_queue_control(
                        self._pp_vpp_new_control(
                            PipelineControlKind.REQUEST,
                            payload={"requests": request_snapshot},
                        ).forwarded()
                    )

                round_states = self._pp_vpp_all_gather_cpu_state(
                    (
                        not bootstrap_round_active
                        and resource_gate.can_bootstrap()
                        and bool(self.disagg_prefill_bootstrap_queue.queue),
                        not transfer_round_active
                        and bool(self.disagg_prefill_inflight_queue),
                        int(self.req_to_metadata_buffer_idx_allocator.available_size()),
                    )
                )
                start_bootstrap_round = bool(round_states[:, 0].all())
                if start_bootstrap_round:
                    good, bad = self.get_rids(
                        self.disagg_prefill_bootstrap_queue.queue,
                        True,
                        [KVPoll.WaitingForInput],
                        [KVPoll.Failed],
                    )
                    aborted = {
                        req.rid
                        for req in self.disagg_prefill_bootstrap_queue.queue
                        if isinstance(req.finished_reason, FINISH_ABORT)
                    }
                    good, bad = self._route_aborts_to_bad(good, bad, aborted)
                    metadata_slots = int(round_states[:, 2].min())
                    bootstrap_round_active = True
                    self._pp_vpp_queue_control(
                        self._pp_vpp_new_control(
                            PipelineControlKind.BOOTSTRAP_STATUS,
                            payload={
                                "phase": "collect",
                                "good": sorted(good),
                                "bad": sorted(bad),
                                "prefix_boundaries": (
                                    self._pp_vpp_bootstrap_prefix_boundaries(good)
                                ),
                                "metadata_slots": metadata_slots,
                            },
                        ).forwarded()
                    )

                start_transfer_round = bool(round_states[:, 1].all())
                if start_transfer_round:
                    terminal_rids = self.get_rids(
                        self.disagg_prefill_inflight_queue,
                        True,
                        [KVPoll.Success, KVPoll.Failed],
                    )
                    transfer_round_active = True
                    self._pp_vpp_queue_control(
                        self._pp_vpp_new_control(
                            PipelineControlKind.TRANSFER_STATUS,
                            payload={
                                "phase": "collect",
                                "rids": sorted(terminal_rids),
                            },
                        ).forwarded()
                    )

                pending_abort = self._pending_chunked_abort_req
                self.process_pending_chunked_abort()
                if pending_abort is not None:
                    cancelled_batches = [
                        slot_batch_seqs[slot_id]
                        for slot_id, batch in enumerate(self.mbs)
                        if batch is not None
                        and any(req.rid == pending_abort.rid for req in batch.reqs)
                    ]
                    if not cancelled_batches:
                        cancelled_batches = [-1]
                    for batch_seq in cancelled_batches:
                        rank_schedule.cancel(batch_seq)
                        self._pp_vpp_queue_control(
                            self._pp_vpp_new_control(
                                PipelineControlKind.CANCEL,
                                batch_seq=batch_seq,
                                payload={"rid": pending_abort.rid},
                            ).forwarded()
                        )

                required_activation = token_budget * hidden_size * hc_mult * 2
                continuation = (
                    self.chunked_req is not None and self.chunked_req.kv.holds_kv
                )
                required_kv_tokens = token_budget + (
                    self.page_size if self._pp_vpp_overlap_chunks else 0
                )
                if (
                    (
                        not bootstrap_round_active
                        or continuation
                        or bool(self.waiting_queue)
                    )
                    and (self._pp_vpp_overlap_chunks or not pending_chunk_batches)
                    and self._pp_vpp_can_queue_admit(
                        next_batch_seq,
                        pending_admits,
                        slot_batch_seqs,
                        rank_schedule,
                    )
                    and resource_gate.can_admit(
                        required_request_slots=0 if continuation else 1,
                        required_kv_tokens=required_kv_tokens,
                        required_activation_bytes=required_activation,
                    )
                ):
                    slot_id = next_batch_seq % max_inflight
                    batch = self._pp_vpp_prepare_wavefront_batch(
                        next_batch_seq,
                        slot_batch_seqs,
                    )
                    if batch is None:
                        self.mbs[slot_id] = None
                        self.mb_metadata[slot_id] = None
                        slot_batch_seqs[slot_id] = None
                    else:
                        self._pp_vpp_register_prefix_batch(batch)
                        manifest = self._pp_vpp_batch_manifest(
                            next_batch_seq,
                            slot_id,
                            batch,
                        )
                        envelope = self._pp_vpp_new_control(
                            PipelineControlKind.ADMIT,
                            batch_seq=next_batch_seq,
                            slot_id=slot_id,
                            payload={
                                "manifest": manifest,
                                "errors": [],
                            },
                        )
                        self._pp_vpp_queue_control(envelope.forwarded())
                        pending_admits.add(next_batch_seq)
                        rank_schedule.admit(next_batch_seq)
                        if batch.chunked_req is not None:
                            pending_chunk_batches.add(next_batch_seq)
                        next_batch_seq += 1
                        mark_progress("admission")
                        server_is_idle = False
            else:
                self.process_pending_chunked_abort()

            for batch_seq, ranks in tuple(pending_first_pass.items()):
                if len(ranks) != self.pp_group.world_size:
                    continue
                batch = self.mbs[batch_seq % max_inflight]
                req = None if batch is None else batch.chunked_req
                if req is not None:
                    request_generation = int(
                        getattr(req, "session_generation", None) or 0
                    )
                    end = batch.disagg_prefill_chunk_end_by_rid[req.rid]
                    self._pp_vpp_advance_prefix_mapping(
                        req.rid,
                        request_generation,
                        end,
                    )
                    self._pp_vpp_queue_control(
                        self._pp_vpp_new_control(
                            PipelineControlKind.PREFIX_COMMIT,
                            batch_seq=batch_seq,
                            payload={
                                "rid": req.rid,
                                "request_generation": request_generation,
                                "end": end,
                            },
                        ).forwarded()
                    )
                else:
                    pending_chunk_batches.discard(batch_seq)
                pending_first_pass.pop(batch_seq, None)

            def action_ready(batch_seq: int, stage_id: int) -> bool:
                if self._pp_vpp_overlap_chunks:
                    batch = self.mbs[batch_seq % max_inflight]
                    for req, start in zip(batch.reqs, batch.prefix_lens):
                        generation = int(getattr(req, "session_generation", None) or 0)
                        entry = self._pp_vpp_prefix_registry.get(req.rid, generation)
                        launched = self._pp_vpp_launched_prefix.get(
                            (req.rid, generation, stage_id), entry.committed_end
                        )
                        if launched < start:
                            return False
                if (
                    stage_id < rank_schedule.logical_size - 1
                    and len(activation_send_work) >= max_inflight
                ):
                    return False
                if stage_id == 0:
                    return True
                return (batch_seq, stage_id) in self._pp_vpp_ready_proxies

            received, ready_states = self._pp_vpp_gather_tick_consensus(
                rank_schedule,
                action_ready,
            )
            if received:
                mark_progress("activation")
            while self._pp_vpp_arrivals:
                batch_seq, stage_id = self._pp_vpp_arrivals.popleft()
                rank_schedule.mark_ready(batch_seq, stage_id)

            action = self._pp_vpp_select_tp_action(
                rank_schedule,
                tick,
                action_ready,
                ready_states,
            )
            if action is not None:
                replica_restores = []
                if action.stage_id > 0:
                    proxy = self._pp_vpp_take_ready_proxy(action)
                    source_id = proxy.tensors.get("vpp_source_layer_id")
                    batch = self.mbs[action.slot_id]
                    if source_id is not None and batch is not None:
                        for req in batch.reqs:
                            end = batch.disagg_prefill_chunk_end_by_rid.get(req.rid)
                            if end is None:
                                continue
                            content_id = getattr(
                                req,
                                "_vpp_replica_content_id",
                                None,
                            )
                            if content_id is None:
                                content_id = hashlib.sha256(
                                    pickle.dumps(
                                        tuple(req.origin_input_ids),
                                        protocol=pickle.HIGHEST_PROTOCOL,
                                    )
                                ).hexdigest()
                                req._vpp_replica_content_id = content_id
                            identity = PipelineReplicaIdentity(
                                content_id=content_id,
                                source_id=int(source_id),
                                format_version=1,
                                consumer_rank=self.pp_group.rank_in_group,
                            )
                            state = self._pp_vpp_replica_registry.get(identity)
                            generation = (
                                0 if state is None else state.residency_generation
                            )
                            missing = self._pp_vpp_replica_registry.missing_range(
                                identity,
                                generation,
                                end,
                            )
                            if missing is None:
                                self._pp_vpp_replica_registry.lock(
                                    identity,
                                    end,
                                    owner_id=action.batch_seq,
                                )
                                self._pp_vpp_batch_replicas[action.batch_seq][
                                    identity
                                ] = (generation, end)
                            else:
                                replica_restores.append(
                                    (identity, generation, *missing)
                                )
                    self._pp_tensor_dict_inbox["vpp_proxy"].appendleft(proxy.tensors)
                action_executed, send_work = self._pp_vpp_execute_wavefront_action(
                    action,
                    slot_batch_seqs,
                )
                if send_work:
                    activation_send_work.append(send_work)
                if action_executed:
                    mark_progress(f"stage:{action.stage_id}")
                    for identity, generation, start, end in replica_restores:
                        self._pp_vpp_pending_replica_updates.append(
                            (
                                action.batch_seq,
                                action.stage_id,
                                identity,
                                generation,
                                start,
                                end,
                            )
                        )
                    transition = rank_schedule.complete(action)
                    batch = self.mbs[action.slot_id]
                    if batch is not None:
                        for req in batch.reqs:
                            if req.extend_range is None:
                                continue
                            request_generation = int(
                                getattr(req, "session_generation", None) or 0
                            )
                            end = batch.disagg_prefill_chunk_end_by_rid[req.rid]
                            if self._pp_vpp_overlap_chunks:
                                self._pp_vpp_launched_prefix[
                                    (req.rid, request_generation, action.stage_id)
                                ] = end
                            self._pp_vpp_pending_materialized.append(
                                (
                                    action.batch_seq,
                                    action.stage_id,
                                    req.rid,
                                    request_generation,
                                    end,
                                )
                            )
                    if transition.first_pass_done:
                        if self.pp_group.rank_in_group == 0:
                            pending_first_pass[action.batch_seq].add(0)
                        else:
                            self._pp_vpp_queue_control(
                                self._pp_vpp_new_control(
                                    PipelineControlKind.FIRST_PASS_DONE,
                                    batch_seq=action.batch_seq,
                                ).forwarded()
                            )
                    if transition.batch_complete:
                        output_tensors = self._pp_vpp_take_local_completion(
                            action.batch_seq,
                            self.last_rank_comm_queue,
                        )
                        flush_replica_updates()
                        flush_materialized()
                        completion = self._pp_vpp_new_control(
                            PipelineControlKind.COMPLETION,
                            batch_seq=action.batch_seq,
                            slot_id=action.slot_id,
                        )
                        if batch is not None:
                            evict_batch_replicas(action.batch_seq, batch)
                        self._pp_vpp_finalize_rank_local_batch(
                            action.batch_seq,
                            slot_batch_seqs,
                            output_tensors,
                            release_slot=False,
                        )
                        self._pp_vpp_queue_control(
                            completion.forwarded(),
                            output_tensors,
                        )
                    server_is_idle = False

            flush_replica_updates()
            flush_materialized()

            self._pp_vpp_flush_control_outbox(
                control_send_work,
                max_control_sends,
            )
            now = time.monotonic()
            has_pending_work = (
                bool(pending_admits)
                or bootstrap_round_active
                or bool(pending_chunk_batches)
                or bool(pending_bootstrap_applies)
                or bool(self.waiting_queue)
                or bool(self.disagg_prefill_bootstrap_queue.queue)
                or bool(self.disagg_prefill_inflight_queue)
                or bool(self._pp_vpp_control_outbox)
                or bool(control_send_work)
                or bool(activation_send_work)
                or rank_schedule.inflight_count > 0
            )
            if (
                self.attn_tp_group.rank_in_group == 0
                and has_pending_work
                and now - stall_last_progress_at >= 10
                and now - stall_last_log_at >= 10
            ):
                stall_last_log_at = now
                logger.warning(
                    "[VPP-STALL] no scheduler progress for %.1fs "
                    "tick=%s last_progress=%s@%s pp=%s "
                    "pending_admit=%s bootstrap_round=%s transfer_round=%s "
                    "pending_chunks=%s pending_first_pass=%s "
                    "slots=%s ready=%s running=%s "
                    "waiting=%s bootstrap=%s inflight=%s "
                    "metadata_slots=%s bootstrap_applies=%s "
                    "materialized=%s replica_updates=%s "
                    "control_outbox=%s control_sends=%s activation_sends=%s "
                    "ready_proxies=%s arrivals=%s resource_ranks=%s blocked_ranks=%s "
                    "local_resource=%s",
                    now - stall_last_progress_at,
                    tick,
                    stall_last_progress_event,
                    stall_last_progress_tick,
                    self.pp_group.rank_in_group,
                    sorted(pending_admits),
                    bootstrap_round_active,
                    transfer_round_active,
                    sorted(pending_chunk_batches),
                    {
                        batch_seq: sorted(ranks)
                        for batch_seq, ranks in pending_first_pass.items()
                    },
                    rank_schedule.slot_batch_seqs,
                    rank_schedule.ready_tasks,
                    rank_schedule.running,
                    len(self.waiting_queue),
                    len(self.disagg_prefill_bootstrap_queue.queue),
                    len(self.disagg_prefill_inflight_queue),
                    self.req_to_metadata_buffer_idx_allocator.available_size(),
                    len(pending_bootstrap_applies),
                    len(self._pp_vpp_pending_materialized),
                    len(self._pp_vpp_pending_replica_updates),
                    len(self._pp_vpp_control_outbox),
                    len(control_send_work),
                    len(activation_send_work),
                    len(self._pp_vpp_ready_proxies),
                    len(self._pp_vpp_arrivals),
                    sorted(resource_gate._snapshots),
                    sorted(resource_gate._blocked),
                    snapshot,
                )
            if activation_send_work or control_send_work:
                server_is_idle = False
            if rank_schedule.inflight_count:
                server_is_idle = False
            if server_is_idle and len(self.disagg_prefill_inflight_queue) == 0:
                self.on_idle()
            tick += 1

    def init_pp_loop_state(self: Scheduler):
        super().init_pp_loop_state()
        if not self._pp_vpp_enabled():
            return

        self.pp_loop_size = self._pp_vpp_max_inflight()
        self.mbs = [None] * self.pp_loop_size
        self.last_mbs = [None] * self.pp_loop_size
        self.running_mbs = [
            ScheduleBatch(reqs=[], batch_is_full=False)
            for _ in range(self.pp_loop_size)
        ]
        self.mb_metadata: List[Optional[PPBatchMetadata]] = [None] * self.pp_loop_size

    def _pp_recv_vpp_proxy_tensors(
        self: Scheduler,
        *,
        first_visit: bool,
        expected_batch_seq: Optional[int] = None,
        expected_stage_id: Optional[int] = None,
    ) -> Optional[PPProxyTensors]:
        if first_visit and self.pp_group.is_first_rank:
            return None
        received = self._pp_recv_typed_dict(
            expected_kind="vpp_proxy",
            all_gather_group=self.attn_tp_group,
            batch_p2p=True,
        )
        tensors = received[0] if isinstance(received, tuple) else received
        proxy = PPProxyTensors(tensors)
        if (
            expected_batch_seq is not None
            and int(proxy.tensors.get("vpp_batch_seq", -1)) != expected_batch_seq
        ):
            raise RuntimeError(
                "VPP activation batch mismatch: expected "
                f"{expected_batch_seq}, got {proxy.tensors.get('vpp_batch_seq')}"
            )
        if (
            expected_stage_id is not None
            and int(proxy.tensors.get("vpp_stage_id", -1)) != expected_stage_id
        ):
            raise RuntimeError(
                "VPP activation stage mismatch: expected "
                f"{expected_stage_id}, got {proxy.tensors.get('vpp_stage_id')}"
            )
        return proxy

    def _pp_launch_vpp_batch(self: Scheduler, *args, **kwargs):
        raise RuntimeError("VPP2 batches must run through the ready-queue scheduler")

    def _pp_launch_vpp_stage(
        self: Scheduler,
        action: PipelineWavefrontAction,
        cur_batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        mb_metadata: List[Optional[PPBatchMetadata]],
        last_rank_comm_queue: deque,
    ):
        if get_parallel().pp_virtual_stages != 2:
            raise RuntimeError("the VPP scheduler currently supports VPP2 only")
        if action.physical_rank != self.pp_group.rank_in_group:
            raise RuntimeError(
                f"wavefront action for PP rank {action.physical_rank} "
                f"cannot run on PP rank {self.pp_group.rank_in_group}"
            )
        if action.stage_id == 0:
            if pp_proxy_tensors is not None:
                raise RuntimeError("the first VPP stage must use local model inputs")
        elif pp_proxy_tensors is None:
            raise RuntimeError(
                f"logical stage {action.stage_id} requires an activation"
            )

        send_work = []
        stage_label = f"run_vpp_stage_{action.stage_id}"
        with torch.profiler.record_function(stage_label):
            with self.forward_stream_ctx:
                self.forward_stream.wait_stream(self.schedule_stream)
                if action.stage_id < self.pp_group.world_size:
                    set_time_batch(
                        cur_batch.reqs,
                        "set_run_batch_cpu_start_time",
                        trace_only=True,
                    )
                result = self.run_batch(cur_batch, pp_proxy_tensors)
                is_last_stage = action.stage_id == (
                    self.pp_group.world_size * get_parallel().pp_virtual_stages - 1
                )
                if getattr(cur_batch, "return_logprob", False) or getattr(
                    cur_batch, "return_hidden_states", False
                ):
                    result.extend_input_len_per_req = list(cur_batch.extend_lens)
                if is_last_stage:
                    if result.pp_hidden_states_proxy_tensors is not None:
                        raise RuntimeError("the final VPP stage did not produce logits")
                else:
                    proxy = result.pp_hidden_states_proxy_tensors
                    if proxy is None:
                        raise RuntimeError(
                            "a non-final VPP stage produced no activation"
                        )
                    output_stage_id = int(proxy.tensors.get("vpp_stage_id", -1))
                    if output_stage_id != action.stage_id + 1:
                        raise RuntimeError(
                            "VPP stage produced an unexpected successor: "
                            f"stage {action.stage_id} produced {output_stage_id}"
                        )
                    proxy.tensors["vpp_batch_seq"] = action.batch_seq
                    proxy.tensors["vpp_generation"] = action.batch_seq // len(
                        mb_metadata
                    )
                    proxy.tensors["vpp_protocol_version"] = _VPP_PROTOCOL_VERSION
                    proxy.tensors["vpp_runtime_epoch"] = self._pp_vpp_runtime_epoch
                    proxy.tensors["vpp_layout_digest"] = (
                        self._pp_vpp_control_layout_digest
                    )
                    proxy.tensors["vpp_src_stage_id"] = action.stage_id
                    send_work = self._pp_send_dict_to_next_stage(
                        proxy.tensors,
                        async_send=True,
                        msg_type="vpp_proxy",
                        batch_p2p=True,
                        tag=_VPP_ACTIVATION_TAG,
                    )
                if action.stage_id >= self.pp_group.world_size:
                    set_time_batch(
                        cur_batch.reqs,
                        "set_run_batch_cpu_end_time",
                        trace_only=True,
                        attrs={"pp_mb_id": action.slot_id},
                    )

                mb_metadata[action.slot_id] = PPBatchMetadata(
                    can_run_cuda_graph=result.can_run_cuda_graph,
                )
                if is_last_stage:
                    output_tensors = (
                        self._pp_prepare_tensor_dict(result, cur_batch)
                        if self.attn_tp_group.rank_in_group == 0
                        else None
                    )
                    output_tensors = self.attn_tp_group.broadcast_tensor_dict(
                        output_tensors,
                        src=0,
                    )
                    output_tensors["vpp_batch_seq"] = action.batch_seq
                event = self.device_module.Event()
                event.record(self.device_module.current_stream())
                if is_last_stage:
                    last_rank_comm_queue.append(
                        (
                            event,
                            PPProxyTensors(output_tensors),
                        )
                    )
        return result, event, send_work
