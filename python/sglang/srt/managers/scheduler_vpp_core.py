"""Batch identities and activation lifetimes for rank-local VPP scheduling."""

from __future__ import annotations

import logging
import pickle
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.distributed

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.distributed.vpp_policy import (
    PipelineControlEnvelope,
    PipelineControlKind,
    PipelinePrefixRegistry,
    PipelineRankSchedule,
    PipelineWavefrontAction,
)
from sglang.srt.distributed.vpp_transport import (
    P2PWorkGroup,
    VPPSchedulerActivationReceiver,
)
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

_VPP_PROTOCOL_VERSION = 1
_VPP_ACTIVATION_TAG = 1
logger = logging.getLogger(__name__)


class SchedulingVPPCore:
    def _pp_vpp_remote_admit_slot_available(
        self: Scheduler,
        envelope: PipelineControlEnvelope,
        slot_batch_seqs: List[Optional[int]],
    ) -> bool:
        slot_id = envelope.batch_seq % len(slot_batch_seqs)
        if envelope.batch_seq < 0 or envelope.slot_id != slot_id:
            raise RuntimeError("invalid VPP admission slot identity")

        owner = slot_batch_seqs[slot_id]
        if owner is not None and owner >= envelope.batch_seq:
            # Only an older owner may be waiting for retirement.
            # Duplicate or stale admissions remain protocol errors.
            raise RuntimeError(
                f"VPP wavefront slot {slot_id} is still owned by batch {owner}"
            )
        return owner is None

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

    def _pp_vpp_batch_manifest(
        self: Scheduler,
        batch_seq: int,
        slot_id: int,
        batch: ScheduleBatch,
    ) -> Dict[str, object]:
        requests = []
        for req in batch.reqs:
            index = len(requests)
            end = int(batch.prefix_lens[index] + batch.extend_lens[index])
            requests.append(
                (
                    req.rid,
                    int(batch.prefix_lens[index]),
                    end,
                )
            )
        manifest = {
            "protocol_version": _VPP_PROTOCOL_VERSION,
            "batch_seq": batch_seq,
            "generation": batch_seq // len(self.mbs),
            "slot_id": slot_id,
            "requests": tuple(requests),
            "request_generations": {
                req.rid: int(getattr(req, "session_generation", None) or 0)
                for req in batch.reqs
            },
            "bootstrap_rooms": {
                req.rid: getattr(req, "bootstrap_room", None) for req in batch.reqs
            },
        }
        if getattr(batch, "forward_mode", None) is not None:
            manifest["forward_mode"] = batch.forward_mode.value
        return manifest

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
            and batch_seq > slot_batch_seq
            and self.pp_group.is_last_rank
            and stage_id == self.pp_group.rank_in_group
            and self.mbs[slot_id] is None
            and not any(
                seq % len(self._pp_vpp_slot_batch_seqs) == slot_id and seq != batch_seq
                for seq, _ in self._pp_vpp_early_activations
            )
        )
        if slot_batch_seq not in (None, batch_seq) and not retiring_previous_generation:
            raise RuntimeError(
                f"Stale VPP activation for batch {batch_seq} in slot {slot_id}"
            )
        key = (batch_seq, stage_id)
        if key in self._pp_vpp_ready_proxies or key in self._pp_vpp_early_activations:
            raise RuntimeError(f"Duplicate VPP activation for batch/stage {key}")
        proxy = PPProxyTensors(tensors)
        self._pp_vpp_fixed_received.append(key)
        if slot_batch_seq is None or retiring_previous_generation:
            if len(self._pp_vpp_early_activations) >= len(self._pp_vpp_slot_batch_seqs):
                raise RuntimeError("VPP early-activation quarantine is full")
            self._pp_vpp_early_activations[key] = proxy
        else:
            self._pp_vpp_ready_proxies[key] = proxy
            if hasattr(self, "_pp_vpp_arrivals"):
                self._pp_vpp_arrivals.append(key)
        self._pp_vpp_pending_recv = None

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

    def _pp_vpp_init_fixed_batch(
        self: Scheduler,
        max_inflight: int,
    ) -> None:
        if (
            self.pp_group.world_size != 2
            or self._vpp_layout.virtual_stages != 2
            or max_inflight != 2
        ):
            raise ValueError("VPP scheduling requires PP2/VPP2/max-inflight=2")
        logger.info(
            "VPP fixed batch scheduling enabled: two-batch waves, activation manifests"
        )
        self._pp_vpp_fixed_wave = []
        self._pp_vpp_fixed_second_pass = False
        self._pp_vpp_fixed_received = deque()
        self._pp_vpp_highest_admitted_batch_seq = -1
        self._pp_vpp_pending_batch_controls = deque()
        self._pp_vpp_fixed_activation_received = False
        self._pp_vpp_round_check = True
        logger.info(
            "VPP fixed TP scheduling enabled: local task selection, "
            "scheduler-owned FIFO with GPU event fences, ordered send credits"
        )

    def _pp_vpp_fixed_next_task(
        self: Scheduler,
        rank_schedule: PipelineRankSchedule,
    ) -> Optional[Tuple[int, int]]:
        if self.pp_group.is_first_rank:
            first_pass = sorted(
                seq for seq, stage in rank_schedule.ready_tasks if stage == 0
            )
            if not self._pp_vpp_fixed_second_pass:
                if first_pass:
                    return (first_pass[0], 0)
                if self._pp_vpp_fixed_wave:
                    key = (
                        self._pp_vpp_fixed_wave[0],
                        self.pp_group.world_size,
                    )
                    # Admission was attempted before task selection. Drain a
                    # partial wave once its second pass is ready, even if more
                    # requests are waiting for bootstrap or local capacity.
                    if key in self._pp_vpp_ready_proxies:
                        self._pp_vpp_fixed_second_pass = True

            if self._pp_vpp_fixed_second_pass:
                if self._pp_vpp_fixed_wave:
                    key = (
                        self._pp_vpp_fixed_wave[0],
                        self.pp_group.world_size,
                    )
                    if key in self._pp_vpp_ready_proxies:
                        return key
        elif self._pp_vpp_fixed_received:
            return self._pp_vpp_fixed_received[0]
        return None

    def _pp_vpp_fixed_record_dispatch(
        self: Scheduler,
        action: PipelineWavefrontAction,
        rank_schedule: PipelineRankSchedule,
    ) -> None:
        batch_seq = action.batch_seq
        stage_id = action.stage_id
        if self.pp_group.is_first_rank:
            if stage_id == 0:
                self._pp_vpp_fixed_wave.append(batch_seq)
                if len(self._pp_vpp_fixed_wave) == rank_schedule.max_inflight:
                    self._pp_vpp_fixed_second_pass = True
            else:
                if self._pp_vpp_fixed_wave.pop(0) != batch_seq:
                    raise RuntimeError("fixed VPP second pass is out of order")
                if not self._pp_vpp_fixed_wave:
                    self._pp_vpp_fixed_second_pass = False

        if stage_id > 0:
            self._pp_vpp_fixed_received.remove((batch_seq, stage_id))

    def _pp_vpp_fixed_wait_send_credit(
        self: Scheduler,
        pending_work: deque,
        max_pending: int,
    ) -> None:
        """Release the oldest send before dispatch, independent of local polling."""
        while len(pending_work) >= max_pending:
            group = pending_work.popleft()
            if not isinstance(group, (list, P2PWorkGroup)):
                group.wait()
                continue
            items = group if isinstance(group, list) else group.items
            with self.pp_comm_stream_ctx:
                for item in items:
                    if item.work is not None:
                        item.work.wait()
                    if isinstance(item.payload, torch.Tensor) and item.payload.is_cuda:
                        # GPU wait inserts a stream dependency. Keep the storage
                        # alive until that stream has finished reading it.
                        item.payload.record_stream(self.device_module.current_stream())
            items.clear()

    def _pp_vpp_reap_send_work(self: Scheduler, pending_work: deque) -> None:
        while pending_work:
            work_group = pending_work[0]
            if isinstance(work_group, list):
                work_group = P2PWorkGroup(work_group)
                pending_work[0] = work_group
            if not work_group.poll():
                return
            pending_work.popleft()

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

    def _pp_vpp_validate_admit_manifest(
        self: Scheduler,
        envelope: PipelineControlEnvelope,
    ) -> Dict[str, object]:
        manifest = (envelope.payload or {}).get("manifest")
        if not isinstance(manifest, dict) or (
            manifest.get("protocol_version") != _VPP_PROTOCOL_VERSION
            or manifest.get("batch_seq") != envelope.batch_seq
            or manifest.get("slot_id") != envelope.slot_id
            or manifest.get("generation") != envelope.generation
        ):
            raise RuntimeError("invalid VPP admission manifest identity")
        requests = manifest.get("requests", ())
        generations = manifest.get("request_generations", {})
        rooms = manifest.get("bootstrap_rooms", {})
        rids = [rid for rid, _, _ in requests]
        if (
            not rids
            or len(set(rids)) != len(rids)
            or set(generations) != set(rids)
            or set(rooms) != set(rids)
        ):
            raise RuntimeError("invalid VPP admission request identities")
        return manifest

    def _pp_vpp_remote_manifest_ready(
        self: Scheduler,
        envelope: PipelineControlEnvelope,
    ) -> bool:
        manifest = self._pp_vpp_validate_admit_manifest(envelope)
        requests = manifest["requests"]
        generations = manifest["request_generations"]
        rooms = manifest["bootstrap_rooms"]

        candidates = {req.rid: req for req in self.waiting_queue}
        slot_id = envelope.slot_id
        for batches in (
            getattr(self, "running_mbs", ()),
            getattr(self, "last_mbs", ()),
        ):
            if batches and batches[slot_id] is not None:
                candidates.update((req.rid, req) for req in batches[slot_id].reqs)
        for req in getattr(self, "chunked_reqs", (self.chunked_req,)):
            if req is not None:
                candidates[req.rid] = req

        ready = True
        new_requests = 0
        tokens = 0
        checkpoint_reqs = []
        for rid, start, end in requests:
            if not 0 <= start < end:
                raise RuntimeError("invalid VPP admission request span")

            req = candidates.get(rid)
            if req is None:
                ready = False
                continue
            if any(
                batch is not None and any(row is req for row in batch.reqs)
                for batch in self.mbs
            ):
                ready = False
                continue
            generation = int(getattr(req, "session_generation", None) or 0)
            if generation != int(generations[rid]) or (
                getattr(req, "bootstrap_room", None) != rooms[rid]
            ):
                ready = False
                continue

            if end > len(req.origin_input_ids):
                raise RuntimeError("VPP admission span exceeds request input")

            entry = self._pp_vpp_prefix_registry.get(rid, generation)
            if entry is not None and entry.planned_end != start:
                raise RuntimeError("VPP admission prefix order mismatch")

            ready = ready and not req.finished()
            if (
                req in getattr(self, "chunked_reqs", (self.chunked_req,))
                and req.extend_end == start
                and req.prefix_len < start
            ):
                if entry is None or entry.committed_end < start:
                    ready = False
                else:
                    checkpoint_reqs.append(req)
            new_requests += int(not req.kv.holds_kv)
            tokens += end - start + self.page_size

        if any(start > 0 for _, start, _ in requests):
            # Checkpoint the completed chunk before pricing its continuation.
            # Every TP lane must enter the cache collectives with the same rows.
            states = self._pp_vpp_all_gather_cpu_state(
                (int(ready), len(checkpoint_reqs))
            )
            if not bool(states[:, 0].min()) or (
                states[:, 1].min() != states[:, 1].max()
            ):
                return False
            for req in checkpoint_reqs:
                req.skip_radix_cache_insert = req._vpp_original_skip_radix_cache_insert
                self.stash_chunked_request(req)

        available_tokens = self._pp_vpp_available_prefill_tokens()

        ready = (
            ready
            and self.req_to_token_pool.available_size() >= new_requests
            and available_tokens >= tokens
        )

        # All TP lanes must agree before entering batch-building collectives.
        states = self._pp_vpp_all_gather_cpu_state((int(ready),))
        return bool(states[:, 0].min())

    def _pp_vpp_poll_control_receiver_tp_broadcast(
        self: Scheduler,
    ) -> Optional[Tuple[PipelineControlEnvelope, Dict[str, object]]]:
        # Progress each lane's P2P receive, but never enter a TP collective
        # until all lanes receive the same consume command below.
        self._pp_vpp_activation_receiver.progress()
        wire = None
        if self.attn_tp_group.rank_in_group == 0:
            message = self._pp_vpp_poll_control_receiver()
            if message is not None:
                wire = message[1]
        progress = (
            (
                wire,
                self._pp_vpp_activation_receiver.receive_count(),
                self._pp_vpp_round_candidate,
            )
            if self.attn_tp_group.rank_in_group == 0
            else None
        )
        wire, consume, self._pp_vpp_round_check = self.attn_tp_group.broadcast_object(
            progress, src=0
        )
        for _ in range(consume):
            # Follow exactly the leader's FIFO prefix, even when local
            # metadata arrives later. Device waits stay on the receive stream.
            self._pp_vpp_accept_received_tensors(
                self._pp_vpp_activation_receiver.take()
            )
            self._pp_vpp_fixed_activation_received = True
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

    def _pp_vpp_cancel_target(
        self: Scheduler,
        envelope: PipelineControlEnvelope,
    ) -> Tuple[bool, Optional[Req]]:
        """Wait for admission, then match the exact request incarnation."""
        payload = envelope.payload or {}
        rid = str(payload["rid"])
        generation = int(payload["request_generation"])
        if envelope.batch_seq >= 0:
            if envelope.batch_seq > self._pp_vpp_highest_admitted_batch_seq:
                return False, None
            slot_id = envelope.batch_seq % len(self._pp_vpp_slot_batch_seqs)
            if self._pp_vpp_slot_batch_seqs[slot_id] != envelope.batch_seq:
                return True, None
            batch = self.mbs[slot_id]
            candidates = [] if batch is None else batch.reqs
        else:
            candidates = list(self.waiting_queue)
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                candidates += list(self.disagg_prefill_bootstrap_queue.queue)
            candidates += [
                req for batch in self.mbs if batch is not None for req in batch.reqs
            ]
            if self.chunked_req is not None:
                candidates.append(self.chunked_req)
        for req in candidates:
            if (
                req.rid == rid
                and int(getattr(req, "session_generation", None) or 0) == generation
                and getattr(req, "bootstrap_room", None) == payload["bootstrap_room"]
            ):
                return True, req
        return True, None

    def _pp_vpp_fixed_remote_admit(
        self: Scheduler,
        batch_seq: int,
        rank_schedule: PipelineRankSchedule,
    ) -> bool:
        """Build the downstream batch from its first activation's manifest."""
        slot_id = batch_seq % len(self._pp_vpp_slot_batch_seqs)
        if self._pp_vpp_slot_batch_seqs[slot_id] == batch_seq:
            return True

        key = (batch_seq, self.pp_group.rank_in_group)
        proxy = self._pp_vpp_early_activations.get(key)
        if proxy is None:
            raise RuntimeError("fixed VPP batch is missing its first activation")
        manifest = proxy.tensors.get("vpp_batch_manifest")
        envelope = self._pp_vpp_new_control(
            PipelineControlKind.ADMIT,
            batch_seq=batch_seq,
            slot_id=slot_id,
            payload={"manifest": manifest},
        )
        if not self._pp_vpp_remote_admit_slot_available(
            envelope, self._pp_vpp_slot_batch_seqs
        ):
            return False
        if not self._pp_vpp_remote_manifest_ready(envelope):
            return False

        batch = self._pp_vpp_prepare_wavefront_batch(
            batch_seq,
            self._pp_vpp_slot_batch_seqs,
            expected_manifest=manifest,
        )
        if (
            batch is None
            or self._pp_vpp_batch_manifest(batch_seq, slot_id, batch) != manifest
        ):
            raise RuntimeError("fixed VPP activation batch manifest mismatch")

        self._pp_vpp_register_prefix_batch(batch)
        rank_schedule.admit(batch_seq)
        self._pp_vpp_highest_admitted_batch_seq = max(
            self._pp_vpp_highest_admitted_batch_seq, batch_seq
        )
        self._pp_vpp_promote_early_activations(batch_seq)
        return True

    def _pp_vpp_fixed_select_tp_action(
        self: Scheduler,
        rank_schedule: PipelineRankSchedule,
        tick: int,
        is_ready,
        pending_send_work: Optional[deque] = None,
    ) -> Optional[PipelineWavefrontAction]:
        target = self._pp_vpp_fixed_next_task(rank_schedule)
        if target is None:
            return None
        batch_seq, stage_id = target
        if stage_id > 0:
            key = (batch_seq, stage_id)
            while (
                key not in self._pp_vpp_ready_proxies
                and key not in self._pp_vpp_early_activations
            ):
                self._pp_vpp_accept_received_tensors(
                    self._pp_vpp_activation_receiver.take()
                )
            if not self.pp_group.is_first_rank:
                if not self._pp_vpp_fixed_remote_admit(batch_seq, rank_schedule):
                    return None

        while self._pp_vpp_arrivals:
            rank_schedule.mark_ready(*self._pp_vpp_arrivals.popleft())

        if pending_send_work is not None and stage_id < rank_schedule.logical_size - 1:
            self._pp_vpp_fixed_wait_send_credit(
                pending_send_work, rank_schedule.max_inflight
            )
        if not is_ready(batch_seq, stage_id):
            return None

        action = rank_schedule.dispatch(tick, batch_seq, stage_id)
        self._pp_vpp_fixed_record_dispatch(action, rank_schedule)
        return action

    def _pp_vpp_find_req(self: Scheduler, rid: str) -> Optional[Req]:
        for batch in [*self.mbs, *self.last_mbs]:
            if batch is None:
                continue
            for req in batch.reqs:
                if req.rid == rid:
                    return req
        for req in getattr(self, "chunked_reqs", (self.chunked_req,)):
            if req is not None and req.rid == rid:
                return req
        return None

    def _event_loop_vpp_rank_local(self: Scheduler):
        self._init_scheduling_native_loop_state()
        max_inflight = self._pp_vpp_max_inflight()
        rank_schedule = PipelineRankSchedule(
            physical_rank=self.pp_group.rank_in_group,
            physical_size=self.pp_group.world_size,
            virtual_stages=self._vpp_layout.virtual_stages,
            max_inflight=max_inflight,
            prefill_burst_size=1,
        )
        self._pp_vpp_init_fixed_batch(max_inflight)
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
        self._pp_vpp_prefix_registry = PipelinePrefixRegistry()
        self._pp_vpp_cpu_state_buffers = {}
        self._pp_vpp_control_layout_digest = self._pp_vpp_layout_digest()
        is_epoch_source = self.world_group.rank_in_group == 0
        self._pp_vpp_runtime_epoch = self.world_group.broadcast_object(
            time.time_ns() if is_epoch_source else None,
            src=0,
        )
        source_rank = (self.pp_group.rank_in_group - 1) % self.pp_group.world_size
        self._pp_vpp_activation_receiver = VPPSchedulerActivationReceiver(
            self._pp_vpp_activation_group(source_rank),
            self.pp_group.vpp_activation_tp_group,
            max_inflight,
            batch_p2p=self.pp_group.world_size == 2,
            tag=_VPP_ACTIVATION_TAG,
        )
        self._pp_vpp_activation_receiver.start()
        self._pp_vpp_start_control_receiver()

        activation_send_work = deque()
        control_send_work = deque()
        max_control_sends = max_inflight * rank_schedule.logical_size
        max_control_sends = max(64, max_control_sends)
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
            if (
                envelope.batch_seq > self._pp_vpp_highest_admitted_batch_seq
                and envelope.kind
                in (
                    PipelineControlKind.CANCEL,
                    PipelineControlKind.COMPLETION,
                    PipelineControlKind.PREFIX_MATERIALIZED,
                    PipelineControlKind.PREFIX_COMMIT,
                )
            ):
                # These messages may overtake the activation that admits their
                # batch. Keep receiving so they cannot block its bootstrap.
                self._pp_vpp_pending_batch_controls.append((envelope, wire))
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
                        final_outcomes = dict(payload["outcomes"])
                        released = self.process_disagg_prefill_transfer_queue(
                            final_outcomes
                        )
                        final_outcomes = {
                            req.rid: final_outcomes[req.rid] for req in released
                        }
                        self._pp_vpp_queue_control(
                            self._pp_vpp_new_control(
                                PipelineControlKind.TRANSFER_STATUS,
                                payload={"phase": "apply", "outcomes": final_outcomes},
                            ).forwarded()
                        )
                        return
                    local = self._vpp_transfer_outcomes()
                    payload["outcomes"] = {
                        rid: failed or local[rid]
                        for rid, failed in payload["outcomes"].items()
                        if rid in local
                    }
                    forward_control(envelope, wire)
                    return
                if phase != "apply":
                    raise RuntimeError(f"invalid transfer phase {phase}")
                if returned_to_source:
                    transfer_round_active = False
                    return
                released = self.process_disagg_prefill_transfer_queue(
                    payload["outcomes"]
                )
                payload["outcomes"] = {
                    req.rid: payload["outcomes"][req.rid] for req in released
                }
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
                if not self._pp_vpp_remote_admit_slot_available(
                    envelope, slot_batch_seqs
                ):
                    # Keep the admission pending until the old slot is retired.
                    pending_remote_admits.appendleft((envelope, wire))
                    return

                if not self._pp_vpp_remote_manifest_ready(envelope):
                    pending_remote_admits.appendleft((envelope, wire))
                    return
                batch = self._pp_vpp_prepare_wavefront_batch(
                    envelope.batch_seq,
                    slot_batch_seqs,
                    expected_manifest=payload["manifest"],
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
                    pending_remote_admits.appendleft((envelope, wire))
                    return
                if local_manifest != expected_manifest:
                    raise RuntimeError(
                        f"VPP admission manifest mismatch: expected "
                        f"{expected_manifest}, got {local_manifest}"
                    )
                self._pp_vpp_register_prefix_batch(batch)
                rank_schedule.admit(envelope.batch_seq)
                self._pp_vpp_highest_admitted_batch_seq = max(
                    self._pp_vpp_highest_admitted_batch_seq, envelope.batch_seq
                )
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

            if envelope.kind == PipelineControlKind.CANCEL:
                if returned_to_source:
                    return
                _, req = self._pp_vpp_cancel_target(envelope)
                if req is not None:
                    if req in getattr(self, "chunked_reqs", (self.chunked_req,)):
                        self._pending_chunked_abort_req = req
                        self.process_pending_chunked_abort()
                    else:
                        # Avoid the generic RID-based abort path, which may
                        # target a newer incarnation in a different queue.
                        req.to_finish = FINISH_ABORT()
                if req is not None:
                    rank_schedule.cancel(envelope.batch_seq)
                forward_control(envelope, wire)
                return

            if envelope.kind == PipelineControlKind.COMPLETION:
                if (
                    slot_batch_seqs[envelope.batch_seq % max_inflight]
                    != envelope.batch_seq
                ):
                    raise RuntimeError("VPP completion does not own its slot")
                if returned_to_source:
                    slot_id = envelope.batch_seq % max_inflight
                    slot_batch_seqs[slot_id] = None
                    rank_schedule.retire(envelope.batch_seq)
                    pending_chunk_batches.discard(envelope.batch_seq)
                    return
                output_tensors = control_extras(envelope, wire)
                rank_schedule.mark_completed(envelope.batch_seq)
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

        def process_control_messages() -> None:
            # TP0's false rules out an all-TP true result. Share it through
            # the existing control broadcast; newly ingested requests are
            # eligible for bootstrap/transfer on the next tick.
            self._pp_vpp_round_candidate = (
                (
                    (
                        not bootstrap_round_active
                        and bool(self.disagg_prefill_bootstrap_queue.queue)
                    )
                    or (
                        not transfer_round_active
                        and bool(self.disagg_prefill_transfer_queue)
                    )
                )
                if self.pp_group.is_first_rank
                else False
            )
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
                    PipelineControlKind.BOOTSTRAP_STATUS,
                    PipelineControlKind.TRANSFER_STATUS,
                ):
                    mark_progress(f"control:{envelope.kind.value}")
            if pending_remote_admits:
                envelope, wire = pending_remote_admits[0]
                if self._pp_vpp_remote_admit_slot_available(envelope, slot_batch_seqs):
                    pending_remote_admits.popleft()
                    handle_control(envelope, wire)
            if pending_bootstrap_applies:
                apply_args = pending_bootstrap_applies.popleft()
                if not try_apply_bootstrap(*apply_args):
                    pending_bootstrap_applies.append(apply_args)
            for _ in range(len(self._pp_vpp_pending_batch_controls)):
                handle_control(*self._pp_vpp_pending_batch_controls.popleft())

        def admit_batches() -> None:
            nonlocal bootstrap_round_active, transfer_round_active
            nonlocal next_batch_seq, server_is_idle

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

                round_states = (
                    self._pp_vpp_all_gather_cpu_state(
                        (
                            not bootstrap_round_active
                            and bool(self.disagg_prefill_bootstrap_queue.queue),
                            not transfer_round_active
                            and bool(self.disagg_prefill_transfer_queue),
                            int(
                                self.req_to_metadata_buffer_idx_allocator.available_size()
                            ),
                        )
                    )
                    if self._pp_vpp_round_check
                    else None
                )
                start_bootstrap_round = round_states is not None and bool(
                    round_states[:, 0].all()
                )
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

                start_transfer_round = round_states is not None and bool(
                    round_states[:, 1].all()
                )
                if start_transfer_round:
                    terminal_outcomes = self._vpp_transfer_outcomes()
                    transfer_round_active = True
                    self._pp_vpp_queue_control(
                        self._pp_vpp_new_control(
                            PipelineControlKind.TRANSFER_STATUS,
                            payload={
                                "phase": "collect",
                                "outcomes": terminal_outcomes,
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
                        and slot_batch_seqs[slot_id] is not None
                        and any(req is pending_abort for req in batch.reqs)
                    ]
                    if not cancelled_batches:
                        cancelled_batches = [-1]
                    for batch_seq in cancelled_batches:
                        rank_schedule.cancel(batch_seq)
                        self._pp_vpp_queue_control(
                            self._pp_vpp_new_control(
                                PipelineControlKind.CANCEL,
                                batch_seq=batch_seq,
                                payload={
                                    "rid": pending_abort.rid,
                                    **(
                                        {
                                            "request_generation": int(
                                                getattr(
                                                    pending_abort,
                                                    "session_generation",
                                                    None,
                                                )
                                                or 0
                                            ),
                                            "bootstrap_room": getattr(
                                                pending_abort, "bootstrap_room", None
                                            ),
                                        }
                                    ),
                                },
                            ).forwarded()
                        )

                continuation = (
                    self.chunked_req is not None and self.chunked_req.kv.holds_kv
                )
                if (
                    (
                        not bootstrap_round_active
                        or continuation
                        or bool(self.waiting_queue)
                    )
                    and self.pause_state.may_launch()
                    and self._pp_vpp_can_queue_admit(
                        next_batch_seq,
                        pending_admits,
                        slot_batch_seqs,
                        rank_schedule,
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
                        batch._vpp_fixed_manifest = manifest
                        rank_schedule.admit(next_batch_seq)
                        self._pp_vpp_highest_admitted_batch_seq = max(
                            self._pp_vpp_highest_admitted_batch_seq, next_batch_seq
                        )
                        if batch.chunked_req is not None:
                            pending_chunk_batches.add(next_batch_seq)
                        next_batch_seq += 1
                        mark_progress("admission")
                        server_is_idle = False
            else:
                self.process_pending_chunked_abort()

        def commit_first_passes() -> None:
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

        def dispatch_stage() -> None:
            nonlocal server_is_idle

            def action_ready(batch_seq: int, stage_id: int) -> bool:
                if (
                    stage_id < rank_schedule.logical_size - 1
                    and len(activation_send_work) >= max_inflight
                ):
                    return False
                if stage_id == 0:
                    return True
                return (batch_seq, stage_id) in self._pp_vpp_ready_proxies

            received = self._pp_vpp_fixed_activation_received
            self._pp_vpp_fixed_activation_received = False
            if received:
                mark_progress("activation")
            while self._pp_vpp_arrivals:
                batch_seq, stage_id = self._pp_vpp_arrivals.popleft()
                rank_schedule.mark_ready(batch_seq, stage_id)

            action = self._pp_vpp_fixed_select_tp_action(
                rank_schedule,
                tick,
                action_ready,
                activation_send_work,
            )
            if action is not None:
                proxy = (
                    self._pp_vpp_take_ready_proxy(action)
                    if action.stage_id > 0
                    else None
                )
                action_executed, send_work = self._pp_vpp_execute_wavefront_action(
                    action,
                    slot_batch_seqs,
                    proxy,
                )
                if send_work:
                    activation_send_work.append(send_work)
                if action_executed:
                    mark_progress(f"stage:{action.stage_id}")
                    transition = rank_schedule.complete(action)
                    batch = self.mbs[action.slot_id]
                    if batch is not None:
                        for req in batch.reqs:
                            if req.extend_end is None:
                                continue
                            request_generation = int(
                                getattr(req, "session_generation", None) or 0
                            )
                            end = batch.disagg_prefill_chunk_end_by_rid[req.rid]
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
                        flush_materialized()
                        completion = self._pp_vpp_new_control(
                            PipelineControlKind.COMPLETION,
                            batch_seq=action.batch_seq,
                            slot_id=action.slot_id,
                        )
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

        def finish_tick() -> None:
            nonlocal stall_last_log_at, server_is_idle

            flush_materialized()

            self._pp_vpp_flush_control_outbox(
                control_send_work,
                max_control_sends,
            )
            now = time.monotonic()
            has_pending_work = (
                bool(pending_admits)
                or bool(self._pp_vpp_pending_batch_controls)
                or bootstrap_round_active
                or bool(pending_chunk_batches)
                or bool(pending_bootstrap_applies)
                or bool(self.waiting_queue)
                or bool(self.disagg_prefill_bootstrap_queue.queue)
                or bool(self.disagg_prefill_transfer_queue)
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
                    "materialized=%s "
                    "control_outbox=%s control_sends=%s activation_sends=%s "
                    "ready_proxies=%s arrivals=%s",
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
                    len(self.disagg_prefill_transfer_queue),
                    self.req_to_metadata_buffer_idx_allocator.available_size(),
                    len(pending_bootstrap_applies),
                    len(self._pp_vpp_pending_materialized),
                    len(self._pp_vpp_control_outbox),
                    len(control_send_work),
                    len(activation_send_work),
                    len(self._pp_vpp_ready_proxies),
                    len(self._pp_vpp_arrivals),
                )
            if activation_send_work or control_send_work:
                server_is_idle = False
            if rank_schedule.inflight_count or self._pp_vpp_pending_batch_controls:
                server_is_idle = False
            if server_is_idle and not self.disagg_prefill_transfer_queue:
                self.on_idle()

        while True:
            server_is_idle = True
            process_control_messages()
            self._pp_vpp_reap_send_work(activation_send_work)
            self._pp_vpp_flush_control_outbox(
                control_send_work,
                max_control_sends,
            )
            admit_batches()
            commit_first_passes()
            dispatch_stage()
            finish_tick()
            tick += 1
