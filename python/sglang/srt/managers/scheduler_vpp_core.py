"""Batch identities and activation lifetimes for rank-local VPP scheduling."""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.distributed.vpp_policy import (
    PipelineControlEnvelope,
    PipelineRankSchedule,
    PipelineWavefrontAction,
)
from sglang.srt.distributed.vpp_transport import P2PWorkGroup
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

_VPP_PROTOCOL_VERSION = 1
logger = logging.getLogger(__name__)


class SchedulingVPPCore:
    _pp_vpp_fixed_tp = True

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
            requests.append(
                (
                    req.rid,
                    int(batch.prefix_lens[index]),
                    int(batch.prefix_lens[index] + batch.extend_lens[index]),
                )
            )
        return {
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
        if getattr(self, "_pp_vpp_fixed_batch", False):
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
        self._pp_vpp_fixed_batch = True
        self._pp_vpp_fixed_tp = True
        if self._pp_vpp_fixed_tp and not self._pp_vpp_fixed_batch:
            raise ValueError("SGLANG_VPP_FIXED_TP requires SGLANG_VPP_FIXED_BATCH=1")
        if self._pp_vpp_fixed_batch:
            if (
                self.pp_group.world_size != 2
                or self._vpp_layout.virtual_stages != 2
                or max_inflight != 2
            ):
                raise ValueError(
                    "SGLANG_VPP_FIXED_BATCH requires PP2/VPP2/max-inflight=2"
                )
            if self._pp_vpp_overlap_chunks:
                raise ValueError(
                    "SGLANG_VPP_FIXED_BATCH does not support overlapping chunks"
                )
            logger.info(
                "VPP fixed batch scheduling enabled: two-batch waves, "
                "activation manifests, cross-rank RESOURCE gate disabled"
            )
        self._pp_vpp_resource_control_enabled = not self._pp_vpp_fixed_batch
        self._pp_vpp_fixed_wave = []
        self._pp_vpp_fixed_second_pass = False
        self._pp_vpp_fixed_received = deque()
        self._pp_vpp_highest_admitted_batch_seq = -1
        self._pp_vpp_pending_batch_controls = deque()
        self._pp_vpp_fixed_activation_received = False
        self._pp_vpp_round_check = True
        self._pp_vpp_static_send_done = None
        if self._pp_vpp_fixed_tp:
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
