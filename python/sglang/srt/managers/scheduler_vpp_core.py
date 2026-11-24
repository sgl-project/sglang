"""Batch identities and activation lifetimes for rank-local VPP scheduling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from sglang.srt.distributed.vpp_policy import (
    PipelineControlEnvelope,
    PipelineRankSchedule,
    PipelineWavefrontAction,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

_VPP_PROTOCOL_VERSION = 1


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
