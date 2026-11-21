"""Rank-local VPP batch identities retained until output processing finishes."""

from dataclasses import dataclass
from typing import Any

from sglang.srt.distributed.vpp_layout import VPPLayout
from sglang.srt.distributed.vpp_schedule import VPPTask


@dataclass
class VPPBatchState:
    batch_seq: int
    slot_id: int
    manifest: tuple
    batch: Any
    forward_batch: Any = None
    metadata: Any = None
    output_tensors: dict | None = None
    dispatched_passes: int = 0


class VPPBatchLedger:
    """A slot stays occupied through all local passes and result processing.

    A manifest contains request IDs, session generations, forward mode and
    sequence/chunk lengths. It detects a differently composed remote batch
    before any activation is consumed by the model.
    """

    def __init__(self, layout: VPPLayout, physical_rank: int, max_inflight: int):
        self.stages = layout.stages_for_rank(physical_rank)
        if type(max_inflight) is not int or max_inflight <= 0:
            raise ValueError("max_inflight must be a positive integer")
        self.layout = layout
        self.max_inflight = max_inflight
        self._batches: dict[int, VPPBatchState] = {}
        self._slots: dict[int, int] = {}
        self._last_batch_seq = -1

    def admit(self, state: VPPBatchState) -> None:
        if type(state.batch_seq) is not int or state.batch_seq <= self._last_batch_seq:
            raise ValueError("Batch sequences must be nonnegative and increasing")
        if type(state.slot_id) is not int or not 0 <= state.slot_id < self.max_inflight:
            raise ValueError("Invalid VPP batch slot")
        if state.slot_id in self._slots:
            raise RuntimeError("VPP slot is still awaiting result processing")
        if state.dispatched_passes:
            raise ValueError("An admitted batch must not have dispatched passes")
        self._batches[state.batch_seq] = state
        self._slots[state.slot_id] = state.batch_seq
        self._last_batch_seq = state.batch_seq

    def get(self, task: VPPTask) -> VPPBatchState:
        state = self._batches.get(task.batch_seq)
        if state is None:
            raise RuntimeError("Unknown or retired VPP batch")
        if (
            state.dispatched_passes >= len(self.stages)
            or self.stages[state.dispatched_passes].stage_id != task.stage_id
        ):
            raise RuntimeError("VPP batch stage is out of order")
        return state

    def check_activation(self, task: VPPTask, tensors: dict) -> VPPBatchState:
        state = self.get(task)
        expected = (task.batch_seq, task.stage_id, state.slot_id, state.manifest)
        if tensors.get("vpp_identity") != expected:
            raise RuntimeError("VPP activation batch identity or manifest mismatch")
        return state

    def record_dispatch(self, task: VPPTask) -> None:
        self.get(task).dispatched_passes += 1

    def complete(self, batch_seq: int) -> VPPBatchState:
        state = self._batches.get(batch_seq)
        if state is None or state.dispatched_passes != len(self.stages):
            raise RuntimeError("VPP batch cannot retire before its final local pass")
        del self._batches[batch_seq]
        del self._slots[state.slot_id]
        return state
