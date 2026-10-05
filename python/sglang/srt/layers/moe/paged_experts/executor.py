"""Runs a paged forward step: page in, masked GEMM per wave, merge and finish per the runner
contract.

``EagerExecutor`` decides residency on the host and serves any step, in as many waves as it
needs. ``DeviceExecutor`` decides and pages on the GPU without a host sync, so CUDA graphs can
capture it; it serves the steps whose routed entries fit the K slots (one wave), which covers
decode up to ``K // top_k`` requests. Both finish through the same contract.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.moe.paged_experts.residency import (
    DeviceResidency,
    LRUPolicy,
    plan_waves,
)
from sglang.srt.layers.moe.paged_experts.runners import RunnerContract
from sglang.srt.layers.moe.paged_experts.store import ExpertStore


def run_wave(base_method, layer, dispatch_output, slots):
    """Run the base method over the K-slot table: routed entry i goes to GPU slot
    ``slots[i]``; entries at -1 are outside the wave and contribute nothing (slot 0, weight 0).
    """
    masked = slots < 0
    topk_output = dispatch_output.topk_output
    wave_output = dispatch_output._replace(
        topk_output=topk_output._replace(
            topk_ids=slots.masked_fill(masked, 0),
            topk_weights=topk_output.topk_weights.masked_fill(masked, 0),
        ),
    )
    return base_method.apply(layer=layer, dispatch_output=wave_output).hidden_states


class EagerExecutor:
    def __init__(
        self,
        *,
        base_method,
        store: ExpertStore,
        policy: LRUPolicy,
        contract: RunnerContract,
        num_experts: int,
        routed_scaling_factor: float,
        device_residency: Optional[DeviceResidency] = None,
    ):
        self.base_method = base_method
        self.store = store
        self.policy = policy
        self.contract = contract
        self.num_experts = num_experts
        self.routed_scaling_factor = routed_scaling_factor
        # Shared with a DeviceExecutor, whose steps (and graph replays) move it on the GPU.
        self.device_residency = device_residency

    def run(self, layer, dispatch_output):
        if self.device_residency is not None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Paged experts: a captured decode batch routes more entries than the K "
                    "resident slots; capture batch sizes must stay at or below K // top_k"
                )
            self.device_residency.load_into(self.policy)
        out = self._run(layer, dispatch_output)
        if self.device_residency is not None:
            self.device_residency.store_from(self.policy)
        return out

    def _run(self, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids
        distinct = [e for e in torch.unique(topk_ids).tolist() if e >= 0]
        merged = None
        for wave in plan_waves(policy=self.policy, distinct=distinct):
            if wave.loads:
                src, dst = torch.tensor(wave.loads, dtype=torch.int64).unbind(dim=1)
                self.store.page_in(layer=layer, src=src, dst=dst)
            # Index E (and -1, which wraps to it) stays -1: padded routing entries are masked.
            to_slot = torch.full((self.num_experts + 1,), -1, dtype=torch.int32)
            to_slot[wave.experts] = torch.tensor(wave.slots, dtype=torch.int32)
            slots = to_slot.to(topk_ids.device)[topk_ids]
            partial = run_wave(self.base_method, layer, dispatch_output, slots)
            masked = slots < 0
            merged = (
                partial
                if merged is None
                else self.contract.merge(merged=merged, partial=partial, masked=masked)
            )
        return StandardCombineInput(
            hidden_states=self.contract.finish(
                merged=merged,
                hidden_states=dispatch_output.hidden_states,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        )


class DeviceExecutor:
    """One wave, decided and paged on the GPU: a step with at most K routed entries needs at
    most K distinct experts, so they all fit the slots at once."""

    def __init__(
        self,
        *,
        base_method,
        store: ExpertStore,
        residency: DeviceResidency,
        contract: RunnerContract,
        routed_scaling_factor: float,
    ):
        self.base_method = base_method
        self.store = store
        self.residency = residency
        self.contract = contract
        self.routed_scaling_factor = routed_scaling_factor

    def run(self, layer, dispatch_output):
        from sglang.kernels.ops.moe.paged_experts import paged_experts_decide
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        r = self.residency
        topk_ids = dispatch_output.topk_output.topk_ids
        paged_experts_decide(
            topk_ids.flatten().to(torch.int32),
            r.step,
            r.slot_expert,
            r.expert_slot,
            r.slot_lastuse,
            r.src,
            r.dst,
            r.count,
        )
        self.store.gather(src=r.src, dst=r.dst, count=r.count)
        # Every routed expert is now resident; padding (-1) stays masked.
        slots = r.expert_slot[topk_ids.clamp(min=0)].masked_fill(topk_ids < 0, -1)
        partial = run_wave(self.base_method, layer, dispatch_output, slots)
        return StandardCombineInput(
            hidden_states=self.contract.finish(
                merged=partial,
                hidden_states=dispatch_output.hidden_states,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        )
