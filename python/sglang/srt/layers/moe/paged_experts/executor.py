"""Runs a paged forward step: page in, the wave's GEMMs, merge and finish per the runner
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
from sglang.srt.layers.moe.paged_experts.runners import RunnerContract, run_on_slots
from sglang.srt.layers.moe.paged_experts.store import HostExpertStore


class EagerExecutor:
    def __init__(
        self,
        *,
        base_method,
        store: HostExpertStore,
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
        r = self.device_residency
        if r is not None and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Paged experts: a captured decode batch routes more entries than the K "
                "resident slots; capture batch sizes must stay at or below K // top_k"
            )
        out = self._run(layer, dispatch_output)
        if r is not None:
            r.store_from(self.policy)
        return out

    def _run(self, layer, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        topk_output = dispatch_output.topk_output
        topk_ids = topk_output.topk_ids
        # Routed entries per expert (padding at index E), read with the device residency
        # state in one sync.
        ids = topk_ids.flatten().long()
        ids = ids.masked_fill(ids < 0, self.num_experts)
        counts = torch.zeros(self.num_experts + 1, dtype=torch.int32, device=ids.device)
        counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
        r = self.device_residency
        routed = counts.tolist() if r is None else r.load_into(self.policy, counts)
        distinct = [e for e in range(self.num_experts) if routed[e]]
        merged = None
        waves = plan_waves(policy=self.policy, distinct=distinct)
        # Every wave's slot map in one upload that does not wait for the GPU. Index E (and -1,
        # which wraps to it) stays -1: padded routing entries are masked.
        to_slot = torch.full((len(waves), self.num_experts + 1), -1, dtype=torch.int32)
        for w, wave in enumerate(waves):
            to_slot[w, wave.experts] = torch.tensor(wave.slots, dtype=torch.int32)
        if topk_ids.is_cuda:
            to_slot = to_slot.pin_memory()
        to_slot = to_slot.to(topk_ids.device, non_blocking=True)
        for w, wave in enumerate(waves):
            self.store.page_in(layer=layer, loads=wave.loads)
            slots = to_slot[w][topk_ids]
            if len(waves) == 1:  # only padding is outside the wave
                merged = run_on_slots(self.base_method, layer, dispatch_output, slots)
                break
            merged = self.contract.run_wave(
                base_method=self.base_method,
                layer=layer,
                dispatch_output=dispatch_output,
                slots=slots,
                merged=merged,
                num_entries=sum(routed[e] for e in wave.experts),
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
        store: HostExpertStore,
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
        topk_output = dispatch_output.topk_output
        topk_ids, topk_weights = topk_output.topk_ids, topk_output.topk_weights
        run_ids = torch.empty(topk_ids.shape, dtype=torch.int32, device=topk_ids.device)
        run_w = torch.empty(topk_ids.shape, dtype=torch.float32, device=topk_ids.device)
        paged_experts_decide(
            topk_ids.flatten().to(torch.int32),
            topk_weights.flatten().to(torch.float32),
            r.step,
            r.slot_expert,
            r.expert_slot,
            r.slot_lastuse,
            r.src,
            r.dst,
            r.count,
            run_ids.view(-1),
            run_w.view(-1),
        )
        self.store.gather(src=r.src, dst=r.dst, count=r.count)
        # Every routed expert is now resident; padding got slot 0 and weight 0.
        wave_output = dispatch_output._replace(
            topk_output=topk_output._replace(
                topk_ids=run_ids, topk_weights=run_w.to(topk_weights.dtype)
            )
        )
        partial = self.base_method.apply(
            layer=layer, dispatch_output=wave_output
        ).hidden_states
        return StandardCombineInput(
            hidden_states=self.contract.finish(
                merged=partial,
                hidden_states=dispatch_output.hidden_states,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        )
