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
        """Serve a step in waves, with one host sync: the read of each expert's routed
        entries (and of the device residency state, if any) the waves are planned from."""
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        topk_ids = dispatch_output.topk_output.topk_ids
        r = self.device_residency
        if r is not None and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Paged experts: a captured decode batch routes more entries than the K "
                "resident slots; capture batch sizes must stay at or below K // top_k"
            )
        counts = r.counts if r is not None else self._counts(topk_ids.device)
        counts.zero_()
        # Padding (-1) counts at index E.
        ids = topk_ids.flatten().long()
        ids = ids.masked_fill((ids < 0) | (ids >= self.num_experts), self.num_experts)
        counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
        host = (r.readback if r is not None else counts).cpu()
        if r is not None:
            r.load_into(self.policy, host)
            host = host[-(self.num_experts + 1) :]
        routed = host.tolist()
        waves = plan_waves(
            policy=self.policy,
            distinct=[e for e in range(self.num_experts) if routed[e]],
        )
        merged = self._run_waves(layer, dispatch_output, waves, routed, ids)
        if r is not None:
            r.store_from(self.policy)
        return StandardCombineInput(
            hidden_states=self.contract.finish(
                merged=merged,
                hidden_states=dispatch_output.hidden_states,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        )

    def _counts(self, device) -> torch.Tensor:
        if getattr(self, "_counts_buffer", None) is None:
            self._counts_buffer = torch.zeros(
                self.num_experts + 1, dtype=torch.int32, device=device
            )
        return self._counts_buffer

    def _run_waves(self, layer, dispatch_output, waves, routed, ids):
        """Upload every wave's slot map, page-in plan and (for several waves) each entry's wave
        in one copy that does not wait for the GPU, then page in and run the waves."""
        topk_ids = dispatch_output.topk_output.topk_ids
        e, k, n = self.num_experts, self.policy.num_slots, len(waves)
        # Per wave: the slot of each of its experts (-1 elsewhere, and at E for padding).
        table = [-1] * (n * (e + 1)) + [n] * (e + 1)
        # Per wave: count, src[K], dst[K] of its page-ins.
        plans = [0] * (n * (2 * k + 1))
        for w, wave in enumerate(waves):
            for expert, slot in zip(wave.experts, wave.slots):
                table[w * (e + 1) + expert] = slot
                table[n * (e + 1) + expert] = w
            base = w * (2 * k + 1)
            plans[base] = len(wave.loads)
            for i, (expert, slot) in enumerate(wave.loads):
                plans[base + 1 + i] = expert
                plans[base + 1 + k + i] = slot
        host = torch.tensor(table + plans, dtype=torch.int32)
        if topk_ids.is_cuda:
            host = host.pin_memory()
        device = host.to(topk_ids.device, non_blocking=True)
        maps = device[: (n + 1) * (e + 1)].view(n + 1, e + 1)
        plans = device[(n + 1) * (e + 1) :].view(n, 2 * k + 1)
        plans = [(p[1 : 1 + k], p[1 + k :], p[:1]) for p in plans]
        if n == 1:  # only padding is outside the wave
            self.store.page_in(layer=layer, loads=waves[0].loads, plan=plans[0])
            return run_on_slots(
                self.base_method, layer, dispatch_output, maps[0][topk_ids]
            )
        # Each wave's entries, in entry order: one stable sort by wave, sliced at offsets
        # known from the routing counts.
        order = torch.argsort(maps[-1][ids], stable=True)
        start = 0
        merged = None
        for w, wave in enumerate(waves):
            num = sum(routed[x] for x in wave.experts)
            self.store.page_in(layer=layer, loads=wave.loads, plan=plans[w])
            merged = self.contract.run_wave(
                base_method=self.base_method,
                layer=layer,
                dispatch_output=dispatch_output,
                slots=maps[w][topk_ids],
                entries=order[start : start + num],
                merged=merged,
            )
            start += num
        return merged


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
        partial = run_on_slots(self.base_method, layer, dispatch_output, slots)
        return StandardCombineInput(
            hidden_states=self.contract.finish(
                merged=partial,
                hidden_states=dispatch_output.hidden_states,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        )
