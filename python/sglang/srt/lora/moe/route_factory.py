"""Build exactly the route products consumed by one MoE-LoRA plan."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.lora.moe.execution_plan import (
    MoeLoraExecutionPlan,
    RouteBuilderFamily,
    RouteRequirement,
)
from sglang.srt.lora.moe.joint_routing import build_joint_shared_routes
from sglang.srt.lora.moe.routing import (
    RouteView,
    RouteViewKind,
    build_virtual_expert_routing,
)
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


@dataclass(frozen=True, slots=True)
class MoeLoraRoutes:
    """The distinct row-domain views materialized for one forward."""

    raw_per_expert: RouteView | None = None
    raw_shared_outer: RouteView | None = None
    aligned_per_expert: RouteView | None = None
    aligned_shared_outer: RouteView | None = None
    shared_token: RouteView | None = None

    def raw(self, is_shared_outer: bool) -> RouteView:
        if is_shared_outer:
            return self._require(self.raw_shared_outer, "raw_shared_outer")
        return self._require(self.raw_per_expert, "raw_per_expert")

    def aligned(self, is_shared_outer: bool) -> RouteView:
        if is_shared_outer:
            return self._require(self.aligned_shared_outer, "aligned_shared_outer")
        return self._require(self.aligned_per_expert, "aligned_per_expert")

    @staticmethod
    def _require(route: RouteView | None, field: str) -> RouteView:
        """Name the FIELD, so the message is greppable straight to the slot."""
        if route is None:
            raise ValueError(f"the execution plan did not request {field}")
        return route


def build_routes(
    plan: MoeLoraExecutionPlan,
    *,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    workspace: MoeLoraWorkspace,
) -> MoeLoraRoutes:
    """Construct the exact route bundle declared by ``plan``.

    Raw views are descriptor-only and therefore both ownership forms are
    cheap.  Aligned views launch their builders.  The shared token plan uses a
    synthetic top-k-one expert column because gate/up-A is adapter-owned; the
    original pair route remains authoritative for every B consumer.
    """
    requirements = plan.route_requirements()
    values: dict[str, object] = {}
    if RouteRequirement.RAW_PER_EXPERT in requirements:
        values["raw_per_expert"] = build_virtual_expert_routing(
            topk_ids,
            token_lora_mapping,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            view=RouteViewKind.RAW,
        )
    if RouteRequirement.RAW_SHARED_OUTER in requirements:
        # One LoRA expert per adapter, with the local expert count restoring
        # the ownership bound (see the shared-outer note below).
        values["raw_shared_outer"] = build_virtual_expert_routing(
            topk_ids,
            token_lora_mapping,
            num_local_experts=num_local_experts,
            is_shared_outer=True,
            max_loras=max_loras,
            block_size=block_size,
            view=RouteViewKind.RAW,
        )

    if plan.route_builder is RouteBuilderFamily.JOINT_SHARED_OUTER:
        per_expert, shared = build_joint_shared_routes(
            topk_ids,
            token_lora_mapping,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            workspace=workspace,
        )
        values["aligned_per_expert"] = per_expert
        values["aligned_shared_outer"] = shared
    else:
        if RouteRequirement.ALIGNED_PER_EXPERT in requirements:
            values["aligned_per_expert"] = build_virtual_expert_routing(
                topk_ids,
                token_lora_mapping,
                num_local_experts=num_local_experts,
                max_loras=max_loras,
                block_size=block_size,
                view=RouteViewKind.ALIGNED,
                workspace=workspace,
                scratch_prefix="route:aligned_per_expert",
            )
        if RouteRequirement.ALIGNED_SHARED_OUTER in requirements:
            # A shared adapter has exactly ONE LoRA expert, so the generic
            # validity test would end in ``lora_expert_id < 1`` -- always true.
            # The local expert count restores the bound, without which a routed
            # expert this rank does not own would pass as a valid pair.
            values["aligned_shared_outer"] = build_virtual_expert_routing(
                topk_ids,
                token_lora_mapping,
                num_local_experts=num_local_experts,
                is_shared_outer=True,
                max_loras=max_loras,
                block_size=block_size,
                view=RouteViewKind.ALIGNED,
                workspace=workspace,
                scratch_prefix="route:aligned_shared_outer",
            )

    if RouteRequirement.SHARED_TOKEN_PLAN in requirements:
        # A token may have its first local top-k entry masked while another is
        # valid, so deriving this route from topk_ids[:, :1] would incorrectly
        # drop it.  Conversely, a token with no local pair must not schedule
        # shared-A work: the repeated-pair control has no work for that token,
        # and no downstream B reads its bridge.  This is the same sparse-EP
        # contract qualified by Step 3's token-dedup candidate.
        has_local_pair = ((topk_ids >= 0) & (topk_ids < num_local_experts)).any(dim=1)
        shared_token_lora_mapping = workspace.tensor(
            "route:shared_token_lora_mapping",
            token_lora_mapping.shape,
            dtype=token_lora_mapping.dtype,
            device=token_lora_mapping.device,
        )
        torch.where(
            has_local_pair,
            token_lora_mapping,
            torch.full_like(token_lora_mapping, -1),
            out=shared_token_lora_mapping,
        )
        # The shared A factor is adapter-owned, so the expert column itself is
        # synthetic once local participation has been encoded in the slots.
        token_experts = workspace.tensor(
            "route:shared_token_experts",
            (topk_ids.shape[0], 1),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
            zero_on_first_allocation=True,
        )
        # Shared-outer, so the same expert-range bound as above applies.
        values["shared_token"] = build_virtual_expert_routing(
            token_experts,
            shared_token_lora_mapping,
            num_local_experts=num_local_experts,
            is_shared_outer=True,
            max_loras=max_loras,
            block_size=block_size,
            view=RouteViewKind.ALIGNED,
            workspace=workspace,
            scratch_prefix="route:shared_token",
        )

    return MoeLoraRoutes(**values)
