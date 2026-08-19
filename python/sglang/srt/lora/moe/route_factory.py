from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.lora.moe import aligned_route
from sglang.srt.lora.moe.execution_plan import (
    MoeLoraExecutionPlan,
    RouteBuilderFamily,
    RouteRequirement,
)
from sglang.srt.lora.moe.routing import (
    RouteView,
    RouteViewKind,
    build_virtual_expert_routing,
)
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


@dataclass(frozen=True, slots=True)
class MoeLoraRoutes:
    """Every route the plan asked for; the rest stay None."""

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
        per_expert, shared = aligned_route.build(
            topk_ids,
            token_lora_mapping,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            workspace=workspace,
            tensor_prefix="joint_route",
            need_per_expert=True,
            need_shared=True,
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
                tensor_prefix="route:aligned_per_expert",
            )
        if RouteRequirement.ALIGNED_SHARED_OUTER in requirements:
            values["aligned_shared_outer"] = build_virtual_expert_routing(
                topk_ids,
                token_lora_mapping,
                num_local_experts=num_local_experts,
                is_shared_outer=True,
                max_loras=max_loras,
                block_size=block_size,
                view=RouteViewKind.ALIGNED,
                workspace=workspace,
                tensor_prefix="route:aligned_shared_outer",
            )

    if RouteRequirement.SHARED_TOKEN_PLAN in requirements:
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
        token_experts = workspace.tensor(
            "route:shared_token_experts",
            (topk_ids.shape[0], 1),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
            zero_on_first_allocation=True,
        )
        values["shared_token"] = build_virtual_expert_routing(
            token_experts,
            shared_token_lora_mapping,
            num_local_experts=num_local_experts,
            is_shared_outer=True,
            max_loras=max_loras,
            block_size=block_size,
            view=RouteViewKind.ALIGNED,
            workspace=workspace,
            tensor_prefix="route:shared_token",
        )

    return MoeLoraRoutes(**values)
