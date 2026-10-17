"""The MoE LoRA runner's route bundle: the views its execution plan asks for,
built from the shared route builders in ``sglang.kernels.ops.lora.common.routing``."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.kernels.ops.lora.common.route_view import RouteView, RouteViewKind
from sglang.kernels.ops.lora.common.routing import build_route
from sglang.srt.lora.moe.plan import (
    MoePlan,
    RouteBuilderFamily,
    RouteRequirement,
)
from sglang.srt.lora.workspace import LoraWorkspace


@dataclass(frozen=True, slots=True)
class MoeLoraRoutes:
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


def build_moe_routes(
    plan: MoePlan,
    *,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    workspace: LoraWorkspace,
) -> MoeLoraRoutes:
    requirements = plan.route_requirements()
    values: dict[str, object] = {}
    if RouteRequirement.RAW_PER_EXPERT in requirements:
        values["raw_per_expert"] = build_route(
            token_lora_mapping,
            group_ids=topk_ids,
            groups_per_slot=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            view=RouteViewKind.RAW,
        )
    if RouteRequirement.RAW_SHARED_OUTER in requirements:
        values["raw_shared_outer"] = build_route(
            token_lora_mapping,
            group_ids=topk_ids,
            groups_per_slot=1,
            max_loras=max_loras,
            block_size=block_size,
            view=RouteViewKind.RAW,
        )

    if (
        plan.route_builder is RouteBuilderFamily.PARALLEL_SHARED_OUTER
        and RouteRequirement.ALIGNED_PER_EXPERT in requirements
        and RouteRequirement.ALIGNED_SHARED_OUTER in requirements
    ):

        def _build_per_expert() -> RouteView:
            return build_route(
                token_lora_mapping,
                group_ids=topk_ids,
                groups_per_slot=num_local_experts,
                max_loras=max_loras,
                block_size=block_size,
                view=RouteViewKind.ALIGNED,
                workspace=workspace,
                tensor_prefix="route:aligned_per_expert",
            )

        def _build_shared() -> None:
            values["aligned_shared_outer"] = build_route(
                token_lora_mapping,
                group_ids=topk_ids,
                groups_per_slot=1,
                max_loras=max_loras,
                block_size=block_size,
                view=RouteViewKind.ALIGNED,
                workspace=workspace,
                tensor_prefix="route:aligned_shared_outer",
            )

        values["aligned_per_expert"] = workspace.run_parallel(
            name="route:parallel",
            device=topk_ids.device,
            compute=_build_per_expert,
            side=_build_shared,
        )
    else:
        if RouteRequirement.ALIGNED_PER_EXPERT in requirements:
            values["aligned_per_expert"] = build_route(
                token_lora_mapping,
                group_ids=topk_ids,
                groups_per_slot=num_local_experts,
                max_loras=max_loras,
                block_size=block_size,
                view=RouteViewKind.ALIGNED,
                workspace=workspace,
                tensor_prefix="route:aligned_per_expert",
            )
        if RouteRequirement.ALIGNED_SHARED_OUTER in requirements:
            values["aligned_shared_outer"] = build_route(
                token_lora_mapping,
                group_ids=topk_ids,
                groups_per_slot=1,
                max_loras=max_loras,
                block_size=block_size,
                view=RouteViewKind.ALIGNED,
                workspace=workspace,
                tensor_prefix="route:aligned_shared_outer",
            )

    if RouteRequirement.SHARED_TOKEN_PLAN in requirements:

        def build_token_route():
            # Shared experts use one row per token, grouped by adapter slot.
            return build_route(
                token_lora_mapping,
                max_loras=max_loras,
                block_size=block_size,
                view=RouteViewKind.ALIGNED,
                workspace=workspace,
                tensor_prefix=f"route:shared_token:sorted:{block_size}",
            )

        values["shared_token"] = workspace.route(
            token_lora_mapping, ("sorted", block_size, max_loras), build_token_route
        )
    return MoeLoraRoutes(**values)
