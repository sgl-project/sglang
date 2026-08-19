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
    ROUTE_ALIGNED,
    ROUTE_RAW,
    FusedAlignScratch,
    RouteView,
    build_virtual_expert_routing,
    uses_fused_align,
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


def _fused_align_scratch(
    workspace: MoeLoraWorkspace,
    *,
    prefix: str,
    num_buckets: int,
    device: torch.device,
) -> FusedAlignScratch:
    """Return route-owned metadata whose count-zero invariant is self-restoring."""
    return FusedAlignScratch(
        counts=workspace.tensor(
            f"{prefix}:counts",
            (num_buckets,),
            dtype=torch.int32,
            device=device,
            zero_on_first_allocation=True,
        ),
        block_cumulative=workspace.tensor(
            f"{prefix}:block_cumulative",
            (num_buckets + 1,),
            dtype=torch.int32,
            device=device,
        ),
        cursor=workspace.tensor(
            f"{prefix}:cursor",
            (num_buckets,),
            dtype=torch.int32,
            device=device,
        ),
        bucket_end=workspace.tensor(
            f"{prefix}:bucket_end",
            (num_buckets,),
            dtype=torch.int32,
            device=device,
        ),
    )


def _aligned_pair_route(
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
    *,
    is_shared_outer: bool,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    workspace: MoeLoraWorkspace,
    scratch_prefix: str,
) -> RouteView:
    """Build an aligned route with caller-owned fused metadata.

    The scan kernel restores ``counts`` to zero after its final read, so the
    workspace initializes that tensor only when storage is first allocated.
    Supplying the retained scalar directly also avoids a follow-up device copy.
    The dispatch is checked before allocation, so the JIT path keeps
    its own metadata without paying for unused fused scratch.
    """
    lora_experts_per_adapter = 1 if is_shared_outer else num_local_experts
    padded_count = None
    scratch = None
    if uses_fused_align(
        topk_ids,
        num_virtual_experts=lora_experts_per_adapter * max_loras,
    ):
        padded_count = workspace.tensor(
            f"{scratch_prefix}:padded_pairs",
            (1,),
            dtype=torch.int32,
            device=topk_ids.device,
        )
        scratch = _fused_align_scratch(
            workspace,
            prefix=scratch_prefix,
            num_buckets=lora_experts_per_adapter * max_loras + 1,
            device=topk_ids.device,
        )
    # These two move together, and validate_shared_outer enforces it: a shared
    # adapter has exactly ONE LoRA expert, so the validity test would end in
    # ``lora_expert_id < 1`` -- always true. Passing the local expert count
    # restores the bound, without which a routed expert this rank does not own
    # would be accepted as a valid pair.
    return build_virtual_expert_routing(
        topk_ids,
        token_slots,
        lora_experts_per_adapter=lora_experts_per_adapter,
        shared_outer_local_expert_count=num_local_experts if is_shared_outer else None,
        max_loras=max_loras,
        block_size=block_size,
        view=ROUTE_ALIGNED,
        num_pairs_post_padded_out=padded_count,
        fused_align_scratch=scratch,
    )


def build_routes(
    plan: MoeLoraExecutionPlan,
    *,
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
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
            token_slots,
            lora_experts_per_adapter=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            view=ROUTE_RAW,
        )
    if RouteRequirement.RAW_SHARED_OUTER in requirements:
        # One LoRA expert per adapter, with the local expert count restoring
        # the ownership bound (see _aligned_pair_route).
        values["raw_shared_outer"] = build_virtual_expert_routing(
            topk_ids,
            token_slots,
            lora_experts_per_adapter=1,
            shared_outer_local_expert_count=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            view=ROUTE_RAW,
        )

    need_per_expert = RouteRequirement.ALIGNED_PER_EXPERT in requirements
    need_shared = RouteRequirement.ALIGNED_SHARED_OUTER in requirements
    if plan.route_builder is RouteBuilderFamily.JOINT_SHARED_OUTER:
        per_expert, shared = build_joint_shared_routes(
            topk_ids,
            token_slots,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            workspace=workspace,
        )
        values["aligned_per_expert"] = per_expert
        values["aligned_shared_outer"] = shared
    else:
        if need_per_expert:
            values["aligned_per_expert"] = _aligned_pair_route(
                topk_ids,
                token_slots,
                is_shared_outer=False,
                num_local_experts=num_local_experts,
                max_loras=max_loras,
                block_size=block_size,
                workspace=workspace,
                scratch_prefix="route:aligned_per_expert",
            )
        if need_shared:
            values["aligned_shared_outer"] = _aligned_pair_route(
                topk_ids,
                token_slots,
                is_shared_outer=True,
                num_local_experts=num_local_experts,
                max_loras=max_loras,
                block_size=block_size,
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
        shared_token_slots = workspace.tensor(
            "route:shared_token_slots",
            token_slots.shape,
            dtype=token_slots.dtype,
            device=token_slots.device,
        )
        torch.where(
            has_local_pair,
            token_slots,
            torch.full_like(token_slots, -1),
            out=shared_token_slots,
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
        values["shared_token"] = _aligned_pair_route(
            token_experts,
            shared_token_slots,
            is_shared_outer=True,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            workspace=workspace,
            scratch_prefix="route:shared_token",
        )

    return MoeLoraRoutes(**values)
