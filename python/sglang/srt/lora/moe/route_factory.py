"""Build exactly the route products consumed by one MoE-LoRA plan."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.lora.moe.execution_plan import (
    LoraAFamily,
    MoeLoraExecutionPlan,
    RouteBuilderFamily,
    RouteRequirement,
)
from sglang.srt.lora.moe.joint_routing import build_joint_shared_routes
from sglang.srt.lora.moe.routing import (
    ROUTE_ALIGNED,
    ROUTE_FUSED_IDS,
    ROUTE_RAW,
    FusedAlignScratch,
    RouteView,
    build_dual_granularity_aligned_routes,
    build_virtual_expert_routing,
    uses_fused_align,
)
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace


@dataclass(frozen=True, slots=True)
class MoeLoraRoutes:
    """The distinct row-domain views materialized for one forward."""

    raw_per_expert: RouteView | None = None
    raw_shared_outer: RouteView | None = None
    fused_per_expert: RouteView | None = None
    fused_shared_outer: RouteView | None = None
    aligned_per_expert: RouteView | None = None
    aligned_shared_outer: RouteView | None = None
    # Optional second per-expert plan used only by grouped gate/up-A when its
    # best M tile differs from the canonical route shared by the other sites.
    gate_a_aligned_per_expert: RouteView | None = None
    shared_token: RouteView | None = None

    def raw(self, is_shared_outer: bool) -> RouteView:
        route = self.raw_shared_outer if is_shared_outer else self.raw_per_expert
        if route is None:
            raise ValueError(
                f"the execution plan did not request raw {_ownership_name(is_shared_outer)}"
            )
        return route

    def aligned(self, is_shared_outer: bool) -> RouteView:
        route = (
            self.aligned_shared_outer if is_shared_outer else self.aligned_per_expert
        )
        if route is None:
            raise ValueError(
                f"the execution plan did not request aligned {_ownership_name(is_shared_outer)}"
            )
        return route

    def fused(self, is_shared_outer: bool) -> RouteView:
        route = self.fused_shared_outer if is_shared_outer else self.fused_per_expert
        if route is None:
            raise ValueError(
                f"the execution plan did not request fused IDs for {_ownership_name(is_shared_outer)}"
            )
        return route


def _ownership_name(is_shared_outer: bool) -> str:
    return "shared_outer" if is_shared_outer else "per_expert"


def _pair_route(
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
    *,
    is_shared_outer: bool,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    view: str,
    use_pdl: bool | None,
    num_pairs_post_padded_out: torch.Tensor | None = None,
    fused_align_scratch: FusedAlignScratch | None = None,
) -> RouteView:
    if not is_shared_outer:
        return build_virtual_expert_routing(
            topk_ids,
            token_slots,
            lora_experts_per_adapter=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            view=view,
            use_pdl=use_pdl,
            num_pairs_post_padded_out=num_pairs_post_padded_out,
            fused_align_scratch=fused_align_scratch,
        )
    return build_virtual_expert_routing(
        topk_ids,
        token_slots,
        lora_experts_per_adapter=1,
        max_loras=max_loras,
        block_size=block_size,
        shared_outer_local_expert_count=num_local_experts,
        view=view,
        use_pdl=use_pdl,
        num_pairs_post_padded_out=num_pairs_post_padded_out,
        fused_align_scratch=fused_align_scratch,
    )


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
    use_pdl: bool | None,
    workspace: MoeLoraWorkspace,
    scratch_prefix: str,
) -> RouteView:
    """Build an aligned route with caller-owned fused metadata.

    The scan kernel restores ``counts`` to zero after its final read, so the
    workspace initializes that tensor only when storage is first allocated.
    Supplying the retained scalar directly also avoids a follow-up device copy.
    The canonical dispatch is checked before allocation, so the JIT path keeps
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
    return _pair_route(
        topk_ids,
        token_slots,
        is_shared_outer=is_shared_outer,
        num_local_experts=num_local_experts,
        max_loras=max_loras,
        block_size=block_size,
        view=ROUTE_ALIGNED,
        use_pdl=use_pdl,
        num_pairs_post_padded_out=padded_count,
        fused_align_scratch=scratch,
    )


def _dual_granularity_aligned_routes(
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
    *,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    gate_a_block_size: int,
    use_pdl: bool | None,
    workspace: MoeLoraWorkspace,
) -> tuple[RouteView, RouteView]:
    """Build the canonical and gate-A per-expert aligned routes in ONE pass.

    Both views cover the same ``(topk_ids, token_slots)`` pairs with the same
    per-expert key; only the M granularity differs, so the fused dual builder
    replaces two full hist -> scan -> expand triples with one.  Scratch and
    padded-count scalars live under the SAME workspace names the standalone
    builds use, so alternating between this pass and the single-route path
    (e.g. across capture buckets) reuses storage and preserves each counts
    tensor's zero invariant — every scan restores it regardless of which
    builder ran last.
    """
    device = topk_ids.device
    num_buckets = num_local_experts * max_loras + 1
    padded_counts: list[torch.Tensor] = []
    scratches: list[FusedAlignScratch] = []
    for prefix in ("route:aligned_per_expert", "route:gate_a_aligned_per_expert"):
        padded_counts.append(
            workspace.tensor(
                f"{prefix}:padded_pairs",
                (1,),
                dtype=torch.int32,
                device=device,
            )
        )
        scratches.append(
            _fused_align_scratch(
                workspace,
                prefix=prefix,
                num_buckets=num_buckets,
                device=device,
            )
        )
    return build_dual_granularity_aligned_routes(
        topk_ids,
        token_slots,
        lora_experts_per_adapter=num_local_experts,
        max_loras=max_loras,
        block_sizes=(block_size, gate_a_block_size),
        num_pairs_post_padded_outs=(padded_counts[0], padded_counts[1]),
        scratches=(scratches[0], scratches[1]),
        use_pdl=use_pdl,
    )


def build_routes(
    plan: MoeLoraExecutionPlan,
    *,
    topk_ids: torch.Tensor,
    token_slots: torch.Tensor,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    gate_a_block_size: int | None = None,
    workspace: MoeLoraWorkspace,
) -> MoeLoraRoutes:
    """Construct the exact route bundle declared by ``plan``.

    Raw views are descriptor-only and therefore both ownership forms are
    cheap.  Aligned views launch their builders.  The shared token plan uses a
    synthetic top-k-one expert column because gate/up-A is adapter-owned; the
    original pair route remains authoritative for every B consumer.
    """
    requirements = plan.route_requirements()
    gate_a_block_size = (
        block_size if gate_a_block_size is None else int(gate_a_block_size)
    )
    separate_gate_a_route = gate_a_block_size != block_size
    if separate_gate_a_route and not (
        plan.gate_a.family is LoraAFamily.GROUPED
        and not plan.gate_a.is_shared_outer
        and RouteRequirement.ALIGNED_PER_EXPERT in plan.gate_a.route_requirements()
    ):
        raise ValueError(
            "a separate gate-A route is qualified only for grouped, "
            "per-expert gate/up-A"
        )
    # Route PDL is always on where the architecture supports it: measured
    # +0.5..+2.0% decode on every model and GPU, prefill within +-1.3%
    # (2026-08 twins); no per-plan knob.
    from sglang.kernels.jit.utils import is_arch_support_pdl

    use_pdl = is_arch_support_pdl()
    values: dict[str, object] = {}
    if RouteRequirement.RAW in requirements:
        values["raw_per_expert"] = _pair_route(
            topk_ids,
            token_slots,
            is_shared_outer=False,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            view=ROUTE_RAW,
            use_pdl=use_pdl,
        )
        values["raw_shared_outer"] = _pair_route(
            topk_ids,
            token_slots,
            is_shared_outer=True,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            view=ROUTE_RAW,
            use_pdl=use_pdl,
        )

    for requirement, fused_is_shared_outer, name in (
        (
            RouteRequirement.FUSED_PER_EXPERT,
            False,
            "fused_per_expert",
        ),
        (
            RouteRequirement.FUSED_SHARED_OUTER,
            True,
            "fused_shared_outer",
        ),
    ):
        if requirement in requirements:
            values[name] = _pair_route(
                topk_ids,
                token_slots,
                is_shared_outer=fused_is_shared_outer,
                num_local_experts=num_local_experts,
                max_loras=max_loras,
                block_size=block_size,
                view=ROUTE_FUSED_IDS,
                use_pdl=use_pdl,
            )

    need_per_expert = RouteRequirement.ALIGNED_PER_EXPERT in requirements
    need_shared = RouteRequirement.ALIGNED_SHARED_OUTER in requirements
    downstream_requirements: set[RouteRequirement] = set()
    for stage in (plan.gate_b, plan.down_a, plan.down_b):
        if stage is not None:
            downstream_requirements.update(stage.route_requirements())
    downstream_requirements.update(plan.middle.route_requirements())
    downstream_requirements.update(plan.finalize.route_requirements())
    gate_only_per_expert = (
        separate_gate_a_route
        and plan.route_builder is RouteBuilderFamily.STANDARD
        and RouteRequirement.ALIGNED_PER_EXPERT not in downstream_requirements
    )
    dual_granularity_fused = False
    if plan.route_builder is RouteBuilderFamily.JOINT_SHARED_OUTER:
        per_expert, shared = build_joint_shared_routes(
            topk_ids,
            token_slots,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            workspace=workspace,
            use_pdl=use_pdl,
        )
        values["aligned_per_expert"] = per_expert
        values["aligned_shared_outer"] = shared
    else:
        # Both the canonical and gate-A per-expert routes cover identical
        # pair data with identical keys; when both are retained and the shape
        # dispatches to the fused builder anyway, one dual-granularity pass
        # replaces the two standalone triples (6 route launches -> 3).  The
        # JIT small-shape regime keeps its measured standalone paths.
        dual_granularity_fused = (
            separate_gate_a_route
            and need_per_expert
            and not gate_only_per_expert
            and uses_fused_align(
                topk_ids,
                num_virtual_experts=num_local_experts * max_loras,
            )
        )
        if dual_granularity_fused:
            (
                values["aligned_per_expert"],
                values["gate_a_aligned_per_expert"],
            ) = _dual_granularity_aligned_routes(
                topk_ids,
                token_slots,
                num_local_experts=num_local_experts,
                max_loras=max_loras,
                block_size=block_size,
                gate_a_block_size=gate_a_block_size,
                use_pdl=use_pdl,
                workspace=workspace,
            )
        elif need_per_expert and not gate_only_per_expert:
            values["aligned_per_expert"] = _aligned_pair_route(
                topk_ids,
                token_slots,
                is_shared_outer=False,
                num_local_experts=num_local_experts,
                max_loras=max_loras,
                block_size=block_size,
                use_pdl=use_pdl,
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
                use_pdl=use_pdl,
                workspace=workspace,
                scratch_prefix="route:aligned_shared_outer",
            )

    if separate_gate_a_route and not dual_granularity_fused:
        values["gate_a_aligned_per_expert"] = _aligned_pair_route(
            topk_ids,
            token_slots,
            is_shared_outer=False,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=gate_a_block_size,
            use_pdl=(
                use_pdl if plan.route_builder is RouteBuilderFamily.STANDARD else False
            ),
            workspace=workspace,
            scratch_prefix="route:gate_a_aligned_per_expert",
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
            use_pdl=(
                use_pdl if plan.route_builder is RouteBuilderFamily.STANDARD else False
            ),
            workspace=workspace,
            scratch_prefix="route:shared_token",
        )

    return MoeLoraRoutes(**values)
