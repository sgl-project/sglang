"""Build raw or block-aligned routes keyed by adapter and LoRA expert."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton

from sglang.kernels.ops.moe.virtual_experts import (
    _align_block_size_jit,
)
from sglang.srt.lora.moe.execution_plan import (
    MoeLoraExecutionPlan,
    RouteBuilderFamily,
    RouteRequirement,
)
from sglang.srt.lora.moe.kernels.routing import (
    _build_virtual_topk_ids_kernel,
    _hist_kernel,
    _place_kernel,
    _scan_kernel,
    _segment_token_route_kernel,
)
from sglang.srt.lora.moe.route_view import RouteView, RouteViewKind
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

# Use Triton beyond these crossover points; CUDA alignment also caps at 8192 buckets.
FUSED_ALIGN_MIN_VIRTUAL_EXPERTS = 8192
FUSED_ALIGN_MIN_PAIRS = 16384

# Tuning evidence: configs/README.md.
HIST_BLOCK = 512
HIST_WARPS = 8
EXPAND_BLOCK = 128
EXPAND_WARPS = 4
SCAN_CHUNK = 2048
SCAN_WARPS = 4

# Use in-block histograms to reduce global atomics within these limits.
COUNT_MAX_BINS = 512
COUNT_MIN_PAIRS = 16384
CLAIM_MIN_PAIRS_PER_BUCKET = 12288


def count_bins(num_buckets: int, num_pairs: int) -> int:
    # 0 = one atomic per pair; otherwise in-block counting over 2^k bins.
    if num_buckets >= COUNT_MAX_BINS or num_pairs < COUNT_MIN_PAIRS:
        return 0
    return 1 << num_buckets.bit_length()  # the extra bin holds the masked-off lanes


def _routing_capacity(
    num_pairs: int,
    block_size: int,
    num_virtual_experts: int,
) -> int:
    if num_pairs == 0:
        return 0
    max_nonempty_buckets = min(num_pairs, num_virtual_experts + 1)
    upper_bound = num_pairs + max_nonempty_buckets * (block_size - 1)
    return triton.cdiv(triton.cdiv(upper_bound, block_size) * block_size, 4) * 4


def _plan_scratch(
    workspace: MoeLoraWorkspace,
    *,
    prefix: str,
    num_buckets: int,
    capacity: int,
    block_size: int,
    device: torch.device,
) -> dict[str, object]:
    # The scan clears counts for the next replay.
    scratch: dict[str, object] = {
        "num_buckets": num_buckets,
        "capacity": capacity,
        "counts": workspace.tensor(
            f"{prefix}:counts",
            (num_buckets,),
            dtype=torch.int32,
            device=device,
            zero_on_first_allocation=True,
        ),
        "block_cumulative": workspace.tensor(
            f"{prefix}:block_cumulative",
            (num_buckets + 1,),
            dtype=torch.int32,
            device=device,
        ),
        "cursor": workspace.tensor(
            f"{prefix}:cursor", (num_buckets,), dtype=torch.int32, device=device
        ),
        "bucket_end": workspace.tensor(
            f"{prefix}:bucket_end", (num_buckets,), dtype=torch.int32, device=device
        ),
        "padded_pairs": workspace.tensor(
            f"{prefix}:padded_pairs", (1,), dtype=torch.int32, device=device
        ),
    }
    scratch["sorted"] = workspace.tensor(
        f"{prefix}:sorted", (capacity,), dtype=torch.int32, device=device
    )
    scratch["block_ids"] = workspace.tensor(
        f"{prefix}:block_ids",
        (capacity // block_size,),
        dtype=torch.int32,
        device=device,
    )
    return scratch


def _build_aligned(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    workspace: MoeLoraWorkspace,
    tensor_prefix: str,
    is_shared_outer: bool,
) -> RouteView:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    num_pairs = topk_ids.numel()
    name = "shared" if is_shared_outer else "per_expert"
    virtual = max_loras if is_shared_outer else num_local_experts * max_loras
    capacity = _routing_capacity(num_pairs, block_size, virtual)
    if virtual + 1 >= 2**31 or capacity >= 2**31:
        raise ValueError(
            f"aligned routes use int32 plan math: {name} needs {virtual + 1} "
            f"buckets and {capacity} slots, both must be < 2**31"
        )
    own = _plan_scratch(
        workspace,
        prefix=f"{tensor_prefix}:{name}",
        num_buckets=virtual + 1,
        capacity=capacity,
        block_size=block_size,
        device=topk_ids.device,
    )
    num_buckets = own["num_buckets"]
    bound = num_local_experts if is_shared_outer else 0
    experts_per_adapter = 1 if is_shared_outer else num_local_experts

    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    _hist_kernel[(triton.cdiv(max(num_pairs, 1), HIST_BLOCK),)](
        topk_ids,
        token_lora_mapping,
        own["counts"],
        num_pairs,
        bound,
        NUM_BUCKETS=num_buckets,
        LORA_EXPERTS_PER_ADAPTER=experts_per_adapter,
        MAX_LORAS=max_loras,
        TOP_K=topk_ids.shape[1],
        SHARED_OUTER=is_shared_outer,
        BLOCK=HIST_BLOCK,
        BINS=count_bins(num_buckets, num_pairs),
        USE_PDL=use_pdl,
        num_warps=HIST_WARPS,
    )
    _scan_kernel[(1,)](
        own["counts"],
        own["block_cumulative"],
        own["cursor"],
        own["bucket_end"],
        own["padded_pairs"],
        num_buckets,
        BLOCK_SIZE_M=block_size,
        CHUNK=SCAN_CHUNK,
        USE_PDL=use_pdl,
        num_warps=SCAN_WARPS,
        **pdl_kwargs,
    )
    num_blocks = own["capacity"] // block_size
    label_programs = triton.cdiv(max(num_blocks, 1), EXPAND_BLOCK)
    _place_kernel[(label_programs + triton.cdiv(max(num_pairs, 1), EXPAND_BLOCK),)](
        topk_ids,
        token_lora_mapping,
        own["cursor"],
        own["bucket_end"],
        own["block_cumulative"],
        own["sorted"],
        own["block_ids"],
        num_blocks,
        label_programs,
        num_pairs,
        bound,
        NUM_BUCKETS=num_buckets,
        NUM_VIRTUAL=num_buckets - 1,
        LORA_EXPERTS_PER_ADAPTER=experts_per_adapter,
        MAX_LORAS=max_loras,
        TOP_K=topk_ids.shape[1],
        SHARED_OUTER=is_shared_outer,
        BLOCK=EXPAND_BLOCK,
        BLOCK_SIZE_M=block_size,
        # Include the sentinel in the binary-search depth.
        SEARCH_STEPS=num_buckets.bit_length(),
        CLAIM_PER_BLOCK=num_pairs >= CLAIM_MIN_PAIRS_PER_BUCKET * num_buckets,
        USE_PDL=use_pdl,
        num_warps=EXPAND_WARPS,
        **pdl_kwargs,
    )
    return RouteView(
        view=RouteViewKind.ALIGNED,
        block_size=block_size,
        topk_ids=topk_ids,
        token_lora_mapping=token_lora_mapping,
        num_local_experts=num_local_experts,
        is_shared_outer=is_shared_outer,
        max_loras=max_loras,
        maybe_sorted_pair_ids=own["sorted"],
        maybe_block_virtual_expert_ids=own["block_ids"],
        maybe_num_pairs_post_padded=own["padded_pairs"],
    )


def _build_virtual_topk_ids(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_local_experts: int,
    max_loras: int,
    is_shared_outer: bool = False,
) -> torch.Tensor:
    virtual_topk_ids = torch.empty_like(topk_ids)
    if topk_ids.numel() == 0:
        return virtual_topk_ids

    block_size = 1024
    _build_virtual_topk_ids_kernel[(triton.cdiv(topk_ids.numel(), block_size),)](
        topk_ids,
        token_lora_mapping,
        virtual_topk_ids,
        topk_ids.numel(),
        num_local_experts,
        LORA_EXPERTS_PER_ADAPTER=1 if is_shared_outer else num_local_experts,
        MAX_LORAS=max_loras,
        TOP_K=topk_ids.shape[1],
        SHARED_OUTER=is_shared_outer,
        BLOCK_SIZE=block_size,
    )
    return virtual_topk_ids


def build_virtual_expert_routing(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    is_shared_outer: bool = False,
    view: RouteViewKind = RouteViewKind.ALIGNED,
    workspace: MoeLoraWorkspace | None = None,
    tensor_prefix: str | None = None,
) -> RouteView:
    if view not in RouteViewKind:
        raise ValueError(
            f"unknown route view {view!r}; expected one of "
            f"{tuple(kind.value for kind in RouteViewKind)}"
        )
    view = RouteViewKind(view)
    common = {
        "view": view,
        "block_size": block_size,
        "topk_ids": topk_ids,
        "token_lora_mapping": token_lora_mapping,
        "num_local_experts": num_local_experts,
        "is_shared_outer": is_shared_outer,
        "max_loras": max_loras,
    }
    lora_experts_per_adapter = 1 if is_shared_outer else num_local_experts
    if view is RouteViewKind.RAW:
        route = RouteView(**common)
        return route

    num_virtual = lora_experts_per_adapter * max_loras
    if (
        num_virtual >= FUSED_ALIGN_MIN_VIRTUAL_EXPERTS
        or topk_ids.numel() >= FUSED_ALIGN_MIN_PAIRS
    ):
        route = _build_aligned(
            topk_ids,
            token_lora_mapping,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            workspace=workspace,
            tensor_prefix=tensor_prefix,
            is_shared_outer=is_shared_outer,
        )
        return route

    virtual_topk_ids = _build_virtual_topk_ids(
        topk_ids,
        token_lora_mapping,
        num_local_experts,
        max_loras,
        is_shared_outer=is_shared_outer,
    )
    sorted_pair_ids, block_virtual_expert_ids, num_pairs_post_padded = (
        _align_block_size_jit(virtual_topk_ids, block_size, num_virtual)
    )
    route = RouteView(
        **common,
        maybe_sorted_pair_ids=sorted_pair_ids,
        maybe_block_virtual_expert_ids=block_virtual_expert_ids,
        maybe_num_pairs_post_padded=num_pairs_post_padded,
    )
    return route


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


def _build_segment_token_route(
    *,
    seg_indptr: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_tokens: int,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    workspace: MoeLoraWorkspace,
) -> RouteView:
    """Pad each request segment into blocks sharing one adapter slot."""
    num_segments = seg_indptr.shape[0] - 1
    # Each request adds at most block_size - 1 padding rows.
    capacity = triton.cdiv(num_tokens + num_segments * (block_size - 1), block_size)
    capacity = triton.cdiv(capacity * block_size, 4) * 4
    device = token_lora_mapping.device
    sorted_ids = workspace.tensor(
        "route:shared_token:sorted", (capacity,), dtype=torch.int32, device=device
    )
    block_ids = workspace.tensor(
        "route:shared_token:block_ids",
        (capacity // block_size,),
        dtype=torch.int32,
        device=device,
    )
    padded = workspace.tensor(
        "route:shared_token:padded_pairs", (1,), dtype=torch.int32, device=device
    )
    # Represent tokens as top-1 pairs with the shared LoRA expert 0.
    token_experts = workspace.tensor(
        "route:shared_token_experts",
        (num_tokens, 1),
        dtype=torch.int32,
        device=device,
        zero_on_first_allocation=True,
    )
    _segment_token_route_kernel[(1,)](
        seg_indptr,
        token_lora_mapping,
        sorted_ids,
        block_ids,
        padded,
        num_segments,
        num_tokens,
        capacity // block_size,
        BLOCK_SIZE_M=block_size,
        CHUNK=256,
        num_warps=4,
    )
    return RouteView(
        view=RouteViewKind.ALIGNED,
        block_size=block_size,
        topk_ids=token_experts,
        token_lora_mapping=token_lora_mapping,
        num_local_experts=num_local_experts,
        is_shared_outer=True,
        max_loras=max_loras,
        maybe_sorted_pair_ids=sorted_ids,
        maybe_block_virtual_expert_ids=block_ids,
        maybe_num_pairs_post_padded=padded,
    )


def build_routes(
    plan: MoeLoraExecutionPlan,
    *,
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    seg_indptr: torch.Tensor,
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

    if plan.route_builder is RouteBuilderFamily.PARALLEL_SHARED_OUTER:

        def _build_per_expert() -> RouteView:
            return build_virtual_expert_routing(
                topk_ids,
                token_lora_mapping,
                num_local_experts=num_local_experts,
                max_loras=max_loras,
                block_size=block_size,
                view=RouteViewKind.ALIGNED,
                workspace=workspace,
                tensor_prefix="route:aligned_per_expert",
            )

        def _build_shared() -> None:
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

        values["aligned_per_expert"] = workspace.run_parallel(
            name="route:parallel",
            device=topk_ids.device,
            compute=_build_per_expert,
            side=_build_shared,
        )
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
        values["shared_token"] = _build_segment_token_route(
            seg_indptr=seg_indptr,
            token_lora_mapping=token_lora_mapping,
            num_tokens=topk_ids.shape[0],
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            workspace=workspace,
        )
    return MoeLoraRoutes(**values)
