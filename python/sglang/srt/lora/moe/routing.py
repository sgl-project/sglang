"""Turn a forward's token/expert pairs into the routes a plan's kernels read.

A route groups those pairs -- by virtual expert, or by adapter alone for shared
outer factors -- either raw or sorted into whole blocks.
"""

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
from sglang.srt.lora.moe.route_kernels import (
    _build_virtual_topk_ids_kernel,
    _hist_kernel,
    _place_kernel,
    _scan_kernel,
)
from sglang.srt.lora.moe.route_view import RouteView, RouteViewKind
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

# Smallest bucket count and pair count at which our three kernels beat the shared
# CUDA align; the bucket count is also that kernel's own hard ceiling.
FUSED_ALIGN_MIN_VIRTUAL_EXPERTS = 8192
FUSED_ALIGN_MIN_PAIRS = 16384

# Launch tiles; see configs/README.md before changing them.
HIST_BLOCK = 512
HIST_WARPS = 8
EXPAND_BLOCK = 128
EXPAND_WARPS = 4
SCAN_CHUNK = 2048
SCAN_WARPS = 4

# Bin ceiling and smallest pair counts at which counting a block's pairs beats
# one atomic per pair.
COUNT_MAX_BINS = 512
COUNT_MIN_PAIRS = 16384
CLAIM_MIN_PAIRS_PER_BUCKET = 12288


def count_bins(num_buckets: int, num_pairs: int) -> int:
    """Bins for counting inside a block, or 0 to add one pair at a time."""
    if num_buckets >= COUNT_MAX_BINS or num_pairs < COUNT_MIN_PAIRS:
        return 0
    return 1 << num_buckets.bit_length()  # one spare bin, for masked-off lanes


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
    """Route-owned scratch; counts are zeroed once because the scan restores that."""
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
    need_per_expert: bool,
    need_shared: bool,
) -> tuple[RouteView | None, RouteView | None]:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    num_pairs = topk_ids.numel()
    scratch: dict[str, dict[str, object]] = {}
    for name, virtual, wanted in (
        ("per_expert", num_local_experts * max_loras, need_per_expert),
        ("shared", max_loras, need_shared),
    ):
        if not wanted:
            continue
        capacity = _routing_capacity(num_pairs, block_size, virtual)
        if virtual + 1 >= 2**31 or capacity >= 2**31:
            raise ValueError(
                f"aligned routes use int32 plan math: {name} needs {virtual + 1} "
                f"buckets and {capacity} slots, both must be < 2**31"
            )
        scratch[name] = _plan_scratch(
            workspace,
            prefix=f"{tensor_prefix}:{name}",
            num_buckets=virtual + 1,
            capacity=capacity,
            block_size=block_size,
            device=topk_ids.device,
        )
    # An unbuilt route's slot mirrors the built one; its branches are compiled
    # out, so those pointers are never read.
    per_expert = scratch.get("per_expert") or scratch["shared"]
    shared = scratch.get("shared") or scratch["per_expert"]
    pe_buckets = per_expert["num_buckets"]
    sh_buckets = shared["num_buckets"]

    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    num_pairs = topk_ids.numel()
    pe_buckets = per_expert["num_buckets"]
    sh_buckets = shared["num_buckets"]
    shape = dict(
        NEED_PER_EXPERT=need_per_expert,
        NEED_SHARED=need_shared,
        NUM_PER_EXPERT_BUCKETS=pe_buckets,
        NUM_SHARED_BUCKETS=sh_buckets,
        E_LOCAL=num_local_experts,
        MAX_LORAS=max_loras,
        TOP_K=topk_ids.shape[1],
        USE_PDL=use_pdl,
    )

    _hist_kernel[(triton.cdiv(max(num_pairs, 1), HIST_BLOCK),)](
        topk_ids,
        token_lora_mapping,
        per_expert["counts"],
        shared["counts"],
        num_pairs,
        num_local_experts,
        BLOCK=HIST_BLOCK,
        PER_EXPERT_BINS=count_bins(pe_buckets, num_pairs),
        SHARED_BINS=count_bins(sh_buckets, num_pairs),
        num_warps=HIST_WARPS,
        **shape,
    )
    _scan_kernel[(int(need_per_expert) + int(need_shared),)](
        per_expert["counts"],
        per_expert["block_cumulative"],
        per_expert["cursor"],
        per_expert["bucket_end"],
        per_expert["padded_pairs"],
        pe_buckets,
        shared["counts"],
        shared["block_cumulative"],
        shared["cursor"],
        shared["bucket_end"],
        shared["padded_pairs"],
        sh_buckets,
        NEED_PER_EXPERT=need_per_expert,
        NEED_SHARED=need_shared,
        BLOCK_SIZE_M=block_size,
        CHUNK=SCAN_CHUNK,
        USE_PDL=use_pdl,
        num_warps=SCAN_WARPS,
        **pdl_kwargs,
    )
    pe_blocks = per_expert["capacity"] // block_size
    sh_blocks = shared["capacity"] // block_size
    pe_labels = triton.cdiv(max(pe_blocks, 1), EXPAND_BLOCK) if need_per_expert else 0
    sh_labels = triton.cdiv(max(sh_blocks, 1), EXPAND_BLOCK) if need_shared else 0
    _place_kernel[
        (pe_labels + sh_labels + triton.cdiv(max(num_pairs, 1), EXPAND_BLOCK),)
    ](
        topk_ids,
        token_lora_mapping,
        per_expert["cursor"],
        per_expert["bucket_end"],
        per_expert["block_cumulative"],
        per_expert["sorted"],
        per_expert["block_ids"],
        pe_blocks,
        pe_labels,
        shared["cursor"],
        shared["bucket_end"],
        shared["block_cumulative"],
        shared["sorted"],
        shared["block_ids"],
        sh_blocks,
        sh_labels,
        num_pairs,
        num_local_experts,
        NUM_PER_EXPERT_VIRTUAL=pe_buckets - 1,
        NUM_SHARED_VIRTUAL=sh_buckets - 1,
        BLOCK=EXPAND_BLOCK,
        BLOCK_SIZE_M=block_size,
        # The search picks one of NUM_BUCKETS + 1 answers, so it needs
        # num_buckets.bit_length() steps -- one fewer and a sentinel reads as 0.
        PER_EXPERT_SEARCH_STEPS=pe_buckets.bit_length(),
        SHARED_SEARCH_STEPS=sh_buckets.bit_length(),
        PER_EXPERT_CLAIM_PER_BLOCK=num_pairs >= CLAIM_MIN_PAIRS_PER_BUCKET * pe_buckets,
        SHARED_CLAIM_PER_BLOCK=num_pairs >= CLAIM_MIN_PAIRS_PER_BUCKET * sh_buckets,
        num_warps=EXPAND_WARPS,
        **pdl_kwargs,
        **shape,
    )

    def route(name: str, *, is_shared_outer: bool) -> RouteView:
        own = scratch[name]
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

    return (
        route("per_expert", is_shared_outer=False) if need_per_expert else None,
        route("shared", is_shared_outer=True) if need_shared else None,
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
        # An idle DP rank arrives with zero tokens; cdiv(0, block) is no grid.
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
    is_joint_routing: bool = False,
    view: RouteViewKind = RouteViewKind.ALIGNED,
    workspace: MoeLoraWorkspace | None = None,
    tensor_prefix: str | None = None,
) -> tuple[RouteView | None, RouteView | None]:
    if view not in RouteViewKind:
        raise ValueError(
            f"unknown route view {view!r}; expected one of "
            f"{tuple(kind.value for kind in RouteViewKind)}"
        )
    view = RouteViewKind(view)
    if topk_ids.ndim != 2 or token_lora_mapping.shape != (topk_ids.shape[0],):
        raise ValueError("expected topk_ids [T,K] and token_lora_mapping [T]")
    if min(num_local_experts, max_loras, block_size) <= 0:
        raise ValueError(
            "local expert count, adapter capacity, and block size must all "
            "be positive"
        )
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
        return (None, route) if is_shared_outer else (route, None)

    num_virtual = lora_experts_per_adapter * max_loras
    if (
        is_joint_routing
        or num_virtual >= FUSED_ALIGN_MIN_VIRTUAL_EXPERTS
        or topk_ids.numel() >= FUSED_ALIGN_MIN_PAIRS
    ):
        if workspace is None or tensor_prefix is None:
            raise ValueError(
                "the fused aligned builder needs workspace and tensor_prefix "
                f"(view={view.value}, virtual experts={num_virtual}, "
                f"pairs={topk_ids.numel()})"
            )
        return _build_aligned(
            topk_ids,
            token_lora_mapping,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            workspace=workspace,
            tensor_prefix=tensor_prefix,
            need_per_expert=is_joint_routing or not is_shared_outer,
            need_shared=is_joint_routing or is_shared_outer,
        )

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
    return (None, route) if is_shared_outer else (route, None)


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
        values["raw_per_expert"], _ = build_virtual_expert_routing(
            topk_ids,
            token_lora_mapping,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            view=RouteViewKind.RAW,
        )
    if RouteRequirement.RAW_SHARED_OUTER in requirements:
        _, values["raw_shared_outer"] = build_virtual_expert_routing(
            topk_ids,
            token_lora_mapping,
            num_local_experts=num_local_experts,
            is_shared_outer=True,
            max_loras=max_loras,
            block_size=block_size,
            view=RouteViewKind.RAW,
        )

    if plan.route_builder is RouteBuilderFamily.JOINT_SHARED_OUTER:
        per_expert, shared = build_virtual_expert_routing(
            topk_ids,
            token_lora_mapping,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            is_joint_routing=True,
            workspace=workspace,
            tensor_prefix="joint_route",
        )
        values["aligned_per_expert"] = per_expert
        values["aligned_shared_outer"] = shared
    else:
        if RouteRequirement.ALIGNED_PER_EXPERT in requirements:
            values["aligned_per_expert"], _ = build_virtual_expert_routing(
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
            _, values["aligned_shared_outer"] = build_virtual_expert_routing(
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
        _, values["shared_token"] = build_virtual_expert_routing(
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
