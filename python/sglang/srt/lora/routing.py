"""Shared route builders for dense and MoE LoRA."""

from __future__ import annotations

import torch
import triton

from sglang.kernels.ops.moe.virtual_experts import (
    _align_block_size_jit,
)
from sglang.srt.lora.kernels.routing import (
    build_route_group_ids_kernel,
    route_histogram_kernel,
    route_place_kernel,
    route_scan_kernel,
    segment_token_route_kernel,
)
from sglang.srt.lora.route_view import RouteView, RouteViewKind
from sglang.srt.lora.workspace import LoraWorkspace

# Use Triton beyond these crossover points; CUDA alignment also caps at 8192 buckets.
_FUSED_ALIGN_MIN_GROUPS = 8192
_FUSED_ALIGN_MIN_PAIRS = 16384

# Tuning evidence: moe/configs/README.md.
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


def _histogram_bins(num_buckets: int, num_pairs: int) -> int:
    # 0 = one atomic per pair; otherwise in-block counting over 2^k bins.
    if num_buckets >= COUNT_MAX_BINS or num_pairs < COUNT_MIN_PAIRS:
        return 0
    return 1 << num_buckets.bit_length()  # the extra bin holds the masked-off lanes


def aligned_route_capacity(
    num_pairs: int,
    block_size: int,
    num_groups: int,
) -> int:
    """Bound padded pairs, including the invalid-ID bucket."""
    if num_pairs == 0:
        return 0
    max_nonempty_buckets = min(num_pairs, num_groups + 1)
    upper_bound = num_pairs + max_nonempty_buckets * (block_size - 1)
    return triton.cdiv(triton.cdiv(upper_bound, block_size) * block_size, 4) * 4


def _aligned_route_scratch(
    workspace: LoraWorkspace,
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


def _build_aligned_route(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    workspace: LoraWorkspace,
    tensor_prefix: str,
    is_shared_outer: bool,
) -> RouteView:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    num_pairs = topk_ids.numel()
    name = "shared" if is_shared_outer else "per_expert"
    num_groups = max_loras if is_shared_outer else num_local_experts * max_loras
    capacity = aligned_route_capacity(num_pairs, block_size, num_groups)
    if num_groups + 1 >= 2**31 or capacity >= 2**31:
        raise ValueError(
            f"aligned routes use int32 plan math: {name} needs {num_groups + 1} "
            f"buckets and {capacity} slots, both must be < 2**31"
        )
    own = _aligned_route_scratch(
        workspace,
        prefix=f"{tensor_prefix}:{name}",
        num_buckets=num_groups + 1,
        capacity=capacity,
        block_size=block_size,
        device=topk_ids.device,
    )
    num_buckets = own["num_buckets"]
    bound = num_local_experts if is_shared_outer else 0
    experts_per_adapter = 1 if is_shared_outer else num_local_experts

    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    route_histogram_kernel[(triton.cdiv(max(num_pairs, 1), HIST_BLOCK),)](
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
        BINS=_histogram_bins(num_buckets, num_pairs),
        USE_PDL=use_pdl,
        num_warps=HIST_WARPS,
    )
    route_scan_kernel[(1,)](
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
    route_place_kernel[
        (label_programs + triton.cdiv(max(num_pairs, 1), EXPAND_BLOCK),)
    ](
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
        NUM_GROUPS=num_buckets - 1,
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
        maybe_block_group_ids=own["block_ids"],
        maybe_num_pairs_post_padded=own["padded_pairs"],
    )


def _build_route_group_ids(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_local_experts: int,
    max_loras: int,
    is_shared_outer: bool = False,
) -> torch.Tensor:
    group_ids = torch.empty_like(topk_ids)
    if topk_ids.numel() == 0:
        return group_ids

    block_size = 1024
    build_route_group_ids_kernel[(triton.cdiv(topk_ids.numel(), block_size),)](
        topk_ids,
        token_lora_mapping,
        group_ids,
        topk_ids.numel(),
        num_local_experts,
        LORA_EXPERTS_PER_ADAPTER=1 if is_shared_outer else num_local_experts,
        MAX_LORAS=max_loras,
        TOP_K=topk_ids.shape[1],
        SHARED_OUTER=is_shared_outer,
        BLOCK_SIZE=block_size,
    )
    return group_ids


def build_group_route(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    is_shared_outer: bool = False,
    view: RouteViewKind = RouteViewKind.ALIGNED,
    workspace: LoraWorkspace | None = None,
    tensor_prefix: str | None = None,
) -> RouteView:
    """Build raw or group-aligned routes."""
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

    num_groups = lora_experts_per_adapter * max_loras
    if (
        num_groups >= _FUSED_ALIGN_MIN_GROUPS
        or topk_ids.numel() >= _FUSED_ALIGN_MIN_PAIRS
    ):
        route = _build_aligned_route(
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

    group_ids = _build_route_group_ids(
        topk_ids,
        token_lora_mapping,
        num_local_experts,
        max_loras,
        is_shared_outer=is_shared_outer,
    )
    sorted_pair_ids, block_group_ids, num_pairs_post_padded = _align_block_size_jit(
        group_ids, block_size, num_groups
    )
    route = RouteView(
        **common,
        maybe_sorted_pair_ids=sorted_pair_ids,
        maybe_block_group_ids=block_group_ids,
        maybe_num_pairs_post_padded=num_pairs_post_padded,
    )
    return route


def build_segmented_token_route(
    *,
    seg_indptr: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_tokens: int,
    num_local_experts: int,
    max_loras: int,
    block_size: int,
    workspace: LoraWorkspace,
) -> RouteView:
    """Pad each contiguous, single-adapter request into aligned token blocks."""
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
    segment_token_route_kernel[(1,)](
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
        maybe_block_group_ids=block_ids,
        maybe_num_pairs_post_padded=padded,
    )
