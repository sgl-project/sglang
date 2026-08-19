from __future__ import annotations

import torch
import triton

from sglang.kernels.ops.moe.virtual_experts import (
    _align_block_size_jit,
)
from sglang.srt.lora.moe.route_kernels import _build_virtual_topk_ids_kernel
from sglang.srt.lora.moe.route_view import RouteView, RouteViewKind
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

# Smallest bucket count (V) and pair count (P) at which the fused align
# builder beats the JIT one; either alone is enough to switch.
FUSED_ALIGN_MIN_VIRTUAL_EXPERTS = 8192
FUSED_ALIGN_MIN_PAIRS = 16384


def _build_virtual_topk_ids(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_local_experts: int,
    max_loras: int,
    is_shared_outer: bool = False,
) -> torch.Tensor:
    # The sole caller validated these same objects before branching.
    virtual_topk_ids = torch.empty_like(topk_ids)
    if topk_ids.numel() == 0:
        # An idle DP rank arrives with zero tokens; cdiv(0, block) is no grid.
        return virtual_topk_ids

    block_size = 1024  # pairs per program
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
        return RouteView(**common)

    num_virtual = lora_experts_per_adapter * max_loras
    if (
        num_virtual >= FUSED_ALIGN_MIN_VIRTUAL_EXPERTS
        or topk_ids.numel() >= FUSED_ALIGN_MIN_PAIRS
    ):
        from sglang.srt.lora.moe import aligned_route

        if workspace is None or tensor_prefix is None:
            raise ValueError(
                "the fused aligned builder needs workspace and tensor_prefix "
                f"(view={view.value}, virtual experts={num_virtual}, "
                f"pairs={topk_ids.numel()})"
            )
        per_expert, shared = aligned_route.build(
            topk_ids,
            token_lora_mapping,
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            workspace=workspace,
            tensor_prefix=tensor_prefix,
            need_per_expert=not is_shared_outer,
            need_shared=is_shared_outer,
        )
        return shared if is_shared_outer else per_expert

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
    return RouteView(
        **common,
        maybe_sorted_pair_ids=sorted_pair_ids,
        maybe_block_virtual_expert_ids=block_virtual_expert_ids,
        maybe_num_pairs_post_padded=num_pairs_post_padded,
    )
