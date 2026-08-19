from __future__ import annotations

from enum import Enum

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.moe.virtual_experts import (
    _align_block_size_jit,
)
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

# Smallest bucket count (V) and pair count (P) at which the fused align
# builder beats the JIT one; either alone is enough to switch.
FUSED_ALIGN_MIN_VIRTUAL_EXPERTS = 8192
FUSED_ALIGN_MIN_PAIRS = 16384


class RouteViewKind(str, Enum):
    RAW = "raw"
    ALIGNED = "aligned"


class RouteView(msgspec.Struct, frozen=True, kw_only=True):
    view: RouteViewKind
    block_size: int
    topk_ids: torch.Tensor
    token_lora_mapping: torch.Tensor
    num_local_experts: int
    is_shared_outer: bool
    max_loras: int
    # Present only for `aligned`.
    maybe_sorted_pair_ids: torch.Tensor | None = None
    maybe_block_virtual_expert_ids: torch.Tensor | None = None
    maybe_num_pairs_post_padded: torch.Tensor | None = None

    @property
    def lora_experts_per_adapter(self) -> int:
        return 1 if self.is_shared_outer else self.num_local_experts

    @property
    def num_virtual_experts(self) -> int:
        return self.lora_experts_per_adapter * self.max_loras

    def _require(self, value, field: str, needed: RouteViewKind):
        if value is None:
            raise ValueError(
                f"route view {self.view.value!r} did not build {field}; the "
                f"consumer must request view {needed.value!r} or derive it inline"
            )
        return value

    @property
    def sorted_pair_ids(self) -> torch.Tensor:
        return self._require(
            self.maybe_sorted_pair_ids, "sorted_pair_ids", RouteViewKind.ALIGNED
        )

    @property
    def block_virtual_expert_ids(self) -> torch.Tensor:
        return self._require(
            self.maybe_block_virtual_expert_ids,
            "block_virtual_expert_ids",
            RouteViewKind.ALIGNED,
        )

    @property
    def num_pairs_post_padded(self) -> torch.Tensor:
        return self._require(
            self.maybe_num_pairs_post_padded,
            "num_pairs_post_padded",
            RouteViewKind.ALIGNED,
        )


@triton.jit
def virtual_expert_ids_inline(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    pair_ids,
    pair_mask,
    routed_expert_id_bound,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
):
    token_ids = pair_ids // TOP_K
    adapter_ids = tl.load(
        token_lora_mapping_ptr + token_ids,
        mask=pair_mask,
        other=-1,
    ).to(tl.int32)
    routed_expert_ids = tl.load(
        topk_ids_ptr + pair_ids,
        mask=pair_mask,
        other=-1,
    ).to(tl.int32)
    if SHARED_OUTER:
        in_range = (routed_expert_ids >= 0) & (
            routed_expert_ids < routed_expert_id_bound
        )
        lora_expert_ids = tl.where(in_range, 0, -1)
    else:
        lora_expert_ids = routed_expert_ids

    valid = (
        (adapter_ids >= 0)
        & (adapter_ids < MAX_LORAS)
        & (lora_expert_ids >= 0)
        & (lora_expert_ids < LORA_EXPERTS_PER_ADAPTER)
    )
    return tl.where(valid, adapter_ids * LORA_EXPERTS_PER_ADAPTER + lora_expert_ids, -1)


@triton.jit
def _build_virtual_topk_ids_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    virtual_topk_ids_ptr,
    num_pairs,
    routed_expert_id_bound,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pair_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    pair_mask = pair_ids < num_pairs
    virtual_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_lora_mapping_ptr,
        pair_ids,
        pair_mask,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        SHARED_OUTER=SHARED_OUTER,
    )
    tl.store(virtual_topk_ids_ptr + pair_ids, virtual_ids, mask=pair_mask)


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
    scratch_prefix: str | None = None,
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
        from sglang.srt.lora.moe.fused_align import fused_align_block_size

        if workspace is None or scratch_prefix is None:
            raise ValueError(
                "the fused aligned builder needs workspace and scratch_prefix "
                f"(view={view.value}, virtual experts={num_virtual}, "
                f"pairs={topk_ids.numel()})"
            )
        sorted_pair_ids, block_virtual_expert_ids, num_pairs_post_padded = (
            fused_align_block_size(
                topk_ids,
                token_lora_mapping,
                num_local_experts=num_local_experts,
                is_shared_outer=is_shared_outer,
                max_loras=max_loras,
                block_size=block_size,
                capacity=_routing_capacity(topk_ids.numel(), block_size, num_virtual),
                workspace=workspace,
                scratch_prefix=scratch_prefix,
            )
        )
        return RouteView(
            **common,
            maybe_sorted_pair_ids=sorted_pair_ids,
            maybe_block_virtual_expert_ids=block_virtual_expert_ids,
            maybe_num_pairs_post_padded=num_pairs_post_padded,
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
    return RouteView(
        **common,
        maybe_sorted_pair_ids=sorted_pair_ids,
        maybe_block_virtual_expert_ids=block_virtual_expert_ids,
        maybe_num_pairs_post_padded=num_pairs_post_padded,
    )
