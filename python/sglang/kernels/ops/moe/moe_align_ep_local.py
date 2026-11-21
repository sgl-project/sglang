"""One-launch MoE block alignment of one expert-parallel rank's local routes."""

from typing import Tuple

import torch
import triton
import triton.language as tl

# The kernel holds [local experts, routes] int32 tiles in registers. Measured on
# H200 (sm90): 16 x 1024 and 32 x 1024 used 117 and 210 registers, no spills;
# 64 x 512 spilled 20 and 128 x 256 spilled 306, both at 255 registers.
_MAX_LOCAL_EXPERTS_PAD = 32
_MAX_ROUTES = 1024


@triton.jit(do_not_specialize=["num_routes"])
def _moe_align_ep_local_kernel(
    topk_ids_ptr,
    expert_map_ptr,
    local_ids_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    num_post_pad_ptr,
    num_routes,
    num_global_experts,
    E: tl.constexpr,
    E_PAD: tl.constexpr,
    R_PAD: tl.constexpr,
    B: tl.constexpr,
    B_PAD: tl.constexpr,
    NB_PAD: tl.constexpr,
):
    r = tl.arange(0, R_PAD)
    valid = r < num_routes
    g = tl.load(topk_ids_ptr + r, mask=valid, other=-1)
    in_range = valid & (g >= 0) & (g < num_global_experts)
    local = tl.load(expert_map_ptr + g, mask=in_range, other=-1).to(tl.int32)
    local = tl.where((local >= 0) & (local < E), local, -1)
    tl.store(local_ids_ptr + r, local, mask=valid)

    ex = tl.arange(0, E_PAD)
    hit = (local[None, :] == ex[:, None]).to(tl.int32)
    count = tl.sum(hit, axis=1)
    padded = (count + (B - 1)) // B * B
    end = tl.cumsum(padded, axis=0)
    start = end - padded
    rank = tl.cumsum(hit, axis=1) - hit + start[:, None]
    # A route hits at most one expert row, so its column sum is its slot in sorted_ids.
    tl.store(sorted_ids_ptr + tl.sum(hit * rank, axis=0), r, mask=local >= 0)

    lane = tl.arange(0, B_PAD)
    pad = (start + count)[:, None] + lane[None, :]
    pad_mask = (lane[None, :] < B) & (pad < end[:, None])
    tl.store(
        sorted_ids_ptr + pad,
        tl.zeros([E_PAD, B_PAD], dtype=tl.int32) + num_routes,
        mask=pad_mask,
    )

    total = tl.sum(padded, axis=0)
    b = tl.arange(0, NB_PAD)
    # Block b belongs to the first expert whose blocks end after b;
    # empty experts end where the previous one does.
    block_expert = tl.sum(((end // B)[None, :] <= b[:, None]).to(tl.int32), axis=1)
    tl.store(expert_ids_ptr + b, block_expert, mask=b * B < total)
    tl.store(num_post_pad_ptr, total)


def ep_local_align_supported(*, num_routes: int, num_local_experts: int) -> bool:
    """Whether moe_align_ep_local takes a step of num_routes routes."""
    return (
        num_routes <= _MAX_ROUTES
        and triton.next_power_of_2(num_local_experts) <= _MAX_LOCAL_EXPERTS_PAD
    )


def moe_align_ep_local(
    *,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    num_local_experts: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align this rank's top-k routes into local expert blocks in one launch.

    Replaces mapping topk_ids through expert_map followed by
    moe_align_block_size(local_ids, block_size, num_local_experts,
    ignore_invalid_expert=True), with the same buffer sizes. topk_ids is
    [num_tokens, top_k] int32 or int64 global expert ids, and expert_map is
    [num_global_experts] int32, global id to local id or -1. Returns int32
    tensors (local_ids, sorted_token_ids, expert_ids, num_tokens_post_padded):
    local_ids has topk_ids' shape and holds each route's local expert or -1;
    sorted_token_ids groups route indices by local expert in index order, each
    expert padded to a multiple of block_size with the sentinel
    topk_ids.numel(); expert_ids holds one local expert per block, ascending;
    and num_tokens_post_padded is the [1] padded total.

    Covers the steps ep_local_align_supported admits (one CTA) and raises on
    others. Deviations from the pair it replaces: global ids outside
    [0, num_global_experts) and map values outside [0, num_local_experts) are
    dropped; routes keep index order inside an expert, which
    moe_align_block_size sets by atomics; and sorted_token_ids and expert_ids
    are left unwritten past the padded total, which a GEMM that walks
    num_tokens_post_padded never reads.
    """
    numel = topk_ids.numel()
    num_global_experts = expert_map.numel()
    if not ep_local_align_supported(
        num_routes=numel, num_local_experts=num_local_experts
    ):
        raise ValueError(
            f"moe_align_ep_local does not cover {numel} routes over "
            f"{num_local_experts} local experts; see ep_local_align_supported"
        )
    # moe_align_block_size's sizing over the local experts.
    if numel < num_local_experts + 1:
        max_padded = numel * block_size
    else:
        max_padded = numel + (num_local_experts + 1) * (block_size - 1)
    max_padded = (max_padded + 3) // 4 * 4
    # Every block holds a route, and each expert adds at most one partial block.
    num_blocks_bound = min(numel, triton.cdiv(numel, block_size) + num_local_experts)
    device = topk_ids.device
    local_ids = torch.empty(topk_ids.shape, dtype=torch.int32, device=device)
    sorted_ids = torch.empty((max_padded,), dtype=torch.int32, device=device)
    expert_ids = torch.empty(
        (triton.cdiv(max_padded, block_size),), dtype=torch.int32, device=device
    )
    num_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    routes_pad = max(triton.next_power_of_2(numel), 16)
    _moe_align_ep_local_kernel[(1,)](
        topk_ids.contiguous(),
        expert_map,
        local_ids,
        sorted_ids,
        expert_ids,
        num_post_pad,
        numel,
        num_global_experts,
        E=num_local_experts,
        E_PAD=triton.next_power_of_2(num_local_experts),
        R_PAD=routes_pad,
        B=block_size,
        B_PAD=triton.next_power_of_2(block_size),
        NB_PAD=triton.next_power_of_2(max(num_blocks_bound, 2)),
        # 8 warps keep the 32 x 1024 tile spill-free on sm90; the 512-route
        # switch point is arbitrary.
        num_warps=8 if routes_pad >= 512 else 4,
    )
    return local_ids, sorted_ids, expert_ids, num_post_pad


__all__ = ["ep_local_align_supported", "moe_align_ep_local"]
