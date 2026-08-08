"""Backend dispatch for MoE align-block-size.

Picks between the AOT / JIT ``moe.moe_align_block_size_out`` kernels and the
pure-torch fallback, and owns the output buffers they write into. The public
entry point is ``sglang.kernels.ops.moe.moe_align_block_size``.
"""

from __future__ import annotations

from typing import Tuple

import torch

from sglang.kernels.registry import registry
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import KernelBackend, PlatformInfo
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.kernels.ops.moe.moe_align import V2_SMALL_NUMEL_LIMIT, v2_supported
else:
    V2_SMALL_NUMEL_LIMIT = 0
    v2_supported = lambda *_: False

# How wide each backend's expert scan reaches. The AOT kernel gives one thread
# per bucket out of a 1024-thread block and checks nothing, so it silently
# overruns past its limit; the JIT kernel's per-thread multi-expert path goes to
# 8192.
_MAX_BUCKETS = {
    KernelBackend.AOT: 1024,
    KernelBackend.JIT: 8192,
}

_VEC_SIZE = 4


# NOTE: int32 align4, typically required by backend implementation
def _align4(n: int) -> int:
    return (n + 3) & ~3


def _default_out_backend() -> KernelBackend:
    """JIT wherever its spec says it runs (CUDA only), AOT everywhere else.

    The platform rule lives in the JIT spec's ``capabilities``, not here, so the
    two cannot drift.
    """
    jit = registry.get_backend("moe.moe_align_block_size_out", KernelBackend.JIT)
    if jit.is_available(PlatformInfo.detect()):
        return KernelBackend.JIT
    return KernelBackend.AOT


_OUT_BACKEND = _default_out_backend()


@torch.compile(dynamic=True)
def align_block_size_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch align_block_size for num_experts > 1024, compiled via torch.compile.

    Fallback for platforms where the CUDA JIT kernel is unavailable (e.g. AMD/ROCm).

    Out-of-range topk_ids (negative sentinels left by EP dispatch, or virtual-
    expert IDs >= num_experts produced when those sentinels are combined with
    a per-adapter offset) are routed into a dedicated sentinel bucket. Without
    this, indexing ``padded_offsets[sorted_expert_ids]`` would wrap (-1) or
    OOB-read, and the bad expert ids would propagate into the downstream LoRA
    GEMM as real expert slots.
    """
    device = topk_ids.device
    flat_topk_ids = topk_ids.reshape(-1).to(torch.int64)
    num_total_tokens = flat_topk_ids.numel()

    sentinel = num_experts
    valid_mask = (flat_topk_ids >= 0) & (flat_topk_ids < num_experts)
    safe_topk_ids = torch.where(
        valid_mask,
        flat_topk_ids,
        torch.full_like(flat_topk_ids, sentinel),
    )

    bucket_count = num_experts + 1
    max_total_padded_tokens = (
        (num_total_tokens + bucket_count * (block_size - 1) + block_size - 1)
        // block_size
    ) * block_size
    max_num_blocks = max_total_padded_tokens // block_size

    sorted_token_ids = torch.full(
        (max_total_padded_tokens,),
        num_total_tokens,
        dtype=torch.int32,
        device=device,
    )
    expert_ids = torch.full(
        (max_num_blocks,),
        -1,
        dtype=torch.int32,
        device=device,
    )

    if num_total_tokens == 0:
        num_tokens_post_padded = torch.zeros((1,), dtype=torch.int32, device=device)
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    sorted_order = torch.argsort(safe_topk_ids)
    sorted_expert_ids = safe_topk_ids[sorted_order]
    expert_range = torch.arange(bucket_count, device=device, dtype=torch.int64)
    counts_offsets = torch.searchsorted(sorted_expert_ids, expert_range, right=False)
    counts_end = torch.searchsorted(sorted_expert_ids, expert_range, right=True)
    counts = counts_end - counts_offsets
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    total_padded_tokens = padded_counts.sum().to(torch.int32).reshape(1)
    padded_offsets = torch.cumsum(padded_counts, dim=0) - padded_counts

    token_ranks = (
        torch.arange(num_total_tokens, device=device, dtype=torch.int64)
        - counts_offsets[sorted_expert_ids]
    )
    output_positions = padded_offsets[sorted_expert_ids] + token_ranks
    sorted_token_ids.scatter_(
        0,
        output_positions.to(torch.int64),
        sorted_order.to(torch.int32),
    )

    block_counts = padded_counts // block_size
    real_block_counts = block_counts.clone()
    real_block_counts[sentinel] = 0
    actual_num_blocks = real_block_counts.sum()

    if max_num_blocks <= 0:
        return sorted_token_ids, expert_ids, total_padded_tokens

    block_offsets = torch.cumsum(real_block_counts, dim=0)
    all_block_positions = torch.arange(max_num_blocks, device=device, dtype=torch.int64)
    assigned_experts = torch.searchsorted(
        block_offsets, all_block_positions, right=True
    ).to(torch.int32)
    expert_ids.copy_(
        torch.where(
            all_block_positions < actual_num_blocks,
            assigned_experts,
            torch.full_like(assigned_experts, -1),
        )
    )

    return sorted_token_ids, expert_ids, total_padded_tokens


def align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    ignore_invalid_expert: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implementation of ``sglang.kernels.ops.moe.moe_align_block_size``.

    Lives here rather than in the group package so that the module-level setup
    it needs (torch, the resolved backend) is paid once at import instead of on
    every call; see that wrapper for the contract.
    """
    # Only the AOT kernel implements ignore_invalid_expert, so that flag pins the
    # backend; otherwise take this platform's default.
    backend = KernelBackend.AOT if ignore_invalid_expert else _OUT_BACKEND
    num_buckets = num_experts + 1
    device = topk_ids.device
    numel = topk_ids.numel()

    # Tiny batches (bs=1 decode) take v2's single-CTA kernels, which work on the
    # pair axis and never scan the bucket axis -- so alone among the paths here
    # they are not bounded by _MAX_BUCKETS.
    jit_tiny_batch = (
        backend == KernelBackend.JIT
        and numel <= V2_SMALL_NUMEL_LIMIT
        and v2_supported(topk_ids, num_buckets, block_size)
    )

    # No kernel on this platform scans that wide; the pure-torch path allocates
    # its own buffers, so take it before we allocate ours.
    if not jit_tiny_batch and num_buckets > _MAX_BUCKETS[backend]:
        return align_block_size_torch(topk_ids, block_size, num_experts)

    if numel == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty, torch.zeros(1, dtype=torch.int32, device=device)

    if numel < num_buckets:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + num_buckets * (block_size - 1)

    # sorted_token_ids is exposed at its exact length, but reserved rounded up to
    # VEC_SIZE: the kernels clear it with ceil(len / VEC_SIZE) int4 stores, so an
    # exact length is written up to 3 int32s past its end. Reserving the tail
    # keeps that spill inside our own buffer; exposing the exact length keeps
    # shape[0] identical to what separate allocations gave, which matters because
    # some consumers derive a block count from it by *floor* division.
    sorted_len = max_num_tokens_padded
    sorted_reserved = _align4(sorted_len)
    num_m_blocks = (sorted_len + block_size - 1) // block_size
    # In EP, expert_ids for filtered experts are -1. We have num_experts + 1 ids in total.
    # For safety of cumlen, we need 1 more (num_experts + 2) to avoid out-of-bound access.
    cumsum_len = num_buckets + 1
    buf = torch.empty(
        sorted_reserved
        + _align4(num_m_blocks)
        + _VEC_SIZE  # num_tokens_post_pad, one element
        + _align4(cumsum_len),
        dtype=torch.int32,
        device=device,
    )
    off = 0
    sorted_ids = buf[off : off + sorted_len]
    off += sorted_reserved
    expert_ids = buf[off : off + num_m_blocks]
    off += _align4(num_m_blocks)
    num_tokens_post_pad = buf[off : off + 1]
    off += _VEC_SIZE
    cumsum_buffer = buf[off : off + cumsum_len]

    # Pass ignore_invalid_expert only when it is actually asked for: wheels built
    # without it bind an 8-arg signature, and a 9th positional argument is a
    # TypeError there. XPU and MUSA images ship such a wheel.
    extra_args = (ignore_invalid_expert,) if ignore_invalid_expert else ()
    get_kernel("moe.moe_align_block_size_out", backend)(
        topk_ids,
        num_buckets,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,  # NOTE: pad_sorted_token_ids
        *extra_args,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad
