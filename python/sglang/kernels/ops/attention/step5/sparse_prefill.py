# SPDX-License-Identifier: Apache-2.0
"""Step5 grouped-union CuTe attention over SGLang-owned native KV storage."""

import functools
import math

import torch

from sglang.kernels.ops.attention.step5 import options as step5_options


@functools.cache
def use_sm100_sparse_prefill() -> bool:
    return (
        # TODO(yhyang201): frozen SGLANG_OPT_STEP5_SPARSE_PREFILL_SM100; prune dead paths.
        step5_options.SPARSE_PREFILL_SM100
        and torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] == 10
    )


def step5_sparse_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    union_physical: torch.Tensor,
    union_logical: torch.Tensor,
    union_counts: torch.Tensor,
    membership: torch.Tensor,
    group_query_start: torch.Tensor,
    group_query_end: torch.Tensor,
    query_positions: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float = 1.0 / math.sqrt(192),
    sm100_workspace=None,
) -> None:
    """Write BF16 attention and FP32 natural LSE for exact sparse unions.

    Q/out are [T,H,192], H in (8,16); native contiguous K/V are [slots,1,192], page64-aligned.
    Metadata comes from step5_build_prefill_union: int32 physical/logical region
    IDs and membership [G,C], counts [G], and packed-query start/end [G]. Logical
    IDs identify eight-token regions. Bit j selects the j-th query in a group.
    Int32 query_positions [T] are logical positions, independent of packed rows.

    The caller supplies disjoint, request-contained query ranges of at most 16
    rows and counts in [0,C]. Each owned row is written, including zero output
    and -inf LSE for empty attention. Rows outside the ranges remain untouched.
    The consumer applies both exact membership and token-level causal masking.

    Caller owns disjoint buffers and stream ordering. Warm compilation before
    capture; this entry allocates no CUDA storage and performs no host reads of
    device values. Shape-polymorphic compiled functions retain no input tensors.
    """
    if sm100_workspace is not None:
        from sglang.kernels.ops.attention.step5.cute.sm100_union.sparse_gqa.token_sparse_attn.prefill_union_paged_sm100_gqa import (
            Sm100UnionExactPrefillWorkspace,
            token_wise_flash_attn_prefill_union_sm100_gqa_func,
        )

        workspace = Sm100UnionExactPrefillWorkspace(
            block_counts=sm100_workspace.sm100_block_counts,
            block_indices=sm100_workspace.sm100_block_indices,
            full_block_counts=sm100_workspace.sm100_full_block_counts,
            exact_region_words=sm100_workspace.sm100_region_words,
            causal_token_limits=sm100_workspace.sm100_token_limits,
            max_union_windows=union_physical.shape[1],
            q_heads=q.shape[1],
        )
        token_wise_flash_attn_prefill_union_sm100_gqa_func(
            q,
            k.view(-1, 16, 1, 192),
            v.view(-1, 16, 1, 192),
            union_counts=union_counts,
            union_phys_indices=union_physical,
            union_logical_indices=union_logical,
            exact_mask_bits=membership,
            causal_limits=workspace.causal_token_limits,
            work_q_global=query_positions,
            work_q_local=group_query_start,
            work_q_len=group_query_end,
            workspace=workspace,
            out=out,
            lse=lse,
            softmax_scale=softmax_scale,
        )
        return

    if q.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 sparse prefill requires NVIDIA CUDA")
    if q.ndim != 3 or q.shape[1] not in (8, 16) or q.shape[2] != 192:
        raise ValueError("Step5 Q requires shape [T,8 or 16,192]")
    if k.ndim != 3 or tuple(k.shape[1:]) != (1, 192) or k.shape[0] % 64:
        raise ValueError("Step5 K requires page64-aligned native [slots,1,192]")
    if union_physical.ndim != 2 or not 1 <= union_physical.shape[1] <= 16 * 513:
        raise ValueError("Step5 prefill union requires shape [G,C], C in [1,8208]")
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError("Step5 softmax scale must be finite and positive")
    rows, heads, _ = q.shape
    groups, capacity = union_physical.shape
    for name, tensor, shape, dtype, alignment in (
        ("Q", q, (rows, heads, 192), torch.bfloat16, 16),
        ("K", k, tuple(k.shape), torch.bfloat16, 16),
        ("V", v, tuple(k.shape), torch.bfloat16, 16),
        ("union_physical", union_physical, (groups, capacity), torch.int32, 4),
        ("union_logical", union_logical, (groups, capacity), torch.int32, 4),
        ("union_counts", union_counts, (groups,), torch.int32, 4),
        ("membership", membership, (groups, capacity), torch.int32, 4),
        ("group_query_start", group_query_start, (groups,), torch.int32, 4),
        ("group_query_end", group_query_end, (groups,), torch.int32, 4),
        ("query_positions", query_positions, (rows,), torch.int32, 4),
        ("out", out, (rows, heads, 192), torch.bfloat16, 16),
        ("lse", lse, (rows, heads), torch.float32, 4),
    ):
        if (
            tensor.device != q.device
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.data_ptr() % alignment
        ):
            raise ValueError(f"Step5 sparse prefill invalid {name}")
    if groups:
        from sglang.kernels.ops.attention.step5.cute.cute_utils import device_cache_key
        from sglang.kernels.ops.attention.step5.cute.prefill_dispatch import (
            compile_prefill,
            compile_prefill_sm90,
        )

        device_key = device_cache_key(q.device)
        if device_key[2][0] == 9:
            compile_prefill_sm90(capacity, device_key, logical_q_heads=heads)(
                q,
                k.view(-1, 16, 1, 192),
                v.view(-1, 16, 1, 192),
                union_physical,
                union_logical,
                union_counts,
                membership,
                query_positions,
                group_query_start,
                group_query_end,
                None,
                out,
                lse,
                softmax_scale,
            )
            return
        compile_prefill(capacity, device_key, logical_q_heads=heads)(
            q,
            k,
            v,
            union_physical,
            union_logical,
            union_counts,
            membership,
            group_query_start,
            group_query_end,
            query_positions,
            out,
            lse,
            softmax_scale,
        )


def step5_sparse_prefill_selected(
    q,
    k,
    v,
    *,
    packed,
    counts,
    query_positions,
    group_query_start,
    group_query_end,
    num_region_bins,
    logical_starts,
    physical_regions,
    union_physical,
    union_logical,
    union_counts,
    membership,
    bitset,
    out,
    lse,
    scale,
    sm100_workspace=None,
):
    """Unpack selected regions, build exact unions, then execute sparse prefill.

    Union contents depend on this layer's selection. All scratch is allocated
    before layer execution by the caller; no indexer/cache ownership is retained.
    """
    from sglang.kernels.ops.attention.step5.prefill_metadata import (
        step5_build_prefill_union,
    )

    # Values depend on this layer's Top-K; only their storage is shared.
    # CUDA stores int64 low/high words in this order. Copy the two strided
    # views into contiguous kernel inputs; an int32 out= right-shift would
    # narrow the operand before shifting and lose the physical region ID.
    words = packed.view(torch.int32)
    step5_build_prefill_union(
        words[:, 0::2],
        words[:, 1::2],
        counts,
        group_query_start=group_query_start,
        group_query_end=group_query_end,
        union_physical=union_physical,
        union_logical=union_logical,
        union_counts=union_counts,
        membership=membership,
        bitset=bitset,
        num_region_bins=num_region_bins,
    )
    step5_sparse_prefill(
        q,
        k,
        v,
        union_physical=union_physical,
        union_logical=union_logical,
        union_counts=union_counts,
        membership=membership,
        group_query_start=group_query_start,
        group_query_end=group_query_end,
        query_positions=query_positions,
        out=out,
        lse=lse,
        softmax_scale=scale,
        sm100_workspace=sm100_workspace,
    )
