# SPDX-License-Identifier: Apache-2.0
"""Step5 grouped-union metadata over request-contained packed-query ranges."""

import torch


def step5_build_prefill_union(
    logical_starts: torch.Tensor,
    physical_regions: torch.Tensor,
    counts: torch.Tensor,
    *,
    group_query_start: torch.Tensor,
    group_query_end: torch.Tensor,
    union_physical: torch.Tensor,
    union_logical: torch.Tensor,
    union_counts: torch.Tensor,
    membership: torch.Tensor,
    bitset: torch.Tensor,
    num_region_bins: int,
) -> None:
    """Build sorted region unions with exact membership for up to 16 queries.

    Inputs are int32 logical token starts / physical region IDs [T,513], counts
    [T], and packed-query start/end [G]. End is exclusive; each range contains
    at most 16 rows of ONE request and one consistent physical mapping. Logical
    query positions are separate attention inputs, not packed-query offsets.

    Outputs: physical/logical region IDs and membership bits [G,capacity], union
    counts [G]. Bit j denotes selection by query slot j, not permission for all
    queries to attend the union. Consumers also apply per-token causal masks.
    Only slots below union_counts are valid; all membership slots are cleared
    before construction. The int32 bitset workspace is [G,ceil(region_bins/32)].

    Caller owns disjoint buffers and stream ordering. Ranges and selections are
    device inputs, so replay may change their values without changing addresses.
    Capacity must cover the worst-case union. Warm compilation before capture;
    this entry neither allocates storage nor reads device values on the host.
    """
    if logical_starts.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 prefill union requires NVIDIA CUDA")
    if logical_starts.ndim != 2 or logical_starts.shape[1] != 513:
        raise ValueError("Step5 prefill selection requires shape [T,513]")
    if group_query_start.ndim != 1 or union_physical.ndim != 2:
        raise ValueError("Step5 prefill groups require [G] ranges and [G,C] outputs")
    if not 1 <= num_region_bins <= 131072 or num_region_bins & (num_region_bins - 1):
        raise ValueError("Step5 region bins must be a power of two through 131072")
    rows, groups = logical_starts.shape[0], group_query_start.shape[0]
    capacity = union_physical.shape[1]
    if capacity < min(16 * 513, num_region_bins):
        raise ValueError("Step5 prefill union capacity could truncate selected regions")
    selection_stride = logical_starts.stride(1) if rows else 1
    for name, tensor in (
        ("logical_starts", logical_starts),
        ("physical_regions", physical_regions),
    ):
        if (
            tensor.device != logical_starts.device
            or tensor.dtype != torch.int32
            or tuple(tensor.shape) != (rows, 513)
            or (rows and tensor.stride() != (513 * selection_stride, selection_stride))
            or selection_stride not in (1, 2)
        ):
            raise ValueError(f"Step5 prefill union invalid {name}")
    for name, tensor, shape in (
        ("counts", counts, (rows,)),
        ("group_query_start", group_query_start, (groups,)),
        ("group_query_end", group_query_end, (groups,)),
        ("union_physical", union_physical, (groups, capacity)),
        ("union_logical", union_logical, (groups, capacity)),
        ("union_counts", union_counts, (groups,)),
        ("membership", membership, (groups, capacity)),
        ("bitset", bitset, (groups, (num_region_bins + 31) // 32)),
    ):
        if (
            tensor.device != logical_starts.device
            or tensor.dtype != torch.int32
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Step5 prefill union invalid {name}")
    if groups:
        from sglang.kernels.ops.attention.step5.cute.cute_utils import device_cache_key
        from sglang.kernels.ops.attention.step5.cute.prefill_union import _compile_union

        _compile_union(
            num_region_bins,
            capacity,
            device_cache_key(logical_starts.device),
            selection_stride,
        )(
            logical_starts,
            physical_regions,
            counts,
            group_query_start,
            group_query_end,
            bitset,
            union_physical,
            union_logical,
            union_counts,
            membership,
            groups,
            rows,
        )
