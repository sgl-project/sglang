# SPDX-License-Identifier: Apache-2.0
"""Adapt SGLang logical Step5 selections to the specialized sparse kernel ABI."""

import torch


def step5_pack_decode_regions(
    region_ids: torch.Tensor,
    counts: torch.Tensor,
    *,
    req_to_token: torch.Tensor,
    row_to_req: torch.Tensor,
    positions: torch.Tensor,
    packed_regions: torch.Tensor,
    causal_lengths: torch.Tensor,
) -> None:
    """Pack physical region and logical token start without copying raw K/V.

    Selection supplies sorted int32 IDs [T,513] and counts [T], including current.
    Request slot zero is padding and must have count zero. The native FULL pool
    uses page64; req_to_token[request, 8*region] is the aligned physical start.
    Its page16 view uses the same physical token numbering. Each int64 output
    stores physical_start/8 in its high word and logical_start in its low word.

    The caller owns disjoint output buffers and keeps all inputs alive until
    consumption. Output widths 513/514/516 support the source split policies.
    Slots beyond count become -1; padded rows get causal length zero. Logical
    IDs, live request slots, aligned mappings, and positions below the model's
    context limit are producer contracts, as in step5_paged_scores. No tensor
    data is read by Python, so changed-input Graph replay remains supported.
    """
    if region_ids.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 sparse metadata requires NVIDIA CUDA")
    if region_ids.ndim != 2 or region_ids.shape[1] != 513:
        raise ValueError("Step5 region_ids must have shape [T,513]")
    rows = region_ids.shape[0]
    for name, tensor, shape, dtypes in (
        ("region_ids", region_ids, (rows, 513), (torch.int32,)),
        ("counts", counts, (rows,), (torch.int32,)),
        ("row_to_req", row_to_req, (rows,), (torch.int32, torch.int64)),
        ("positions", positions, (rows,), (torch.int32, torch.int64)),
        ("causal_lengths", causal_lengths, (rows,), (torch.int32,)),
    ):
        if (
            tensor.device != region_ids.device
            or tensor.dtype not in dtypes
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Step5 sparse metadata invalid {name}")
    if (
        req_to_token.device != region_ids.device
        or req_to_token.dtype not in (torch.int32, torch.int64)
        or req_to_token.ndim != 2
        or req_to_token.stride(1) != 1
    ):
        raise ValueError("Step5 req_to_token requires CUDA integer [requests, context]")
    if (
        packed_regions.device != region_ids.device
        or packed_regions.dtype != torch.int64
        or packed_regions.ndim != 2
        or packed_regions.shape[0] != rows
        or packed_regions.shape[1] not in (513, 514, 516)
        or not packed_regions.is_contiguous()
    ):
        raise ValueError("Step5 packed_regions requires int64 [T,513/514/516]")
    if rows:
        from sglang.kernels.ops.attention.step5.sparse_metadata_triton import (
            pack_decode_regions,
        )

        pack_decode_regions(
            region_ids,
            counts,
            req_to_token,
            row_to_req,
            positions,
            packed_regions,
            causal_lengths,
        )
