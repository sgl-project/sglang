# SPDX-License-Identifier: Apache-2.0
"""Step5 region selection using the existing FlashInfer deterministic top-k."""

import torch


def can_use_step5_selection_cute(scores: torch.Tensor) -> bool:
    return (
        scores.device.type == "cuda"
        and torch.version.hip is None
        and torch.cuda.get_device_capability(scores.device)[0] == 10
    )


def step5_select_regions(
    scores: torch.Tensor,
    *,
    positions: torch.Tensor,
    row_to_req: torch.Tensor,
    region_ids: torch.Tensor,
    counts: torch.Tensor,
    use_cute: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select <=512 historical regions, sort IDs, then append current exactly once.

    Inputs follow step5_paged_scores: finite history and -inf future scores.
    Width covers positions//8. Slot zero is padding. Output IDs are logical,
    ascending, int32 [T,513], padded with -1; counts are int32 [T]. Caller must
    keep outputs disjoint from inputs and each other. Warm FlashInfer outside
    capture and use its shared device scratch on the sequential forward stream.
    """
    if scores.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 selection requires NVIDIA CUDA")
    if (
        scores.ndim != 2
        or scores.dtype != torch.float32
        or not scores.is_contiguous()
        or not 1 <= scores.shape[1] < 2**31
    ):
        raise ValueError(
            "Step5 selection requires contiguous FP32 scores [T,R], 1<=R<2**31"
        )
    rows, width = scores.shape
    for name, value, shape in (
        ("positions", positions, (rows,)),
        ("row_to_req", row_to_req, (rows,)),
        ("region_ids", region_ids, (rows, 513)),
        ("counts", counts, (rows,)),
    ):
        dtype_ok = (
            value.dtype in (torch.int32, torch.int64)
            if name in ("positions", "row_to_req")
            else value.dtype == torch.int32
        )
        if (
            value.device != scores.device
            or value.layout != torch.strided
            or tuple(value.shape) != shape
            or not value.is_contiguous()
            or not dtype_ok
        ):
            raise ValueError(f"Step5 selection invalid {name} metadata")
    if rows == 0:
        return region_ids, counts
    for out, others in (
        (region_ids, (scores, positions, row_to_req, counts)),
        (counts, (scores, positions, row_to_req)),
    ):
        a = out.data_ptr()
        b = a + out.numel() * out.element_size()
        for value in others:
            lo = value.data_ptr()
            if max(a, lo) < min(b, lo + value.numel() * value.element_size()):
                raise ValueError(
                    "Step5 selection outputs must not alias inputs or each other"
                )
    if use_cute is None:
        use_cute = can_use_step5_selection_cute(scores)
    if use_cute:
        from sglang.kernels.ops.attention.step5.selection_cute import (
            step5_select_regions_cute,
        )

        return step5_select_regions_cute(
            scores, positions, row_to_req, region_ids, counts
        )
    selected_k = 0
    indices = region_ids
    if width > 512:
        import flashinfer

        from sglang.kernels.ops.attention.step5.selection_triton import (
            canonicalize_zero,
        )

        # Numerical equality includes +0/-0; the library's radix key does not.
        # Keep the caller's scores intact, including during graph replay.
        ordered_scores = torch.empty_like(scores)
        canonicalize_zero(scores, ordered_scores)
        _, indices = flashinfer.top_k(
            ordered_scores,
            512,
            sorted=False,
            deterministic=True,
            tie_break=1,
            dsa_graph_safe=True,
        )
        selected_k = 512
    from sglang.kernels.ops.attention.step5.selection_triton import launch

    launch(indices, positions, row_to_req, region_ids, counts, selected_k=selected_k)
    return region_ids, counts
