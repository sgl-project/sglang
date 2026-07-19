"""Paged chunked SGMV expand kernel — reads LoRA-B weights from page storage.

Instead of a flat ``(num_lora, output_dim, max_rank)`` weight tensor, the paged
variant reads from ``B_pages`` (``(total_pages, output_dim, page_rank_size)``)
indexed via a ``page_table`` that maps each adapter's logical pages to physical
page indices.

The input ``x`` is the output of :func:`chunked_sgmv_lora_shrink_forward_paged`,
with shape ``(S, max_pages_per_lora * num_slices * page_rank_size)``.

The inner loop iterates over logical pages, loading one page of B weights at a
time.  This design ensures that each contraction dimension aligns with a single
physical page (no cross-page boundary handling needed).
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.lora_tuning_config import get_lora_expand_config
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.utils import cached_triton_kernel, next_power_of_2


@cached_triton_kernel(
    lambda _, kwargs: (
        kwargs["OUTPUT_DIM"],
        kwargs["NUM_SLICES"],
        kwargs["BLOCK_M"],
        kwargs["BLOCK_N"],
        kwargs["PAGE_RANK_SIZE"],
        kwargs["MAX_PAGES_PER_LORA"],
    )
)
@triton.jit(
    do_not_specialize=[
        "num_segs",
        "output_stride_0",
        "output_stride_1",
        "b_page_stride",
    ]
)
def _chunked_lora_expand_kernel_paged(
    # Pointers to matrices
    x,
    B_pages,
    output,
    # Output strides may differ from OUTPUT_DIM when compact LoRA output is
    # accumulated into a wider base projection.
    output_stride_0,
    output_stride_1,
    # Page table: (num_adapters, MAX_PAGES_PER_LORA) int32
    page_table,
    # B_pages stride(0) — must be the actual tensor stride, not derived from
    # OUTPUT_DIM, because B_pages.shape[1] may exceed max_slice_size for
    # multi-slice modules (e.g. qkv_proj stores q+k+v in one page).
    b_page_stride,
    # Information on sequence lengths and weight id
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    # For fused output scaling
    scalings,
    # Offsets of q/k/v slice on output dimension
    slice_offsets,
    # Meta parameters
    NUM_SLICES: tl.constexpr,
    OUTPUT_DIM: tl.constexpr,
    PAGE_RANK_SIZE: tl.constexpr,
    MAX_PAGES_PER_LORA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Paged expand kernel.

    Grid: ``(triton.cdiv(max_slice_size, BLOCK_N), num_slices, num_segments)``

    The inner loop iterates over logical pages (``0 .. MAX_PAGES_PER_LORA-1``).
    For each page that is resident (page-table entry != -1), the kernel reads
    ``BLOCK_K`` elements of the input ``x`` (where the first ``PAGE_RANK_SIZE``
    values are valid; remaining ``BLOCK_K - PAGE_RANK_SIZE`` are zero-padded
    because ``tl.dot`` requires K >= 16) and one page of B weights, then
    accumulates the matmul product.

    Performance note: the inner loop is fully unrolled at compile time.
    Pages beyond the adapter's actual rank are skipped via the page table
    (which has -1 for unallocated pages).
    """
    pid_s = tl.program_id(2)
    if pid_s >= num_segs:
        return

    # Adapter info for this segment
    w_index = tl.load(weight_indices + pid_s)
    cur_rank = tl.load(lora_ranks + w_index)

    if cur_rank == 0:
        return

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)

    slice_id = tl.program_id(1)
    slice_start = tl.load(slice_offsets + slice_id)
    slice_end = tl.load(slice_offsets + slice_id + 1)

    scaling = tl.load(scalings + w_index)

    # Map logical sequence index to physical index
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end
    )

    pid_n = tl.program_id(0)
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + slice_start
    k_offset = tl.arange(0, BLOCK_K)

    # x row stride: max_pages_per_lora * num_slices * page_rank_size
    x_stride_0 = MAX_PAGES_PER_LORA * NUM_SLICES * PAGE_RANK_SIZE

    # B_pages stride between pages — passed from Python to handle
    # multi-slice modules where full_output_dim > max_slice_size.
    # Do NOT recompute as OUTPUT_DIM * PAGE_RANK_SIZE here.

    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over logical pages.  The loop is fully unrolled (constexpr bound).
    for page_idx in range(MAX_PAGES_PER_LORA):
        # Skip if this page is beyond the adapter's rank or swapped out
        phys_page = tl.load(page_table + w_index * MAX_PAGES_PER_LORA + page_idx)
        if phys_page != -1 and page_idx * PAGE_RANK_SIZE < cur_rank:
            # K-dim mask: only PAGE_RANK_SIZE values per page are valid;
            # BLOCK_K >= PAGE_RANK_SIZE (and >= 16 for tl.dot).
            # Extra K values are zero-padded via the mask.
            k_valid = k_offset < PAGE_RANK_SIZE

            # Column offset in x for this page + slice
            # Layout of x: [page0_s0_r0..r{PR-1}, page0_s1_r0..r{PR-1}, ...]
            x_col_offset = (page_idx * NUM_SLICES + slice_id) * PAGE_RANK_SIZE

            x_ptrs = x + (
                s_offset_physical[:, None] * x_stride_0
                + x_col_offset
                + k_offset[None, :]
            )

            w_ptrs = (B_pages + phys_page * b_page_stride) + (
                n_offset[None, :] * PAGE_RANK_SIZE + k_offset[:, None]
            )

            x_tile = tl.load(
                x_ptrs,
                mask=(s_offset_logical[:, None] < seg_end) & k_valid[None, :],
                other=0.0,
            )
            w_tile = tl.load(
                w_ptrs,
                mask=(n_offset[None, :] < slice_end) & k_valid[:, None],
                other=0.0,
            )
            partial_sum += tl.dot(x_tile, w_tile)

    # Store result and accumulate into output
    partial_sum *= scaling
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + n_offset[None, :] * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (
        n_offset[None, :] < slice_end
    )
    partial_sum += tl.load(output_ptr, mask=output_mask, other=0.0)
    tl.store(output_ptr, partial_sum, mask=output_mask)


def chunked_sgmv_lora_expand_forward_paged(
    x: torch.Tensor,
    B_pages: torch.Tensor,
    batch_info: LoRABatchInfo,
    slice_offsets: torch.Tensor,
    max_slice_size: int,
    base_output: Optional[torch.Tensor],
    page_table: torch.Tensor,
    max_pages_per_lora: int,
    page_rank_size: int = 8,
) -> torch.Tensor:
    """Paged LoRA-B (expand) forward.

    Args:
        x: Input (output of paged shrink), shape
            ``(S, max_pages_per_lora * num_slices * page_rank_size)``.
        B_pages: Paged B weights, shape
            ``(total_pages, output_dim, page_rank_size)``.
        batch_info: Batch metadata (segments, ranks, etc.).
        slice_offsets: Boundaries for different slices in the output
            dimension, shape ``(num_slices + 1,)``.
        max_slice_size: Total output dimension.
        base_output: Optional pre-initialized output tensor.  If provided,
            results are accumulated in-place.
        page_table: Page table, shape ``(num_adapters, max_pages_per_lora)``
            int32.  Entry ``page_table[i, j]`` is the physical page index for
            adapter *i*, logical page *j*, or -1 if swapped out.
        max_pages_per_lora: Max logical pages any adapter in the batch uses.
        page_rank_size: Ranks per page (default 8).

    Returns:
        Output tensor, shape ``(S, output_dim)``.
    """
    assert x.is_contiguous()
    assert B_pages.is_contiguous()
    assert len(x.shape) == 2
    assert len(B_pages.shape) == 3
    assert page_table.is_contiguous()
    assert page_table.dtype == torch.int32

    S = x.shape[0]
    OUTPUT_DIM = B_pages.shape[1]
    PR = page_rank_size
    MAX_PAGES = max_pages_per_lora
    num_slices = len(slice_offsets) - 1

    # Validate x shape
    expected_x_dim = MAX_PAGES * num_slices * PR
    assert x.shape[1] == expected_x_dim, (
        f"x has {x.shape[1]} columns but expected {expected_x_dim} "
        f"(max_pages={MAX_PAGES}, slices={num_slices}, PR={PR})"
    )

    # ── DEBUG GUARDS: catch corrupt server-state args before kernel launch.
    # Enabled by env SGLANG_PAGED_DEBUG=1. Cheap (a few reductions) and only
    # on the crash path. Each check names exactly which arg is corrupt.
    import os as _os

    if _os.environ.get("SGLANG_PAGED_DEBUG") == "1":
        _total_pages = B_pages.shape[0]
        _num_slots = page_table.shape[0]
        _pt_cols = page_table.shape[1]
        # page_table physical indices must be -1 or in [0, total_pages)
        _pt = page_table
        _bad_mask = (_pt != -1) & ((_pt < 0) | (_pt >= _total_pages))
        if bool(_bad_mask.any().item()):
            _bad = _bad_mask.nonzero()
            raise RuntimeError(
                f"[paged-expand guard] page_table has out-of-range entry: "
                f"value={int(_pt[_bad[0,0], _bad[0,1]].item())} "
                f"at (slot={int(_bad[0,0].item())},page={int(_bad[0,1].item())}); "
                f"total_pages={_total_pages} "
                f"page_table.shape={tuple(page_table.shape)} "
                f"(pt_cols={_pt_cols} vs MAX_PAGES={MAX_PAGES})"
            )
        assert _pt_cols == MAX_PAGES, (
            f"[paged-expand guard] page_table cols={_pt_cols} != "
            f"MAX_PAGES_PER_LORA={MAX_PAGES} (stride mismatch in kernel)"
        )
        # weight_indices must be valid slots and reference existing pages
        _wi = batch_info.weight_indices[: batch_info.num_segments]
        if _wi.numel() > 0:
            _wmax = int(_wi.max().item())
            _wmin = int(_wi.min().item())
            assert _wmax < _num_slots, (
                f"[paged-expand guard] weight_indices max={_wmax} >= "
                f"num_slots={_num_slots} (OOB page_table row read)"
            )
            assert _wmin >= 0, f"[paged-expand guard] weight_indices min={_wmin} < 0"
        # lora_ranks tensor must be wide enough for the max slot used
        _lr = batch_info.lora_ranks
        if _wi.numel() > 0:
            _wmax = int(_wi.max().item())
            assert _wmax < _lr.shape[0], (
                f"[paged-expand guard] max weight_index={_wmax} >= "
                f"lora_ranks len={_lr.shape[0]}"
            )
        # slice_end must not exceed OUTPUT_DIM (else w_ptrs OOB on B_pages)
        _so = slice_offsets
        assert int(_so[-1].item()) <= OUTPUT_DIM, (
            f"[paged-expand guard] slice_offsets[-1]={int(_so[-1].item())} > "
            f"OUTPUT_DIM={OUTPUT_DIM} (B_pages dim1={B_pages.shape[1]})"
        )
        # permutation values must be in [0, S)
        _perm = batch_info.permutation
        if _perm is not None and _perm.numel() > 0:
            _pmax = int(_perm.max().item())
            _pmin = int(_perm.min().item())
            assert _pmax < S, (
                f"[paged-expand guard] permutation max={_pmax} >= S={S} "
                f"(wild x/output row pointer)"
            )
            assert _pmin >= 0, f"[paged-expand guard] permutation min={_pmin} < 0"

    BLOCK_M = next_power_of_2(batch_info.max_len)
    effective_max_rank = max_pages_per_lora * page_rank_size
    config = get_lora_expand_config(
        K=OUTPUT_DIM,
        R=effective_max_rank,
        num_slices=num_slices,
        chunk_size=BLOCK_M,
    )
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = next_power_of_2(max(PR, 16))

    num_segments = batch_info.num_segments

    grid = (
        triton.cdiv(max_slice_size, BLOCK_N),
        num_slices,
        batch_info.bs if batch_info.use_cuda_graph else num_segments,
    )

    if base_output is None:
        output = torch.zeros((S, OUTPUT_DIM), device=x.device, dtype=x.dtype)
    else:
        output = base_output

    _chunked_lora_expand_kernel_paged[grid](
        x=x,
        B_pages=B_pages,
        output=output,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        page_table=page_table,
        b_page_stride=B_pages.stride(0),
        seg_indptr=batch_info.seg_indptr,
        weight_indices=batch_info.weight_indices,
        lora_ranks=batch_info.lora_ranks,
        permutation=batch_info.permutation,
        num_segs=num_segments,
        scalings=batch_info.scalings,
        slice_offsets=slice_offsets,
        # constants
        NUM_SLICES=num_slices,
        OUTPUT_DIM=OUTPUT_DIM,
        PAGE_RANK_SIZE=PR,
        MAX_PAGES_PER_LORA=MAX_PAGES,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output
