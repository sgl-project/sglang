"""Paged chunked SGMV shrink kernel — reads LoRA-A weights from page storage.

Instead of a flat ``(num_lora, N, K)`` weight tensor, the paged variant reads
from ``A_pages`` (``(total_pages, N_per_page, input_dim)``) indexed via a
``page_table`` that maps each adapter's logical pages to physical page indices.

Output shape: ``(S, max_pages_per_lora * num_slices * page_rank_size)``
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.lora_tuning_config import get_lora_shrink_config
from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.utils import cached_triton_kernel, next_power_of_2


@cached_triton_kernel(
    lambda _, kwargs: (
        kwargs["INPUT_DIM"],
        kwargs["N_PER_PAGE"],
        kwargs["BLOCK_N"],
        kwargs["BLOCK_M"],
        kwargs["MAX_PAGES_PER_LORA"],
    )
)
@triton.jit(do_not_specialize=["num_segs"])
def _chunked_lora_shrink_kernel_paged(
    # Pointers to matrices
    x,
    A_pages,
    output,
    output_stride_0,
    output_stride_1,
    # Page table: (num_adapters, MAX_PAGES_PER_LORA) int32
    # -1 means the logical page is swapped out
    page_table,
    # Information on sequence lengths, ranks and weight id
    seg_indptr,
    weight_indices,
    lora_ranks,
    permutation,
    num_segs,
    # Meta parameters
    INPUT_DIM: tl.constexpr,
    PAGE_RANK_SIZE: tl.constexpr,
    N_PER_PAGE: tl.constexpr,
    MAX_PAGES_PER_LORA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Paged shrink kernel.

    Grid: ``(max_pages_per_lora, num_segments)``  (or
    ``(max_pages_per_lora, bs)`` in CUDA graph mode).

    Each block handles one logical page of one segment.  It reads the
    page-table entry for the segment's adapter at the given logical-page index,
    and if the page is resident (not -1), computes ``x @ A_pages[phys_page]``
    and writes the result into the corresponding output columns.  When a page
    is swapped out, the output region for that page stays zero (output must be
    pre-zeroed).  The cur_rank check is placed before the page-table tl.load
    so that programs whose page is beyond the adapter's rank return without
    any memory access.
    """
    pid_page = tl.program_id(0)
    pid_s = tl.program_id(1)

    if pid_s >= num_segs:
        return

    # Adapter info for this segment
    w_index = tl.load(weight_indices + pid_s)
    cur_rank = tl.load(lora_ranks + w_index)

    # If this page is entirely beyond the adapter's rank, skip
    if pid_page * PAGE_RANK_SIZE >= cur_rank:
        return

    # Look up physical page in the page table
    phys_page = tl.load(page_table + w_index * MAX_PAGES_PER_LORA + pid_page)
    if phys_page == -1:
        return  # page swapped out — output stays zero

    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)

    # Map logical sequence index to physical (permuted) index
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end
    )

    # Column offsets in output for this page
    output_page_offset = pid_page * N_PER_PAGE
    n_offset = tl.arange(0, BLOCK_N)
    k_offset = tl.arange(0, BLOCK_K)

    # Base pointer within A_pages for this physical page
    # A_pages layout: (total_pages, N_PER_PAGE, INPUT_DIM)
    A_page_base = A_pages + phys_page * N_PER_PAGE * INPUT_DIM

    # x and weight pointers for the matmul
    x_ptrs = x + (s_offset_physical[:, None] * INPUT_DIM + k_offset[None, :])
    w_ptrs = A_page_base + (k_offset[:, None] + n_offset[None, :] * INPUT_DIM)

    # Accumulate x @ A_pages[phys_page] over the input dimension
    partial_sum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(INPUT_DIM, BLOCK_K)):
        x_tile = tl.load(
            x_ptrs,
            mask=(s_offset_logical[:, None] < seg_end)
            & (k_offset[None, :] < INPUT_DIM - k * BLOCK_K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(k_offset[:, None] < INPUT_DIM - k * BLOCK_K)
            & (n_offset[None, :] < N_PER_PAGE),
            other=0.0,
        )
        partial_sum += tl.dot(x_tile, w_tile)

        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    # Store result in output tensor
    partial_sum = partial_sum.to(x.dtype.element_ty)
    output_ptr = output + (
        s_offset_physical[:, None] * output_stride_0
        + (output_page_offset + n_offset[None, :]) * output_stride_1
    )
    output_mask = (s_offset_logical[:, None] < seg_end) & (
        n_offset[None, :] < N_PER_PAGE
    )
    tl.store(output_ptr, partial_sum, mask=output_mask)


def chunked_sgmv_lora_shrink_forward_paged(
    x: torch.Tensor,
    A_pages: torch.Tensor,
    batch_info: LoRABatchInfo,
    num_slices: int,
    page_table: torch.Tensor,
    max_pages_per_lora: int,
    page_rank_size: int = 8,
) -> torch.Tensor:
    """Paged LoRA-A (shrink) forward.

    Args:
        x: Input activations, shape ``(S, input_dim)``.
        A_pages: Paged A weights, shape
            ``(total_pages, page_rank_size * num_slices, input_dim)``.
        batch_info: Batch metadata (segments, ranks, etc.).
        num_slices: Stacked-multiply factor (1 for o_proj, 2 for gate_up, 3 for qkv).
        page_table: Page table, shape ``(num_adapters, max_pages_per_lora)`` int32.
            Entry ``page_table[i, j]`` is the physical page index for adapter *i*,
            logical page *j*, or -1 if the page is swapped out.
        max_pages_per_lora: Max logical pages any adapter in the batch uses.
        page_rank_size: Ranks per page (default 8).

    Returns:
        Output tensor, shape ``(S, max_pages_per_lora * num_slices * page_rank_size)``.
    """
    assert x.is_contiguous()
    assert A_pages.is_contiguous()
    assert len(x.shape) == 2
    assert len(A_pages.shape) == 3
    assert page_table.is_contiguous()
    assert page_table.dtype == torch.int32

    S = x.shape[0]
    INPUT_DIM = x.shape[1]
    PR = page_rank_size
    N_PER_PAGE = num_slices * PR
    MAX_PAGES = max_pages_per_lora
    N_OUT = MAX_PAGES * N_PER_PAGE

    import os as _os

    if _os.environ.get("SGLANG_PAGED_DEBUG") == "1":
        _total_pages = A_pages.shape[0]
        _num_slots = page_table.shape[0]
        _pt = page_table
        _bad = (_pt != -1) & ((_pt < 0) | (_pt >= _total_pages))
        if bool(
            _bad.any().item()
        ):  # debug-only GPU-CPU sync, acceptable for SGLANG_PAGED_DEBUG
            _b = _bad.nonzero()
            raise RuntimeError(
                f"[paged-shrink guard] page_table OOB entry "
                f"value={int(_pt[_b[0,0],_b[0,1]].item())} "
                f"at (slot={int(_b[0,0].item())},page={int(_b[0,1].item())}) "
                f"total_pages={_total_pages}"
            )
        assert page_table.shape[1] == MAX_PAGES, (
            f"[paged-shrink guard] page_table cols={page_table.shape[1]} "
            f"!= MAX_PAGES={MAX_PAGES}"
        )
        _wi = batch_info.weight_indices[: batch_info.num_segments]
        if _wi.numel() > 0:
            assert int(_wi.max().item()) < _num_slots, (
                f"[paged-shrink guard] weight_index "
                f"max={int(_wi.max().item())} >= num_slots={_num_slots}"
            )

    # Block shapes — use flat kernel's auto-tuned config (same max_lora_rank)
    BLOCK_M = next_power_of_2(batch_info.max_len)
    effective_max_rank = max_pages_per_lora * page_rank_size
    config = get_lora_shrink_config(
        K=INPUT_DIM,
        R=effective_max_rank,
        num_slices=num_slices,
        chunk_size=BLOCK_M,
    )
    BLOCK_K = config["BLOCK_K"]
    BLOCK_N = min(config["BLOCK_N"], next_power_of_2(N_PER_PAGE))

    num_segments = batch_info.num_segments

    grid = (
        MAX_PAGES,
        batch_info.bs if batch_info.use_cuda_graph else num_segments,
    )

    # Use zeros so that columns corresponding to swapped-out or
    # unallocated pages are initialized to zero.
    output = torch.zeros((S, N_OUT), device=x.device, dtype=x.dtype)

    _chunked_lora_shrink_kernel_paged[grid](
        x=x,
        A_pages=A_pages,
        output=output,
        output_stride_0=N_OUT,
        output_stride_1=1,
        page_table=page_table,
        seg_indptr=batch_info.seg_indptr,
        weight_indices=batch_info.weight_indices,
        lora_ranks=batch_info.lora_ranks,
        permutation=batch_info.permutation,
        num_segs=num_segments,
        # constants
        INPUT_DIM=INPUT_DIM,
        PAGE_RANK_SIZE=PR,
        N_PER_PAGE=N_PER_PAGE,
        MAX_PAGES_PER_LORA=MAX_PAGES,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output
