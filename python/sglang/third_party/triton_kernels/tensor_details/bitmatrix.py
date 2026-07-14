from dataclasses import dataclass
import triton
import triton.language as tl
import torch
from .bitmatrix_details.sum_bitmatrix_rows import sum_bitmatrix_rows


@dataclass
class BitmatrixMetadata:
    """
    Example:
    `bitmatrix` = [0 0 1 0 1 1 0
                   0 1 0 0 0 1 0
                   1 1 1 0 0 0 1
                   0 0 1 0 1 0 0]
    `col_sum` = [1 2 3 0 2 2 1]
    `col_sorted_indx` = cat([5], [3 6], [0 7], [], [9 1 10], [2 4], [8])
    `row_sorted_indx` = cat([3 6 8], [1 9], [0 2 4 10], [5 7])
    """
    # the number of entries equal to 1 in each column
    col_sum: torch.Tensor
    # indices of nonzero values numbered row-major, grouped by cols, concatenated
    col_sorted_indx: torch.Tensor
    # indices of nonzero values numbered col-major, grouped by rows, concatenated
    row_sorted_indx: torch.Tensor


# `make_bitmatrix_metadata`: entry point for optimized implementation
# ---------------------------------------------------------------------------- #


@triton.jit
def _keyed_add(x, y):
    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xffff0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _bitmatrix_metadata_compute_stage2(ColSortedIndx, RowSortedIndx, NonzeroIndx, n_tokens, ColPartialSum, stride_pm,
                                       stride_pn, ColOffs, TOKS_PER_ROW: tl.constexpr, BLOCK_PER_TOK: tl.constexpr):
    BLOCK_SIZE: tl.constexpr = BLOCK_PER_TOK * TOKS_PER_ROW
    tl.static_assert(BLOCK_SIZE <= 32768)
    if isinstance(n_tokens, tl.tensor) and n_tokens.dtype.is_ptr():
        n_tokens = tl.load(n_tokens)
    nonzero_indx_size = n_tokens * TOKS_PER_ROW
    pid_m = tl.program_id(0)
    # load column indices
    offs_local = tl.arange(0, BLOCK_SIZE)
    offs_global = pid_m * BLOCK_SIZE + offs_local
    mask = offs_global < nonzero_indx_size
    col_indx = tl.load(NonzeroIndx + offs_global, mask=mask, other=-1).to(tl.uint32)
    # stable-sort by columns index
    kv_pairs = ((col_indx << 16) | offs_local).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    col_indx = kv_pairs >> 16
    offs_global = pid_m * BLOCK_SIZE + (kv_pairs & 0xffff)
    mask = col_indx != 0xffff
    # compute run lengths in column-sorted order:
    x = (kv_pairs & 0xffff0000 | 0x00000001)
    cols_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
    exclusive_run_lengths = (cols_and_inclusive_run_lengths - 1) & 0xffff
    # compute output
    row_sorted_indx = tl.load(ColPartialSum + pid_m * stride_pm + col_indx * stride_pn, mask=mask)
    row_sorted_indx += tl.load(ColOffs + col_indx, mask=mask)
    row_sorted_indx += exclusive_run_lengths
    # write back output
    tl.store(RowSortedIndx + offs_global, row_sorted_indx, mask=mask)
    tl.store(ColSortedIndx + row_sorted_indx, offs_global, mask=mask)


@triton.jit
def _bitmatrix_metadata_compute_stage1(CombinedIndx, n_combined_indx, sentinel, BLOCK: tl.constexpr, ColSum, ColOffs,
                                       n_cols, PartialColSum, shape_pm, stride_pm, stride_pn, BLOCK_M: tl.constexpr,
                                       BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    # compute col_partial_sums
    if pid < n_cols:
        PartialColSum += pid * stride_pn
        curr_sum = 0
        for start in range(0, shape_pm, BLOCK_M):
            offs = start + tl.arange(0, BLOCK_M) * stride_pm
            partial_col_sum = tl.load(PartialColSum + offs, mask=offs < shape_pm)
            out = tl.cumsum(partial_col_sum, 0) - partial_col_sum + curr_sum
            curr_sum += tl.sum(partial_col_sum, 0)
            tl.store(PartialColSum + offs, out, mask=offs < shape_pm)
    # compute col_offs
    elif pid == n_cols:
        curr_sum = 0
        for start in range(0, n_cols, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            col_sum = tl.load(ColSum + offs, mask=offs < n_cols)
            col_offs = tl.cumsum(col_sum, 0) - col_sum + curr_sum
            curr_sum += tl.sum(col_sum, 0)
            tl.store(ColOffs + offs, col_offs, mask=offs < n_cols)
    # memset `combined_indx` to `sentinel`
    else:
        offs = (pid - n_cols - 1) * BLOCK + tl.arange(0, BLOCK)
        tl.store(CombinedIndx + offs, sentinel, mask=offs < n_combined_indx)


def cdiv(x, y):
    return (x + y - 1) // y


def make_bitmatrix_metadata(nonzero_indx, bitmatrix):
    assert nonzero_indx.ndim == 2
    PARTIAL_BLOCK_M = 32
    col_sum, col_partial_sum = sum_bitmatrix_rows(bitmatrix, partials_block_size=PARTIAL_BLOCK_M)
    # allocate memory
    device = bitmatrix.device
    n_indx = nonzero_indx.numel()
    n_cols = bitmatrix.shape[1]
    col_offs = torch.empty(n_cols, dtype=torch.int32, device=device)
    combined_indx = torch.empty(n_indx * 2, dtype=torch.int32, device=device)
    col_sorted_indx = combined_indx[:n_indx]
    row_sorted_indx = combined_indx[n_indx:]
    # this kernel:
    # - initializes `{row,col}_sorted_indx` to `sentinel`
    # - computes col_offs; necessary for computing `{row,col}_sorted_indx`
    # - computes col_partial_sums; necessary for computing `{row,col}_sorted_indx`
    MEMSET_BLOCK = 1024
    memset_grid = (cdiv(n_indx * 2, MEMSET_BLOCK) + n_cols + 1, )
    _bitmatrix_metadata_compute_stage1[memset_grid](
        combined_indx, n_indx * 2, -1, MEMSET_BLOCK, col_sum,  #
        col_offs, col_sum.shape[0], col_partial_sum,  # inputs
        col_partial_sum.shape[0], col_partial_sum.stride(0), col_partial_sum.stride(1),  # outputs
        BLOCK_M=512, BLOCK_N=512,  # tunable parameters
    )
    # this kernel computes valid entries of `{row,col}_sorted_indx`
    # using `col_offs` and `col_partial_sums`
    n_indx = nonzero_indx.numel()
    toks_per_row = nonzero_indx.shape[-1]
    compute_grid = (cdiv(bitmatrix.shape_max[0], PARTIAL_BLOCK_M), )
    _bitmatrix_metadata_compute_stage2[compute_grid](
        col_sorted_indx, row_sorted_indx,  # outputs
        nonzero_indx, bitmatrix.shape[0], col_partial_sum, col_partial_sum.stride(0),
        col_partial_sum.stride(1),  # inputs
        col_offs,  #
        TOKS_PER_ROW=toks_per_row, BLOCK_PER_TOK=PARTIAL_BLOCK_M,  #
    )
    return BitmatrixMetadata(
        col_sum=col_sum,
        col_sorted_indx=col_sorted_indx,
        row_sorted_indx=row_sorted_indx,
    )


# `make_bitmatrix_metadata_torch`: entry point for reference implementation
# ---------------------------------------------------------------------------- #


def make_bitmatrix_metadata_torch(nonzero_indx, bitmatrix):
    n_batches = bitmatrix.shape[1]
    nonzero_indx = nonzero_indx.reshape(-1).to(torch.int32)
    pad = lambda x, total_size: torch.cat((x, torch.full((total_size - x.shape[0], ), -1, device=x.device)))
    col_sorted_indx = pad(torch.argsort(nonzero_indx[nonzero_indx != -1], stable=True), nonzero_indx.numel())
    row_sorted_indx = pad(torch.argsort(col_sorted_indx[col_sorted_indx != -1], stable=True), nonzero_indx.numel())
    col_sum = torch.histc(nonzero_indx, bins=n_batches, max=n_batches - 1).int()
    return BitmatrixMetadata(
        col_sum=col_sum,
        col_sorted_indx=col_sorted_indx,
        row_sorted_indx=row_sorted_indx,
    )
