"""DCP owned-slot sparse MLA on the aiter fp8 qh64 non-persistent asm kernel.

Each query row attends only the top-k slots this rank owns. The asm kernel
takes CSR indices (no work metadata) and fp8 Q/KV at unit scale, and returns
a natural-log LSE. A row that owns no slot gets a zero output and a -inf LSE.
"""

import functools
import os
from typing import Tuple

import torch
import triton
import triton.language as tl

# The fp32 split partials of the kernel are addressed with 32-bit offsets.
ASM_MAX_ROWS = 32768
# The shape the qh64 kernel is built for: 64 query heads over a 512 + 64 row.
ASM_NUM_HEADS = 64
ASM_KV_LORA_RANK = 512
ASM_ROPE_DIM = 64
# fp8 Q/KV, 64 query heads, non-persistent, qseqlen 1 (ROCm/aiter#6132).
_ASM_KERNEL_ROW = f"fp8,fp8,{ASM_NUM_HEADS},0,1,0,1,0,0,"


@functools.cache
def asm_sparse_mla_dcp_available() -> bool:
    try:
        from aiter.jit.core import get_asm_dir

        with open(os.path.join(get_asm_dir(), "mla", "mla_asm.csv")) as f:
            return any(line.startswith(_ASM_KERNEL_ROW) for line in f)
    except (ImportError, OSError):
        return False


def is_asm_q8(q_nope: torch.Tensor, q_rope: torch.Tensor) -> bool:
    """Whether ``q_nope`` and ``q_rope`` are the two halves of one fp8
    ``[q_nope | q_rope]`` row, the query ``asm_sparse_mla_dcp`` reads."""
    return (
        q_nope.dtype == torch.float8_e4m3fn
        and q_rope.data_ptr()
        == q_nope.data_ptr() + q_nope.shape[2] * q_nope.element_size()
        and q_rope.stride()[:2] == q_nope.stride()[:2]
    )


@triton.jit
def _dcp_csr_kernel(
    page_table_ptr,
    indptr_ptr,
    out_ptr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(indptr_ptr + row)
    count = tl.load(indptr_ptr + row + 1) - start
    offs = tl.arange(0, BLOCK)
    mask = offs < count
    vals = tl.load(page_table_ptr + row * TOPK + offs, mask=mask)
    tl.store(out_ptr + start + offs, vals, mask=mask)


def build_dcp_csr(
    local_page_table: torch.Tensor, counts: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CSR (indptr, indices) of the owned slots ``transform_index_page_table_dcp``
    packed to the front of each row."""
    rows, topk = local_page_table.shape
    indptr = torch.zeros(rows + 1, dtype=torch.int32, device=counts.device)
    torch.cumsum(counts, 0, dtype=torch.int32, out=indptr[1:])
    indices = torch.empty(rows * topk, dtype=torch.int32, device=counts.device)
    if rows > 0:
        _dcp_csr_kernel[(rows,)](
            local_page_table,
            indptr,
            indices,
            TOPK=topk,
            BLOCK=triton.next_power_of_2(topk),
        )
    return indptr, indices


@triton.jit
def _empty_row_kernel(
    indptr_ptr,
    lse_ptr,
    o_ptr,
    s_o_t,
    H: tl.constexpr,
    DV: tl.constexpr,
):
    row = tl.program_id(0)
    count = tl.load(indptr_ptr + row + 1) - tl.load(indptr_ptr + row)
    if count == 0:
        tl.store(lse_ptr + row * H + tl.arange(0, H), float("-inf"))
        offs_d = tl.arange(0, DV)
        zeros = tl.zeros((DV,), dtype=o_ptr.dtype.element_ty)
        for h in range(H):
            tl.store(o_ptr + row * s_o_t + h * DV + offs_d, zeros)


@functools.cache
def _row_index_buffers(device) -> Tuple[torch.Tensor, torch.Tensor]:
    """qo_indptr (0, 1, ..., ASM_MAX_ROWS) and unit last-page lengths: one query
    row and one-token pages for every call the kernel takes."""
    qo_indptr = torch.arange(ASM_MAX_ROWS + 1, dtype=torch.int32, device=device)
    ones = torch.ones(ASM_MAX_ROWS, dtype=torch.int32, device=device)
    return qo_indptr, ones


def asm_sparse_mla_dcp(
    q8: torch.Tensor,
    kv: torch.Tensor,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    one_scale: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (out bf16 [T, H, d_v], lse natural-log fp32 [T, H])."""
    import aiter

    T, H, D = q8.shape
    assert T <= ASM_MAX_ROWS and H == ASM_NUM_HEADS and q8.is_contiguous()
    device = q8.device
    out = torch.empty((T, H, d_v), dtype=torch.bfloat16, device=device)
    lse = torch.empty((T, H), dtype=torch.float32, device=device)
    if T == 0:
        return out, lse
    split_lse = torch.empty((T, 1, H, 1), dtype=torch.float32, device=device)
    qo_indptr, ones = _row_index_buffers(device)
    kv_view = kv.view(torch.float8_e4m3fn).view(-1, 1, 1, D)
    aiter.mla_decode_stage1_asm_fwd(
        q8,
        kv_view,
        qo_indptr[: T + 1],
        indptr,
        indices,
        ones[:T],
        qo_indptr[: T + 1],
        None,
        None,
        None,
        1,
        1,
        1,
        sm_scale,
        out.view(T, 1, H, d_v),
        split_lse,
        out,
        lse,
        one_scale,
        one_scale,
        None,
        1,
        0,
        None,
        0,
        True,
    )
    # The kernel leaves the output of a row with no slot unwritten and its LSE
    # at 0; a NaN there survives the zero merge weight.
    _empty_row_kernel[(T,)](indptr, lse, out, out.stride(0), H=H, DV=d_v)
    return out, lse
