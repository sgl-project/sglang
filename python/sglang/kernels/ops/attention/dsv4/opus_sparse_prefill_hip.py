"""DeepSeek-V4 sparse prefill attention on gfx950 through aiter's OPUS kernels.

OPUS attends over two row sources, each addressed by a CSR index list. The caller
gathers rows into workspaces and builds padded per-query lists
(``SparsePrefillChunkCache``); this module turns those lists into CSR and runs the
kernel: bf16 rows through ``pa_sparse_prefill_opus``, or fp8 rows in aiter's DMA
layout (512 e4m3 per row at unit scale, RoPE in the last 64) through
``pa_sparse_prefill_fp8_opus``.
"""

import functools
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import fp8_dtype

# (int32 indices, int32 indptr [T + 1])
Csr = Tuple[torch.Tensor, torch.Tensor]

_D = 512
_D_NOPE = 448
_D_ROPE = 64
_Q_BLK = 32


@functools.cache
def opus_fp8_available() -> bool:
    """aiter's fp8 OPUS kernel reading the DMA KV layout on this GPU."""
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime
        from aiter.ops.pa_sparse_prefill_opus import fp8_kv_dma_layout_enabled
    except ImportError:
        return False
    return fp8_kv_dma_layout_enabled(get_gfx_runtime())


@triton.jit
def _quant_q_mxfp8_kernel(
    q_ptr,
    nope_ptr,
    scale_ptr,
    rope_ptr,
    q_stride_t,
    q_stride_h,
    num_heads,
    NBLK: tl.constexpr,
    NBLK_P2: tl.constexpr,
    BLK: tl.constexpr,
    D: tl.constexpr,
    D_NOPE: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    # one program per (token, head) row
    row = tl.program_id(0).to(tl.int64)
    t = row // num_heads
    h = row % num_heads
    src = q_ptr + t * q_stride_t + h * q_stride_h
    blk = tl.arange(0, NBLK_P2)
    col = tl.arange(0, BLK)
    ok = (blk < NBLK)[:, None]
    x = tl.load(src + blk[:, None] * BLK + col[None, :], mask=ok, other=0.0).to(
        tl.float32
    )
    # E8M0 putting each block's max in (224, 448], as aiter's reference quantizer
    amax = tl.max(tl.abs(x), axis=1)
    e = tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 448.0))
    e = tl.where(amax == 0.0, 0.0, tl.clamp(e, -127.0, 127.0))
    y = tl.clamp(x * tl.exp2(-e)[:, None], -448.0, 448.0)
    dst = nope_ptr + row * D
    tl.store(
        dst + blk[:, None] * BLK + col[None, :],
        y.to(nope_ptr.dtype.element_ty),
        mask=ok,
    )
    # one byte per block at [D_NOPE, D_NOPE + NBLK); the two after them are read as
    # the RoPE blocks' slots, which the kernel fills itself
    sblk = tl.arange(0, NBLK_P2)
    sval = tl.where(sblk < NBLK, e, 0.0) + 127.0
    tl.store(scale_ptr + row * D + D_NOPE + sblk, sval.to(tl.uint8))
    rc = tl.arange(0, D_ROPE)
    tl.store(rope_ptr + row * D_ROPE + rc, tl.load(src + D_NOPE + rc))


def quantize_q_opus_fp8(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16 q [T, H, 512] -> the fp8 OPUS kernel's query pair: q_nope [T, H, 512] fp8
    (448 NoPE values, one E8M0 per 32 at bytes 448..461) and q_rope [T, H, 64] bf16."""
    num_tokens, num_heads, head_dim = q.shape
    assert head_dim == _D and q.stride(2) == 1, (q.shape, q.stride())
    q_nope = torch.empty((num_tokens, num_heads, _D), dtype=fp8_dtype, device=q.device)
    q_rope = torch.empty(
        (num_tokens, num_heads, _D_ROPE), dtype=q.dtype, device=q.device
    )
    if num_tokens * num_heads:
        nblk = _D_NOPE // _Q_BLK
        _quant_q_mxfp8_kernel[(num_tokens * num_heads,)](
            q,
            q_nope,
            q_nope.view(torch.uint8),
            q_rope,
            q.stride(0),
            q.stride(1),
            num_heads,
            NBLK=nblk,
            NBLK_P2=triton.next_power_of_2(nblk),
            BLK=_Q_BLK,
            D=_D,
            D_NOPE=_D_NOPE,
            D_ROPE=_D_ROPE,
        )
    return q_nope, q_rope


@triton.jit
def _combined_to_csr_kernel(
    indices_ptr,
    indices_stride,
    lens_ptr,
    indptr_ptr,
    out_ptr,
    row_start,
    row_end,
    BLOCK: tl.constexpr,
):
    # OPUS reads every CSR entry, so -1 holes inside the scanned prefix and
    # entries outside [row_start, row_end) are dropped here.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    length = tl.load(lens_ptr + row)
    idx = tl.load(
        indices_ptr + row * indices_stride + cols, mask=cols < length, other=-1
    )
    keep = (idx >= row_start) & (idx < row_end)
    dst = tl.cumsum(keep.to(tl.int32), axis=0) - 1
    start = tl.load(indptr_ptr + row)
    tl.store(out_ptr + start + dst, idx - row_start, mask=keep)


def combined_to_csr(
    indices: torch.Tensor, lens: torch.Tensor, row_start: int, row_end: int
) -> Csr:
    """int32 [T, W] padded index lists with per-row scanned lengths [T] -> the CSR
    (T * W capacity) of their entries in [row_start, row_end), rebased to row_start."""
    num_rows, width = indices.shape
    device = indices.device
    cols = torch.arange(width, device=device, dtype=torch.int32)
    keep = (
        (cols[None, :] < lens[:, None]) & (indices >= row_start) & (indices < row_end)
    )
    indptr = torch.zeros(num_rows + 1, dtype=torch.int32, device=device)
    torch.cumsum(keep.sum(dim=1, dtype=torch.int32), dim=0, out=indptr[1:])
    csr = torch.empty(max(num_rows * width, 1), dtype=torch.int32, device=device)
    if num_rows > 0:
        _combined_to_csr_kernel[(num_rows,)](
            indices,
            indices.stride(0),
            lens,
            indptr,
            csr,
            row_start,
            row_end,
            BLOCK=triton.next_power_of_2(width),
        )
    return csr, indptr


def opus_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    csr: Csr,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    extend_kv: Optional[torch.Tensor] = None,
    extend_csr: Optional[Csr] = None,
) -> torch.Tensor:
    """Attention of q [T, H, 512] bf16 over the bf16 rows of kv [R, 512] named by
    csr, then those of extend_kv named by extend_csr, with a per-head sink."""
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_opus

    indices, indptr = csr
    if extend_kv is None:
        # every query reads an empty row of the unused extend source
        extend_kv = kv[:1]
        extend_csr = (indices[:1], torch.zeros_like(indptr))
    extend_indices, extend_indptr = extend_csr
    return pa_sparse_prefill_opus(
        q.contiguous(),
        kv,
        indices,
        indptr,
        extend_kv,
        extend_indices,
        extend_indptr,
        attn_sink,
        softmax_scale,
    )


def opus_sparse_prefill_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    csr: Csr,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    extend_kv: Optional[torch.Tensor] = None,
    extend_csr: Optional[Csr] = None,
) -> torch.Tensor:
    """``opus_sparse_prefill`` over fp8 rows: q from ``quantize_q_opus_fp8``, kv and
    extend_kv [R, 512] fp8 in the DMA layout (unit scale, fp8 RoPE in the last 64).
    At most 32 heads: wider launches take a kernel that also reads bf16 RoPE."""
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_fp8_opus

    assert q_nope.shape[1] <= 32, q_nope.shape
    indices, indptr = csr
    if extend_kv is None:
        extend_kv = kv[:1]
        extend_csr = (indices[:1], torch.zeros_like(indptr))
    extend_indices, extend_indptr = extend_csr
    # The kernel takes QK's RoPE from the rows themselves; the bf16 RoPE operands are
    # only shape-checked, so one zero row stands in for them. The two must share a row
    # stride, which expand() leaves at 64 for a single row, hence as_strided.
    rope_stub = q_rope.new_zeros((1, _D_ROPE))
    return pa_sparse_prefill_fp8_opus(
        q_nope,
        q_rope,
        kv,
        rope_stub.as_strided((kv.shape[0], _D_ROPE), (0, 1)),
        indices,
        indptr,
        extend_kv,
        rope_stub.as_strided((extend_kv.shape[0], _D_ROPE), (0, 1)),
        extend_indices,
        extend_indptr,
        attn_sink,
        softmax_scale,
        kv_dma_layout=True,
    )
