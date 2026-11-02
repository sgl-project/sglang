"""FP8 MQA logits on DeepGEMM's ``fp8_fp4_*mqa_logits`` API, shaped like the removed
``fp8_mqa_logits`` / ``fp8_paged_mqa_logits``.

On SM100 DeepGEMM takes MXFP8 Q/K (int32 packed UE8M0 scales, one byte per 32
channels), BF16 weights, returns BF16 logits and needs one ``indices`` entry per
paged query row; SM90 keeps E4M3 with FP32 scales. Callers here pass E4M3 Q whose
scale is folded into the weights, and E4M3 K with one scale per token: an FP32
power of two (repacked here) or already packed UE8M0 as int32. Logits come back
FP32 with 16-byte-aligned rows, as the top-k kernels require.
"""

from __future__ import annotations

import functools
from typing import Any, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda

UE8M0_ONE_X4 = 0x7F7F7F7F
_ROW_ALIGN = 256


def uses_mx_mqa_logits(device: Optional[torch.device]) -> bool:
    if device is None:
        return False
    device = torch.device(device)
    return (
        is_cuda()
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] == 10
    )


@triton.jit
def _pack_ue8m0_x4_kernel(scale_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    bits = tl.load(scale_ptr + offs, mask=mask).to(tl.uint32, bitcast=True)
    packed = ((bits >> 23) & 0xFF) * 0x01010101
    tl.store(out_ptr + offs, packed.to(tl.int32, bitcast=True), mask=mask)


def pack_ue8m0_x4(scale: torch.Tensor) -> torch.Tensor:
    """Power-of-two FP32 scales as their exponent byte replicated into an int32."""
    scale = scale.contiguous()
    out = torch.empty(scale.shape, dtype=torch.int32, device=scale.device)
    n = scale.numel()
    if n:
        block = 1024
        _pack_ue8m0_x4_kernel[(triton.cdiv(n, block),)](scale, out, n, BLOCK=block)
    return out


def mx_q_and_weights(q: Any, weights: torch.Tensor) -> Tuple[Any, torch.Tensor]:
    """BF16 weights and, for a plain E4M3 ``q``, unit UE8M0 scales on SM100;
    unchanged elsewhere. A ``(q, q_sf)`` tuple is already MX-quantized."""
    q_fp8 = q[0] if isinstance(q, (tuple, list)) else q
    if not uses_mx_mqa_logits(q_fp8.device):
        return q, weights
    if isinstance(q, (tuple, list)):
        return tuple(q), weights.to(torch.bfloat16)
    q_sf = torch.full(
        q_fp8.shape[:-1], UE8M0_ONE_X4, dtype=torch.int32, device=q_fp8.device
    )
    return (q_fp8, q_sf), weights.to(torch.bfloat16)


@functools.cache
def dsv4_fp8_index_cache_is_ue8m0() -> bool:
    """Whether the DSV4 C4 FP8 index cache stores packed UE8M0 scales: on SM100,
    unless a non-DeepGEMM indexer (tilelang / aiter / torch) reads it."""
    return uses_mx_mqa_logits(torch.device("cuda")) and not (
        envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()
        or envs.SGLANG_OPT_USE_AITER_INDEXER.get()
        or envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
    )


def mx_kv_scale(k_scale: torch.Tensor) -> torch.Tensor:
    return k_scale if k_scale.dtype == torch.int32 else pack_ue8m0_x4(k_scale)


def paged_mqa_logits_indices(
    num_rows: int, device: torch.device, rows_per_request: int = 1
) -> Optional[torch.Tensor]:
    """The request of each paged query row on SM100, ``rows_per_request``
    consecutive rows per request; SM90 takes none."""
    if not uses_mx_mqa_logits(device):
        return None
    indices = torch.arange(num_rows, dtype=torch.int32, device=device)
    return indices // rows_per_request if rows_per_request > 1 else indices


def _empty_logits(num_rows: int, num_cols: int, device) -> torch.Tensor:
    stride = triton.cdiv(max(num_cols, 1), _ROW_ALIGN) * _ROW_ALIGN
    return torch.empty((num_rows, stride), dtype=torch.float32, device=device)[
        :, :num_cols
    ]


def to_fp32_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dtype == torch.float32:
        return logits
    out = _empty_logits(logits.shape[0], logits.shape[1], logits.device)
    return out.copy_(logits)


def _to_logits_dtype(logits: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.float32:
        return to_fp32_logits(logits)
    return logits.to(dtype)


@triton.jit
def _widen_logits_prefix_kernel(
    src_ptr, dst_ptr, lens_ptr, src_stride, dst_stride, num_cols, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    n = tl.minimum(tl.load(lens_ptr + row), num_cols)
    src = src_ptr + row.to(tl.int64) * src_stride
    dst = dst_ptr + row.to(tl.int64) * dst_stride
    for start in range(0, n, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n
        values = tl.load(src + cols, mask=mask)
        tl.store(dst + cols, values.to(tl.float32), mask=mask)


def widen_logits_prefix(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """FP32 copy of ``logits`` holding only each row's first ``lengths[i]`` columns,
    garbage past them, so the cost follows the context, not the logits width."""
    if logits.dtype == torch.float32:
        return logits
    out = _empty_logits(logits.shape[0], logits.shape[1], logits.device)
    if logits.shape[0] and logits.shape[1]:
        _widen_logits_prefix_kernel[(logits.shape[0],)](
            logits,
            out,
            lengths,
            logits.stride(0),
            out.stride(0),
            logits.shape[1],
            BLOCK=1024,
        )
    return out


@triton.jit
def _expand_compressed_logits_kernel(
    src_ptr,
    dst_ptr,
    starts_ptr,
    ends_ptr,
    src_stride,
    dst_stride,
    num_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(starts_ptr + row)
    end = tl.load(ends_ptr + row)
    valid = (cols >= start) & (cols < end)
    values = tl.load(
        src_ptr + row.to(tl.int64) * src_stride + (cols - start),
        mask=valid,
        other=-float("inf"),
    )
    tl.store(
        dst_ptr + row.to(tl.int64) * dst_stride + cols,
        values.to(tl.float32),
        mask=cols < num_cols,
    )


def expand_compressed_logits(
    logits: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, num_cols: int
) -> torch.Tensor:
    """``[rows, num_cols]`` FP32 with row ``i`` at columns ``[starts[i], ends[i])``
    and ``-inf`` elsewhere, from compressed logits whose row ``i`` starts at 0."""
    out = _empty_logits(logits.shape[0], num_cols, logits.device)
    if logits.shape[0] == 0 or num_cols == 0:
        return out
    block = 1024
    _expand_compressed_logits_kernel[(logits.shape[0], triton.cdiv(num_cols, block))](
        logits,
        out,
        starts,
        ends,
        logits.stride(0),
        out.stride(0),
        num_cols,
        BLOCK=block,
    )
    return out


def fp8_mqa_logits(
    q_fp8: torch.Tensor,
    kv: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    clean_logits: bool = False,
    max_seqlen_k: int = 0,
    logits_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Ragged logits: full width ``[rows, num_keys]`` with ``-inf`` outside
    ``[ks, ke)`` when ``max_seqlen_k`` is 0, else compressed ``[rows, max_seqlen_k]``
    (column ``j`` of row ``i`` is key ``ks[i] + j``, garbage past ``ke[i] - ks[i]``).
    ``logits_dtype`` applies to compressed logits; full-width ones are FP32."""
    import deep_gemm

    k_fp8, k_scale = kv
    if uses_mx_mqa_logits(k_fp8.device):
        k_scale = mx_kv_scale(k_scale)
    q, weights = mx_q_and_weights(q_fp8, weights)
    num_keys = k_fp8.shape[0]
    logits = deep_gemm.fp8_fp4_mqa_logits(
        q, (k_fp8, k_scale), weights, ks, ke, max_seqlen_k or num_keys
    )
    if max_seqlen_k:
        return _to_logits_dtype(logits, logits_dtype)
    return expand_compressed_logits(logits, ks, ke, num_keys)


def fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_seq_len: int,
    clean_logits: bool = False,
    logits_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[batch * next_n, max_seq_len]`` logits of ``q_fp8`` ``[batch, next_n, heads,
    dim]``, garbage past each row's context length. SM100 scores the ``next_n`` rows
    of a request as varlen rows, matching :func:`get_paged_mqa_logits_metadata`."""
    import deep_gemm

    assert not clean_logits, "DeepGEMM no longer cleans MQA logits"
    q, weights = mx_q_and_weights(q_fp8, weights)
    q_lead = q_fp8[0] if isinstance(q_fp8, (tuple, list)) else q_fp8
    batch_size = q_lead.shape[0]
    next_n = q_lead.shape[1] if q_lead.dim() == 4 else 1
    if next_n > 1 and uses_mx_mqa_logits(kv_cache.device):
        q = tuple(t.flatten(0, 1).unsqueeze(1) for t in q)
        context_lens = context_lens.reshape(-1, 1)
        block_table = block_table.repeat_interleave(next_n, dim=0)
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_seq_len,
        indices=paged_mqa_logits_indices(batch_size * next_n, kv_cache.device, next_n),
    )
    if logits_dtype == torch.float32:
        return widen_logits_prefix(logits, context_lens.reshape(-1))
    return logits.to(logits_dtype)


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_kv: int,
    num_sms: int,
    indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """On SM100 ``[batch, next_n]`` context lengths are flattened to ``next_n`` varlen
    rows per request, matching :func:`fp8_paged_mqa_logits`."""
    import deep_gemm

    if indices is None:
        next_n = context_lens.shape[1] if context_lens.dim() == 2 else 1
        indices = paged_mqa_logits_indices(
            context_lens.numel(), context_lens.device, next_n
        )
        if indices is not None:
            context_lens = context_lens.reshape(-1, 1)
    extra = () if indices is None else (indices,)
    return deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, block_kv, num_sms, *extra
    )
