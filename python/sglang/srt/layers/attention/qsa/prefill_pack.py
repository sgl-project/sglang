"""Row-tiled QSA prefill K/V pack (opt-in, default off), bitwise equal to stock.

``SGLANG_QSA_PREFILL_PACK_ROWS=1`` makes ``pack_qsa_prefill_kv`` copy each
context token's K and V row with a program per ``TOKENS`` tokens instead of a
program per 1,024 elements. The request lookup (the same binary search over
``cu_k``) and the ``req_to_token`` slot load run once per token instead of once
per element, and each row (heads x dim, contiguous in the pool) is read and
written in 16-byte vectors; stock reads one element per thread through a
gathered address it cannot vectorize.

Every output element is stock's: the same source element of the same slot, the
same cast to the output dtype (the pool's FP8 to BF16 widening is exact), the
same packed rows written. Pools whose rows are not contiguous (heads x dim with
unit inner stride), or whose row is not a power of two of at most 4,096
elements, keep the stock kernel.
"""

import functools
import logging
import os
import time

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


ENABLED = _env_flag("SGLANG_QSA_PREFILL_PACK_ROWS")
# Tokens per program and warps (VR200 microbench, h449).
_TOKENS = int(os.environ.get("SGLANG_QSA_PREFILL_PACK_ROWS_TOKENS", "16"))
_WARPS = int(os.environ.get("SGLANG_QSA_PREFILL_PACK_ROWS_WARPS", "4"))
# Debug: the first N eligible eager calls also run the stock pack and log time and
# bitwise equality. Leave unset in served runs: it syncs and runs both kernels.
_CHECK_CALLS = int(os.environ.get("SGLANG_QSA_PREFILL_PACK_ROWS_CHECK_CALLS", "0"))
_state = {"checked": 0, "bypass": False}


@triton.jit(do_not_specialize=["TOTAL_K", "BS"])
def _pack_qsa_prefill_kv_rows(
    k,
    v,
    packed_k,
    packed_v,
    req_to_token,
    req_indices,
    cu_k,
    SK0: tl.constexpr,
    SV0: tl.constexpr,
    SR0: tl.constexpr,
    SR1: tl.constexpr,
    ROW: tl.constexpr,
    TOKENS: tl.constexpr,
    TOTAL_K,
    BS,
    COMPACT: tl.constexpr,
):
    # ``_pack_qsa_prefill_kv`` with its per-element token index made a row index:
    # the same request search, slot and validity, once per token.
    rows = tl.program_id(0).to(tl.int64) * TOKENS + tl.arange(0, TOKENS)
    cols = tl.arange(0, ROW)
    if COMPACT:
        valid = rows < TOTAL_K
        lo = tl.full((TOKENS,), 0, tl.int32)
        hi = lo + BS
        steps = BS
        while steps > 0:
            mid = (lo + hi + 1) // 2
            start = tl.load(cu_k + mid).to(tl.int64)
            after = rows >= start
            lo = tl.where(after, mid, lo)
            hi = tl.where(after, hi, mid - 1)
            steps = steps // 2
        start = tl.load(cu_k + lo, valid, 0).to(tl.int64)
        req = tl.load(req_indices + lo, valid, 0).to(tl.int64)
        token = rows - start
        dst = rows
    else:
        batch = tl.program_id(1)
        start = tl.load(cu_k + batch).to(tl.int64)
        end = tl.load(cu_k + batch + 1).to(tl.int64)
        valid = rows < end - start
        req = tl.load(req_indices + batch).to(tl.int64)
        token = rows
        dst = start + rows
    slot = tl.load(req_to_token + req * SR0 + token * SR1, valid, 0).to(tl.int64)
    mask = valid[:, None]
    keys = tl.load(k + slot[:, None] * SK0 + cols[None, :], mask, 0.0)
    values = tl.load(v + slot[:, None] * SV0 + cols[None, :], mask, 0.0)
    out = dst[:, None] * ROW + cols[None, :]
    tl.store(packed_k + out, keys, mask)
    tl.store(packed_v + out, values, mask)


def _rows_contiguous(t: torch.Tensor) -> bool:
    return t.dim() == 3 and t.stride(2) == 1 and t.stride(1) == t.shape[2]


def applies(k: torch.Tensor, v: torch.Tensor, req_to_token: torch.Tensor) -> bool:
    row = k.shape[1] * k.shape[2] if k.dim() == 3 else 0
    return (
        ENABLED
        and not _state["bypass"]
        and k.is_cuda
        and v.shape == k.shape
        and _rows_contiguous(k)
        and _rows_contiguous(v)
        and req_to_token.dim() == 2
        and 16 <= row <= 4096
        and row & (row - 1) == 0
    )


@functools.lru_cache(maxsize=None)
def _log_once(row: int, k_dtype, out_dtype) -> None:
    logger.info(
        "QSA prefill pack rows: on (tokens=%d warps=%d row=%d %s->%s check_calls=%d) "
        "(SGLANG_QSA_PREFILL_PACK_ROWS)",
        _TOKENS,
        _WARPS,
        row,
        k_dtype,
        out_dtype,
        _CHECK_CALLS,
    )


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def _checked(kwargs, total_k: int, max_k: int):
    """The first ``_CHECK_CALLS`` eager calls: both packs, timed and compared."""
    from sglang.srt.layers.attention.qsa.sparse_attn import pack_qsa_prefill_kv

    _state["checked"] += 1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    new = _launch(**kwargs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    _state["bypass"] = True
    try:
        ref = pack_qsa_prefill_kv(**kwargs)
    finally:
        _state["bypass"] = False
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    logger.info(
        "QSA prefill pack rows check %d/%d: total_k=%d max_k=%d rows_ms=%.3f "
        "stock_ms=%.3f bitwise=%s",
        _state["checked"],
        _CHECK_CALLS,
        total_k,
        max_k,
        (t1 - t0) * 1e3,
        (t2 - t1) * 1e3,
        _same_bits(new[0], ref[0]) and _same_bits(new[1], ref[1]),
    )
    return new


def pack_rows(*, k, v, req_to_token, req_indices, cu_k, total_k, max_k, output_dtype=None):
    """``pack_qsa_prefill_kv``'s outputs, or None when ``applies`` is False."""
    if not applies(k, v, req_to_token):
        return None
    kwargs = dict(
        k=k,
        v=v,
        req_to_token=req_to_token,
        req_indices=req_indices,
        cu_k=cu_k,
        total_k=total_k,
        max_k=max_k,
        output_dtype=output_dtype,
    )
    if _state["checked"] < _CHECK_CALLS and not torch.cuda.is_current_stream_capturing():
        return _checked(kwargs, total_k, max_k)
    return _launch(**kwargs)


def _launch(*, k, v, req_to_token, req_indices, cu_k, total_k, max_k, output_dtype=None):
    heads, dim = k.shape[1:]
    row = heads * dim
    packed_k = torch.empty(
        (total_k, heads, dim), dtype=output_dtype or k.dtype, device=k.device
    )
    packed_v = torch.empty(
        (total_k, heads, dim), dtype=output_dtype or v.dtype, device=v.device
    )
    batch_size = req_indices.numel()
    # Stock's mode rule, so both kernels take the same request lookup.
    blocks_per_request = triton.cdiv(max_k * row, 1024)
    compact = (
        max_k * batch_size > 2 * total_k and blocks_per_request * batch_size >= 4096
    )
    grid = (
        (triton.cdiv(total_k, _TOKENS),)
        if compact
        else (triton.cdiv(max_k, _TOKENS), batch_size)
    )
    _log_once(row, k.dtype, packed_k.dtype)
    _pack_qsa_prefill_kv_rows[grid](
        k,
        v,
        packed_k,
        packed_v,
        req_to_token,
        req_indices,
        cu_k,
        SK0=k.stride(0),
        SV0=v.stride(0),
        SR0=req_to_token.stride(0),
        SR1=req_to_token.stride(1),
        ROW=row,
        TOKENS=_TOKENS,
        TOTAL_K=total_k,
        BS=batch_size,
        COMPACT=compact,
        num_warps=_WARPS,
    )
    return packed_k, packed_v
