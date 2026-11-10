"""QSA chunk-prefill sparse attention with TMA gathers (opt-in, default off).

Enabled by SGLANG_ENABLE_QSA_TMA_PREFILL. ``_sparse_gqa_chunk_prefill``'s arithmetic,
unchanged: one program per (query row, KV head), the same 16-token online-softmax
steps, the same two dots. Only the K/V loads change: each step's selected rows
arrive by TMA gather4 instead of per-thread ``cp.async`` gathers, which leaves the
load/store unit free and lets the software pipeline keep two steps in flight.

It reads the BF16 context ``pack_qsa_prefill_kv`` builds and is bitwise identical
to the stock kernel on the passes where stock runs that tile: ``_get_best_config``
picks it by the pass's row count (more than 512 rows; never in batch-invariant
mode), and ``applies`` leaves every other pass to stock.
"""

import logging
import os
import time

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.qsa.sparse_attn import _get_best_config

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


ENABLED = _env_flag("SGLANG_ENABLE_QSA_TMA_PREFILL")
# Pipeline depth; 3 keeps two gathers in flight per program (VR200 microbench).
# The stock kernel's tile (16 tokens, 1 warp) is kept, so the output stays bitwise.
_STAGES = int(os.environ.get("SGLANG_QSA_TMA_PREFILL_STAGES", "3"))
# Debug: the first N eligible calls also run the stock path and log the comparison.
_CHECK_CALLS = int(os.environ.get("SGLANG_QSA_TMA_PREFILL_CHECK_CALLS", "0"))
_BLOCK_N = 16
_HEAD_DIM = 256
_state = {"logged": False, "checked": 0}


# oob_row (the packed row count) changes every call; unspecialized, it never recompiles.
@triton.jit(do_not_specialize=["oob_row"])
def _chunk_prefill_tma(
    q,
    k_desc,
    v_desc,
    out,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    oob_row,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    si_m: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGES: tl.constexpr,
):
    """``_sparse_gqa_chunk_prefill`` with TMA-gathered K/V.

    ``k_desc``/``v_desc`` view the packed K/V as [rows, kv_heads * head_dim] with
    block [1, head_dim]; ``indices`` hold positions relative to row ``cu_k[batch]``.
    Unselected entries read row ``oob_row``, one past the descriptor, which TMA
    zero-fills: the zeros the stock kernel's masked loads produce."""
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    # TMA gather row offsets are int32; the caller's cu_k is an int64 cumsum.
    k_start = tl.load(cu_k + batch).to(tl.int32)
    kv_len = tl.load(kv_lens + batch).to(tl.int64)
    visible = query_relative + kv_len - (q_end - q_start) + 1
    row_topk = tl.minimum(topk, visible)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_values = tl.load(
        q
        + query * sq_m
        + (group * GROUP_SIZE + offs_h[:, None]) * sq_h
        + offs_d[None, :],
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    idx_row = indices + query * si_m
    head_col = (group * HEAD_DIM).to(tl.int32)
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in tl.range(0, row_limit, BLOCK_N, num_stages=STAGES):
        current = start + offs_n
        token = tl.load(idx_row + current, mask=current < topk, other=-1)
        valid = token >= 0
        rows = tl.where(valid, k_start + token, oob_row)
        keys = tl.trans(k_desc.gather(rows, head_col).to(q_values.dtype))
        values = v_desc.gather(rows, head_col).to(q_values.dtype)
        scores = tl.where(valid[None, :], tl.dot(q_values, keys), -float("inf"))
        next_max = tl.maximum(max_value, tl.max(scores, 1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.math.exp2(scores - next_max[:, None])
        accumulator = tl.dot(
            probabilities.to(values.dtype), values, accumulator * alpha[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, 1)
        max_value = next_max
    output = accumulator / normalizer[:, None]
    tl.store(
        out
        + query * so_m
        + (group * GROUP_SIZE + offs_h[:, None]) * so_h
        + offs_d[None, :],
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def _descriptor(kv: torch.Tensor):
    from triton.tools.tensor_descriptor import TensorDescriptor

    return TensorDescriptor.from_tensor(kv.view(kv.shape[0], -1), [1, _HEAD_DIM])


def _shapes_ok(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor
) -> bool:
    if not (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[2] == _HEAD_DIM
    ):
        return False
    if (
        k.ndim != 3
        or k.shape != v.shape
        or k.dtype != v.dtype
        or k.shape[2] != _HEAD_DIM
    ):
        return False
    if (
        not (k.is_contiguous() and v.is_contiguous())
        or k.shape[1] < 1
        or q.shape[1] % k.shape[1]
    ):
        return False
    if q.shape[1] // k.shape[1] > 16 or q.stride(2) != 1:
        return False
    if indices.dtype != torch.int32 or indices.ndim != 2 or indices.stride(1) != 1:
        return False
    # TMA: int32 row coordinates (one past the end is the zero-filled row),
    # 16-byte aligned base and row stride.
    if (
        k.shape[0] >= 2**31 - 1
        or k.data_ptr() % 16
        or v.data_ptr() % 16
        or (k.stride(0) * k.element_size()) % 16
    ):
        return False
    return torch.cuda.get_device_capability(q.device)[0] == 10


def applies(q, packed_k, packed_v, indices) -> bool:
    """BF16 q and packed K/V, head dim 256, <= 16 query heads per KV head, int32
    indices, SM100 family, and a pass on which stock runs this kernel's tile."""
    if _get_best_config(q.shape[0]) != (_BLOCK_N, 1, 2):
        return False
    return packed_k.dtype == torch.bfloat16 and _shapes_ok(
        q, packed_k, packed_v, indices
    )


def chunk_prefill(
    q, packed_k, packed_v, indices, cu_q, cu_k, kv_lens, scale, *, max_q: int
):
    """``sparse_gqa_fwd_interface_triton_ck`` with a host ``max_q``; bitwise equal."""
    out = torch.empty_like(q)
    num_entries = cu_q.shape[0] - 1
    if q.shape[0] == 0 or num_entries == 0 or max_q == 0:
        return out
    num_kv_heads = packed_k.shape[1]
    _chunk_prefill_tma[(max_q, num_entries * num_kv_heads)](
        q,
        _descriptor(packed_k),
        _descriptor(packed_v),
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        packed_k.shape[0],
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        indices.stride(0),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=q.shape[1] // num_kv_heads,
        BLOCK_M=16,
        BLOCK_N=_BLOCK_N,
        HEAD_DIM=_HEAD_DIM,
        STAGES=_STAGES,
        num_warps=1,
    )
    return out


def log_once(q, packed_k, indices) -> None:
    if _state["logged"]:
        return
    _state["logged"] = True
    logger.info(
        "QSA TMA prefill: on (stages=%d heads=%d/%d topk=%d check_calls=%d) "
        "(SGLANG_ENABLE_QSA_TMA_PREFILL)",
        _STAGES,
        q.shape[1],
        packed_k.shape[1],
        indices.shape[-1],
        _CHECK_CALLS,
    )


def checked(run_new, run_stock, rows: int, kv_max: int):
    """Serve ``run_new``; the first ``_CHECK_CALLS`` calls also time ``run_stock``
    and log the comparison."""
    if _state["checked"] >= _CHECK_CALLS:
        return run_new()
    _state["checked"] += 1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    new = run_new()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ref = run_stock()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    diff = new.float() - ref.float()
    rel = float(diff.norm() / ref.float().norm().clamp(min=1e-30))
    logger.info(
        "QSA TMA prefill check %d/%d: rows=%d kv_max=%d tma_ms=%.3f stock_ms=%.3f "
        "bitwise=%s rel_l2=%.3e max_abs=%.3e nonfinite=%d",
        _state["checked"],
        _CHECK_CALLS,
        rows,
        kv_max,
        (t1 - t0) * 1e3,
        (t2 - t1) * 1e3,
        bool(torch.equal(new, ref)),
        rel,
        float(diff.abs().max()),
        int((~torch.isfinite(new)).sum()),
    )
    return new
