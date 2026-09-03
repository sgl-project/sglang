"""Validated sparse GQA operators migrated from the QSA reference branch."""

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

KV_INT8_GROUP = (
    64  # channels per fp16 scale (group 0 = the 64 rotary dims, 1-3 = pass-through)
)

_H20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (1024, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]
_L20_CONFIGS = [
    (32, (32, 8, 2)),
    (64, (64, 8, 2)),
    (128, (64, 4, 2)),
    (512, (32, 4, 2)),
    (float("inf"), (16, 1, 2)),
]


def _get_best_config(total_q: int):
    table = _H20_CONFIGS if "H20" in torch.cuda.get_device_name(0) else _L20_CONFIGS
    return next(cfg for limit, cfg in table if total_q <= limit)


@triton.jit
def _sparse_gqa_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_seqlens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    seq_start = tl.load(cu_seqlens + batch).to(tl.int64)
    seq_end = tl.load(cu_seqlens + batch + 1).to(tl.int64)
    query_relative = tl.program_id(0).to(tl.int64)
    query = seq_start + query_relative
    if query >= seq_end:
        return

    row_topk = tl.minimum(topk, query_relative + 1)
    row_limit = tl.minimum(topk, ((row_topk + BLOCK_N - 1) // BLOCK_N) * BLOCK_N)
    offs_h = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    head_start = group * GROUP_SIZE
    q_values = tl.load(
        q
        + query * sq_m
        + (head_start + offs_h[:, None]) * sq_h
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + seq_start * sk_n + group * sk_h
    v_base = v + seq_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
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
        + (head_start + offs_h[:, None]) * so_h
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton(q, k, v, max_seqlen_k, indices, cu_seqlens, scale):
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_prefill[(max_seqlen_k, (cu_seqlens.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_seqlens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _sparse_gqa_chunk_prefill(
    q,
    k,
    v,
    out,
    indices,
    cu_q,
    cu_k,
    kv_lens,
    scale,
    topk,
    sq_m: tl.constexpr,
    sq_h: tl.constexpr,
    sq_d: tl.constexpr,
    sk_n: tl.constexpr,
    sk_h: tl.constexpr,
    sk_d: tl.constexpr,
    sv_n: tl.constexpr,
    sv_h: tl.constexpr,
    sv_d: tl.constexpr,
    so_m: tl.constexpr,
    so_h: tl.constexpr,
    so_d: tl.constexpr,
    si_m: tl.constexpr,
    si_g: tl.constexpr,
    si_n: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    query_relative = tl.program_id(0).to(tl.int64)
    batch_group = tl.program_id(1)
    group = batch_group % NUM_KV_HEADS
    batch = batch_group // NUM_KV_HEADS
    q_start = tl.load(cu_q + batch)
    q_end = tl.load(cu_q + batch + 1)
    query = (q_start + query_relative).to(tl.int64)
    if query >= q_end:
        return
    k_start = tl.load(cu_k + batch).to(tl.int64)
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
        + offs_d[None, :] * sq_d,
        mask=(offs_h < GROUP_SIZE)[:, None],
        other=0.0,
    )
    q_values = (q_values * scale * 1.4426950408).to(q_values.dtype)
    k_base = k + k_start * sk_n + group * sk_h
    v_base = v + k_start * sv_n + group * sv_h
    idx_row = indices + query * si_m + group * si_g
    max_value = tl.full([BLOCK_M], -float("inf"), tl.float32)
    normalizer = tl.zeros([BLOCK_M], tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    for start in range(0, row_limit, BLOCK_N):
        current = start + offs_n
        token = tl.load(idx_row + current * si_n, mask=current < topk, other=-1)
        valid = token >= 0
        keys = tl.load(
            k_base + token[None, :] * sk_n + offs_d[:, None] * sk_d,
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_base + token[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=valid[:, None],
            other=0.0,
        )
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
        + offs_d[None, :] * so_d,
        output,
        mask=(offs_h < GROUP_SIZE)[:, None],
    )


def sparse_gqa_fwd_interface_triton_ck(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    max_q = int((cu_q[1:] - cu_q[:-1]).max().item())
    block_m = max(16, triton.next_power_of_2(group_size))
    block_n, warps, stages = _get_best_config(total_q)
    out = torch.empty_like(q)
    _sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q,
        k,
        v,
        out,
        indices,
        cu_q,
        cu_k,
        kv_lens,
        scale,
        indices.shape[-1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        indices.stride(0),
        indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_DIM=head_dim,
        num_warps=warps,
        num_stages=stages,
    )
    return out


@triton.jit
def _fa2_valid_counts(
    seq_lens,
    indices,
    counts,
    topk: tl.constexpr,
    stride_i: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_TOPK)
    length = tl.load(seq_lens + row)
    positions = tl.load(
        indices + row * stride_i + cols,
        mask=cols < topk,
        other=-1,
    )
    valid = (positions >= 0) & (positions < length)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), axis=0))


@triton.jit
def _fa2_prefix_sum(counts, cu_k, batch, BLOCK_B: tl.constexpr):
    rows = tl.arange(0, BLOCK_B)
    valid_rows = rows < batch
    row_counts = tl.load(counts + rows, mask=valid_rows, other=0)
    tl.store(cu_k, 0)
    tl.store(cu_k + rows + 1, tl.cumsum(row_counts, 0), mask=valid_rows)


def qwen_sparse_fa2_cu_seqlens_triton(
    seq_lens, indices, counts, cu_k, batch, topk, block_b: Optional[int] = None
):
    block_b = block_b or triton.next_power_of_2(batch)
    # Count one request per program. The previous implementation formed a
    # [next_power_of_2(topk), next_power_of_2(batch)] tensor in one program;
    # topk=2051 and batch=512 therefore exceeded Triton's 1M-element limit.
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )
    # Prefix sum is only over the batch dimension and remains a small 1-D
    # tensor, including during CUDA graph capture.
    _fa2_prefix_sum[(1,)](
        counts,
        cu_k,
        batch,
        BLOCK_B=block_b,
        num_warps=8,
    )


@triton.jit
def _compact_kv(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    src = slots[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (pack_start + cols)[:, None] * heads * dim + head * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    tl.store(out_k + dst, tl.load(k + src, mask=mask, other=0.0), mask=mask)
    tl.store(out_v + dst, tl.load(v + src, mask=mask, other=0.0), mask=mask)


@triton.jit
def _compact_kv_fp8(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """_compact_kv for an fp8_e4m3 pool viewed as uint8: dequantize into the bf16 scratch."""
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    )
    src = slots[:, None] * heads * dim + head * dim + dims[None, :]
    dst = (pack_start + cols)[:, None] * heads * dim + head * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    kk = (
        tl.load(k + src, mask=mask, other=0)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.bfloat16)
    )
    vv = (
        tl.load(v + src, mask=mask, other=0)
        .to(tl.float8e4nv, bitcast=True)
        .to(tl.bfloat16)
    )
    tl.store(out_k + dst, kk, mask=mask)
    tl.store(out_v + dst, vv, mask=mask)


@triton.jit
def _quant_store_kv_int8(
    k,
    v,
    loc,
    k_buf,
    v_buf,
    k_scale,
    v_scale,
    sm_k_inv,
    sm_v_inv,
    sk_n,
    sk_h,
    sv_n,
    sv_h,
    H: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Program (token, head): quantize one K row and one V row (per-GROUP absmax/127, fp16 scale)
    and scatter payload + scales into slot loc[token].  Grid (N, H) is static -> capture-safe.
    """
    NG: tl.constexpr = D // GROUP
    t, h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(loc + t).to(tl.int64)
    offs = tl.arange(0, D)
    goffs = tl.arange(0, NG)
    row = (slot * H + h) * D
    srow = (slot * H + h) * NG

    x = tl.load(k + t * sk_n + h * sk_h + offs).to(tl.float32)
    x = x * tl.load(sm_k_inv + h * D + offs).to(tl.float32)
    xg = tl.reshape(x, [NG, GROUP])
    a = tl.max(tl.abs(xg), axis=1)
    s = tl.minimum(tl.where(a > 0, a / 127.0, 1.0), 65504.0).to(
        tl.float16
    )  # fp16 max: no inf/NaN on extreme groups            # the stored scale (fp16) is the one used
    q = libdevice.rint(xg / s.to(tl.float32)[:, None])
    q = tl.clamp(q, -127.0, 127.0).to(tl.int8)
    tl.store(k_buf + row + offs, tl.reshape(q, [D]))
    tl.store(k_scale + srow + goffs, s)

    x = tl.load(v + t * sv_n + h * sv_h + offs).to(tl.float32)
    x = x * tl.load(sm_v_inv + h * D + offs).to(tl.float32)
    xg = tl.reshape(x, [NG, GROUP])
    a = tl.max(tl.abs(xg), axis=1)
    s = tl.minimum(tl.where(a > 0, a / 127.0, 1.0), 65504.0).to(
        tl.float16
    )  # fp16 max: no inf/NaN on extreme groups
    q = libdevice.rint(xg / s.to(tl.float32)[:, None])
    q = tl.clamp(q, -127.0, 127.0).to(tl.int8)
    tl.store(v_buf + row + offs, tl.reshape(q, [D]))
    tl.store(v_scale + srow + goffs, s)


def quant_store_kv_int8(k, v, loc, k_buf, v_buf, k_scale, v_scale, sm_k_inv, sm_v_inv):
    """k, v: [N, H, D] (any float dtype, unit last stride); loc: [N] int32/int64 slots;
    k_buf/v_buf: int8 [rows, H, D]; k_scale/v_scale: fp16 [rows, H, D // GROUP]; sm_*_inv: fp16 [H, D].
    """
    N, H, D = k.shape
    assert v.shape == (N, H, D) and k.stride(2) == 1 and v.stride(2) == 1
    assert k_buf.dtype == torch.int8 and v_buf.dtype == torch.int8
    assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16
    assert (
        k_buf.is_contiguous()
        and v_buf.is_contiguous()
        and k_scale.is_contiguous()
        and v_scale.is_contiguous()
    )
    assert k_buf.shape[1:] == (H, D) and k_scale.shape[1:] == (H, D // KV_INT8_GROUP)
    assert sm_k_inv.shape == (H, D) and sm_v_inv.shape == (H, D)
    assert loc.numel() == N
    if N == 0:
        return
    _quant_store_kv_int8[(N, H)](
        k,
        v,
        loc,
        k_buf,
        v_buf,
        k_scale,
        v_scale,
        sm_k_inv,
        sm_v_inv,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        H=H,
        D=D,
        GROUP=KV_INT8_GROUP,
        num_warps=4,
    )


@triton.jit
def _compact_kv_int8(
    k,
    v,
    k_scale,
    v_scale,
    sm_k,
    sm_v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """_compact_kv for an int8 pool: gather + dequantize (q * s * sm, fp32) into the bf16 scratch.
    Same store mask as _compact_kv / _compact_kv_fp8: only valid (in-region, 0 <= pos < seq_len)
    columns are written; invalid columns are neither read nor written (on the trtllm strided tables
    cu_k spans the whole page-aligned stride, so zero-filling them would write every unused column).
    """
    NG: tl.constexpr = dim // GROUP
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    ).to(tl.int64)
    dmask = dims < dim
    src = (slots[:, None] * heads + head) * dim + dims[None, :]
    ssrc = (slots[:, None] * heads + head) * NG + dims[None, :] // GROUP
    dst = ((pack_start + cols)[:, None] * heads + head) * dim + dims[None, :]
    mask = valid[:, None] & dmask[None, :]
    smr_k = tl.load(sm_k + head * dim + dims, mask=dmask, other=0).to(tl.float32)
    smr_v = tl.load(sm_v + head * dim + dims, mask=dmask, other=0).to(tl.float32)
    sc = tl.load(k_scale + ssrc, mask=mask, other=0).to(tl.float32)
    kk = tl.load(k + src, mask=mask, other=0).to(tl.float32) * sc * smr_k[None, :]
    tl.store(out_k + dst, kk.to(tl.bfloat16), mask=mask)
    sc = tl.load(v_scale + ssrc, mask=mask, other=0).to(tl.float32)
    vv = tl.load(v + src, mask=mask, other=0).to(tl.float32) * sc * smr_v[None, :]
    tl.store(out_v + dst, vv.to(tl.bfloat16), mask=mask)


@triton.jit
def _gather_dequant_rows_int8(
    k,
    v,
    k_scale,
    v_scale,
    sm_k,
    sm_v,
    req_to_token,
    req_indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Program (batch, row block, head): rows t < seq_lens[batch] of request req_indices[batch]
    (slots from req_to_token) -> dequantized bf16 rows cu_k[batch] + t of out_k/out_v.  The row-block
    count is a grid dimension (runtime max_len), so chunk sizes never recompile the kernel.
    """
    NG: tl.constexpr = dim // GROUP
    batch, block, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    t = block * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = t < length
    slots = tl.load(req_to_token + req * req_stride + t, mask=valid, other=0).to(
        tl.int64
    )
    dims = tl.arange(0, dim)
    src = (slots[:, None] * heads + head) * dim + dims[None, :]
    ssrc = (slots[:, None] * heads + head) * NG + dims[None, :] // GROUP
    dst = ((pack_start + t)[:, None] * heads + head) * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    smr_k = tl.load(sm_k + head * dim + dims).to(tl.float32)
    smr_v = tl.load(sm_v + head * dim + dims).to(tl.float32)
    sc = tl.load(k_scale + ssrc, mask=mask, other=0).to(tl.float32)
    kk = tl.load(k + src, mask=mask, other=0).to(tl.float32) * sc * smr_k[None, :]
    tl.store(out_k + dst, kk.to(tl.bfloat16), mask=mask)
    sc = tl.load(v_scale + ssrc, mask=mask, other=0).to(tl.float32)
    vv = tl.load(v + src, mask=mask, other=0).to(tl.float32) * sc * smr_v[None, :]
    tl.store(out_v + dst, vv.to(tl.bfloat16), mask=mask)


def qwen_sparse_prefix_gather_dequant_int8(
    k,
    v,
    k_scale,
    v_scale,
    sm_k,
    sm_v,
    req_to_token,
    req_indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    max_len,
):
    """Whole-prefix gather-dequant of an int8 pool: rows [0, seq_lens[b]) of every request packed at
    cu_k[b] into the bf16 out_k/out_v.  Rows at or beyond seq_lens[b] are not written.
    """
    _, heads, dim = k.shape
    assert k.dtype == torch.int8 and v.dtype == torch.int8
    assert (
        out_k.dtype == torch.bfloat16 and out_v.dtype == torch.bfloat16
    ), "int8 KV gather needs a bf16 scratch"
    assert dim % KV_INT8_GROUP == 0 and dim == triton.next_power_of_2(dim)
    assert (
        k_scale.shape[1:] == (heads, dim // KV_INT8_GROUP) and k_scale.is_contiguous()
    )
    assert (
        v_scale.shape[1:] == (heads, dim // KV_INT8_GROUP) and v_scale.is_contiguous()
    )
    assert sm_k.shape == (heads, dim) and sm_v.shape == (heads, dim)
    assert out_k.is_contiguous() and out_v.is_contiguous()
    block_t = 16
    if batch == 0 or max_len == 0:
        return
    _gather_dequant_rows_int8[(batch, triton.cdiv(int(max_len), block_t), heads)](
        k,
        v,
        k_scale,
        v_scale,
        sm_k,
        sm_v,
        req_to_token,
        req_indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        heads,
        dim,
        req_to_token.stride(0),
        GROUP=KV_INT8_GROUP,
        BLOCK_T=block_t,
        num_warps=8,
    )


KV_INT4_GROUP = (
    32  # channels per fp16 scale for the int4 pool (groups 0-1 = the 64 rotary dims)
)


@triton.jit
def _quant_store_kv_int4(
    k,
    v,
    loc,
    k_buf,
    v_buf,
    k_scale,
    v_scale,
    sm_k_inv,
    sm_v_inv,
    sk_n,
    sk_h,
    sv_n,
    sv_h,
    H: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Program (token, head): quantize one K row and one V row (per-GROUP absmax/7, fp16 scale,
    q = rint(x / s) with an IEEE-exact division, clamped to [-7, 7]; s clamped to fp16 max so it is never inf) and pack channel pairs into bytes (low nibble = even channel,
    high nibble = odd channel, offset-binary q + 8), then scatter D // 2 bytes + D // GROUP scales
    into slot loc[token].  Grid (N, H) is static -> capture-safe."""
    DH: tl.constexpr = D // 2
    NG: tl.constexpr = D // GROUP
    GH: tl.constexpr = GROUP // 2
    t, h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(loc + t).to(tl.int64)
    pairs = tl.arange(0, DH)
    even = 2 * pairs
    odd = even + 1
    goffs = tl.arange(0, NG)
    row = (slot * H + h) * DH
    srow = (slot * H + h) * NG

    base = k + t * sk_n + h * sk_h
    xe = tl.load(base + even).to(tl.float32) * tl.load(sm_k_inv + h * D + even).to(
        tl.float32
    )
    xo = tl.load(base + odd).to(tl.float32) * tl.load(sm_k_inv + h * D + odd).to(
        tl.float32
    )
    ge = tl.reshape(xe, [NG, GH])  # (g, j) = channel 32 g + 2 j
    go = tl.reshape(xo, [NG, GH])  # (g, j) = channel 32 g + 2 j + 1
    a = tl.maximum(tl.max(tl.abs(ge), axis=1), tl.max(tl.abs(go), axis=1))
    # The stored scale (fp16) is the one used.  Clamp to fp16 max BEFORE the cast: a bf16 group absmax
    # above 7 * 65504 would otherwise give s = +inf -> nibble 8 -> (8 - 8) * inf = NaN on dequant;
    # clamped, such channels saturate to +/- 7 * 65504 (finite) instead.
    s = tl.minimum(tl.where(a > 0, a / 7.0, 1.0), 65504.0).to(tl.float16)
    # IEEE-exact division: Triton's `/` is the approximate div.full (2 ulp), which breaks exact .5 ties
    # (x/s = 6.5 -> 6.5000001 -> 7) that int4's coarse grid hits constantly; torch rounds them to even.
    sf = tl.broadcast_to(s.to(tl.float32)[:, None], [NG, GH])
    qe = tl.clamp(libdevice.rint(tl.div_rn(ge, sf)), -7.0, 7.0).to(tl.int32) + 8
    qo = tl.clamp(libdevice.rint(tl.div_rn(go, sf)), -7.0, 7.0).to(tl.int32) + 8
    packed = (qe | (qo << 4)).to(tl.uint8)
    tl.store(k_buf + row + pairs, tl.reshape(packed, [DH]))
    tl.store(k_scale + srow + goffs, s)

    base = v + t * sv_n + h * sv_h
    xe = tl.load(base + even).to(tl.float32) * tl.load(sm_v_inv + h * D + even).to(
        tl.float32
    )
    xo = tl.load(base + odd).to(tl.float32) * tl.load(sm_v_inv + h * D + odd).to(
        tl.float32
    )
    ge = tl.reshape(xe, [NG, GH])
    go = tl.reshape(xo, [NG, GH])
    a = tl.maximum(tl.max(tl.abs(ge), axis=1), tl.max(tl.abs(go), axis=1))
    s = tl.minimum(tl.where(a > 0, a / 7.0, 1.0), 65504.0).to(
        tl.float16
    )  # fp16-max clamp, see K half
    # IEEE-exact division: Triton's `/` is the approximate div.full (2 ulp), which breaks exact .5 ties
    # (x/s = 6.5 -> 6.5000001 -> 7) that int4's coarse grid hits constantly; torch rounds them to even.
    sf = tl.broadcast_to(s.to(tl.float32)[:, None], [NG, GH])
    qe = tl.clamp(libdevice.rint(tl.div_rn(ge, sf)), -7.0, 7.0).to(tl.int32) + 8
    qo = tl.clamp(libdevice.rint(tl.div_rn(go, sf)), -7.0, 7.0).to(tl.int32) + 8
    packed = (qe | (qo << 4)).to(tl.uint8)
    tl.store(v_buf + row + pairs, tl.reshape(packed, [DH]))
    tl.store(v_scale + srow + goffs, s)


def quant_store_kv_int4(k, v, loc, k_buf, v_buf, k_scale, v_scale, sm_k_inv, sm_v_inv):
    """k, v: [N, H, D] (any float dtype, unit last stride); loc: [N] int32/int64 slots;
    k_buf/v_buf: uint8 [rows, H, D // 2]; k_scale/v_scale: fp16 [rows, H, D // GROUP]; sm_*_inv: fp16 [H, D].
    """
    N, H, D = k.shape
    assert v.shape == (N, H, D) and k.stride(2) == 1 and v.stride(2) == 1
    assert D % KV_INT4_GROUP == 0 and D == triton.next_power_of_2(D)
    assert k_buf.dtype == torch.uint8 and v_buf.dtype == torch.uint8
    assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16
    assert (
        k_buf.is_contiguous()
        and v_buf.is_contiguous()
        and k_scale.is_contiguous()
        and v_scale.is_contiguous()
    )
    assert k_buf.shape[1:] == (H, D // 2) and v_buf.shape[1:] == (H, D // 2)
    assert k_scale.shape[1:] == (H, D // KV_INT4_GROUP) and v_scale.shape[1:] == (
        H,
        D // KV_INT4_GROUP,
    )
    assert sm_k_inv.shape == (H, D) and sm_v_inv.shape == (H, D)
    assert loc.numel() == N
    if N == 0:
        return
    _quant_store_kv_int4[(N, H)](
        k,
        v,
        loc,
        k_buf,
        v_buf,
        k_scale,
        v_scale,
        sm_k_inv,
        sm_v_inv,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        H=H,
        D=D,
        GROUP=KV_INT4_GROUP,
        num_warps=4,
    )


@triton.jit
def _compact_kv_int4(
    k,
    v,
    k_scale,
    v_scale,
    sm_k,
    sm_v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    """_compact_kv for the int4 pool: gather dim // 2 packed bytes per (slot, head), unpack the two
    nibbles ((b & 15) - 8 = even channel, (b >> 4) - 8 = odd channel), dequantize (* s * sm, fp32) and
    interleave into the bf16 scratch.  `dim` is the logical head_dim.  Same store mask as
    _compact_kv / _compact_kv_int8: only valid (in-region, 0 <= pos < seq_len) columns are written;
    invalid columns are neither read nor written."""
    DH: tl.constexpr = dim // 2
    NG: tl.constexpr = dim // GROUP
    GH: tl.constexpr = GROUP // 2
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    pairs = tl.arange(0, DH)
    dims = tl.arange(0, dim)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    ).to(tl.int64)
    src = (slots[:, None] * heads + head) * DH + pairs[None, :]
    ssrc = (slots[:, None] * heads + head) * NG + pairs[None, :] // GH
    dst = ((pack_start + cols)[:, None] * heads + head) * dim + dims[None, :]
    pmask = valid[:, None] & (pairs[None, :] < DH)
    mask = valid[:, None] & (dims[None, :] < dim)
    sm_e = tl.load(sm_k + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_k + head * dim + 2 * pairs + 1).to(tl.float32)
    b = tl.load(k + src, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(k_scale + ssrc, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    tl.store(out_k + dst, tl.interleave(lo, hi).to(tl.bfloat16), mask=mask)
    sm_e = tl.load(sm_v + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_v + head * dim + 2 * pairs + 1).to(tl.float32)
    b = tl.load(v + src, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(v_scale + ssrc, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    tl.store(out_v + dst, tl.interleave(lo, hi).to(tl.bfloat16), mask=mask)


@triton.jit
def _gather_dequant_rows_int4(
    k,
    v,
    k_scale,
    v_scale,
    sm_k,
    sm_v,
    req_to_token,
    req_indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Program (batch, row block, head): rows t < seq_lens[batch] of request req_indices[batch]
    (slots from req_to_token) -> unpacked + dequantized bf16 rows cu_k[batch] + t of out_k/out_v.
    The row-block count is a grid dimension (runtime max_len), so chunk sizes never recompile.
    """
    DH: tl.constexpr = dim // 2
    NG: tl.constexpr = dim // GROUP
    GH: tl.constexpr = GROUP // 2
    batch, block, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    t = block * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = t < length
    slots = tl.load(req_to_token + req * req_stride + t, mask=valid, other=0).to(
        tl.int64
    )
    pairs = tl.arange(0, DH)
    dims = tl.arange(0, dim)
    src = (slots[:, None] * heads + head) * DH + pairs[None, :]
    ssrc = (slots[:, None] * heads + head) * NG + pairs[None, :] // GH
    dst = ((pack_start + t)[:, None] * heads + head) * dim + dims[None, :]
    pmask = valid[:, None] & (pairs[None, :] < DH)
    mask = valid[:, None] & (dims[None, :] < dim)
    sm_e = tl.load(sm_k + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_k + head * dim + 2 * pairs + 1).to(tl.float32)
    b = tl.load(k + src, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(k_scale + ssrc, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    tl.store(out_k + dst, tl.interleave(lo, hi).to(tl.bfloat16), mask=mask)
    sm_e = tl.load(sm_v + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_v + head * dim + 2 * pairs + 1).to(tl.float32)
    b = tl.load(v + src, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(v_scale + ssrc, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    tl.store(out_v + dst, tl.interleave(lo, hi).to(tl.bfloat16), mask=mask)


def qwen_sparse_prefix_gather_dequant_int4(
    k,
    v,
    k_scale,
    v_scale,
    sm_k,
    sm_v,
    req_to_token,
    req_indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    max_len,
):
    """Whole-prefix gather-dequant of the int4 pool (k/v uint8 [rows, H, D // 2]): rows [0, seq_lens[b])
    of every request packed at cu_k[b] into the bf16 out_k/out_v [.., H, D].  Rows at or beyond
    seq_lens[b] are not written."""
    _, heads, dh = k.shape
    dim = 2 * dh
    assert k.dtype == torch.uint8 and v.dtype == torch.uint8
    assert (
        out_k.dtype == torch.bfloat16 and out_v.dtype == torch.bfloat16
    ), "int4 KV gather needs a bf16 scratch"
    assert out_k.shape[1:] == (heads, dim) and out_v.shape[1:] == (heads, dim)
    assert dim % KV_INT4_GROUP == 0 and dim == triton.next_power_of_2(dim)
    assert (
        k_scale.shape[1:] == (heads, dim // KV_INT4_GROUP) and k_scale.is_contiguous()
    )
    assert (
        v_scale.shape[1:] == (heads, dim // KV_INT4_GROUP) and v_scale.is_contiguous()
    )
    assert sm_k.shape == (heads, dim) and sm_v.shape == (heads, dim)
    assert out_k.is_contiguous() and out_v.is_contiguous()
    block_t = 16
    if batch == 0 or max_len == 0:
        return
    _gather_dequant_rows_int4[(batch, triton.cdiv(int(max_len), block_t), heads)](
        k,
        v,
        k_scale,
        v_scale,
        sm_k,
        sm_v,
        req_to_token,
        req_indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        heads,
        dim,
        req_to_token.stride(0),
        GROUP=KV_INT4_GROUP,
        BLOCK_T=block_t,
        num_warps=8,
    )


@triton.jit
def _stamp_ring_owner(loc, owner, N, RING_MASK: tl.constexpr, BLOCK: tl.constexpr):
    """owner[slot & RING_MASK] = slot for every slot of loc[0:N].  Tokens of one write whose slots share a
    ring row race on the same int32 word; one of them wins (definite once the launch is complete) and only
    the winner writes the ring row in the dual-write launch that follows.  Static grid -> capture-safe.
    """
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < N
    slot = tl.load(loc + offs, mask=m, other=0).to(tl.int64)
    tl.store(owner + (slot & RING_MASK), slot.to(tl.int32), mask=m)


@triton.jit
def _quant_store_kv_tiered(
    k,
    v,
    loc,
    k_buf,
    v_buf,
    k_scale,
    v_scale,
    rk,
    rv,
    rks,
    rvs,
    owner,
    sm_k_inv,
    sm_v_inv,
    sk_n,
    sk_h,
    sv_n,
    sv_h,
    H: tl.constexpr,
    D: tl.constexpr,
    GROUP4: tl.constexpr,
    GROUP8: tl.constexpr,
    RING_MASK: tl.constexpr,
):
    """Program (token, head): dual-write of one K row and one V row.  (1) int8-g64 (absmax/127, fp16 scale
    clamped to fp16 max, q = rint(x / s) clamped to [-127, 127]) into ring row r = slot & RING_MASK of
    rk/rv/rks/rvs -- the body of _quant_store_kv_int8 plus the clamp -- stored ONLY if owner[r] == slot
    (stamped by _stamp_ring_owner in the preceding launch: a token that lost a same-write ring-row collision
    stores nothing into the ring and is cold); (2) int4-g32 nibble-packed into the full-context row `slot`
    of k_buf/v_buf/k_scale/v_scale for every token -- the body of _quant_store_kv_int4, verbatim.
    Grid (N, H) is static -> capture-safe."""
    DH: tl.constexpr = D // 2
    NG4: tl.constexpr = D // GROUP4
    GH: tl.constexpr = GROUP4 // 2
    NG8: tl.constexpr = D // GROUP8
    t, h = tl.program_id(0), tl.program_id(1)
    slot = tl.load(loc + t).to(tl.int64)
    r = slot & RING_MASK
    hot = (
        tl.load(owner + r).to(tl.int64) == slot
    )  # this token owns its ring row (stamp launch done)

    # ---- (1) int8 ring row (owner only) ------------------------------------------------------
    offs = tl.arange(0, D)
    hmask = (offs < D) & hot
    hsmask = (tl.arange(0, NG8) < NG8) & hot
    goffs8 = tl.arange(0, NG8)
    row8 = (r * H + h) * D
    srow8 = (r * H + h) * NG8
    x = tl.load(k + t * sk_n + h * sk_h + offs).to(tl.float32)
    x = x * tl.load(sm_k_inv + h * D + offs).to(tl.float32)
    xg = tl.reshape(x, [NG8, GROUP8])
    a = tl.max(tl.abs(xg), axis=1)
    # fp16-max clamp (the int8 kernel lacks it): a bf16 group absmax above 127 * 65504 would give s = inf
    s = tl.minimum(tl.where(a > 0, a / 127.0, 1.0), 65504.0).to(tl.float16)
    q = libdevice.rint(xg / s.to(tl.float32)[:, None])
    q = tl.clamp(q, -127.0, 127.0).to(tl.int8)
    tl.store(rk + row8 + offs, tl.reshape(q, [D]), mask=hmask)
    tl.store(rks + srow8 + goffs8, s, mask=hsmask)

    x = tl.load(v + t * sv_n + h * sv_h + offs).to(tl.float32)
    x = x * tl.load(sm_v_inv + h * D + offs).to(tl.float32)
    xg = tl.reshape(x, [NG8, GROUP8])
    a = tl.max(tl.abs(xg), axis=1)
    s = tl.minimum(tl.where(a > 0, a / 127.0, 1.0), 65504.0).to(
        tl.float16
    )  # fp16-max clamp, see K half
    q = libdevice.rint(xg / s.to(tl.float32)[:, None])
    q = tl.clamp(q, -127.0, 127.0).to(tl.int8)
    tl.store(rv + row8 + offs, tl.reshape(q, [D]), mask=hmask)
    tl.store(rvs + srow8 + goffs8, s, mask=hsmask)

    # ---- (2) int4 full-context row (= _quant_store_kv_int4) ----------------------------------
    pairs = tl.arange(0, DH)
    even = 2 * pairs
    odd = even + 1
    goffs = tl.arange(0, NG4)
    row = (slot * H + h) * DH
    srow = (slot * H + h) * NG4

    base = k + t * sk_n + h * sk_h
    xe = tl.load(base + even).to(tl.float32) * tl.load(sm_k_inv + h * D + even).to(
        tl.float32
    )
    xo = tl.load(base + odd).to(tl.float32) * tl.load(sm_k_inv + h * D + odd).to(
        tl.float32
    )
    ge = tl.reshape(xe, [NG4, GH])  # (g, j) = channel 32 g + 2 j
    go = tl.reshape(xo, [NG4, GH])  # (g, j) = channel 32 g + 2 j + 1
    a = tl.maximum(tl.max(tl.abs(ge), axis=1), tl.max(tl.abs(go), axis=1))
    s = tl.minimum(tl.where(a > 0, a / 7.0, 1.0), 65504.0).to(tl.float16)
    sf = tl.broadcast_to(s.to(tl.float32)[:, None], [NG4, GH])
    qe = tl.clamp(libdevice.rint(tl.div_rn(ge, sf)), -7.0, 7.0).to(tl.int32) + 8
    qo = tl.clamp(libdevice.rint(tl.div_rn(go, sf)), -7.0, 7.0).to(tl.int32) + 8
    packed = (qe | (qo << 4)).to(tl.uint8)
    tl.store(k_buf + row + pairs, tl.reshape(packed, [DH]))
    tl.store(k_scale + srow + goffs, s)

    base = v + t * sv_n + h * sv_h
    xe = tl.load(base + even).to(tl.float32) * tl.load(sm_v_inv + h * D + even).to(
        tl.float32
    )
    xo = tl.load(base + odd).to(tl.float32) * tl.load(sm_v_inv + h * D + odd).to(
        tl.float32
    )
    ge = tl.reshape(xe, [NG4, GH])
    go = tl.reshape(xo, [NG4, GH])
    a = tl.maximum(tl.max(tl.abs(ge), axis=1), tl.max(tl.abs(go), axis=1))
    s = tl.minimum(tl.where(a > 0, a / 7.0, 1.0), 65504.0).to(tl.float16)
    sf = tl.broadcast_to(s.to(tl.float32)[:, None], [NG4, GH])
    qe = tl.clamp(libdevice.rint(tl.div_rn(ge, sf)), -7.0, 7.0).to(tl.int32) + 8
    qo = tl.clamp(libdevice.rint(tl.div_rn(go, sf)), -7.0, 7.0).to(tl.int32) + 8
    packed = (qe | (qo << 4)).to(tl.uint8)
    tl.store(v_buf + row + pairs, tl.reshape(packed, [DH]))
    tl.store(v_scale + srow + goffs, s)


def quant_store_kv_tiered(
    k,
    v,
    loc,
    k_buf,
    v_buf,
    k_scale,
    v_scale,
    rk,
    rv,
    rks,
    rvs,
    owner,
    sm_k_inv,
    sm_v_inv,
    ring_mask,
):
    """k, v: [N, H, D] (any float dtype, unit last stride); loc: [N] int32/int64 slots (N <= R; slots that
    share a ring row within one write are safe: the stamp launch picks one owner per row, only it writes the
    ring row, the others are cold); k_buf/v_buf: uint8 [rows, H, D // 2]; k_scale/v_scale:
    fp16 [rows, H, D // 32]; rk/rv: int8 [R, H, D]; rks/rvs: fp16 [R, H, D // 64]; owner: int32 [R];
    sm_*_inv: fp16 [H, D]; ring_mask = R - 1 (R a power of two)."""
    N, H, D = k.shape
    R = int(ring_mask) + 1
    assert R > 0 and R & (R - 1) == 0, f"ring_mask + 1 = {R} is not a power of two"
    assert v.shape == (N, H, D) and k.stride(2) == 1 and v.stride(2) == 1
    assert (
        D % KV_INT4_GROUP == 0
        and D % KV_INT8_GROUP == 0
        and D == triton.next_power_of_2(D)
    )
    assert k_buf.dtype == torch.uint8 and v_buf.dtype == torch.uint8
    assert k_scale.dtype == torch.float16 and v_scale.dtype == torch.float16
    assert (
        k_buf.is_contiguous()
        and v_buf.is_contiguous()
        and k_scale.is_contiguous()
        and v_scale.is_contiguous()
    )
    assert k_buf.shape[1:] == (H, D // 2) and v_buf.shape[1:] == (H, D // 2)
    assert k_scale.shape[1:] == (H, D // KV_INT4_GROUP) and v_scale.shape[1:] == (
        H,
        D // KV_INT4_GROUP,
    )
    assert (
        rk.dtype == torch.int8
        and rv.dtype == torch.int8
        and rk.shape == (R, H, D)
        and rv.shape == (R, H, D)
    )
    assert rks.dtype == torch.float16 and rvs.dtype == torch.float16
    assert rks.shape == (R, H, D // KV_INT8_GROUP) and rvs.shape == (
        R,
        H,
        D // KV_INT8_GROUP,
    )
    assert (
        rk.is_contiguous()
        and rv.is_contiguous()
        and rks.is_contiguous()
        and rvs.is_contiguous()
    )
    assert owner.dtype == torch.int32 and owner.shape == (R,) and owner.is_contiguous()
    assert sm_k_inv.shape == (H, D) and sm_v_inv.shape == (H, D)
    assert loc.numel() == N
    assert N <= R, f"{N} tokens in one tiered write exceed the ring of {R} slots"
    if N == 0:
        return
    STAMP_BLOCK = 128
    _stamp_ring_owner[(triton.cdiv(N, STAMP_BLOCK),)](
        loc, owner, N, RING_MASK=R - 1, BLOCK=STAMP_BLOCK, num_warps=4
    )
    _quant_store_kv_tiered[(N, H)](
        k,
        v,
        loc,
        k_buf,
        v_buf,
        k_scale,
        v_scale,
        rk,
        rv,
        rks,
        rvs,
        owner,
        sm_k_inv,
        sm_v_inv,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        H=H,
        D=D,
        GROUP4=KV_INT4_GROUP,
        GROUP8=KV_INT8_GROUP,
        RING_MASK=R - 1,
        num_warps=4,
    )


@triton.jit
def _compact_kv_tiered(
    k,
    v,
    k_scale,
    v_scale,
    rk,
    rv,
    rks,
    rvs,
    owner,
    sm_k,
    sm_v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    topk: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    idx_stride: tl.constexpr,
    GROUP4: tl.constexpr,
    GROUP8: tl.constexpr,
    RING_MASK: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    """_compact_kv for the tiered pool.  Per selected slot the tier is decided on the device:
    hot = valid & (owner[slot & RING_MASK] == slot) -> int8 ring row (dequant q * s * sm as _compact_kv_int8),
    cold = valid & ~hot -> int4 full-context row (unpack + dequant as _compact_kv_int4); masked-off lanes of
    the other tier issue no loads; tl.where selects.  Store mask unchanged: only valid columns are written
    (trtllm strided tables rely on unused columns never being touched)."""
    DH: tl.constexpr = dim // 2
    NG4: tl.constexpr = dim // GROUP4
    GH: tl.constexpr = GROUP4 // 2
    NG8: tl.constexpr = dim // GROUP8
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    cols = block * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)
    pairs = tl.arange(0, DH)
    dims = tl.arange(0, dim)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    valid_count = tl.load(cu_k + batch + 1) - pack_start
    positions = tl.load(indices + batch * idx_stride + cols, mask=cols < topk, other=-1)
    valid = (cols < valid_count) & (positions >= 0) & (positions < length)
    slots = tl.load(
        req_to_token + req * req_stride + tl.where(valid, positions, 0),
        mask=valid,
        other=0,
    ).to(tl.int64)
    # device-side tier test
    r = slots & RING_MASK
    o = tl.load(owner + r, mask=valid, other=-1).to(tl.int64)
    hot = valid & (o == slots)
    cold = valid & (o != slots)
    dst = ((pack_start + cols)[:, None] * heads + head) * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    # cold: int4 rows
    src4 = (slots[:, None] * heads + head) * DH + pairs[None, :]
    ssrc4 = (slots[:, None] * heads + head) * NG4 + pairs[None, :] // GH
    pmask = cold[:, None] & (pairs[None, :] < DH)
    # hot: int8 ring rows
    src8 = (r[:, None] * heads + head) * dim + dims[None, :]
    ssrc8 = (r[:, None] * heads + head) * NG8 + dims[None, :] // GROUP8
    mask8 = hot[:, None] & (dims[None, :] < dim)

    sm_e = tl.load(sm_k + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_k + head * dim + 2 * pairs + 1).to(tl.float32)
    smr = tl.load(sm_k + head * dim + dims).to(tl.float32)
    b = tl.load(k + src4, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(k_scale + ssrc4, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    k4 = tl.interleave(lo, hi)
    sc8 = tl.load(rks + ssrc8, mask=mask8, other=0).to(tl.float32)
    k8 = tl.load(rk + src8, mask=mask8, other=0).to(tl.float32) * sc8 * smr[None, :]
    tl.store(out_k + dst, tl.where(hot[:, None], k8, k4).to(tl.bfloat16), mask=mask)

    sm_e = tl.load(sm_v + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_v + head * dim + 2 * pairs + 1).to(tl.float32)
    smr = tl.load(sm_v + head * dim + dims).to(tl.float32)
    b = tl.load(v + src4, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(v_scale + ssrc4, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    v4 = tl.interleave(lo, hi)
    sc8 = tl.load(rvs + ssrc8, mask=mask8, other=0).to(tl.float32)
    v8 = tl.load(rv + src8, mask=mask8, other=0).to(tl.float32) * sc8 * smr[None, :]
    tl.store(out_v + dst, tl.where(hot[:, None], v8, v4).to(tl.bfloat16), mask=mask)


@triton.jit
def _gather_dequant_rows_tiered(
    k,
    v,
    k_scale,
    v_scale,
    rk,
    rv,
    rks,
    rvs,
    owner,
    sm_k,
    sm_v,
    req_to_token,
    req_indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    heads: tl.constexpr,
    dim: tl.constexpr,
    req_stride: tl.constexpr,
    GROUP4: tl.constexpr,
    GROUP8: tl.constexpr,
    RING_MASK: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Program (batch, row block, head): rows t < seq_lens[batch] of request req_indices[batch] (slots from
    req_to_token) -> bf16 rows cu_k[batch] + t of out_k/out_v, each row from its tier (owner test as in
    _compact_kv_tiered).  The row-block count is a grid dimension (runtime max_len): no per-chunk recompile.
    """
    DH: tl.constexpr = dim // 2
    NG4: tl.constexpr = dim // GROUP4
    GH: tl.constexpr = GROUP4 // 2
    NG8: tl.constexpr = dim // GROUP8
    batch, block, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    length = tl.load(seq_lens + batch)
    req = tl.load(req_indices + batch)
    pack_start = tl.load(cu_k + batch)
    t = block * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = t < length
    slots = tl.load(req_to_token + req * req_stride + t, mask=valid, other=0).to(
        tl.int64
    )
    pairs = tl.arange(0, DH)
    dims = tl.arange(0, dim)
    r = slots & RING_MASK
    o = tl.load(owner + r, mask=valid, other=-1).to(tl.int64)
    hot = valid & (o == slots)
    cold = valid & (o != slots)
    dst = ((pack_start + t)[:, None] * heads + head) * dim + dims[None, :]
    mask = valid[:, None] & (dims[None, :] < dim)
    src4 = (slots[:, None] * heads + head) * DH + pairs[None, :]
    ssrc4 = (slots[:, None] * heads + head) * NG4 + pairs[None, :] // GH
    pmask = cold[:, None] & (pairs[None, :] < DH)
    src8 = (r[:, None] * heads + head) * dim + dims[None, :]
    ssrc8 = (r[:, None] * heads + head) * NG8 + dims[None, :] // GROUP8
    mask8 = hot[:, None] & (dims[None, :] < dim)

    sm_e = tl.load(sm_k + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_k + head * dim + 2 * pairs + 1).to(tl.float32)
    smr = tl.load(sm_k + head * dim + dims).to(tl.float32)
    b = tl.load(k + src4, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(k_scale + ssrc4, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    k4 = tl.interleave(lo, hi)
    sc8 = tl.load(rks + ssrc8, mask=mask8, other=0).to(tl.float32)
    k8 = tl.load(rk + src8, mask=mask8, other=0).to(tl.float32) * sc8 * smr[None, :]
    tl.store(out_k + dst, tl.where(hot[:, None], k8, k4).to(tl.bfloat16), mask=mask)

    sm_e = tl.load(sm_v + head * dim + 2 * pairs).to(tl.float32)
    sm_o = tl.load(sm_v + head * dim + 2 * pairs + 1).to(tl.float32)
    smr = tl.load(sm_v + head * dim + dims).to(tl.float32)
    b = tl.load(v + src4, mask=pmask, other=0).to(tl.int32)
    sc = tl.load(v_scale + ssrc4, mask=pmask, other=0).to(tl.float32)
    lo = ((b & 15) - 8).to(tl.float32) * sc * sm_e[None, :]
    hi = ((b >> 4) - 8).to(tl.float32) * sc * sm_o[None, :]
    v4 = tl.interleave(lo, hi)
    sc8 = tl.load(rvs + ssrc8, mask=mask8, other=0).to(tl.float32)
    v8 = tl.load(rv + src8, mask=mask8, other=0).to(tl.float32) * sc8 * smr[None, :]
    tl.store(out_v + dst, tl.where(hot[:, None], v8, v4).to(tl.bfloat16), mask=mask)


def _check_tier_args(
    heads, dim, k_scale, v_scale, ring_k, ring_v, ring_ks, ring_vs, owner, ring_mask
):
    assert (
        ring_k is not None
        and ring_v is not None
        and ring_ks is not None
        and ring_vs is not None
    )
    assert (
        owner is not None and ring_mask is not None
    ), "tiered gather needs the owner table and ring_mask"
    R = int(ring_mask) + 1
    assert R > 0 and R & (R - 1) == 0, f"ring_mask + 1 = {R} is not a power of two"
    assert (
        dim % KV_INT4_GROUP == 0
        and dim % KV_INT8_GROUP == 0
        and dim == triton.next_power_of_2(dim)
    )
    assert (
        k_scale.shape[1:] == (heads, dim // KV_INT4_GROUP) and k_scale.is_contiguous()
    )
    assert (
        v_scale.shape[1:] == (heads, dim // KV_INT4_GROUP) and v_scale.is_contiguous()
    )
    assert ring_k.dtype == torch.int8 and ring_v.dtype == torch.int8
    assert ring_k.shape == (R, heads, dim) and ring_v.shape == (R, heads, dim)
    assert ring_ks.dtype == torch.float16 and ring_vs.dtype == torch.float16
    assert ring_ks.shape == (R, heads, dim // KV_INT8_GROUP) and ring_vs.shape == (
        R,
        heads,
        dim // KV_INT8_GROUP,
    )
    assert (
        ring_k.is_contiguous()
        and ring_v.is_contiguous()
        and ring_ks.is_contiguous()
        and ring_vs.is_contiguous()
    )
    assert owner.dtype == torch.int32 and owner.shape == (R,) and owner.is_contiguous()
    return R


def qwen_sparse_prefix_gather_dequant_tiered(
    k,
    v,
    k_scale,
    v_scale,
    sm_k,
    sm_v,
    req_to_token,
    req_indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    max_len,
    ring_k=None,
    ring_v=None,
    ring_ks=None,
    ring_vs=None,
    owner=None,
    ring_mask=None,
):
    """Whole-prefix gather-dequant of the tiered pool (int4 rows k/v uint8 [rows, H, D // 2] + int8 ring):
    rows [0, seq_lens[b]) of every request packed at cu_k[b] into the bf16 out_k/out_v [.., H, D], each row
    from its tier.  Rows at or beyond seq_lens[b] are not written."""
    _, heads, dh = k.shape
    dim = 2 * dh
    assert k.dtype == torch.uint8 and v.dtype == torch.uint8
    assert (
        out_k.dtype == torch.bfloat16 and out_v.dtype == torch.bfloat16
    ), "tiered KV gather needs a bf16 scratch"
    assert out_k.shape[1:] == (heads, dim) and out_v.shape[1:] == (heads, dim)
    assert sm_k.shape == (heads, dim) and sm_v.shape == (heads, dim)
    assert out_k.is_contiguous() and out_v.is_contiguous()
    R = _check_tier_args(
        heads, dim, k_scale, v_scale, ring_k, ring_v, ring_ks, ring_vs, owner, ring_mask
    )
    block_t = 16
    if batch == 0 or max_len == 0:
        return
    _gather_dequant_rows_tiered[(batch, triton.cdiv(int(max_len), block_t), heads)](
        k,
        v,
        k_scale,
        v_scale,
        ring_k,
        ring_v,
        ring_ks,
        ring_vs,
        owner,
        sm_k,
        sm_v,
        req_to_token,
        req_indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        heads,
        dim,
        req_to_token.stride(0),
        GROUP4=KV_INT4_GROUP,
        GROUP8=KV_INT8_GROUP,
        RING_MASK=R - 1,
        BLOCK_T=block_t,
        num_warps=8,
    )


def qwen_sparse_valid_counts_triton(seq_lens, indices, counts, batch, topk):
    """Valid-count pass alone, for consumers that need per-row lengths but
    not the packed cu_seqlens prefix sum (trtllm paged decode packs rows at
    a fixed page-aligned stride instead)."""
    _fa2_valid_counts[(batch,)](
        seq_lens,
        indices,
        counts,
        topk,
        indices.stride(0),
        BLOCK_TOPK=triton.next_power_of_2(topk),
        num_warps=8,
    )


def qwen_sparse_kv_extraction_compact_triton(
    k,
    v,
    req_to_token,
    req_indices,
    indices,
    seq_lens,
    cu_k,
    out_k,
    out_v,
    batch,
    topk,
    k_scale=None,
    v_scale=None,
    sm_k=None,
    sm_v=None,
    kv_bits=None,
    ring_k=None,
    ring_v=None,
    ring_ks=None,
    ring_vs=None,
    owner=None,
    ring_mask=None,
):
    _, heads, dim = k.shape
    block_topk = 16
    if (
        k.dtype == torch.uint8 and kv_bits == 4 and owner is not None
    ):  # tiered pool: int8 ring over int4 rows
        dim = 2 * dim  # k is [rows, H, D // 2] packed; dim = logical D
        assert (
            out_k.dtype == torch.bfloat16 and out_v.dtype == torch.bfloat16
        ), "tiered KV gather needs a bf16 scratch"
        assert out_k.shape[1:] == (heads, dim) and out_v.shape[1:] == (heads, dim)
        assert (
            k_scale is not None
            and v_scale is not None
            and sm_k is not None
            and sm_v is not None
        )
        R = _check_tier_args(
            heads,
            dim,
            k_scale,
            v_scale,
            ring_k,
            ring_v,
            ring_ks,
            ring_vs,
            owner,
            ring_mask,
        )
        _compact_kv_tiered[(batch, heads, triton.cdiv(topk, block_topk))](
            k,
            v,
            k_scale,
            v_scale,
            ring_k,
            ring_v,
            ring_ks,
            ring_vs,
            owner,
            sm_k,
            sm_v,
            req_to_token,
            req_indices,
            indices,
            seq_lens,
            cu_k,
            out_k,
            out_v,
            topk,
            heads,
            dim,
            req_to_token.stride(0),
            indices.stride(0),
            GROUP4=KV_INT4_GROUP,
            GROUP8=KV_INT8_GROUP,
            RING_MASK=R - 1,
            BLOCK_TOPK=block_topk,
            num_warps=8,
        )
        return
    if (
        k.dtype == torch.uint8 and kv_bits == 4
    ):  # int4_g32 pool (keyed on the pool: fp8 is uint8 too)
        dim = 2 * dim  # k is [rows, H, D // 2] packed; dim = logical D
        assert (
            out_k.dtype == torch.bfloat16 and out_v.dtype == torch.bfloat16
        ), "int4 KV gather needs a bf16 scratch"
        assert out_k.shape[1:] == (heads, dim) and out_v.shape[1:] == (heads, dim)
        assert (
            k_scale is not None
            and v_scale is not None
            and sm_k is not None
            and sm_v is not None
        )
        assert dim % KV_INT4_GROUP == 0 and dim == triton.next_power_of_2(dim)
        assert k_scale.is_contiguous() and v_scale.is_contiguous()
        _compact_kv_int4[(batch, heads, triton.cdiv(topk, block_topk))](
            k,
            v,
            k_scale,
            v_scale,
            sm_k,
            sm_v,
            req_to_token,
            req_indices,
            indices,
            seq_lens,
            cu_k,
            out_k,
            out_v,
            topk,
            heads,
            dim,
            req_to_token.stride(0),
            indices.stride(0),
            GROUP=KV_INT4_GROUP,
            BLOCK_TOPK=block_topk,
            num_warps=8,
        )
        return
    if k.dtype == torch.int8:  # int8_g64 pool: dequantize into the bf16 scratch
        assert (
            out_k.dtype == torch.bfloat16 and out_v.dtype == torch.bfloat16
        ), "int8 KV gather needs a bf16 scratch"
        assert (
            k_scale is not None
            and v_scale is not None
            and sm_k is not None
            and sm_v is not None
        )
        assert (
            dim % KV_INT8_GROUP == 0
            and k_scale.is_contiguous()
            and v_scale.is_contiguous()
        )
        _compact_kv_int8[(batch, heads, triton.cdiv(topk, block_topk))](
            k,
            v,
            k_scale,
            v_scale,
            sm_k,
            sm_v,
            req_to_token,
            req_indices,
            indices,
            seq_lens,
            cu_k,
            out_k,
            out_v,
            topk,
            heads,
            dim,
            req_to_token.stride(0),
            indices.stride(0),
            GROUP=KV_INT8_GROUP,
            BLOCK_TOPK=block_topk,
            BLOCK_D=triton.next_power_of_2(dim),
            num_warps=8,
        )
        return
    if k.dtype == torch.float8_e4m3fn:  # fp8 pool: dequantize into the bf16 scratch
        assert out_k.dtype == torch.bfloat16, "fp8 KV gather needs a bf16 scratch"
        _compact_kv_fp8[(batch, heads, triton.cdiv(topk, block_topk))](
            k.view(torch.uint8),
            v.view(torch.uint8),
            req_to_token,
            req_indices,
            indices,
            seq_lens,
            cu_k,
            out_k,
            out_v,
            topk,
            heads,
            dim,
            req_to_token.stride(0),
            indices.stride(0),
            BLOCK_TOPK=block_topk,
            BLOCK_D=triton.next_power_of_2(dim),
            num_warps=8,
        )
        return
    _compact_kv[(batch, heads, triton.cdiv(topk, block_topk))](
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        topk,
        heads,
        dim,
        req_to_token.stride(0),
        indices.stride(0),
        BLOCK_TOPK=block_topk,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=8,
    )


__all__ = [
    "qwen_sparse_fa2_cu_seqlens_triton",
    "qwen_sparse_valid_counts_triton",
    "qwen_sparse_kv_extraction_compact_triton",
    "sparse_gqa_fwd_interface_triton",
    "sparse_gqa_fwd_interface_triton_ck",
]
