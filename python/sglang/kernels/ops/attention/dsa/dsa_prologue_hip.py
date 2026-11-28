"""Fused Kimi-K3.1 DSA decode / verify prologue kernels (gfx950), after vLLM #67847.

Each kernel replaces a run of small launches with one and keeps the stock arithmetic:

- ``index_qk_prep_fp8``: the indexer key LayerNorm (fp32 affine, as
  ``Fp32LayerNormHip`` / ``triton_layer_norm`` compute it), AITER
  ``indexer_k_quant_and_cache``'s FP8 store into the (preshuffled) index-K pages, and
  ``indexer_q_quant_head_gate`` (query FP8 + BF16 head-gate weights): 3 launches -> 1.
- ``index_qk_prep_fp4``: the same LayerNorm, ``store_fp4_index_k``'s MXFP4 key store
  and ``indexer_q_fp4_head_gate``: 3 launches -> 1.
- ``fp8_ds_mla_store``: ``quantize_k_cache_separate`` (FP8 NoPE + FP32 tile scales +
  BF16 RoPE) and ``set_mla_kv_buffer``'s scatter into the 656-byte fp8_ds_mla rows,
  with the DCP owner rule: 2 launches -> 1.

The query and key halves of an index kernel run as disjoint program ranges of one grid
(vLLM's ``_quantize_index_qk``). Key slots follow the stock owner rule: under index-K
DCP sharding a widened slot s belongs to rank s % N at local row s // N; a negative slot
(or one another rank owns) is skipped.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp4_indexer_quant import (
    _fp4_e2m1_code_rne,
    quantize_fp4_indexer_row,
)

HEAD_DIM = 128
# (token, head) rows per query program of the FP8 kernel, as indexer_q_quant_head_gate
_Q_ROWS_PER_PROGRAM = 32
# heads per query program of the FP4 kernel, as indexer_q_fp4_head_gate
_Q_FP4_HEADS_PER_PROGRAM = 16


@triton.jit
def _local_slot(LOC, t, DCP_SIZE: tl.constexpr, DCP_RANK: tl.constexpr):
    loc = tl.load(LOC + t).to(tl.int64)
    if DCP_SIZE > 1:
        owned = (loc >= 0) & (loc % DCP_SIZE == DCP_RANK)
        loc = tl.where(owned, loc // DCP_SIZE, -1)
    return loc


@triton.jit
def _key_layer_norm(K, t, k_stride, NW, NB, eps, D: tl.constexpr):
    # triton_layer_norm (layernorm_gated._layer_norm_fwd_1pass_kernel, no gate) with
    # fp32 weight / bias, its output rounded to the bf16 the stock store reads
    cols = tl.arange(0, D)
    x = tl.load(K + t * k_stride + cols).to(tl.float32)[None, :]
    mean = tl.sum(x, axis=1) / D
    xbar = x - mean[:, None]
    var = tl.sum(xbar * xbar, axis=1) / D
    rstd = tl.rsqrt(var + eps)
    w = tl.load(NW + cols).to(tl.float32)
    b = tl.load(NB + cols).to(tl.float32)
    y = (x - mean[:, None]) * rstd[:, None] * w[None, :] + b[None, :]
    return y.to(tl.bfloat16).to(tl.float32)


@triton.jit(do_not_specialize=["Q_ROWS", "NUM_Q_PROGRAMS", "k_stride"])
def _index_qk_prep_fp8_kernel(
    Q,
    G,
    Q8,
    W,
    K,
    NW,
    NB,
    LOC,
    CACHE_Q,
    CACHE_S,
    Q_ROWS,
    NUM_Q_PROGRAMS,
    k_stride,
    eps,
    HEADS: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    GATE_SCALE: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    PAGE: tl.constexpr,
    TOK_STRIDE: tl.constexpr,
    PRESHUFFLE: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, 128)
    if pid < NUM_Q_PROGRAMS:
        # indexer_q_quant_head_gate: act_quant of each 128-wide head, BF16 head gate
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        q = tl.load(
            Q + rows[:, None] * 128 + cols[None, :], rows[:, None] < Q_ROWS, other=0
        ).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(q), 1), 1e-4)
        scale = amax * (1.0 / 448.0)
        if ROUND_SCALE:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        quant = tl.minimum(tl.maximum(q / scale[:, None], -448.0), 448.0)
        tl.store(
            Q8 + rows[:, None] * 128 + cols[None, :], quant, rows[:, None] < Q_ROWS
        )
        gate = tl.load(
            G + rows // HEADS * GATE_STRIDE + rows % HEADS, rows < Q_ROWS, other=0
        ).to(tl.float32)
        gate = (gate * GATE_SCALE).to(tl.bfloat16).to(tl.float32)
        tl.store(W + rows, gate * scale * SOFTMAX_SCALE, rows < Q_ROWS)
    else:
        t = pid - NUM_Q_PROGRAMS
        slot = _local_slot(LOC, t, DCP_SIZE, DCP_RANK)
        if slot >= 0:
            y = _key_layer_norm(K, t, k_stride, NW, NB, eps, 128)
            # aiter indexer_k_quant_and_cache: one scale per 128-wide token
            amax = tl.max(tl.abs(y), axis=1)
            scale = tl.maximum(amax, 1e-4) / 448.0
            if ROUND_SCALE:
                scale = tl.exp2(tl.ceil(tl.log2(scale)))
            inv = 1.0 / scale
            kq = tl.clamp(y * inv[:, None], -448.0, 448.0).to(CACHE_Q.dtype.element_ty)
            page_base = (slot // PAGE) * (PAGE * TOK_STRIDE)
            tok = slot % PAGE
            if PRESHUFFLE:
                dst = (
                    page_base
                    + (tok // 16) * (16 * 128)
                    + (cols // 16) * 256
                    + (tok % 16) * 16
                    + cols % 16
                )
            else:
                dst = page_base + tok * 128 + cols
            tl.store(CACHE_Q + dst[None, :], kq)
            tl.store(
                CACHE_S + (page_base + PAGE * 128 + tok * 4) // 4 + tl.arange(0, 1),
                scale,
            )


@triton.jit(do_not_specialize=["NUM_Q_PROGRAMS", "k_stride"])
def _index_qk_prep_fp4_kernel(
    Q,
    G,
    QP,
    QS,
    W,
    K,
    NW,
    NB,
    LOC,
    PAYLOAD,
    SCALE,
    NUM_Q_PROGRAMS,
    k_stride,
    eps,
    HEADS: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    GATE_SCALE: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    HB: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < NUM_Q_PROGRAMS:
        # indexer_q_fp4_head_gate: program (t, hb) covers heads [hb*HB, hb*HB + HB)
        t = pid // (64 // HB)
        h = (pid % (64 // HB)) * HB + tl.arange(0, HB)
        live = h < HEADS
        values = tl.load(
            Q + (t * HEADS + h)[:, None] * 128 + tl.arange(0, 128)[None, :],
            mask=live[:, None],
            other=0.0,
        ).to(tl.float32)
        groups = tl.reshape(values, (HB, 4, 32))
        amax = tl.max(tl.abs(groups), axis=2)
        bits = tl.maximum(amax / 6.0, 1.0e-4).to(tl.int32, bitcast=True)
        exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0)
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        scaled = groups / (exp << 23).to(tl.float32, bitcast=True)[:, :, None]
        lo, hi = tl.split(tl.reshape(scaled, (HB, 4, 16, 2)))
        packed = (_fp4_e2m1_code_rne(lo) & 0x0F) | (
            (_fp4_e2m1_code_rne(hi) & 0x0F) << 4
        )
        tl.store(
            QP + (t * HEADS + h)[:, None] * 64 + tl.arange(0, 64)[None, :],
            tl.reshape(packed, (HB, 64)),
            mask=live[:, None],
        )
        chunks = tl.arange(0, 4)
        tl.store(
            QS + t * 256 + chunks[None, :] * 64 + (h % 16 * 4 + h // 16)[:, None],
            tl.where(live[:, None], exp, 0).to(tl.uint8),
        )
        gate = tl.load(G + t * GATE_STRIDE + h, mask=live, other=0.0).to(tl.float32)
        gate = (gate * GATE_SCALE).to(tl.bfloat16).to(tl.float32)
        tl.store(
            W + t * HEADS + h, (gate * SOFTMAX_SCALE).to(W.dtype.element_ty), mask=live
        )
    else:
        t = pid - NUM_Q_PROGRAMS
        loc = _local_slot(LOC, t, DCP_SIZE, DCP_RANK)
        if loc >= 0:
            # store_fp4_index_k on the LayerNorm output
            values = tl.reshape(
                _key_layer_norm(K, t, k_stride, NW, NB, eps, 128), (128,)
            )
            v0, v1 = tl.split(tl.reshape(values, (64, 2)))
            packed, packed_sf = quantize_fp4_indexer_row(
                values, v0, v1, BLOCK_N=128, GROUP_N=32, RNE=True
            )
            page, slot = loc // 64, loc % 64
            byte = tl.arange(0, 64)
            tl.store(
                PAYLOAD + page * 4096 + byte // 16 * 1024 + slot * 16 + byte % 16,
                packed,
            )
            chunks = tl.arange(0, 4)
            tl.store(
                SCALE + page * 256 + chunks * 64 + slot % 16 * 4 + slot // 16,
                ((packed_sf >> (8 * chunks)) & 0xFF).to(tl.uint8),
            )


def _check_key_norm(key, norm_weight, norm_bias, loc) -> int:
    assert key.dtype == torch.bfloat16 and key.ndim == 2
    assert key.shape[1] == HEAD_DIM and key.stride(1) == 1
    assert norm_weight.shape == (HEAD_DIM,) and norm_bias.shape == (HEAD_DIM,)
    assert norm_weight.dtype == norm_bias.dtype == torch.float32
    assert loc.ndim == 1 and loc.stride(0) == 1
    return min(key.shape[0], loc.shape[0])


def _check_query_gate(query, gate) -> tuple[int, int]:
    assert query.dtype == torch.bfloat16 and query.ndim == 3
    assert query.shape[-1] == HEAD_DIM and query.is_contiguous()
    tokens, heads, _ = query.shape
    assert gate.shape == (tokens, heads) and gate.stride(-1) == 1
    assert gate.dtype == torch.bfloat16
    return tokens, heads


def index_qk_prep_fp8(
    query: torch.Tensor,
    gate: torch.Tensor,
    softmax_scale: float,
    scale_fmt: Optional[str],
    key: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
    index_k_cache: Optional[torch.Tensor],
    loc: torch.Tensor,
    *,
    page_size: int = 64,
    preshuffle: bool = True,
    dcp_size: int = 1,
    dcp_rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One launch for the FP8 indexer prologue.

    query [T, H, 128] bf16 (contiguous), gate [T, H] bf16 (row stride free), key
    [T, 128] bf16 pre-LayerNorm (row stride free), index_k_cache the layer's index-K
    buffer (any view of its ``[pages, page_size * 132]`` bytes; None skips the key
    store) and loc the widened int64 write slots. Returns (q_fp8 [T, H, 128],
    weights [T, H, 1] fp32) as ``indexer_q_quant_head_gate`` does."""
    tokens, heads = _check_query_gate(query, gate)
    num_keys = _check_key_norm(key, norm_weight, norm_bias, loc)
    if index_k_cache is None:
        num_keys = 0
    q8 = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    weights = torch.empty((tokens, heads, 1), device=query.device, dtype=torch.float32)
    q_programs = triton.cdiv(tokens * heads, _Q_ROWS_PER_PROGRAM)
    if q_programs + num_keys == 0:
        return q8, weights
    if index_k_cache is None:
        flat = q8.view(-1)
        scales = weights.view(-1)
    else:
        flat = index_k_cache.reshape(-1).view(torch.uint8).view(torch.float8_e4m3fn)
        scales = flat.view(torch.uint8).view(torch.float32)
    _index_qk_prep_fp8_kernel[(q_programs + num_keys,)](
        query,
        gate,
        q8,
        weights,
        key,
        norm_weight,
        norm_bias,
        loc,
        flat,
        scales,
        tokens * heads,
        q_programs,
        key.stride(0),
        eps,
        HEADS=heads,
        GATE_STRIDE=gate.stride(0),
        GATE_SCALE=heads**-0.5,
        SOFTMAX_SCALE=softmax_scale,
        ROUND_SCALE=scale_fmt is not None,
        BLOCK_M=_Q_ROWS_PER_PROGRAM,
        PAGE=page_size,
        TOK_STRIDE=HEAD_DIM + 4,
        PRESHUFFLE=preshuffle,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
    )
    return q8, weights


def index_qk_prep_fp4(
    query: torch.Tensor,
    gate: torch.Tensor,
    softmax_scale: float,
    key: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float,
    payload: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
    loc: torch.Tensor,
    *,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    weights_dtype: torch.dtype = torch.float32,
):
    """One launch for the MXFP4 indexer prologue: returns (Fp4IndexQuery, weights
    [T, H, 1]) as ``indexer_q_fp4_head_gate`` does and stores the LayerNorm'd keys
    as ``store_fp4_index_k`` does (payload / scale None skip the store)."""
    from sglang.kernels.ops.attention.dsa.fp4_index_hip import Fp4IndexQuery

    tokens, heads = _check_query_gate(query, gate)
    assert heads % 16 == 0 and heads <= 64, f"{heads} heads: AITER takes 16-head groups"
    num_keys = _check_key_norm(key, norm_weight, norm_bias, loc)
    if payload is None:
        num_keys = 0
    q_payload = torch.empty(
        (tokens, heads, HEAD_DIM // 2), dtype=torch.uint8, device=query.device
    )
    q_scale = torch.empty((tokens, 1, 4, 16, 4), dtype=torch.uint8, device=query.device)
    weights = torch.empty((tokens, heads, 1), dtype=weights_dtype, device=query.device)
    hb = _Q_FP4_HEADS_PER_PROGRAM
    q_programs = tokens * (64 // hb)
    if q_programs + num_keys:
        if payload is None:
            payload, scale = q_payload, q_scale
        _index_qk_prep_fp4_kernel[(q_programs + num_keys,)](
            query,
            gate,
            q_payload,
            q_scale,
            weights,
            key,
            norm_weight,
            norm_bias,
            loc,
            payload,
            scale,
            q_programs,
            key.stride(0),
            eps,
            HEADS=heads,
            GATE_STRIDE=gate.stride(0),
            GATE_SCALE=heads**-0.5,
            SOFTMAX_SCALE=softmax_scale,
            HB=hb,
            DCP_SIZE=dcp_size,
            DCP_RANK=dcp_rank,
        )
    return Fp4IndexQuery(q_payload, q_scale), weights


@triton.jit(do_not_specialize=["nope_stride", "rope_stride"])
def _fp8_ds_mla_store_kernel(
    NOPE,
    ROPE,
    LOC,
    DST_Q,
    DST_S,
    DST_R,
    nope_stride,
    rope_stride,
    reserved_skip_index,
    ROW_STRIDE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILES: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    DCP_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    loc = tl.load(LOC + t).to(tl.int64)
    # set_mla_kv_buffer_kernel's owner rule and reserved slot
    if (loc != reserved_skip_index) & (loc % DCP_SIZE == DCP_RANK):
        row = loc // DCP_SIZE
        # _quantize_k_cache_fast_kernel, one [TILES, 128] tile per token
        tiles = tl.arange(0, TILES)
        offs = tl.arange(0, 128)
        y = tl.load(NOPE + t * nope_stride + tiles[:, None] * 128 + offs[None, :]).to(
            tl.float32
        )
        y_s = tl.max(tl.abs(y), axis=1) / FP8_MAX
        if SCALE_UE8M0:
            bits = tl.maximum(y_s, 1e-4).to(tl.int32, bitcast=True)
            e = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
            y_s = (e << 23).to(tl.float32, bitcast=True)
        y_s_inv = 1.0 / y_s
        y_q = tl.clamp(y * y_s_inv[:, None], FP8_MIN, FP8_MAX).to(
            DST_Q.dtype.element_ty
        )
        tl.store(DST_Q + row * ROW_STRIDE + tiles[:, None] * 128 + offs[None, :], y_q)
        tl.store(DST_S + row * (ROW_STRIDE // 4) + DIM_NOPE // 4 + tiles, y_s)
        if DIM_ROPE > 0:
            r = tl.arange(0, DIM_ROPE)
            tl.store(
                DST_R + row * (ROW_STRIDE // 2) + (DIM_NOPE + 4 * TILES) // 2 + r,
                tl.load(ROPE + t * rope_stride + r),
            )


def fp8_ds_mla_store(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: Optional[torch.Tensor],
    *,
    scale_ue8m0: bool,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    reserved_skip_index: int = 0,
) -> None:
    """quantize_k_cache_separate + set_mla_kv_buffer(_dcp_sharded)_triton in one launch.

    kv_buffer: the layer's fp8_ds_mla rows [rows, (1,) row_bytes] in any dtype, each
    ``DIM_NOPE + 4 * DIM_NOPE / 128 + 2 * DIM_ROPE`` bytes at a 4-byte-aligned row
    stride; k_nope [T, (1,) 512] and k_rope [T, (1,) 64] bf16 with free row strides."""
    nope = k_nope.reshape(k_nope.shape[0], -1) if k_nope.ndim == 3 else k_nope
    t = nope.shape[0]
    dim_nope = nope.shape[1]
    if k_rope is None or k_rope.numel() == 0:
        rope, dim_rope = nope, 0
    else:
        rope = k_rope.reshape(k_rope.shape[0], -1) if k_rope.ndim == 3 else k_rope
        dim_rope = rope.shape[1]
    assert nope.dtype == torch.bfloat16 and nope.stride(-1) == 1
    assert dim_rope == 0 or (rope.dtype == torch.bfloat16 and rope.stride(-1) == 1)
    assert dim_nope % 128 == 0 and dim_rope in (0, 64)
    tiles = dim_nope // 128
    row_bytes = dim_nope + 4 * tiles + 2 * dim_rope
    row_stride = kv_buffer.stride(0) * kv_buffer.element_size()
    assert kv_buffer.shape[-1] * kv_buffer.element_size() == row_bytes
    assert row_stride % 4 == 0 and kv_buffer.stride(-1) == 1
    n = min(t, loc.numel())
    if n == 0:
        return
    # a flat byte view of the buffer's storage from its first row
    flat = torch.as_strided(
        kv_buffer.view(torch.uint8),
        ((kv_buffer.shape[0] - 1) * row_stride + row_bytes,),
        (1,),
    )
    fp8 = torch.float8_e4m3fn
    _fp8_ds_mla_store_kernel[(n,)](
        nope,
        rope,
        loc,
        flat.view(fp8),
        flat.view(torch.float32),
        flat.view(torch.bfloat16),
        nope.stride(0),
        rope.stride(0),
        reserved_skip_index,
        ROW_STRIDE=row_stride,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
        TILES=tiles,
        SCALE_UE8M0=scale_ue8m0,
        FP8_MIN=torch.finfo(fp8).min,
        FP8_MAX=torch.finfo(fp8).max,
        DCP_SIZE=dcp_size,
        DCP_RANK=dcp_rank,
    )
