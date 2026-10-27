"""MiniMax-M3 sparse prefill attention on gfx950, two queries per program.

Consecutive queries share most of their top-k blocks, so one single-warp program walks the
union of both queries' blocks and loads each K/V tile once for both; each query's 16 heads
are their own 16-row MFMA tile, which skips the blocks that query did not select. K/V come
from the bf16 SHUFFLE scratch that ``gluon_prefill`` gathers (64-token pages, x = 8; an fp8
pool is widened and an NVFP4 pool dequantized there, both exactly):

- K page [D // 8, 64, 8]: elem(d, t) at (d // 8) * 512 + t * 8 + d % 8
- V page [64 // 8, D, 8]: elem(t, d) at (t // 8) * 8 * D + d * 8 + t % 8

and load straight into the MFMA operand layouts: 8 contiguous dims per token for K and 8
contiguous tokens per dim for V, so no LDS transpose is needed.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna4 import buffer_load, buffer_store, mfma

from ..common.out_quant_gfx950 import store_fp8_out

_LOG2E = 1.4426950408889634
# Measured on MI350X: below these the single-warp programs underfill the GPU; the per-query
# kernel reads an fp8 scratch at half the bytes, so pairs win later there.
PAIR_MIN_TOKENS_FP8 = 4096
PAIR_MIN_TOKENS_BF16 = 2048
# Programs are dispatched round-robin over MI350X's 8 XCDs.
_XCDS = 8


@gluon.jit
def _pair_keys(
    topk_ptr, req_id_ptr, page_start_ptr, q0, c, total_q, topk, stride_t_n, TOPK_P
):
    # Scratch page of each candidate block of the pair; int32 max marks an empty entry.
    cq = q0 + c // TOPK_P
    q_ok = cq < total_q
    ok = q_ok & (c % TOPK_P < topk)
    blk = gl.load(
        topk_ptr + cq.to(gl.int64) * stride_t_n + c % TOPK_P, mask=ok, other=-1
    )
    req = gl.load(req_id_ptr + cq, mask=q_ok, other=0)
    return gl.where(blk >= 0, gl.load(page_start_ptr + req) + 2 * blk, 0x7FFFFFFF)


@gluon.jit
def _pair_union(
    topk_ptr,
    req_id_ptr,
    page_start_ptr,
    q0,
    total_q,
    topk,
    stride_t_n,
    TOPK_P: gl.constexpr,
    pair_layout: gl.constexpr,
):
    """The pair's candidate pages ``key_j`` (query 0's in the first TOPK_P entries), and
    the union of their distinct pages ascending, ``union_j[:n_blocks]``."""
    NC: gl.constexpr = 2 * TOPK_P
    ci = gl.arange(0, NC, layout=gl.SliceLayout(1, pair_layout))
    cj = gl.arange(0, NC, layout=gl.SliceLayout(0, pair_layout))
    key_i = _pair_keys(
        topk_ptr, req_id_ptr, page_start_ptr, q0, ci, total_q, topk, stride_t_n, TOPK_P
    )
    key_j = _pair_keys(
        topk_ptr, req_id_ptr, page_start_ptr, q0, cj, total_q, topk, stride_t_n, TOPK_P
    )
    # Drop repeats, then rank-sort.
    same = key_j[None, :] == key_i[:, None]
    uniq_i = gl.where(
        gl.sum((same & (cj[None, :] < ci[:, None])).to(gl.int32), axis=1) == 0,
        key_i,
        0x7FFFFFFF,
    )
    uniq_j = gl.where(
        gl.sum((same & (ci[:, None] < cj[None, :])).to(gl.int32), axis=0) == 0,
        key_j,
        0x7FFFFFFF,
    )
    # Distinct keys have distinct ranks; repeats and empty entries rank past them.
    rank_i = gl.sum((uniq_j[None, :] < uniq_i[:, None]).to(gl.int32), axis=1)
    union_j = gl.sum(
        gl.where(
            (rank_i[:, None] == cj[None, :]) & (uniq_i != 0x7FFFFFFF)[:, None],
            uniq_i[:, None],
            0,
        ),
        axis=0,
    )
    n_blocks = gl.sum((uniq_i != 0x7FFFFFFF).to(gl.int32), axis=0)
    return key_j, cj, union_j, n_blocks


@gluon.jit
def _query_step(
    q,
    k,
    v,
    m_i,
    l_i,
    acc,
    is_self,
    tail,
    first,
    t_cols,
    qk_scale,
    mfma_layout: gl.constexpr,
    dot_a: gl.constexpr,
    H: gl.constexpr,
    SUB: gl.constexpr,
):
    """One query's online-softmax update over a SUB-token tile starting at token
    ``first`` of a block; only the query's own block masks tokens past it."""
    s = mfma(q, k, gl.zeros([H, SUB], gl.float32, mfma_layout))
    if is_self:
        s = gl.where(((first + t_cols) < tail)[None, :], s * qk_scale, float("-inf"))
        # The running max starts finite, so fully masked rows keep p = 0 and alpha = 1.
        m_new = gl.maximum(m_i, gl.max(s, axis=1))
        p = gl.exp2(s - m_new[:, None])
    else:
        m_new = gl.maximum(m_i, gl.max(s, axis=1) * qk_scale)
        p = gl.exp2(gl.fma(s, qk_scale, -m_new[:, None]))
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + gl.sum(p, axis=1)
    acc = acc * alpha[:, None]
    acc = mfma(gl.convert_layout(p.to(gl.bfloat16), dot_a), v, acc)
    return m_new, l_i, acc


@gluon.jit
def _store_query_out(
    acc,
    l_i,
    v_scale,
    row,
    o_ptr,
    o_off,
    stride_o_n,
    q8_ptr,
    q8_off,
    stride_q8_n,
    scale_ptr,
    scale_off,
    OUT_QUANT: gl.constexpr,
):
    """One query's O: bf16 [H, D], or with OUT_QUANT the per 1 x 128 FP8 bytes and
    scales the o_proj's separate quant of that bf16 O writes (see ``store_fp8_out``)."""
    out = (acc * (v_scale / gl.where(l_i > 0, l_i, 1.0))[:, None]).to(
        o_ptr.dtype.element_ty
    )
    if OUT_QUANT:
        store_fp8_out(
            out.to(gl.float32),
            q8_ptr + q8_off + row * stride_q8_n,
            scale_ptr,
            scale_off + row,
            2,
        )
    else:
        buffer_store(out, ptr=o_ptr, offsets=o_off + row * stride_o_n)


@gluon.jit
def _sparse_prefill_pair_kernel(
    q_ptr,  # [total_q, H, D] bf16
    k_scr_ptr,  # bf16 SHUFFLE K pages
    v_scr_ptr,  # bf16 SHUFFLE V pages
    topk_ptr,  # [total_q, topk] int32 block ids, -1 = empty
    o_ptr,  # [total_q, H, D] bf16; only its dtype is read with OUT_QUANT
    q8_ptr,  # [total_q, H * D] fp8, with OUT_QUANT
    scale_ptr,  # [H, total_q] fp32 (column-major per 1 x 128 scales), with OUT_QUANT
    req_id_ptr,  # [total_q] int32
    abs_pos_ptr,  # [total_q] int32
    page_start_ptr,  # [batch] int32 first scratch page of each request
    total_q,
    topk,
    qk_scale,  # sm_scale * k_scale * log2(e)
    v_scale,
    stride_q_n,
    stride_q_h,
    stride_t_n,
    stride_o_n,
    stride_o_h,
    stride_q8_n,
    stride_s_h,
    H: gl.constexpr,
    D: gl.constexpr,
    TOPK_P: gl.constexpr,
    X: gl.constexpr,  # scratch elements per 16 bytes
    XCDS: gl.constexpr,
    OUT_QUANT: gl.constexpr,
    num_warps: gl.constexpr,
):
    SUB: gl.constexpr = 32
    PAGE: gl.constexpr = 64
    PAGE_ELEMS: gl.constexpr = PAGE * D
    BLOCK_TOKENS: gl.constexpr = 128
    # Each XCD runs a contiguous range of pairs, so the blocks neighbouring queries share
    # stay in its L2.
    pid = gl.program_id(0)
    q0 = ((pid % XCDS) * gl.cdiv(gl.num_programs(0), XCDS) + pid // XCDS) * 2
    if q0 >= total_q:
        return

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    dot_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    pair_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [2, 32], [num_warps, 1], [1, 0]
    )
    key_j, cj, union_j, n_blocks = _pair_union(
        topk_ptr,
        req_id_ptr,
        page_start_ptr,
        q0,
        total_q,
        topk,
        stride_t_n,
        TOPK_P,
        pair_layout,
    )

    q1 = gl.minimum(q0 + 1, total_q - 1)
    q1_ok = q0 + 1 < total_q
    pos0 = gl.load(abs_pos_ptr + q0)
    pos1 = gl.load(abs_pos_ptr + q1)
    req0 = gl.load(req_id_ptr + q0)
    req1 = gl.load(req_id_ptr + q1)
    self0 = gl.load(page_start_ptr + req0) + 2 * (pos0 // BLOCK_TOKENS)
    self1 = gl.load(page_start_ptr + req1) + 2 * (pos1 // BLOCK_TOKENS)
    t_cols = gl.arange(0, SUB, layout=col_layout)

    qh = gl.arange(0, H, layout=gl.SliceLayout(1, dot_a))
    qd = gl.arange(0, D, layout=gl.SliceLayout(0, dot_a))
    q_off = (qh * stride_q_h)[:, None] + qd[None, :]
    q_a = buffer_load(ptr=q_ptr, offsets=q_off + q0 * stride_q_n)
    q_b = buffer_load(ptr=q_ptr, offsets=q_off + q1 * stride_q_n)
    kd = gl.arange(0, D, layout=gl.SliceLayout(1, dot_b))
    kt = gl.arange(0, SUB, layout=gl.SliceLayout(0, dot_b))
    k_off = (kd // X)[:, None] * (PAGE * X) + kt[None, :] * X + (kd % X)[:, None]
    vt = gl.arange(0, SUB, layout=gl.SliceLayout(1, dot_b))
    vd = gl.arange(0, D, layout=gl.SliceLayout(0, dot_b))
    v_off = (vt // X)[:, None] * (X * D) + vd[None, :] * X + (vt % X)[:, None]

    m_a = gl.full([H], -1e30, gl.float32, row_layout)
    l_a = gl.zeros([H], gl.float32, row_layout)
    acc_a = gl.zeros([H, D], gl.float32, mfma_layout)
    m_b = gl.full([H], -1e30, gl.float32, row_layout)
    l_b = gl.zeros([H], gl.float32, row_layout)
    acc_b = gl.zeros([H, D], gl.float32, mfma_layout)
    for it in range(n_blocks):
        key = gl.sum(gl.where(cj == it, union_j, 0), axis=0)
        hit0 = gl.sum(((key_j == key) & (cj < TOPK_P)).to(gl.int32), axis=0) > 0
        hit1 = gl.sum(((key_j == key) & (cj >= TOPK_P)).to(gl.int32), axis=0) > 0
        # A selected block past the query's own block holds only future tokens.
        active0 = hit0 & (key <= self0)
        active1 = hit1 & (key <= self1) & q1_ok
        for sub in gl.static_range(BLOCK_TOKENS // SUB):
            base = (key + sub // 2) * PAGE_ELEMS
            k = buffer_load(ptr=k_scr_ptr, offsets=k_off + base + (sub % 2) * SUB * X)
            v = buffer_load(ptr=v_scr_ptr, offsets=v_off + base + (sub % 2) * SUB * D)
            if active0:
                m_a, l_a, acc_a = _query_step(
                    q_a,
                    k,
                    v,
                    m_a,
                    l_a,
                    acc_a,
                    key == self0,
                    pos0 % BLOCK_TOKENS + 1,
                    sub * SUB,
                    t_cols,
                    qk_scale,
                    mfma_layout,
                    dot_a,
                    H,
                    SUB,
                )
            if active1:
                m_b, l_b, acc_b = _query_step(
                    q_b,
                    k,
                    v,
                    m_b,
                    l_b,
                    acc_b,
                    key == self1,
                    pos1 % BLOCK_TOKENS + 1,
                    sub * SUB,
                    t_cols,
                    qk_scale,
                    mfma_layout,
                    dot_a,
                    H,
                    SUB,
                )

    oh = gl.arange(0, H, layout=row_layout)
    od = gl.arange(0, D, layout=col_layout)
    o_off = (oh * stride_o_h)[:, None] + od[None, :]
    q8_off = (oh * D)[:, None] + od[None, :]
    scale_off = oh * stride_s_h
    _store_query_out(
        acc_a,
        l_a,
        v_scale,
        q0,
        o_ptr,
        o_off,
        stride_o_n,
        q8_ptr,
        q8_off,
        stride_q8_n,
        scale_ptr,
        scale_off,
        OUT_QUANT,
    )
    if q1_ok:
        _store_query_out(
            acc_b,
            l_b,
            v_scale,
            q1,
            o_ptr,
            o_off,
            stride_o_n,
            q8_ptr,
            q8_off,
            stride_q8_n,
            scale_ptr,
            scale_off,
            OUT_QUANT,
        )


def can_use_pair_prefill(q: torch.Tensor, fp8_pool: bool) -> bool:
    min_tokens = PAIR_MIN_TOKENS_FP8 if fp8_pool else PAIR_MIN_TOKENS_BF16
    return (
        q.shape[0] >= min_tokens
        and q.shape[1] == 16
        and q.shape[2] == 128
        and q.dtype == torch.bfloat16
    )


def pair_sparse_prefill(
    q: torch.Tensor,  # [total_q, 16, 128] bf16
    k_scratch: torch.Tensor,  # bf16 SHUFFLE K pages from gluon_prefill
    v_scratch: torch.Tensor,  # bf16 SHUFFLE V pages
    topk_idx: torch.Tensor,  # [1, total_q, topk] int32
    req_id: torch.Tensor,  # [total_q] int32
    abs_pos: torch.Tensor,  # [total_q] int32
    page_start: torch.Tensor,  # [batch] int32
    sm_scale: float,
    k_scale: float,
    v_scale: float,
    out_quant: str = "",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sparse prefill attention, two queries per program, over gluon_prefill's scratch.

    ``out_quant`` is the o_proj's ``fp8_input_quant_format``: for "fp8" O comes back
    as the (fp8 [total_q, H * D], scale [total_q, H] column-major) pair the o_proj's
    separate per 1 x 128 quant would write, else as bf16."""
    total_q, num_heads, head_dim = q.shape
    topk = topk_idx.shape[2]
    topk_rows = topk_idx[0]
    out_fp8 = out_quant == "fp8"
    if out_fp8:
        q8 = torch.empty(
            total_q, num_heads * head_dim, dtype=torch.float8_e4m3fn, device=q.device
        )
        # The block GEMM reads per 1 x 128 scales column-major: [heads, rows] bytes.
        scale = torch.empty(num_heads, total_q, dtype=torch.float32, device=q.device)
        out = q
    else:
        out = torch.empty_like(q)
        q8, scale = out, out
    num_pairs = triton.cdiv(total_q, 2)
    _sparse_prefill_pair_kernel[(triton.cdiv(num_pairs, _XCDS) * _XCDS,)](
        q,
        k_scratch,
        v_scratch,
        topk_rows,
        out,
        q8,
        scale,
        req_id,
        abs_pos,
        page_start,
        total_q,
        topk,
        sm_scale * k_scale * _LOG2E,
        v_scale,
        q.stride(0),
        q.stride(1),
        topk_rows.stride(0),
        out.stride(0),
        out.stride(1),
        q8.stride(0),
        scale.stride(0),
        H=num_heads,
        D=head_dim,
        TOPK_P=max(16, triton.next_power_of_2(topk)),
        X=16 // k_scratch.element_size(),
        XCDS=_XCDS,
        OUT_QUANT=out_fp8,
        num_warps=1,
        # Two waves per SIMD fit in 256 registers here; the default allocation runs one.
        waves_per_eu=2,
    )
    if out_fp8:
        return q8, scale.t()
    return out
