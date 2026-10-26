"""Muse Spark full-attention extend (gfx950, Gluon): FP8 Q (per-tensor scale), paged FP8 KV (page 16), head dim 64,
8 queries per KV head; one program per (16-token query block, KV head).

Transposed form S^T = K Q^T, O^T = V^T P^T on 32x32x64 FP8 MFMAs with the warps splitting the 128 query columns:
the softmax statistics reduce over keys inside a warp, and P^T's accumulator registers are the B operand of the PV
MFMA once its key rows are relabelled (V^T is loaded in the same relabelled key order), so P never round-trips
through LDS. K / V tiles are async-copied into two LDS buffers (one barrier per tile) with page ids fetched two tiles
ahead. P is formed as 448 * exp2(.) (the FP8 E4M3 range), which scales numerator and denominator alike and keeps
small probabilities out of the FP8 subnormals.
"""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

BLOCK_Q = 16
TILE = 128
NUM_WARPS = 4


@gluon.jit
def _find_seq(cu_q, q_block, num_seqs, BLOCK_Q: gl.constexpr):
    # Request owning q-block q_block in the grid's upper-bound numbering (cu_q[s] // BLOCK_Q + s).
    lo = 0
    hi = num_seqs
    while lo < hi:
        mid = (lo + hi) // 2
        if gl.load(cu_q + mid) // BLOCK_Q + mid <= q_block:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


@gluon.jit
def _relabel(x, TILE: gl.constexpr, N: gl.constexpr):
    # Key rows 64 c + 32 t + 8 i + 4 h + j of the 32x32 accumulator -> 64 c + 32 h + 16 t + 4 i + j, the order the
    # k_width-32 B operand holds them in (gl.convert_layout(..., assert_trivial=True) checks it).
    y = gl.permute(gl.reshape(x, [TILE // 64, 2, 4, 2, 4, N]), [0, 3, 1, 2, 4, 5])
    return gl.reshape(y, [TILE, N])


@gluon.jit
def _pages(bt, j, kr, vr, seq_len, PAGE: gl.constexpr, TILE: gl.constexpr, n_tiles):
    # Page ids of tile j's K rows (key order) and V rows (relabelled order); 0 past the end.
    key = j * TILE + kr
    key_v = j * TILE + vr
    live = j < n_tiles
    page = gl.load(bt + key // PAGE, mask=live & (key < seq_len), other=0)
    page_v = gl.load(bt + key_v // PAGE, mask=live & (key_v < seq_len), other=0)
    return page, page_v


@gluon.jit
def _issue(
    K,
    V,
    j,
    page,
    page_v,
    kr,
    vr,
    kd,
    seq_len,
    kvh,
    k_stride0,
    k_stride1,
    k_stride2,
    v_stride0,
    v_stride1,
    v_stride2,
    k_buf,
    v_buf,
    PAGE: gl.constexpr,
    TILE: gl.constexpr,
    n_tiles,
    MASKED: gl.constexpr,
):
    key = j * TILE + kr
    key_v = j * TILE + vr
    k_ptr = (
        K
        + page.to(gl.int64)[:, None] * k_stride0
        + (key % PAGE)[:, None] * k_stride1
        + kvh * k_stride2
        + kd[None, :]
    )
    v_ptr = (
        V
        + page_v.to(gl.int64)[:, None] * v_stride0
        + (key_v % PAGE)[:, None] * v_stride1
        + kvh * v_stride2
        + kd[None, :]
    )
    if MASKED:
        live = j < n_tiles
        gl.amd.cdna4.async_copy.global_load_to_shared(
            k_buf, k_ptr, mask=(live & (key < seq_len))[:, None], other=0.0
        )
        gl.amd.cdna4.async_copy.global_load_to_shared(
            v_buf, v_ptr, mask=(live & (key_v < seq_len))[:, None], other=0.0
        )
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(k_buf, k_ptr)
        gl.amd.cdna4.async_copy.global_load_to_shared(v_buf, v_ptr)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _step(
    j,
    pg,
    pg_v,
    k_bufs,
    v_bufs,
    qt,
    m_i,
    l_i,
    acc,
    c,
    q_pos,
    seq_len,
    K,
    V,
    bt,
    kr,
    vr,
    kd,
    kvh,
    k_stride0,
    k_stride1,
    k_stride2,
    v_stride0,
    v_stride1,
    v_stride2,
    n_tiles,
    mfma: gl.constexpr,
    a_op: gl.constexpr,
    b_op: gl.constexpr,
    PAGE: gl.constexpr,
    TILE: gl.constexpr,
    BM: gl.constexpr,
    MASKED: gl.constexpr,
    NEXT_MASKED: gl.constexpr,
):
    # pg / pg_v: page ids of tile j + 1, loaded one iteration earlier. One barrier: tile j's copies (the only
    # group in flight) are visible to every warp, and every warp is done with tile j - 1's buffer, which takes
    # tile j + 1 while tile j computes.
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.barrier()
    nb = (j + 1) % 2
    _issue(
        K,
        V,
        j + 1,
        pg,
        pg_v,
        kr,
        vr,
        kd,
        seq_len,
        kvh,
        k_stride0,
        k_stride1,
        k_stride2,
        v_stride0,
        v_stride1,
        v_stride2,
        k_bufs.index(nb),
        v_bufs.index(nb),
        PAGE,
        TILE,
        n_tiles,
        NEXT_MASKED,
    )
    pg, pg_v = _pages(bt, j + 2, kr, vr, seq_len, PAGE, TILE, n_tiles)
    cb = j % 2
    ka = gl.amd.cdna4.async_copy.load_shared_relaxed(k_bufs.index(cb), a_op)
    s = gl.amd.cdna4.mfma_scaled(
        ka, None, "e4m3", qt, None, "e4m3", gl.zeros([TILE, BM], gl.float32, layout=mfma)
    )
    if MASKED:
        kpos = j * TILE + gl.arange(0, TILE, layout=gl.SliceLayout(1, mfma))
        ok = (kpos[:, None] <= q_pos[None, :]) & (kpos < seq_len)[:, None]
        s = gl.where(ok, s, float("-inf"))
    m_new = gl.maximum(m_i, gl.max(s, axis=0) * c)
    # log2(448): P = 448 * exp2(s * c - m), the FP8 E4M3 range.
    p = gl.exp2(s * c - (m_new - 8.807354922057604)[None, :])
    alpha = gl.exp2(m_i - m_new)
    l_i = l_i * alpha + gl.sum(p, axis=0)
    pb = gl.convert_layout(
        _relabel(p.to(gl.float8e4nv), TILE, BM), b_op, assert_trivial=True
    )
    vt = gl.amd.cdna4.async_copy.load_shared_relaxed(
        v_bufs.index(cb).permute([1, 0]), a_op
    )
    acc = gl.amd.cdna4.mfma_scaled(
        vt, None, "e4m3", pb, None, "e4m3", acc * alpha[None, :]
    )
    return pg, pg_v, m_new, l_i, acc


@gluon.jit
def _muse_extend_kernel(
    OUT,
    Q,
    K,
    V,
    BT,
    SEQ_LENS,
    CU_Q,
    QD,
    KD,
    VD,
    scale,
    num_seqs,
    bt_stride,
    q_stride0,
    o_stride0,
    k_stride0,
    k_stride1,
    k_stride2,
    v_stride0,
    v_stride1,
    v_stride2,
    GQA: gl.constexpr,
    D: gl.constexpr,
    PAGE: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    TILE: gl.constexpr,
    NW: gl.constexpr,
):
    BM: gl.constexpr = BLOCK_Q * GQA
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 64], transposed=False, warps_per_cta=[1, NW]
    )
    a_op: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=32)
    b_op: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=32)
    col: gl.constexpr = gl.SliceLayout(0, mfma)
    row: gl.constexpr = gl.SliceLayout(1, mfma)
    kv_blk: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [NW, 1], [1, 0])
    q_blk: gl.constexpr = gl.BlockedLayout([16, 1], [4, 16], [1, NW], [0, 1])
    smem_l: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 1, order=[1, 0])

    kvh = gl.program_id(0)
    qb = gl.program_id(1)
    seq = _find_seq(CU_Q, qb, num_seqs, BLOCK_Q)
    q_start = gl.load(CU_Q + seq)
    q_len = gl.load(CU_Q + seq + 1) - q_start
    local = qb - (q_start // BLOCK_Q + seq)
    if local * BLOCK_Q < q_len:
        # int32 loop bounds: the LDS buffer index must be int32 (seq_lens may be int64).
        seq_len = gl.load(SEQ_LENS + seq).to(gl.int32)
        ctx = seq_len - q_len
        c = scale * 1.4426950408889634 * gl.load(QD) * gl.load(KD)

        # Q^T [D, BM]: column r = (token local * BLOCK_Q + r // GQA, head kvh * GQA + r % GQA)
        qr = gl.arange(0, BM, layout=gl.SliceLayout(0, q_blk))
        qdim = gl.arange(0, D, layout=gl.SliceLayout(1, q_blk))
        q_tok = local * BLOCK_Q + qr // GQA
        qt = gl.load(
            Q
            + (q_start + q_tok)[None, :].to(gl.int64) * q_stride0
            + (kvh * GQA + qr % GQA)[None, :] * D
            + qdim[:, None],
            mask=(q_tok < q_len)[None, :],
            other=0.0,
        )
        qt = gl.convert_layout(qt, b_op)

        k_bufs = gl.allocate_shared_memory(K.dtype.element_ty, [2, TILE, D], smem_l)
        v_bufs = gl.allocate_shared_memory(V.dtype.element_ty, [2, TILE, D], smem_l)
        kr = gl.arange(0, TILE, layout=gl.SliceLayout(1, kv_blk))
        kd = gl.arange(0, D, layout=gl.SliceLayout(0, kv_blk))
        vr = gl.reshape(_relabel(gl.reshape(kr, [TILE, 1]), TILE, 1), [TILE])
        vr = gl.convert_layout(vr, gl.SliceLayout(1, kv_blk))
        bt = BT + seq * bt_stride

        q_pos = ctx + local * BLOCK_Q + gl.arange(0, BM, layout=col) // GQA
        m_i = gl.full([BM], float("-inf"), gl.float32, layout=col)
        l_i = gl.zeros([BM], gl.float32, layout=col)
        acc = gl.zeros([D, BM], gl.float32, layout=mfma)
        # Tiles below `interior` are seen in full by every query of the block.
        interior = (ctx + local * BLOCK_Q + 1) // TILE
        n_tiles = gl.cdiv(ctx + gl.minimum(local * BLOCK_Q + BLOCK_Q, q_len), TILE)
        pg, pg_v = _pages(bt, 0, kr, vr, seq_len, PAGE, TILE, n_tiles)
        _issue(
            K,
            V,
            0,
            pg,
            pg_v,
            kr,
            vr,
            kd,
            seq_len,
            kvh,
            k_stride0,
            k_stride1,
            k_stride2,
            v_stride0,
            v_stride1,
            v_stride2,
            k_bufs.index(0),
            v_bufs.index(0),
            PAGE,
            TILE,
            n_tiles,
            True,
        )
        pg, pg_v = _pages(bt, 1, kr, vr, seq_len, PAGE, TILE, n_tiles)
        for j in range(0, interior - 1):
            pg, pg_v, m_i, l_i, acc = _step(
                j, pg, pg_v, k_bufs, v_bufs, qt, m_i, l_i, acc, c, q_pos, seq_len,
                K, V, bt, kr, vr, kd, kvh, k_stride0, k_stride1, k_stride2,
                v_stride0, v_stride1, v_stride2, n_tiles, mfma, a_op, b_op, PAGE, TILE, BM,
                False, False,
            )  # fmt: skip
        for j in range(gl.maximum(interior - 1, 0), interior):
            pg, pg_v, m_i, l_i, acc = _step(
                j, pg, pg_v, k_bufs, v_bufs, qt, m_i, l_i, acc, c, q_pos, seq_len,
                K, V, bt, kr, vr, kd, kvh, k_stride0, k_stride1, k_stride2,
                v_stride0, v_stride1, v_stride2, n_tiles, mfma, a_op, b_op, PAGE, TILE, BM,
                False, True,
            )  # fmt: skip
        for j in range(interior, n_tiles):
            pg, pg_v, m_i, l_i, acc = _step(
                j, pg, pg_v, k_bufs, v_bufs, qt, m_i, l_i, acc, c, q_pos, seq_len,
                K, V, bt, kr, vr, kd, kvh, k_stride0, k_stride1, k_stride2,
                v_stride0, v_stride1, v_stride2, n_tiles, mfma, a_op, b_op, PAGE, TILE, BM,
                True, True,
            )  # fmt: skip
        gl.amd.cdna4.async_copy.wait_group(0)
        o = acc * (gl.load(VD) / l_i)[None, :]
        od = gl.arange(0, D, layout=row)
        orr = gl.arange(0, BM, layout=col)
        o_tok = local * BLOCK_Q + orr // GQA
        gl.store(
            OUT
            + (q_start + o_tok)[None, :].to(gl.int64) * o_stride0
            + (kvh * GQA + orr % GQA)[None, :] * D
            + od[:, None],
            o.to(OUT.dtype.element_ty),
            mask=(o_tok < q_len)[None, :],
        )


def muse_extend_applies(q: torch.Tensor, k_cache: torch.Tensor) -> bool:
    """FP8 Q [tokens, 8 * KV heads, 64] over a page-16 FP8 KV cache [pages, 16, KV heads, 64]."""
    return (
        q.dim() == 3
        and k_cache.dim() == 4
        and q.shape[2] == k_cache.shape[3] == 64
        and k_cache.shape[1] == 16
        and q.shape[1] == 8 * k_cache.shape[2]
        and q.dtype == k_cache.dtype == torch.float8_e4m3fn
    )


def muse_extend_attention(
    q, k_cache, v_cache, out, cu_seqlens_q, seq_lens, block_table, q_descale, k_descale, v_descale, scale
):
    """Causal full-attention extend: ``out [tokens, heads, 64]`` (BF16) from FP8 ``q`` and the paged FP8 cache.

    ``block_table [requests, pages]`` holds page ids, ``seq_lens`` the KV lengths including the new tokens and
    ``cu_seqlens_q`` the query offsets; the descales are per-tensor ``[1]`` FP32."""
    n = q.shape[0]
    hk = k_cache.shape[2]
    bs = seq_lens.numel()
    _muse_extend_kernel[(hk, n // BLOCK_Q + bs)](
        out,
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        cu_seqlens_q,
        q_descale,
        k_descale,
        v_descale,
        scale,
        bs,
        block_table.stride(0),
        q.stride(0),
        out.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        GQA=q.shape[1] // hk,
        D=q.shape[2],
        PAGE=k_cache.shape[1],
        BLOCK_Q=BLOCK_Q,
        TILE=TILE,
        NW=NUM_WARPS,
        num_warps=NUM_WARPS,
        waves_per_eu=2,
    )
    return out
