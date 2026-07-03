"""Triton sparse MLA decode kernel with fp8 KV cache support.

Adapted from aiter's unified_attention_sparse_mla kernel for DSA shapes:
  q:       [bs, H, DIM]     fp8 (DIM=576 = D_V+D_TAIL)
  kv:      [num_pages, 1, DIM]  fp8
  indices: [bs, 1, topk]    int32
  output:  [1, bs, H, D_V]  bf16

Two variants:
  1. Base: single-pass per-token kernel (adapted from aiter)
  2. Split-K: adaptive split-K with fused fast path (adapted from DSv4)
"""

import functools

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Variant 1: Base single-pass kernel (adapted from aiter sparse MLA)
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_mla_decode_base_kernel(
    q_ptr,  # [N, H, Q_DIM] fp8
    kv_ptr,  # [num_pages, 1, KV_DIM] fp8
    idx_ptr,  # [N, topk] int32
    out_ptr,  # [N, H, D_V] bf16
    sm_scale,
    topk: tl.constexpr,
    H: tl.constexpr,
    Q_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_block = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M) + head_block * BLOCK_M
    h_mask = offs_m < H

    offs_nope = tl.arange(0, D_V)
    offs_rope = tl.arange(0, D_TAIL)

    # Load Q nope part [BLOCK_M, D_V]
    q_nope = tl.load(
        q_ptr + token_idx * H * Q_DIM + offs_m[:, None] * Q_DIM + offs_nope[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    # Load Q rope part [BLOCK_M, D_TAIL]
    q_rope = tl.load(
        q_ptr
        + token_idx * H * Q_DIM
        + offs_m[:, None] * Q_DIM
        + (D_V + offs_rope)[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    M_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_V], dtype=tl.float32)

    offs_t = tl.arange(0, TILE_SIZE)
    num_tiles = (topk + TILE_SIZE - 1) // TILE_SIZE

    for t in range(num_tiles):
        tile_start = t * TILE_SIZE
        valid = (tile_start + offs_t) < topk

        idx = tl.load(
            idx_ptr + token_idx * topk + tile_start + offs_t,
            mask=valid,
            other=0,
        )
        valid = valid & (idx >= 0)
        page = tl.where(valid, idx, 0)

        # Load KV — addressed with KV_DIM stride (may differ from Q_DIM)
        kv_base = kv_ptr + page[:, None] * KV_DIM
        kv_nope = tl.load(
            kv_base + offs_nope[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        # Load KV rope [TILE_SIZE, D_TAIL]
        kv_rope = tl.load(
            kv_base + (D_V + offs_rope)[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        # QK = q_nope @ kv_nope.T + q_rope @ kv_rope.T  [BLOCK_M, TILE_SIZE]
        S = tl.dot(q_nope, tl.trans(kv_nope)).to(tl.float32)
        S += tl.dot(q_rope, tl.trans(kv_rope)).to(tl.float32)
        S = S * sm_scale
        S = tl.where(valid[None, :], S, float("-inf"))

        # Online softmax
        m_new = tl.maximum(M_i, tl.max(S, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(M_i - m_safe)
        p = tl.exp(S - m_safe[:, None])
        L_i = L_i * alpha + tl.sum(p, axis=1)

        # PV = P @ V_nope  [BLOCK_M, D_V]
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv_nope).to(tl.float32)
        M_i = m_new

    # Normalize
    l_safe = tl.where(L_i == 0.0, 1.0, L_i)
    acc = acc / l_safe[:, None]

    tl.store(
        out_ptr + token_idx * H * D_V + offs_m[:, None] * D_V + offs_nope[None, :],
        acc.to(tl.bfloat16),
        mask=h_mask[:, None],
    )


def triton_sparse_mla_decode_base(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Base single-pass Triton sparse MLA decode.

    q:       [bs, H, DIM] fp8
    kv:      [num_pages, 1, DIM] fp8
    indices: [bs, 1, topk] int32
    returns: [1, bs, H, d_v] bf16
    """
    bs, H, q_dim = q.shape
    kv_dim = kv.shape[-1]
    d_tail = q_dim - d_v
    topk = indices.shape[-1]
    idx_flat = indices.squeeze(1).contiguous()  # [bs, topk]

    out = torch.empty(bs, H, d_v, device=q.device, dtype=torch.bfloat16)

    BLOCK_M = 16
    n_head_blocks = (H + BLOCK_M - 1) // BLOCK_M
    TILE_SIZE = 64

    _sparse_mla_decode_base_kernel[(bs, n_head_blocks)](
        q,
        kv,
        idx_flat,
        out,
        sm_scale,
        topk=topk,
        H=H,
        Q_DIM=q_dim,
        KV_DIM=kv_dim,
        D_V=d_v,
        D_TAIL=d_tail,
        BLOCK_M=BLOCK_M,
        TILE_SIZE=TILE_SIZE,
        num_warps=4,
        num_stages=1,
    )
    return out.unsqueeze(0)


# ---------------------------------------------------------------------------
# Variant 2: Split-K kernel (adapted from DSv4 paged_decode.py)
# ---------------------------------------------------------------------------

LOG2E = 1.4426950408889634


@functools.lru_cache(maxsize=1)
def _cu_count() -> int:
    from aiter.ops.triton.utils.device_info import get_num_sms

    return get_num_sms()


def _prev_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _kv_splits_heuristic(
    T: int,
    H: int,
    block_h: int,
    num_cu: int | None = None,
    target_wg_per_cu: float = 2.0,
    max_kv_splits: int = 64,
) -> int:
    if num_cu is None:
        num_cu = _cu_count()
    target_wg = max(1, int(target_wg_per_cu * num_cu))
    head_blocks = max(1, (H + block_h - 1) // block_h)
    base_ctas = max(1, T * head_blocks)
    if base_ctas >= target_wg:
        return 1
    splits_to_fill = max(1, target_wg // base_ctas)
    return _prev_pow2(min(splits_to_fill, max_kv_splits))


@triton.jit
def _sparse_mla_decode_fused_kernel(
    q_ptr,  # [N, H, Q_DIM]
    kv_ptr,  # [num_pages, 1, KV_DIM]
    idx_ptr,  # [N, topk]
    out_ptr,  # [N, H, D_V]
    qk_scale,  # sm_scale * LOG2E
    topk: tl.constexpr,
    H: tl.constexpr,
    Q_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    d_nope = tl.arange(0, D_V)
    d_rope = tl.arange(0, D_TAIL)

    # Load Q — addressed with Q_DIM stride
    q_base = q_ptr + t * H * Q_DIM
    q_nope = tl.load(
        q_base + h_offs[:, None] * Q_DIM + d_nope[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_rope = tl.load(
        q_base + h_offs[:, None] * Q_DIM + (D_V + d_rope)[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D_V), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    num_tiles = tl.cdiv(topk, BLOCK_K)

    for j in tl.range(0, num_tiles, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < topk

        slot = tl.load(idx_ptr + t * topk + k_pos, mask=valid, other=0)
        valid = valid & (slot >= 0)
        page = tl.where(valid, slot, 0)

        # KV addressed with KV_DIM stride (may differ from Q_DIM due to padding)
        kv_base = kv_ptr + page[:, None] * KV_DIM
        kv_nope = tl.load(
            kv_base + d_nope[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        kv_rope = tl.load(
            kv_base + (D_V + d_rope)[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q_nope, tl.trans(kv_nope)) + tl.dot(q_rope, tl.trans(kv_rope))
        scores = scores * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv_nope).to(tl.float32)
        m_i = m_new
        l_i = l_new

    denom = tl.maximum(l_i, 1.0e-30)
    out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)

    tl.store(
        out_ptr + t * H * D_V + h_offs[:, None] * D_V + d_nope[None, :],
        out.to(tl.bfloat16),
        mask=h_mask[:, None],
    )


@triton.jit
def _sparse_mla_decode_split_kernel(
    q_ptr,  # [N, H, Q_DIM]
    kv_ptr,  # [num_pages, 1, KV_DIM]
    idx_ptr,  # [N, topk]
    m_partial_ptr,  # [N, KV_SPLITS, H_padded]
    l_partial_ptr,  # [N, KV_SPLITS, H_padded]
    acc_partial_ptr,  # [N, KV_SPLITS, H_padded, D_V]
    qk_scale,
    topk: tl.constexpr,
    H: tl.constexpr,
    Q_DIM: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    d_nope = tl.arange(0, D_V)
    d_rope = tl.arange(0, D_TAIL)

    q_base = q_ptr + t * H * Q_DIM
    q_nope = tl.load(
        q_base + h_offs[:, None] * Q_DIM + d_nope[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    q_rope = tl.load(
        q_base + h_offs[:, None] * Q_DIM + (D_V + d_rope)[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    # Split-K: each CTA handles a portion of topk
    tiles_per_segment = tl.cdiv(topk, KV_SPLITS * BLOCK_K)
    if pid_k * tiles_per_segment * BLOCK_K >= topk:
        return
    num_tiles = tl.cdiv(topk, BLOCK_K)
    tile_start = pid_k * tiles_per_segment
    tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, D_V), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    for j in tl.range(tile_start, tile_end, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < topk

        slot = tl.load(idx_ptr + t * topk + k_pos, mask=valid, other=0)
        valid = valid & (slot >= 0)
        page = tl.where(valid, slot, 0)

        kv_base = kv_ptr + page[:, None] * KV_DIM
        kv_nope = tl.load(
            kv_base + d_nope[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        kv_rope = tl.load(
            kv_base + (D_V + d_rope)[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        scores = tl.dot(q_nope, tl.trans(kv_nope)) + tl.dot(q_rope, tl.trans(kv_rope))
        scores = scores * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv_nope).to(tl.float32)
        m_i = m_new
        l_i = l_new

    # Write partials
    H_padded = tl.cdiv(H, BLOCK_H) * BLOCK_H
    mp_base = t * KV_SPLITS * H_padded + pid_k * H_padded
    tl.store(m_partial_ptr + mp_base + h_offs, m_i, mask=h_mask)
    tl.store(l_partial_ptr + mp_base + h_offs, l_i, mask=h_mask)

    ap_base = t * KV_SPLITS * H_padded * D_V + pid_k * H_padded * D_V
    tl.store(
        acc_partial_ptr + ap_base + h_offs[:, None] * D_V + d_nope[None, :],
        acc,
        mask=h_mask[:, None],
    )


@triton.jit
def _sparse_mla_decode_reduce_kernel(
    m_partial_ptr,  # [N, KV_SPLITS, H_padded]
    l_partial_ptr,  # [N, KV_SPLITS, H_padded]
    acc_partial_ptr,  # [N, KV_SPLITS, H_padded, D_V]
    out_ptr,  # [N, H, D_V]
    H: tl.constexpr,
    D_V: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    D_CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    dc = tl.program_id(2)

    d_offs = dc * D_CHUNK + tl.arange(0, D_CHUNK)
    k_offs = tl.arange(0, KV_SPLITS)
    d_mask = d_offs < D_V

    neg_large = -3.4028234663852886e38

    H_padded = tl.cdiv(H, 16) * 16

    # Load m, l partials [KV_SPLITS]
    mp_base = t * KV_SPLITS * H_padded
    m_p = tl.load(
        m_partial_ptr + mp_base + k_offs * H_padded + h,
    )
    l_p = tl.load(
        l_partial_ptr + mp_base + k_offs * H_padded + h,
    )

    # Load acc partials [KV_SPLITS, D_CHUNK]
    ap_base = t * KV_SPLITS * H_padded * D_V
    a_p = tl.load(
        acc_partial_ptr
        + ap_base
        + k_offs[:, None] * H_padded * D_V
        + h * D_V
        + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    )

    # Combine across splits
    m_max = tl.max(m_p, axis=0)
    alpha_split = tl.exp2(m_p - m_max)
    l_combined = tl.sum(l_p * alpha_split, axis=0)
    acc_combined = tl.sum(a_p * alpha_split[:, None], axis=0)

    denom = tl.maximum(l_combined, 1.0e-30)
    out = tl.where(l_combined > 0.0, acc_combined / denom, 0.0)

    tl.store(
        out_ptr + t * H * D_V + h * D_V + d_offs,
        out.to(tl.bfloat16),
        mask=d_mask,
    )


def triton_sparse_mla_decode_splitk(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    kv_splits: int | None = None,
) -> torch.Tensor:
    """Split-K Triton sparse MLA decode (DSv4 pattern).

    q:       [bs, H, DIM] fp8
    kv:      [num_pages, 1, DIM] fp8
    indices: [bs, 1, topk] int32
    returns: [1, bs, H, d_v] bf16
    """
    bs, H, q_dim = q.shape
    kv_dim = kv.shape[-1]
    d_tail = q_dim - d_v
    topk = indices.shape[-1]
    idx_flat = indices.squeeze(1).contiguous()

    BLOCK_H = 16
    BLOCK_K = 16
    n_head_blocks = (H + BLOCK_H - 1) // BLOCK_H
    h_padded = n_head_blocks * BLOCK_H

    if kv_splits is None:
        kv_splits = _kv_splits_heuristic(bs, H, BLOCK_H)

    qk_scale = float(sm_scale) * LOG2E

    out = torch.empty(bs, H, d_v, device=q.device, dtype=torch.bfloat16)

    if kv_splits == 1:
        _sparse_mla_decode_fused_kernel[(bs, n_head_blocks)](
            q,
            kv,
            idx_flat,
            out,
            qk_scale,
            topk=topk,
            H=H,
            Q_DIM=q_dim,
            KV_DIM=kv_dim,
            D_V=d_v,
            D_TAIL=d_tail,
            BLOCK_H=BLOCK_H,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
        return out.unsqueeze(0)

    # Split-K path
    m_partial = torch.empty(
        bs, kv_splits, h_padded, dtype=torch.float32, device=q.device
    )
    l_partial = torch.empty_like(m_partial)
    acc_partial = torch.empty(
        bs, kv_splits, h_padded, d_v, dtype=torch.float32, device=q.device
    )

    grid_split = (bs, n_head_blocks, kv_splits)
    _sparse_mla_decode_split_kernel[grid_split](
        q,
        kv,
        idx_flat,
        m_partial,
        l_partial,
        acc_partial,
        qk_scale,
        topk=topk,
        H=H,
        Q_DIM=q_dim,
        KV_DIM=kv_dim,
        D_V=d_v,
        D_TAIL=d_tail,
        KV_SPLITS=kv_splits,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    # Reduce
    D_CHUNK = 64
    grid_reduce = (bs, H, (d_v + D_CHUNK - 1) // D_CHUNK)
    _sparse_mla_decode_reduce_kernel[grid_reduce](
        m_partial,
        l_partial,
        acc_partial,
        out,
        H=H,
        D_V=d_v,
        KV_SPLITS=kv_splits,
        D_CHUNK=D_CHUNK,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out.unsqueeze(0)


# ---------------------------------------------------------------------------
# Convenience: auto-select best variant
# ---------------------------------------------------------------------------


def triton_sparse_mla_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Auto-select between base and split-K based on batch size."""
    return triton_sparse_mla_decode_splitk(q, kv, indices, sm_scale, d_v)
