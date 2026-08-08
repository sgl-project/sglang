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

from sglang.kernels.ops.attention.dsa.triton_sparse_mla import _row_strides
from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

_IS_FNUZ = is_fp8_fnuz()
_FP8_MAX = 240.0 if _IS_FNUZ else 448.0
_G = tl.constexpr(128)

_splitk_bufs: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_splitk_bufs(
    bs: int,
    kv_splits: int,
    h_padded: int,
    d_v: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = device
    needed_lse = bs * kv_splits * h_padded
    needed_acc = bs * kv_splits * h_padded * d_v
    if key in _splitk_bufs:
        lse_buf, acc_buf = _splitk_bufs[key]
        if lse_buf.numel() >= needed_lse and acc_buf.numel() >= needed_acc:
            lse = lse_buf[:needed_lse].view(bs, kv_splits, h_padded)
            acc = acc_buf[:needed_acc].view(bs, kv_splits, h_padded, d_v)
            return lse, acc
    cap_bs = max(bs, 128)
    cap_splits = max(kv_splits, 32)
    lse_buf = torch.empty(
        cap_bs * cap_splits * h_padded, dtype=torch.float32, device=device
    )
    acc_buf = torch.empty(
        cap_bs * cap_splits * h_padded * d_v, dtype=torch.bfloat16, device=device
    )
    _splitk_bufs[key] = (lse_buf, acc_buf)
    lse = lse_buf[:needed_lse].view(bs, kv_splits, h_padded)
    acc = acc_buf[:needed_acc].view(bs, kv_splits, h_padded, d_v)
    return lse, acc


# ---------------------------------------------------------------------------
# Split-K kernel (adapted from DSv4 paged_decode.py)
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
    q_nope_ptr,  # [N, H, D_V]
    q_rope_ptr,  # [N, H, D_TAIL]
    kv_ptr,  # [num_pages, 1, KV_DIM]
    idx_ptr,  # [N, topk]
    out_ptr,  # [N, H, D_V]
    qk_scale,
    fp8_max,
    topk: tl.constexpr,
    H: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    STRIDE_QN_T: tl.constexpr,
    STRIDE_QN_H: tl.constexpr,
    STRIDE_QR_T: tl.constexpr,
    STRIDE_QR_H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, _G)

    fp8_type = q_nope_ptr.dtype.element_ty
    inv_fp8_max = 1.0 / fp8_max

    qn_base = q_nope_ptr + t * STRIDE_QN_T
    qn_row = qn_base + h_offs[:, None] * STRIDE_QN_H
    q0 = tl.load(
        qn_row + g[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(fp8_type)
    if NUM_GROUPS >= 2:
        q1 = tl.load(
            qn_row + (_G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    if NUM_GROUPS >= 3:
        q2 = tl.load(
            qn_row + (2 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    if NUM_GROUPS >= 4:
        q3 = tl.load(
            qn_row + (3 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    q_tail = tl.load(
        q_rope_ptr + t * STRIDE_QR_T + h_offs[:, None] * STRIDE_QR_H + dt[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(fp8_type)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 2:
        acc1 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 3:
        acc2 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 4:
        acc3 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    num_tiles = tl.cdiv(topk, BLOCK_K)

    for j in tl.range(0, num_tiles, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < topk

        slot = tl.load(idx_ptr + t * topk + k_pos, mask=valid, other=0)
        valid = valid & (slot >= 0)
        page = tl.where(valid, slot, 0).to(tl.int64)

        kv_base = kv_ptr + page[:, None] * KV_DIM
        kv0 = tl.load(
            kv_base + g[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(fp8_type)
        if NUM_GROUPS >= 2:
            kv1 = tl.load(
                kv_base + (_G + g)[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(fp8_type)
        if NUM_GROUPS >= 3:
            kv2 = tl.load(
                kv_base + (2 * _G + g)[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(fp8_type)
        if NUM_GROUPS >= 4:
            kv3 = tl.load(
                kv_base + (3 * _G + g)[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(fp8_type)
        kv_tail = tl.load(
            kv_base + (D_V + dt)[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(fp8_type)

        scores = tl.dot(q0, tl.trans(kv0))
        if NUM_GROUPS >= 2:
            scores += tl.dot(q1, tl.trans(kv1))
        if NUM_GROUPS >= 3:
            scores += tl.dot(q2, tl.trans(kv2))
        if NUM_GROUPS >= 4:
            scores += tl.dot(q3, tl.trans(kv3))
        scores += tl.dot(q_tail, tl.trans(kv_tail))
        scores = scores * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        p_fp8 = (p * fp8_max).to(fp8_type)
        acc0 = acc0 * alpha[:, None] + tl.dot(p_fp8, kv0).to(tl.float32) * inv_fp8_max
        if NUM_GROUPS >= 2:
            acc1 = (
                acc1 * alpha[:, None] + tl.dot(p_fp8, kv1).to(tl.float32) * inv_fp8_max
            )
        if NUM_GROUPS >= 3:
            acc2 = (
                acc2 * alpha[:, None] + tl.dot(p_fp8, kv2).to(tl.float32) * inv_fp8_max
            )
        if NUM_GROUPS >= 4:
            acc3 = (
                acc3 * alpha[:, None] + tl.dot(p_fp8, kv3).to(tl.float32) * inv_fp8_max
            )
        m_i = m_new
        l_i = l_new

    denom = tl.maximum(l_i, 1.0e-30)
    inv_denom = 1.0 / denom
    acc0 = tl.where(l_i[:, None] > 0.0, acc0 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 2:
        acc1 = tl.where(l_i[:, None] > 0.0, acc1 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 3:
        acc2 = tl.where(l_i[:, None] > 0.0, acc2 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 4:
        acc3 = tl.where(l_i[:, None] > 0.0, acc3 * inv_denom[:, None], 0.0)

    o_base = out_ptr + t * H * D_V
    tl.store(
        o_base + h_offs[:, None] * D_V + g[None, :],
        acc0.to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    if NUM_GROUPS >= 2:
        tl.store(
            o_base + h_offs[:, None] * D_V + (_G + g)[None, :],
            acc1.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
    if NUM_GROUPS >= 3:
        tl.store(
            o_base + h_offs[:, None] * D_V + (2 * _G + g)[None, :],
            acc2.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
    if NUM_GROUPS >= 4:
        tl.store(
            o_base + h_offs[:, None] * D_V + (3 * _G + g)[None, :],
            acc3.to(tl.bfloat16),
            mask=h_mask[:, None],
        )


@triton.jit
def _sparse_mla_decode_split_kernel(
    q_nope_ptr,  # [N, H, D_V]
    q_rope_ptr,  # [N, H, D_TAIL]
    kv_ptr,  # [num_pages, 1, KV_DIM]
    idx_ptr,  # [N, topk]
    lse_partial_ptr,  # [N, KV_SPLITS, H_padded]  fp32
    acc_partial_ptr,  # [N, KV_SPLITS, H_padded, D_V]  bf16
    qk_scale,
    fp8_max,
    topk: tl.constexpr,
    H: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    STRIDE_QN_T: tl.constexpr,
    STRIDE_QN_H: tl.constexpr,
    STRIDE_QR_T: tl.constexpr,
    STRIDE_QR_H: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, _G)

    fp8_type = q_nope_ptr.dtype.element_ty
    inv_fp8_max = 1.0 / fp8_max

    qn_base = q_nope_ptr + t * STRIDE_QN_T
    qn_row = qn_base + h_offs[:, None] * STRIDE_QN_H
    q0 = tl.load(
        qn_row + g[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(fp8_type)
    if NUM_GROUPS >= 2:
        q1 = tl.load(
            qn_row + (_G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    if NUM_GROUPS >= 3:
        q2 = tl.load(
            qn_row + (2 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    if NUM_GROUPS >= 4:
        q3 = tl.load(
            qn_row + (3 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    q_tail = tl.load(
        q_rope_ptr + t * STRIDE_QR_T + h_offs[:, None] * STRIDE_QR_H + dt[None, :],
        mask=h_mask[:, None],
        other=0.0,
    ).to(fp8_type)

    tiles_per_segment = tl.cdiv(topk, KV_SPLITS * BLOCK_K)
    if pid_k * tiles_per_segment * BLOCK_K >= topk:
        return
    num_tiles = tl.cdiv(topk, BLOCK_K)
    tile_start = pid_k * tiles_per_segment
    tile_end = tl.minimum((pid_k + 1) * tiles_per_segment, num_tiles)

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc0 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 2:
        acc1 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 3:
        acc2 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)
    if NUM_GROUPS >= 4:
        acc3 = tl.zeros((BLOCK_H, _G), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    for j in tl.range(tile_start, tile_end, num_stages=3):
        k_start = j * BLOCK_K
        k_pos = k_start + k_offs
        valid = k_pos < topk

        slot = tl.load(idx_ptr + t * topk + k_pos, mask=valid, other=0)
        valid = valid & (slot >= 0)
        page = tl.where(valid, slot, 0).to(tl.int64)

        kv_base = kv_ptr + page[:, None] * KV_DIM
        kv0 = tl.load(
            kv_base + g[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(fp8_type)
        if NUM_GROUPS >= 2:
            kv1 = tl.load(
                kv_base + (_G + g)[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(fp8_type)
        if NUM_GROUPS >= 3:
            kv2 = tl.load(
                kv_base + (2 * _G + g)[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(fp8_type)
        if NUM_GROUPS >= 4:
            kv3 = tl.load(
                kv_base + (3 * _G + g)[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(fp8_type)
        kv_tail = tl.load(
            kv_base + (D_V + dt)[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(fp8_type)

        scores = tl.dot(q0, tl.trans(kv0))
        if NUM_GROUPS >= 2:
            scores += tl.dot(q1, tl.trans(kv1))
        if NUM_GROUPS >= 3:
            scores += tl.dot(q2, tl.trans(kv2))
        if NUM_GROUPS >= 4:
            scores += tl.dot(q3, tl.trans(kv3))
        scores += tl.dot(q_tail, tl.trans(kv_tail))
        scores = scores * qk_scale
        scores = tl.where(valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)

        p_fp8 = (p * fp8_max).to(fp8_type)
        acc0 = acc0 * alpha[:, None] + tl.dot(p_fp8, kv0).to(tl.float32) * inv_fp8_max
        if NUM_GROUPS >= 2:
            acc1 = (
                acc1 * alpha[:, None] + tl.dot(p_fp8, kv1).to(tl.float32) * inv_fp8_max
            )
        if NUM_GROUPS >= 3:
            acc2 = (
                acc2 * alpha[:, None] + tl.dot(p_fp8, kv2).to(tl.float32) * inv_fp8_max
            )
        if NUM_GROUPS >= 4:
            acc3 = (
                acc3 * alpha[:, None] + tl.dot(p_fp8, kv3).to(tl.float32) * inv_fp8_max
            )
        m_i = m_new
        l_i = l_new

    neg_large = -1073741824.0
    denom = tl.maximum(l_i, 1.0e-30)
    inv_denom = 1.0 / denom
    has_data = l_i > 0.0
    acc0 = tl.where(has_data[:, None], acc0 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 2:
        acc1 = tl.where(has_data[:, None], acc1 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 3:
        acc2 = tl.where(has_data[:, None], acc2 * inv_denom[:, None], 0.0)
    if NUM_GROUPS >= 4:
        acc3 = tl.where(has_data[:, None], acc3 * inv_denom[:, None], 0.0)

    lse = tl.where(has_data, tl.log2(l_i) + m_i, neg_large)

    H_padded = tl.cdiv(H, BLOCK_H) * BLOCK_H
    lse_base = t * KV_SPLITS * H_padded + pid_k * H_padded
    tl.store(lse_partial_ptr + lse_base + h_offs, lse, mask=h_mask)

    ap_base = t * KV_SPLITS * H_padded * D_V + pid_k * H_padded * D_V
    tl.store(
        acc_partial_ptr + ap_base + h_offs[:, None] * D_V + g[None, :],
        acc0.to(tl.bfloat16),
        mask=h_mask[:, None],
    )
    if NUM_GROUPS >= 2:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (_G + g)[None, :],
            acc1.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
    if NUM_GROUPS >= 3:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (2 * _G + g)[None, :],
            acc2.to(tl.bfloat16),
            mask=h_mask[:, None],
        )
    if NUM_GROUPS >= 4:
        tl.store(
            acc_partial_ptr + ap_base + h_offs[:, None] * D_V + (3 * _G + g)[None, :],
            acc3.to(tl.bfloat16),
            mask=h_mask[:, None],
        )


@triton.jit
def _sparse_mla_decode_reduce_kernel(
    lse_partial_ptr,  # [N, KV_SPLITS, H_padded]  fp32
    acc_partial_ptr,  # [N, KV_SPLITS, H_padded, D_V]  bf16
    out_ptr,  # [N, H, D_V]
    H: tl.constexpr,
    D_V: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    ACTIVE_SPLITS: tl.constexpr,
    D_CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    dc = tl.program_id(2)

    d_offs = dc * D_CHUNK + tl.arange(0, D_CHUNK)
    k_offs = tl.arange(0, ACTIVE_SPLITS)
    d_mask = d_offs < D_V

    H_padded = tl.cdiv(H, 16) * 16

    lse_base = t * KV_SPLITS * H_padded
    lse_p = tl.load(lse_partial_ptr + lse_base + k_offs * H_padded + h)

    ap_base = t * KV_SPLITS * H_padded * D_V
    a_p = tl.load(
        acc_partial_ptr
        + ap_base
        + k_offs[:, None] * H_padded * D_V
        + h * D_V
        + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    lse_max = tl.max(lse_p, axis=0)
    weights = tl.exp2(lse_p - lse_max)
    w_sum = tl.sum(weights, axis=0)
    scale = tl.exp2(lse_p - lse_max - tl.log2(tl.maximum(w_sum, 1.0e-30)))
    out = tl.sum(a_p * scale[:, None], axis=0)

    tl.store(
        out_ptr + t * H * D_V + h * D_V + d_offs,
        out.to(tl.bfloat16),
        mask=d_mask,
    )


def triton_sparse_mla_decode_splitk(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    kv_splits: int | None = None,
) -> torch.Tensor:
    """Split-K Triton sparse MLA decode (DSv4 pattern).

    q_nope:  [bs, H, d_v] fp8
    q_rope:  [bs, H, d_tail] fp8
    kv:      [num_pages, 1, DIM] fp8
    indices: [bs, 1, topk] int32
    returns: [1, bs, H, d_v] bf16
    """
    bs, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    d_tail = q_rope.shape[-1]
    kv_dim = kv.shape[-1]
    topk = indices.shape[-1]
    idx_flat = indices.squeeze(1).contiguous()
    # The kernels address q by explicit row strides, so a packed [N, H, D] layout
    # is not required -- only a unit-stride last dim. Callers that pass an
    # already-concatenated q (dsa_backend, GLM-5.2 path) hand us two strided views
    # of one [N, H, D_V + D_TAIL] buffer; copying those would cost two extra
    # device kernels per layer per forward for nothing.
    q_nope, stride_qn_t, stride_qn_h = _row_strides(q_nope)
    q_rope, stride_qr_t, stride_qr_h = _row_strides(q_rope)

    BLOCK_H = 16
    BLOCK_K = 64
    n_head_blocks = (H + BLOCK_H - 1) // BLOCK_H
    h_padded = n_head_blocks * BLOCK_H

    assert d_v % 128 == 0, f"d_v must be divisible by 128, got {d_v}"
    num_groups = d_v // 128

    max_kv_splits = topk // BLOCK_K
    if kv_splits is None:
        kv_splits = min(
            _kv_splits_heuristic(
                bs, H, BLOCK_H, target_wg_per_cu=1.0, max_kv_splits=max_kv_splits
            ),
            max_kv_splits,
        )
    else:
        kv_splits = min(kv_splits, max_kv_splits)

    qk_scale = float(sm_scale) * LOG2E

    if kv_splits == 1:
        out = torch.empty(bs, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
        _sparse_mla_decode_fused_kernel[(bs, n_head_blocks)](
            q_nope,
            q_rope,
            kv,
            idx_flat,
            out,
            qk_scale,
            _FP8_MAX,
            topk=topk,
            H=H,
            KV_DIM=kv_dim,
            D_V=d_v,
            D_TAIL=d_tail,
            NUM_GROUPS=num_groups,
            STRIDE_QN_T=stride_qn_t,
            STRIDE_QN_H=stride_qn_h,
            STRIDE_QR_T=stride_qr_t,
            STRIDE_QR_H=stride_qr_h,
            BLOCK_H=BLOCK_H,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
        return out.unsqueeze(0)

    tiles_per_split = (topk + kv_splits * BLOCK_K - 1) // (kv_splits * BLOCK_K)
    active_splits = (topk + tiles_per_split * BLOCK_K - 1) // (
        tiles_per_split * BLOCK_K
    )
    active_splits = min(active_splits, kv_splits)

    lse_partial, acc_partial = _get_splitk_bufs(
        bs, kv_splits, h_padded, d_v, q_nope.device
    )
    out = torch.empty(bs, H, d_v, device=q_nope.device, dtype=torch.bfloat16)

    grid_split = (bs, n_head_blocks, kv_splits)
    _sparse_mla_decode_split_kernel[grid_split](
        q_nope,
        q_rope,
        kv,
        idx_flat,
        lse_partial,
        acc_partial,
        qk_scale,
        _FP8_MAX,
        topk=topk,
        H=H,
        KV_DIM=kv_dim,
        D_V=d_v,
        D_TAIL=d_tail,
        NUM_GROUPS=num_groups,
        STRIDE_QN_T=stride_qn_t,
        STRIDE_QN_H=stride_qn_h,
        STRIDE_QR_T=stride_qr_t,
        STRIDE_QR_H=stride_qr_h,
        KV_SPLITS=kv_splits,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    D_CHUNK = 64
    grid_reduce = (bs, H, (d_v + D_CHUNK - 1) // D_CHUNK)
    _sparse_mla_decode_reduce_kernel[grid_reduce](
        lse_partial,
        acc_partial,
        out,
        H=H,
        D_V=d_v,
        KV_SPLITS=kv_splits,
        ACTIVE_SPLITS=active_splits,
        D_CHUNK=D_CHUNK,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )
    return out.unsqueeze(0)


# ---------------------------------------------------------------------------
# Convenience: auto-select best variant
# ---------------------------------------------------------------------------


def triton_sparse_mla_decode(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    return triton_sparse_mla_decode_splitk(q_nope, q_rope, kv, indices, sm_scale, d_v)
