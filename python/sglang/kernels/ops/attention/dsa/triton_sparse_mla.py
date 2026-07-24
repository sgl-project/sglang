"""Triton sparse-MLA forward for the DSA fp8 path.

Two strategies, auto-selected by sequence length:
  1. Single-pass: grid=(seq,), best when seq is large enough to fill CUs.
  2. Split-K: grid=(seq, head_blocks, kv_splits) + reduce, best for short
     sequences (MTP verify/draft with seq=1-6) where single-pass starves the GPU.

Both use the split-dim pattern: D_V processed in NUM_GROUPS chunks of 128
for native CDNA4 fp8 MFMA tile alignment.
"""

import functools

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

_IS_FNUZ = is_fp8_fnuz()
_FP8_MAX = 240.0 if _IS_FNUZ else 448.0
_LOG2E = 1.4426950408889634
_G = tl.constexpr(128)


# ---------------------------------------------------------------------------
# Helper functions for split-K heuristic
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _cu_count() -> int:
    from aiter.ops.triton.utils.device_info import get_num_sms

    return get_num_sms()


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


def _prune_configs(configs, named_args, **kwargs):
    """Drop configs whose KV tile exceeds topk (pure waste)."""
    topk = named_args["topk"]
    keep = [c for c in configs if c.kwargs["BLOCK_N"] <= topk]
    return keep or [configs[0]]


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=ns)
    for bn in (32, 64, 128)
    for w in (1, 2, 4)
    for ns in (1, 2)
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["topk", "H", "DIM"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _sparse_mla_fwd_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    o_ptr,
    sm_scale,
    fp8_max,
    topk,
    H: tl.constexpr,
    DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    s_i = tl.program_id(0)

    h = tl.arange(0, H)
    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)
    q_main = tl.load(q_nope_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :]).to(
        q_nope_ptr.dtype.element_ty
    )
    q_tail = tl.load(
        q_rope_ptr + s_i * H * D_TAIL + h[:, None] * D_TAIL + dt[None, :]
    ).to(q_nope_ptr.dtype.element_ty)

    m_i = tl.full([H], -float("inf"), tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    for k0 in range(0, topk, BLOCK_N):
        kmask = (k0 + n) < topk
        idx = tl.load(idx_ptr + s_i * topk + k0 + n, mask=kmask, other=-1)
        valid = (idx >= 0) & kmask
        page = tl.where(valid, idx, 0).to(tl.int64)
        kbase = kv_ptr + page[:, None] * DIM
        kv_main = tl.load(kbase + dv[None, :], mask=valid[:, None], other=0.0).to(
            q_nope_ptr.dtype.element_ty
        )
        kv_tail = tl.load(
            kbase + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
        ).to(q_nope_ptr.dtype.element_ty)

        qk = tl.dot(q_main, tl.trans(kv_main)).to(tl.float32)
        qk += tl.dot(q_tail, tl.trans(kv_tail)).to(tl.float32)
        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        p_fp8 = (p * fp8_max).to(q_nope_ptr.dtype.element_ty)
        pv = tl.dot(p_fp8, kv_main).to(tl.float32) * (1.0 / fp8_max)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :],
        acc.to(o_ptr.dtype.element_ty),
    )


# ---------------------------------------------------------------------------
# Single-pass split-dim kernel (autotuned, for long sequences)
# grid=(seq,), processes D_V in NUM_GROUPS chunks of 128
# ---------------------------------------------------------------------------

_SPLIT_DIM_CONFIGS = [
    triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=ns)
    for bn in (32, 64)
    for w in (2, 4)
    for ns in (1, 2)
]


@triton.autotune(
    configs=_SPLIT_DIM_CONFIGS,
    key=["topk", "H"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _sparse_mla_fwd_split_dim_kernel(
    q_nope_ptr,  # [seq, H, D_V]   fp8
    q_rope_ptr,  # [seq, H, D_TAIL] fp8
    kv_ptr,  # [num_pages, 1, DIM] fp8
    idx_ptr,  # [seq, topk]      int32
    o_ptr,  # [seq, H, D_V]    bf16
    qk_scale,  # sm_scale * LOG2E (prescaled for exp2)
    fp8_max,
    topk,
    H: tl.constexpr,
    DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    s_i = tl.program_id(0)

    h = tl.arange(0, H)
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, _G)

    q_base = q_nope_ptr + s_i * H * D_V
    q0 = tl.load(q_base + h[:, None] * D_V + g[None, :]).to(q_nope_ptr.dtype.element_ty)
    if NUM_GROUPS >= 2:
        q1 = tl.load(q_base + h[:, None] * D_V + (_G + g)[None, :]).to(
            q_nope_ptr.dtype.element_ty
        )
    if NUM_GROUPS >= 3:
        q2 = tl.load(q_base + h[:, None] * D_V + (2 * _G + g)[None, :]).to(
            q_nope_ptr.dtype.element_ty
        )
    if NUM_GROUPS >= 4:
        q3 = tl.load(q_base + h[:, None] * D_V + (3 * _G + g)[None, :]).to(
            q_nope_ptr.dtype.element_ty
        )
    q_tail = tl.load(
        q_rope_ptr + s_i * H * D_TAIL + h[:, None] * D_TAIL + dt[None, :]
    ).to(q_nope_ptr.dtype.element_ty)

    m_i = tl.full([H], -float("inf"), tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc0 = tl.zeros([H, _G], tl.float32)
    if NUM_GROUPS >= 2:
        acc1 = tl.zeros([H, _G], tl.float32)
    if NUM_GROUPS >= 3:
        acc2 = tl.zeros([H, _G], tl.float32)
    if NUM_GROUPS >= 4:
        acc3 = tl.zeros([H, _G], tl.float32)

    inv_fp8_max = 1.0 / fp8_max
    n = tl.arange(0, BLOCK_N)
    for k0 in range(0, topk, BLOCK_N):
        kmask = (k0 + n) < topk
        idx = tl.load(idx_ptr + s_i * topk + k0 + n, mask=kmask, other=-1)
        valid = (idx >= 0) & kmask
        page = tl.where(valid, idx, 0).to(tl.int64)
        kbase = kv_ptr + page[:, None] * DIM

        kv0 = tl.load(kbase + g[None, :], mask=valid[:, None], other=0.0).to(
            q_nope_ptr.dtype.element_ty
        )
        if NUM_GROUPS >= 2:
            kv1 = tl.load(kbase + (_G + g)[None, :], mask=valid[:, None], other=0.0).to(
                q_nope_ptr.dtype.element_ty
            )
        if NUM_GROUPS >= 3:
            kv2 = tl.load(
                kbase + (2 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(q_nope_ptr.dtype.element_ty)
        if NUM_GROUPS >= 4:
            kv3 = tl.load(
                kbase + (3 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(q_nope_ptr.dtype.element_ty)
        kv_tail = tl.load(
            kbase + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
        ).to(q_nope_ptr.dtype.element_ty)

        qk = tl.dot(q0, tl.trans(kv0))
        if NUM_GROUPS >= 2:
            qk += tl.dot(q1, tl.trans(kv1))
        if NUM_GROUPS >= 3:
            qk += tl.dot(q2, tl.trans(kv2))
        if NUM_GROUPS >= 4:
            qk += tl.dot(q3, tl.trans(kv3))
        qk += tl.dot(q_tail, tl.trans(kv_tail))
        qk = qk * qk_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))

        m_block = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        p_fp8 = (p * fp8_max).to(q_nope_ptr.dtype.element_ty)
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

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    inv_l = 1.0 / l_safe
    acc0 = acc0 * inv_l[:, None]
    if NUM_GROUPS >= 2:
        acc1 = acc1 * inv_l[:, None]
    if NUM_GROUPS >= 3:
        acc2 = acc2 * inv_l[:, None]
    if NUM_GROUPS >= 4:
        acc3 = acc3 * inv_l[:, None]

    o_base = o_ptr + s_i * H * D_V
    tl.store(o_base + h[:, None] * D_V + g[None, :], acc0.to(o_ptr.dtype.element_ty))
    if NUM_GROUPS >= 2:
        tl.store(
            o_base + h[:, None] * D_V + (_G + g)[None, :],
            acc1.to(o_ptr.dtype.element_ty),
        )
    if NUM_GROUPS >= 3:
        tl.store(
            o_base + h[:, None] * D_V + (2 * _G + g)[None, :],
            acc2.to(o_ptr.dtype.element_ty),
        )
    if NUM_GROUPS >= 4:
        tl.store(
            o_base + h[:, None] * D_V + (3 * _G + g)[None, :],
            acc3.to(o_ptr.dtype.element_ty),
        )


def _triton_sparse_mla_fwd_single(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Single-pass prefill: grid=(seq,), loops over all topk per CTA."""
    seq, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    assert d_v % 128 == 0, f"Triton sparse MLA requires d_v divisible by 128, got {d_v}"
    d_tail = q_rope.shape[-1]
    dim = kv.shape[-1]
    topk = indices.shape[-1]
    q_nope = q_nope.contiguous()
    q_rope = q_rope.contiguous()
    idx_flat = indices.squeeze(1).contiguous() if indices.dim() == 3 else indices
    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
    qk_scale = float(sm_scale) * _LOG2E

    num_groups = d_v // 128
    if H < 16:
        # Pad H to 16 so fp8 tl.dot maps to native MFMA tiles on CDNA4.
        # Without padding, M=H<16 fp8 dots fall back to a scalar path.
        H_pad = 16
        q_nope_pad = torch.zeros(
            seq, H_pad, d_v, device=q_nope.device, dtype=q_nope.dtype
        )
        q_rope_pad = torch.zeros(
            seq, H_pad, d_tail, device=q_rope.device, dtype=q_rope.dtype
        )
        q_nope_pad[:, :H, :] = q_nope
        q_rope_pad[:, :H, :] = q_rope
        out_pad = torch.empty(
            seq, H_pad, d_v, device=q_nope.device, dtype=torch.bfloat16
        )
        _sparse_mla_fwd_split_dim_kernel[(seq,)](
            q_nope_pad,
            q_rope_pad,
            kv,
            idx_flat,
            out_pad,
            qk_scale,
            _FP8_MAX,
            topk,
            H=H_pad,
            DIM=dim,
            D_V=d_v,
            D_TAIL=d_tail,
            NUM_GROUPS=num_groups,
        )
        out = out_pad[:, :H, :].contiguous()
    else:
        _sparse_mla_fwd_split_dim_kernel[(seq,)](
            q_nope,
            q_rope,
            kv,
            idx_flat,
            out,
            qk_scale,
            _FP8_MAX,
            topk,
            H=H,
            DIM=dim,
            D_V=d_v,
            D_TAIL=d_tail,
            NUM_GROUPS=num_groups,
        )
    return out.unsqueeze(0)


def _prev_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


# ---------------------------------------------------------------------------
# Split-K kernels (for short sequences: MTP verify/draft, decode)
# grid=(seq, head_blocks, kv_splits) + reduce
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_mla_fused_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    out_ptr,
    qk_scale,
    fp8_max,
    topk: tl.constexpr,
    H: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Single-pass with head-block splitting. grid=(seq, head_blocks)."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, _G)

    fp8_type = q_nope_ptr.dtype.element_ty
    inv_fp8_max = 1.0 / fp8_max

    qn_base = q_nope_ptr + t * H * D_V
    q0 = tl.load(
        qn_base + h_offs[:, None] * D_V + g[None, :], mask=h_mask[:, None], other=0.0
    ).to(fp8_type)
    if NUM_GROUPS >= 2:
        q1 = tl.load(
            qn_base + h_offs[:, None] * D_V + (_G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    if NUM_GROUPS >= 3:
        q2 = tl.load(
            qn_base + h_offs[:, None] * D_V + (2 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    if NUM_GROUPS >= 4:
        q3 = tl.load(
            qn_base + h_offs[:, None] * D_V + (3 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    q_tail = tl.load(
        q_rope_ptr + t * H * D_TAIL + h_offs[:, None] * D_TAIL + dt[None, :],
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
        kv0 = tl.load(kv_base + g[None, :], mask=valid[:, None], other=0.0).to(fp8_type)
        if NUM_GROUPS >= 2:
            kv1 = tl.load(
                kv_base + (_G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(fp8_type)
        if NUM_GROUPS >= 3:
            kv2 = tl.load(
                kv_base + (2 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(fp8_type)
        if NUM_GROUPS >= 4:
            kv3 = tl.load(
                kv_base + (3 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(fp8_type)
        kv_tail = tl.load(
            kv_base + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
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
        l_i = l_i * alpha + tl.sum(p, axis=1)

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
def _sparse_mla_split_k_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    lse_partial_ptr,
    acc_partial_ptr,
    qk_scale,
    fp8_max,
    topk: tl.constexpr,
    H: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Split-K partial kernel. grid=(seq, head_blocks, kv_splits)."""
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_k = tl.program_id(2)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, _G)

    fp8_type = q_nope_ptr.dtype.element_ty
    inv_fp8_max = 1.0 / fp8_max

    qn_base = q_nope_ptr + t * H * D_V
    q0 = tl.load(
        qn_base + h_offs[:, None] * D_V + g[None, :], mask=h_mask[:, None], other=0.0
    ).to(fp8_type)
    if NUM_GROUPS >= 2:
        q1 = tl.load(
            qn_base + h_offs[:, None] * D_V + (_G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    if NUM_GROUPS >= 3:
        q2 = tl.load(
            qn_base + h_offs[:, None] * D_V + (2 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    if NUM_GROUPS >= 4:
        q3 = tl.load(
            qn_base + h_offs[:, None] * D_V + (3 * _G + g)[None, :],
            mask=h_mask[:, None],
            other=0.0,
        ).to(fp8_type)
    q_tail = tl.load(
        q_rope_ptr + t * H * D_TAIL + h_offs[:, None] * D_TAIL + dt[None, :],
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
        kv0 = tl.load(kv_base + g[None, :], mask=valid[:, None], other=0.0).to(fp8_type)
        if NUM_GROUPS >= 2:
            kv1 = tl.load(
                kv_base + (_G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(fp8_type)
        if NUM_GROUPS >= 3:
            kv2 = tl.load(
                kv_base + (2 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(fp8_type)
        if NUM_GROUPS >= 4:
            kv3 = tl.load(
                kv_base + (3 * _G + g)[None, :], mask=valid[:, None], other=0.0
            ).to(fp8_type)
        kv_tail = tl.load(
            kv_base + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
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

    neg_large = -3.4028234663852886e38
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
def _sparse_mla_reduce_kernel(
    lse_partial_ptr,
    acc_partial_ptr,
    out_ptr,
    H: tl.constexpr,
    D_V: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    D_CHUNK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Reduce split-K partials via log-space combine. grid=(seq, H, d_v_chunks)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    dc = tl.program_id(2)

    d_offs = dc * D_CHUNK + tl.arange(0, D_CHUNK)
    k_offs = tl.arange(0, KV_SPLITS)
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


def _triton_sparse_mla_fwd_splitk(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    kv_splits: int,
) -> torch.Tensor:
    """Split-K path for short sequences."""
    seq, H, d_v_in = q_nope.shape
    assert d_v_in == d_v
    d_tail = q_rope.shape[-1]
    kv_dim = kv.shape[-1]
    topk = indices.shape[-1]
    idx_flat = indices.squeeze(1).contiguous() if indices.dim() == 3 else indices
    q_nope = q_nope.contiguous()
    q_rope = q_rope.contiguous()

    BLOCK_H = 16
    BLOCK_K = 64
    n_head_blocks = (H + BLOCK_H - 1) // BLOCK_H
    h_padded = n_head_blocks * BLOCK_H

    num_groups = d_v // 128
    qk_scale = float(sm_scale) * _LOG2E

    max_kv_splits = topk // BLOCK_K
    kv_splits = min(kv_splits, max_kv_splits)

    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)

    if kv_splits == 1:
        _sparse_mla_fused_kernel[(seq, n_head_blocks)](
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
            BLOCK_H=BLOCK_H,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )
        return out.unsqueeze(0)

    lse_partial = torch.empty(
        seq, kv_splits, h_padded, dtype=torch.float32, device=q_nope.device
    )
    acc_partial = torch.empty(
        seq, kv_splits, h_padded, d_v, dtype=torch.bfloat16, device=q_nope.device
    )

    _sparse_mla_split_k_kernel[(seq, n_head_blocks, kv_splits)](
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
        KV_SPLITS=kv_splits,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    D_CHUNK = 64
    _sparse_mla_reduce_kernel[(seq, H, (d_v + D_CHUNK - 1) // D_CHUNK)](
        lse_partial,
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
# Unified entry point
# ---------------------------------------------------------------------------


def triton_sparse_mla_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """Unified sparse MLA forward. Auto-selects single-pass vs split-K.

    q_nope: [seq, H, d_v] fp8, q_rope: [seq, H, dim-d_v] fp8,
    kv: [num_pages, 1, dim] fp8, indices: [seq, 1, topk].

    Returns [1, seq, H, d_v] bf16 to match tilelang_sparse_fwd.
    """
    seq = q_nope.shape[0]
    H = q_nope.shape[1]
    num_cu = _cu_count()
    BLOCK_H = 16
    BLOCK_K = 64
    topk = indices.shape[-1]
    max_kv_splits = topk // BLOCK_K
    head_blocks = max(1, (H + BLOCK_H - 1) // BLOCK_H)
    base_ctas = seq * head_blocks
    kv_work_per_cta = topk // BLOCK_K
    if base_ctas >= 2 * num_cu and kv_work_per_cta <= 4:
        return _triton_sparse_mla_fwd_single(q_nope, q_rope, kv, indices, sm_scale, d_v)
    kv_splits = min(
        _kv_splits_heuristic(
            seq, H, BLOCK_H, target_wg_per_cu=1.0, max_kv_splits=max_kv_splits
        ),
        max_kv_splits,
    )
    return _triton_sparse_mla_fwd_splitk(
        q_nope, q_rope, kv, indices, sm_scale, d_v, kv_splits
    )
