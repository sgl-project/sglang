"""Triton sparse-MLA forward for the DSA fp8 prefill path.

Single-pass split-dim kernel: grid=(seq,), processes D_V in NUM_GROUPS
chunks of 128 for native CDNA4 fp8 MFMA tile alignment.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

_IS_FNUZ = is_fp8_fnuz()
_FP8_MAX = 240.0 if _IS_FNUZ else 448.0
_LOG2E = 1.4426950408889634


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
        page = tl.where(valid, idx, 0)
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
# N×128 split-dim kernel: breaks MFMA accumulation chains for better ILP
# NUM_GROUPS = D_V // 128 (e.g. 4 for D_V=512, 2 for D_V=256)
# ---------------------------------------------------------------------------

_G = tl.constexpr(128)

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
        page = tl.where(valid, idx, 0)
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


def triton_sparse_mla_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """q_nope: [seq, H, d_v] fp8, q_rope: [seq, H, dim-d_v] fp8,
    kv: [num_pages, 1, dim] fp8, indices: [seq, 1, topk].

    Returns [1, seq, H, d_v] bf16 to match tilelang_sparse_fwd.
    """
    return _triton_sparse_mla_fwd_single(q_nope, q_rope, kv, indices, sm_scale, d_v)
