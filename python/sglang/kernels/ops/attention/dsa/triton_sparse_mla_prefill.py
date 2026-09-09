"""Triton sparse-MLA forward for the DSA fp8 prefill path."""

import math

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import is_gfx95_supported, is_hip

_IS_FNUZ = is_fp8_fnuz()
_LOG2_FP8_MAX = math.log2(240.0 if _IS_FNUZ else 448.0)


@triton.jit
def _sparse_mla_prefill_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    o_ptr,
    sm_scale,
    log2_fp8_max,
    topk,
    n_tok,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
    DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    Q_MAIN_PITCH: tl.constexpr,
    Q_TAIL_PITCH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_XCD: tl.constexpr,
    WIDE_KV_OFFSET: tl.constexpr,
):
    pid = tl.program_id(0)
    # XCD swizzle: pid -> token bijection even when n_tok % N_XCD != 0.
    per = n_tok // N_XCD
    rem = n_tok % N_XCD
    xcd = pid % N_XCD
    s_i = xcd * per + tl.minimum(xcd, rem) + pid // N_XCD
    s_ok = s_i < n_tok

    h = tl.arange(0, H_PAD)
    hm = h < H
    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)
    q_main = tl.load(
        q_nope_ptr + s_i * H * Q_MAIN_PITCH + h[:, None] * Q_MAIN_PITCH + dv[None, :],
        mask=hm[:, None] & s_ok,
        other=0.0,
    ).to(q_nope_ptr.dtype.element_ty)
    q_tail = tl.load(
        q_rope_ptr + s_i * H * Q_TAIL_PITCH + h[:, None] * Q_TAIL_PITCH + dt[None, :],
        mask=hm[:, None] & s_ok,
        other=0.0,
    ).to(q_rope_ptr.dtype.element_ty)

    qk_scale = sm_scale * 1.4426950408889634
    m_i = tl.full([H_PAD], -float("inf"), tl.float32)
    l_i = tl.zeros([H_PAD], tl.float32)
    acc = tl.zeros([H_PAD, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    # Prefetch next block's page ids to hide index load latency.
    kmask = (n < topk) & s_ok
    idx = tl.load(idx_ptr + s_i * topk + n, mask=kmask, other=-1)

    for k0 in range(0, topk, BLOCK_N):
        kmask = ((k0 + n) < topk) & s_ok
        valid_k = (idx >= 0) & kmask
        valid_qk = valid_k[None, :]
        page = tl.where(valid_k, idx, 0)

        nk = k0 + BLOCK_N
        nmask = ((nk + n) < topk) & s_ok
        idx = tl.load(idx_ptr + s_i * topk + nk + n, mask=nmask, other=-1)

        if WIDE_KV_OFFSET:
            kbase = kv_ptr + page[:, None].to(tl.int64) * DIM
        else:
            kbase = kv_ptr + page[:, None] * DIM
        # Invalid lanes read page 0; softmax mask zeroes their contribution.
        kv_main = tl.load(kbase + dv[None, :]).to(q_nope_ptr.dtype.element_ty)
        kv_tail = tl.load(kbase + (D_V + dt)[None, :]).to(q_nope_ptr.dtype.element_ty)

        qk = tl.dot(q_main, tl.trans(kv_main)).to(tl.float32)
        qk += tl.dot(q_tail, tl.trans(kv_tail)).to(tl.float32)
        # Softmax in log2 space; fp8_max folded into running max cancels in final divide.
        qk = qk * qk_scale + tl.where(valid_qk, 0.0, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - (m_safe - log2_fp8_max)[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)

        p_fp8 = p.to(q_nope_ptr.dtype.element_ty)
        pv = tl.dot(p_fp8, kv_main).to(tl.float32)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc * (1.0 / l_safe)[:, None]
    tl.store(
        o_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=hm[:, None] & s_ok,
    )


# Fixed config tuned on MI355X; autotune hits a Triton 3.7.0 LLVM abort on gfx950.
_MAX_WARPS = 4


def _pick_cfg(h_pad: int) -> dict:
    return dict(
        BLOCK_N=64,
        num_warps=max(1, min(_MAX_WARPS, h_pad // 16)),
        num_stages=1,
    )


# XCD count on gfx950. N_XCD=1 disables the swizzle (s_i == pid).
_N_XCD = 8


def _require_gfx950() -> None:
    if not is_hip() or not is_gfx95_supported():
        raise ValueError("Triton sparse MLA kernels are gfx950-only.")


def _q_pitch(q: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return q and its head-row pitch; copy only when strides are not kernel-safe."""
    pitch = q.stride(1)
    if q.stride(2) == 1 and q.stride(0) == q.shape[1] * pitch and pitch >= q.shape[2]:
        return q, pitch
    return q.contiguous(), q.stride(1)


def _sanitize_indices(indices: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """Clamp page ids to valid KV range; negatives stay masked."""
    if indices.numel() == 0:
        return indices
    max_page = kv.shape[0] - 1
    if max_page < 0:
        return indices
    return indices.clamp(min=-1, max=max_page)


def _launch_sparse_mla_prefill(
    q_nope,
    q_rope,
    kv,
    indices,
    out,
    sm_scale,
    d_v,
    *,
    n_xcd=_N_XCD,
    wide_kv_offset=None,
):
    seq, H, d_v_in = q_nope.shape
    if d_v_in != d_v:
        raise ValueError(f"expected d_v={d_v}, got {d_v_in}")
    q_nope, q_main_pitch = _q_pitch(q_nope)
    q_rope, q_tail_pitch = _q_pitch(q_rope)
    d_tail = q_rope.shape[-1]
    dim = kv.shape[-1]
    topk = indices.shape[-1]
    h_pad = max(16, 1 << (H - 1).bit_length())
    args = (
        q_nope,
        q_rope,
        kv,
        indices,
        out,
        sm_scale,
        _LOG2_FP8_MAX,
        topk,
        seq,
    )
    if wide_kv_offset is None:
        wide_kv_offset = kv.shape[0] > (2**31 - 1) // dim
    kw = dict(
        H=H,
        H_PAD=h_pad,
        DIM=dim,
        D_V=d_v,
        D_TAIL=d_tail,
        Q_MAIN_PITCH=q_main_pitch,
        Q_TAIL_PITCH=q_tail_pitch,
        N_XCD=n_xcd,
        WIDE_KV_OFFSET=wide_kv_offset,
    )
    _sparse_mla_prefill_kernel[(seq,)](*args, **kw, **_pick_cfg(h_pad))


def triton_sparse_mla_prefill_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    _require_gfx950()
    seq, H, d_v_in = q_nope.shape
    if d_v_in != d_v:
        raise ValueError(f"expected d_v={d_v}, got {d_v_in}")
    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
    indices = _sanitize_indices(indices, kv)
    _launch_sparse_mla_prefill(q_nope, q_rope, kv, indices, out, sm_scale, d_v)
    return out.unsqueeze(0)
