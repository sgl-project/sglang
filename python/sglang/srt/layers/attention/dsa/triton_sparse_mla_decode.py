"""Per-query Triton sparse-MLA decode for the fp8 KV + CP path (gfx950).

Split-K flash-decode over the indexer top-2048; per-query grid (CP-safe).
DSA fp8 KV pool: 576-wide raw fp8_e4m3fn [512 nope | 64 rope], kv_scale==1.0.
Q bf16 (a16w8). K/V dequant = fp8->bf16. MLA: V = nope(512).
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

_FP8_MAX = 240.0 if is_fp8_fnuz() else 448.0
_LOG2E = 1.4426950408889634


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=1),
    ],
    key=["H", "topk", "BLOCK_H"],
)
@triton.jit
def _splitk_partial_kernel(
    q_nope_ptr, q_rope_ptr, kv_ptr, idx_ptr,
    pacc_ptr, pm_ptr, pl_ptr,
    sm_scale, topk, skv, fp8_max,
    K_SPLITS: tl.constexpr, H: tl.constexpr, D_V: tl.constexpr, D_TAIL: tl.constexpr,
    BLOCK_H: tl.constexpr, USE_FP8_PV: tl.constexpr, BLOCK_N: tl.constexpr,
):
    s_i = tl.program_id(0)
    hb = tl.program_id(1)
    sk = tl.program_id(2)
    h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)
    per = (topk + K_SPLITS - 1) // K_SPLITS
    k_start = sk * per
    k_end = tl.minimum(k_start + per, topk)

    q_main = tl.load(q_nope_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :])
    q_tail = tl.load(q_rope_ptr + s_i * H * D_TAIL + h[:, None] * D_TAIL + dt[None, :])
    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    for k0 in range(k_start, k_end, BLOCK_N):
        kk = k0 + n
        kmask = kk < k_end
        idx = tl.load(idx_ptr + s_i * topk + kk, mask=kmask, other=-1)
        valid = (idx >= 0) & kmask
        page = tl.where(valid, idx, 0)
        kbase = kv_ptr + page[:, None] * skv
        kv_raw = tl.load(kbase + dv[None, :], mask=valid[:, None], other=0.0)  # fp8
        kv_main = kv_raw.to(tl.bfloat16)
        kv_tail = tl.load(kbase + (D_V + dt)[None, :], mask=valid[:, None], other=0.0).to(tl.bfloat16)
        qk = tl.dot(q_main, tl.trans(kv_main)).to(tl.float32)
        qk += tl.dot(q_tail, tl.trans(kv_tail)).to(tl.float32)
        qk = qk * sm_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.math.exp2((m_i - m_safe) * 1.4426950408889634)
        p = tl.math.exp2((qk - m_safe[:, None]) * 1.4426950408889634)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        if USE_FP8_PV:
            p_fp8 = (p * fp8_max).to(kv_raw.dtype)
            acc = acc * alpha[:, None] + tl.dot(p_fp8, kv_raw).to(tl.float32) * (1.0 / fp8_max)
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), kv_main).to(tl.float32)
        m_i = m_new

    base = (s_i * H + h) * K_SPLITS + sk
    tl.store(pm_ptr + base, m_i)
    tl.store(pl_ptr + base, l_i)
    tl.store(pacc_ptr + base[:, None] * D_V + dv[None, :], acc)


@triton.jit
def _splitk_combine_kernel(
    pacc_ptr, pm_ptr, pl_ptr, o_ptr,
    K_SPLITS: tl.constexpr, H: tl.constexpr, D_V: tl.constexpr, BLOCK_H: tl.constexpr,
):
    s_i = tl.program_id(0)
    hb = tl.program_id(1)
    h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    dv = tl.arange(0, D_V)
    gm = tl.full([BLOCK_H], -float("inf"), tl.float32)
    gl = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], tl.float32)
    for k in range(0, K_SPLITS):
        off = (s_i * H + h) * K_SPLITS + k
        mk = tl.load(pm_ptr + off)
        lk = tl.load(pl_ptr + off)
        pak = tl.load(pacc_ptr + off[:, None] * D_V + dv[None, :])
        new_m = tl.maximum(gm, mk)
        new_m_safe = tl.where(new_m == -float("inf"), 0.0, new_m)
        a = tl.exp(gm - new_m_safe)
        b = tl.exp(mk - new_m_safe)
        gl = gl * a + lk * b
        acc = acc * a[:, None] + pak * b[:, None]
        gm = new_m
    gl_safe = tl.where(gl == 0.0, 1.0, gl)
    acc = acc / gl_safe[:, None]
    tl.store(o_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :], acc.to(o_ptr.dtype.element_ty))


def triton_sparse_mla_decode_fp8(
    q_nope, q_rope, kv_cache, kv_indices, sm_scale,
    out=None, TOPK=None, BLOCK_H=64, K_SPLITS=8, USE_FP8_PV=True,
):
    S, H, D_V = q_nope.shape
    D_TAIL = q_rope.shape[-1]
    kv = kv_cache.reshape(-1, kv_cache.shape[-1])
    P, dim = kv.shape
    assert dim == D_V + D_TAIL, f"expected {D_V + D_TAIL}, got {dim}"
    if TOPK is None:
        TOPK = kv_indices.shape[-1]
    idx = kv_indices.reshape(S, TOPK).contiguous().to(torch.int32)
    q_nope = q_nope.contiguous()
    q_rope = q_rope.contiguous()
    if out is None:
        out = torch.empty((S, H, D_V), device=q_nope.device, dtype=torch.bfloat16)
    bh = BLOCK_H if H % BLOCK_H == 0 else 16
    ks = K_SPLITS if S <= 128 else 1
    pacc = torch.empty((S, H, ks, D_V), device=q_nope.device, dtype=torch.float32)
    pm = torch.empty((S, H, ks), device=q_nope.device, dtype=torch.float32)
    pl = torch.empty((S, H, ks), device=q_nope.device, dtype=torch.float32)
    _splitk_partial_kernel[(S, H // bh, ks)](
        q_nope, q_rope, kv, idx, pacc, pm, pl, sm_scale, TOPK, kv.stride(0), _FP8_MAX,
        K_SPLITS=ks, H=H, D_V=D_V, D_TAIL=D_TAIL, BLOCK_H=bh, USE_FP8_PV=USE_FP8_PV,
    )
    _splitk_combine_kernel[(S, H // bh)](
        pacc, pm, pl, out, K_SPLITS=ks, H=H, D_V=D_V, BLOCK_H=bh,
        num_warps=4, num_stages=1,
    )
    return out
