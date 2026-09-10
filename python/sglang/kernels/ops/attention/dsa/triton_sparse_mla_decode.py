"""Triton sparse-MLA forward for the DSA fp8 decode path (gfx950).

Prefill packs all H heads into one program per token. Decode tiles heads by 16
and splits the topk reduction across groups so the grid fills the machine at
the token counts MTP actually produces. Enable with `--dsa-decode-backend triton`.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
    _LOG2_FP8_MAX,
    _N_XCD,
    _q_pitch,
    _require_gfx950,
)

_HEAD_TILE = 16


@triton.jit
def _sparse_mla_decode_splitk_partial_kernel(
    q_nope_ptr,
    q_rope_ptr,
    kv_ptr,
    idx_ptr,
    po_ptr,
    pm_ptr,
    pl_ptr,
    o_ptr,
    sm_scale,
    log2_fp8_max,
    topk,
    seq,
    max_page,
    H: tl.constexpr,
    H_TILE: tl.constexpr,
    DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    Q_MAIN_PITCH: tl.constexpr,
    Q_TAIL_PITCH: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_GROUPS: tl.constexpr,
    KEYS_PER_GROUP: tl.constexpr,
    N_XCD: tl.constexpr,
    WIDE_KV_OFFSET: tl.constexpr,
):
    """One (token, head tile, key group) partial. Emits unnormalised acc + m + l."""
    pid = tl.program_id(0)
    head_blocks = (H + H_TILE - 1) // H_TILE
    n_prog = seq * head_blocks * N_GROUPS

    # XCD swizzle over (seq, head_blocks, groups) programs.
    per = n_prog // N_XCD
    rem = n_prog % N_XCD
    xcd = pid % N_XCD
    flat = xcd * per + tl.minimum(xcd, rem) + pid // N_XCD
    ok = flat < n_prog

    g = flat % N_GROUPS
    rest = flat // N_GROUPS
    s_i = rest // head_blocks
    h_tile = rest % head_blocks
    h = h_tile * H_TILE + tl.arange(0, H_TILE)
    hm = (h < H) & ok

    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)
    # Read q halves via head-row pitch to avoid materialising slices on the host.
    q_main = tl.load(
        q_nope_ptr + s_i * H * Q_MAIN_PITCH + h[:, None] * Q_MAIN_PITCH + dv[None, :],
        mask=hm[:, None],
        other=0.0,
    ).to(q_nope_ptr.dtype.element_ty)
    q_tail = tl.load(
        q_rope_ptr + s_i * H * Q_TAIL_PITCH + h[:, None] * Q_TAIL_PITCH + dt[None, :],
        mask=hm[:, None],
        other=0.0,
    ).to(q_rope_ptr.dtype.element_ty)

    # Transpose q once; dot(kv, q_t) avoids transposing kv each iteration.
    q_main_t = tl.trans(q_main)
    q_tail_t = tl.trans(q_tail)

    qk_scale = sm_scale * 1.4426950408889634
    m_i = tl.full([H_TILE], -float("inf"), tl.float32)
    l_i = tl.zeros([H_TILE], tl.float32)
    acc = tl.zeros([H_TILE, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    k_lo = g * KEYS_PER_GROUP
    for k0 in range(0, KEYS_PER_GROUP, BLOCK_N):
        kpos = k_lo + k0 + n
        if KEYS_PER_GROUP % BLOCK_N == 0:
            kmask = (kpos < topk) & ok
        else:
            # Trailing tile must not read into the next group's keys.
            kmask = (k0 + n < KEYS_PER_GROUP) & (kpos < topk) & ok
        idx = tl.load(idx_ptr + s_i * topk + kpos, mask=kmask, other=-1)
        # Range-check in-loop instead of a host-side index clamp launch.
        valid_k = (idx >= 0) & (idx <= max_page) & kmask
        page = tl.where(valid_k, idx, 0)

        if WIDE_KV_OFFSET:
            kbase = kv_ptr + page[:, None].to(tl.int64) * DIM
        else:
            kbase = kv_ptr + page[:, None] * DIM
        kv_main = tl.load(kbase + dv[None, :]).to(q_nope_ptr.dtype.element_ty)
        kv_tail = tl.load(kbase + (D_V + dt)[None, :]).to(q_nope_ptr.dtype.element_ty)

        qkt = tl.dot(kv_main, q_main_t).to(tl.float32)
        qkt += tl.dot(kv_tail, q_tail_t).to(tl.float32)
        qkt = qkt * qk_scale + tl.where(valid_k[:, None], 0.0, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qkt, axis=0))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        pt = tl.exp2(qkt - (m_safe - log2_fp8_max)[None, :])
        l_i = l_i * alpha + tl.sum(pt, axis=0)
        acc = acc * alpha[:, None] + tl.dot(
            tl.trans(pt).to(q_nope_ptr.dtype.element_ty), kv_main
        ).to(tl.float32)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)

    if N_GROUPS == 1:
        # Single group writes the final output directly.
        tl.store(
            o_ptr + s_i * H * D_V + h[:, None] * D_V + dv[None, :],
            (acc * (1.0 / l_safe)[:, None]).to(o_ptr.dtype.element_ty),
            mask=hm[:, None],
        )
        return

    # Rescale partial to its own max before bf16 store; combine folds m back in.
    po_off = (s_i * H + h) * N_GROUPS * D_V + g * D_V
    tl.store(
        po_ptr + po_off[:, None] + dv[None, :],
        (acc * (1.0 / l_safe)[:, None]).to(po_ptr.dtype.element_ty),
        mask=hm[:, None],
    )
    pml_off = (s_i * H + h) * N_GROUPS + g
    tl.store(pm_ptr + pml_off, m_i, mask=hm)
    tl.store(pl_ptr + pml_off, l_i, mask=hm)


@triton.jit
def _sparse_mla_decode_splitk_combine_kernel(
    po_ptr,
    pm_ptr,
    pl_ptr,
    o_ptr,
    seq,
    H: tl.constexpr,
    D_V: tl.constexpr,
    N_GROUPS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    DV_BLOCKS: tl.constexpr,
):
    """Merge per-group partials for one (token, head, d_v slice)."""
    pid = tl.program_id(0)
    ok = pid < seq * H * DV_BLOCKS

    dv_blk = pid % DV_BLOCKS
    rest = pid // DV_BLOCKS
    s_i = rest // H
    h = rest % H

    dv = dv_blk * BLOCK_DV + tl.arange(0, BLOCK_DV)
    dvm = (dv < D_V) & ok
    base = (s_i * H + h) * N_GROUPS

    # Load all groups at once; a python loop over g is latency-bound here.
    g = tl.arange(0, N_GROUPS)
    m_all = tl.load(pm_ptr + base + g, mask=ok, other=-float("inf"))
    l_all = tl.load(pl_ptr + base + g, mask=ok, other=0.0)

    m = tl.max(m_all, axis=0)
    m_safe = tl.where(m == -float("inf"), 0.0, m)

    # Partial store divided by l_g; reweight by l_g * 2^(m_g - m) to merge.
    w = tl.exp2(tl.where(m_all == -float("inf"), -float("inf"), m_all - m_safe))
    w = tl.where(w == w, w, 0.0)
    wl = l_all * w
    lsum = tl.sum(wl, axis=0)

    o_g = tl.load(
        po_ptr + base * D_V + g[:, None] * D_V + dv[None, :],
        mask=dvm[None, :],
        other=0.0,
    ).to(tl.float32)
    acc = tl.sum(o_g * wl[:, None], axis=0)

    l_safe = tl.where(lsum == 0.0, 1.0, lsum)
    acc = acc * (1.0 / l_safe)
    tl.store(
        o_ptr + (s_i * H + h) * D_V + dv,
        acc.to(o_ptr.dtype.element_ty),
        mask=dvm,
    )


def _head_blocks(h: int) -> int:
    return (h + _HEAD_TILE - 1) // _HEAD_TILE


_CUS = 256


def pick_combine_block_dv(tokens: int, heads: int, d_v: int = 512) -> int:
    """Largest power-of-two BLOCK_DV in [64, d_v] targeting ~768 merge programs."""
    target = d_v * tokens * heads // 768
    block_dv = 64
    while block_dv * 2 <= min(target, d_v):
        block_dv *= 2
    return block_dv


def pick_block_n_warps(
    tokens: int, heads: int, n_groups: int, topk: int = 2048
) -> tuple[int, int]:
    """Pick block_n (power of two, <= group size) and num_warps from trip count.

    Avoid (32,2), (64,1), (128,1): they miscompile silently on this Triton.
    """
    keys_per_group = topk // n_groups
    block_n = 1 << (min(128, keys_per_group).bit_length() - 1)
    if keys_per_group <= block_n:
        return block_n, 1 if block_n <= 32 else 2
    return block_n, 4


def pick_num_stages(
    tokens: int, heads: int, n_groups: int, block_n: int, topk: int = 2048
) -> int:
    """Return 3 stages when grid <= CU count and keys divide block_n; else 1."""
    keys_per_group = topk // n_groups
    if keys_per_group <= block_n or keys_per_group % block_n:
        return 1
    grid = tokens * _head_blocks(heads) * n_groups
    return 3 if grid <= _CUS else 1


def pick_n_groups(tokens: int, heads: int, topk: int, cap: int = 16) -> int:
    """Pick split-K group count from token/head grid and topk divisibility."""
    base = tokens * _head_blocks(heads)
    n = 8
    while n > 1 and topk % n:
        n //= 2
    while n > 1 and base * n > 512:
        n //= 2
    grow_lim = 384 if heads <= 8 else 256
    while n < cap and base * n * 2 <= grow_lim and topk % (n * 2) == 0:
        n *= 2
    if tokens <= 2 and topk % 64 == 0:
        n = 64
    return n


def triton_sparse_mla_decode_splitk_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    *,
    block_n: int = 0,
    n_groups: int = 0,
    num_warps: int = 0,
    combine_warps: int = 1,
    combine_block_dv: int = 0,
    num_stages: int = 0,
    partial_dtype: torch.dtype = torch.bfloat16,
    n_xcd: int = _N_XCD,
) -> torch.Tensor:
    """Decode sparse MLA. Returns [1, seq, H, d_v] bf16 (gfx950, fp8 KV)."""
    _require_gfx950()
    seq, h, d_v_in = q_nope.shape
    if d_v_in != d_v:
        raise ValueError(f"expected d_v={d_v}, got {d_v_in}")
    if n_groups <= 0:
        n_groups = pick_n_groups(seq, h, indices.shape[-1])
    auto_block_n, auto_warps = pick_block_n_warps(
        seq, h, n_groups, indices.shape[-1]
    )
    block_n = block_n or auto_block_n
    num_warps = num_warps or auto_warps
    num_stages = num_stages or pick_num_stages(
        seq, h, n_groups, block_n, indices.shape[-1]
    )
    combine_block_dv = combine_block_dv or pick_combine_block_dv(seq, h, d_v)
    if topk_rem := indices.shape[-1] % n_groups:
        raise ValueError(
            f"topk {indices.shape[-1]} not divisible by n_groups {n_groups} "
            f"(remainder {topk_rem})"
        )
    q_nope, q_main_pitch = _q_pitch(q_nope)
    q_rope, q_tail_pitch = _q_pitch(q_rope)
    if indices.dim() == 3:
        indices = indices.squeeze(1)
    indices = indices.contiguous()
    dim = kv.shape[-1]
    d_tail = q_rope.shape[-1]
    topk = indices.shape[-1]
    wide = kv.shape[0] > (2**31 - 1) // dim
    max_page = kv.shape[0] - 1
    head_blocks = _head_blocks(h)

    single = n_groups == 1
    if single:
        partial_o = partial_m = partial_l = q_nope
    else:
        partial_o = torch.empty(
            seq, h, n_groups, d_v, device=q_nope.device, dtype=partial_dtype
        )
        partial_m = torch.empty(
            seq, h, n_groups, device=q_nope.device, dtype=torch.float32
        )
        partial_l = torch.empty_like(partial_m)

    grid = seq * head_blocks * n_groups
    out = torch.empty(seq, h, d_v, device=q_nope.device, dtype=torch.bfloat16)

    _sparse_mla_decode_splitk_partial_kernel[(grid,)](
        q_nope,
        q_rope,
        kv,
        indices,
        partial_o,
        partial_m,
        partial_l,
        out,
        sm_scale,
        _LOG2_FP8_MAX,
        topk,
        seq,
        max_page,
        H=h,
        H_TILE=_HEAD_TILE,
        DIM=dim,
        D_V=d_v,
        D_TAIL=d_tail,
        Q_MAIN_PITCH=q_main_pitch,
        Q_TAIL_PITCH=q_tail_pitch,
        BLOCK_N=block_n,
        N_GROUPS=n_groups,
        KEYS_PER_GROUP=topk // n_groups,
        N_XCD=n_xcd,
        WIDE_KV_OFFSET=wide,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if single:
        return out.unsqueeze(0)

    dv_blocks = (d_v + combine_block_dv - 1) // combine_block_dv
    _sparse_mla_decode_splitk_combine_kernel[(seq * h * dv_blocks,)](
        partial_o,
        partial_m,
        partial_l,
        out,
        seq,
        H=h,
        D_V=d_v,
        N_GROUPS=n_groups,
        BLOCK_DV=combine_block_dv,
        DV_BLOCKS=dv_blocks,
        num_warps=combine_warps,
        num_stages=1,
    )
    return out.unsqueeze(0)
