"""K3 MLA target verify (QLEN=8, 12 heads, FP8 KV latent 576) on gfx950.

An on-device switch picks per batch between AITER's Gluon MLA kernel (short
context) and the HipKittens kernel k3_mla_verify_hk (long context): the Gluon
prologue computes bs * max(L), publishes the regime and exits in the HK regime,
and the HK stage 1 is a no-op in the Gluon regime. The grid depends only on bs,
so this is CUDA-graph safe.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# gfx950-only module, imported lazily by the backend.
from aiter.ops.triton.gluon.mla_gluon import mla_gluon  # noqa: E402

from sglang.kernels.ops.attention.k3_mla_verify_common import (
    _HAS_GLUON,
)
from sglang.kernels.ops.attention.k3_mla_verify_common import (
    _MIN_CHUNK as _SPLIT_MIN_CHUNK,
)
from sglang.kernels.ops.attention.k3_mla_verify_common import (
    _ones,
    _split_chunk,
    default_num_splits,
)

_GLUON_BLOCK_N = 128
_INT32_MAX = 2**31 - 1

# Gluon vs HK crossover (bs * max_L tokens; HK regime iff eff >= thresh).
_THRESH_HK = {1: 27000, 2: 28000, 4: 27000, 8: 29000, 16: 30000}
_THRESH_HK_DEFAULT = 30000


def auto_threshold_hk(bs: int) -> int:
    for k in sorted(_THRESH_HK):
        if bs <= k:
            return _THRESH_HK[k]
    return _THRESH_HK_DEFAULT


@triton.jit
def _k3_auto_reduce(
    FLAG,
    KV_INDPTR,
    O_PART,
    LSE_PART,
    G_LOGITS,
    G_LSE,
    O,
    stride_o_tok,
    stride_o_h,
    g_sl_b,
    g_sl_qs,
    g_sl_h,
    g_sl_s,
    g_ml_b,
    g_ml_qs,
    g_ml_h,
    g_ml_s,
    H: tl.constexpr,
    QLEN: tl.constexpr,
    D_V: tl.constexpr,
    NSPLIT: tl.constexpr,
    NSPLIT_P2: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_CHUNK: tl.constexpr,
    S_BLK: tl.constexpr,
    G_NSPLIT: tl.constexpr,
    G_S_BLK: tl.constexpr,
    G_BLOCK_N: tl.constexpr,
    D_BLK: tl.constexpr,
):
    """One program per (request, row = q_pos*H + h, D_BLK columns)."""
    ROWS: tl.constexpr = H * QLEN
    b = tl.program_id(0)
    r = tl.program_id(1)
    d0 = tl.program_id(2) * D_BLK
    offs_d = d0 + tl.arange(0, D_BLK)
    qpos = r // H
    h = r % H
    o_off = (b * QLEN + qpos).to(tl.int64) * stride_o_tok + h * stride_o_h
    use_v2 = tl.load(FLAG)
    if use_v2 != 0:
        # base-2 LSE merge of the v2/v3 splits
        L = tl.load(KV_INDPTR + b + 1) - tl.load(KV_INDPTR + b)
        chunk = _split_chunk(L, NSPLIT, BLOCK_N, MIN_CHUNK)
        nact = tl.minimum(tl.cdiv(L, chunk), NSPLIT)
        base = (b * NSPLIT) * ROWS + r
        offs_all = tl.arange(0, NSPLIT_P2)
        lse_all = tl.load(
            LSE_PART + base + offs_all * ROWS, mask=offs_all < nact, other=float("-inf")
        )
        m = tl.max(lse_all, axis=0)
        m = tl.where(m == float("-inf"), 0.0, m)
        wsum = tl.sum(tl.where(offs_all < nact, tl.math.exp2(lse_all - m), 0.0), axis=0)
        offs_s = tl.arange(0, S_BLK)
        acc = tl.zeros([D_BLK], dtype=tl.float32)
        for s0 in range(0, nact, S_BLK):
            sid = s0 + offs_s
            sm = sid < nact
            lse = tl.load(LSE_PART + base + sid * ROWS, mask=sm, other=float("-inf"))
            w = tl.where(sm, tl.math.exp2(lse - m), 0.0)
            po = tl.load(
                O_PART
                + (base + sid * ROWS)[:, None].to(tl.int64) * D_V
                + offs_d[None, :],
                mask=sm[:, None],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w[:, None] * po, axis=0)
        o = acc / tl.where(wsum > 0.0, wsum, 1.0)
        tl.store(O + o_off + offs_d, o.to(O.dtype.element_ty))
    elif G_NSPLIT > 1:
        # Gluon splits, as aiter _mla_softmax_reducev_kernel
        gL = tl.load(KV_INDPTR + b + 1) - tl.load(KV_INDPTR + b)
        gper = tl.maximum(G_BLOCK_N, gL // G_NSPLIT)
        gnact = tl.minimum(tl.cdiv(gL, gper), G_NSPLIT)
        base_l = b * g_sl_b + qpos * g_sl_qs + h * g_sl_h
        base_ml = b * g_ml_b + qpos * g_ml_qs + h * g_ml_h
        goffs_s = tl.arange(0, G_S_BLK)
        e_sum = 0.0
        e_max = -float("inf")
        gacc = tl.zeros([D_BLK], dtype=tl.float32)
        for s0 in range(0, gnact, G_S_BLK):
            gsid = s0 + goffs_s
            gsm = gsid < gnact
            glse = tl.load(
                G_LSE + base_ml + gsid * g_ml_s, mask=gsm, other=-float("inf")
            )
            lg = tl.load(
                G_LOGITS + base_l + gsid[:, None] * g_sl_s + offs_d[None, :],
                mask=gsm[:, None],
                other=0.0,
            ).to(tl.float32)
            tile_max = tl.max(glse, axis=0)
            n_e_max = tl.maximum(e_max, tile_max)
            old_scale = tl.where(e_max == -float("inf"), 0.0, tl.exp(e_max - n_e_max))
            gw = tl.where(glse == -float("inf"), 0.0, tl.exp(glse - n_e_max))
            lg = tl.where(glse[:, None] == -float("inf"), 0.0, lg)
            gacc = gacc * old_scale + tl.sum(gw[:, None] * lg, axis=0)
            e_sum = e_sum * old_scale + tl.sum(gw, axis=0)
            e_max = n_e_max
        go = gacc / tl.where(e_sum > 0.0, e_sum, 1.0)
        tl.store(O + o_off + offs_d, go.to(O.dtype.element_ty))


def k3_mla_verify_auto(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    kv_scale=None,
    kv_scale_float: Optional[float] = None,
    *,
    num_heads: int = 12,
    qlen: int = 8,
    v_head_dim: int = 512,
    out: Optional[torch.Tensor] = None,
    thresh: Optional[int] = None,
) -> torch.Tensor:
    """MLA target verify with on-device Gluon/HipKittens regime selection.

    kv_scale: fp32 [1] tensor (or float / None); kv_scale_float: the same scale
    as a host float for the Gluon body. thresh: HK regime iff
    bs * max(L) >= thresh (default auto_threshold_hk(bs))."""
    assert _HAS_GLUON, "k3_mla_verify_auto needs triton.experimental.gluon (gfx950)"
    D = kv_buffer.shape[-1]
    assert D == 576 and v_head_dim == 512 and num_heads * qlen <= 128
    kv = kv_buffer.view(-1, D)
    q = q.view(-1, num_heads, D)
    assert q.stride(-1) == 1 and kv.stride(-1) == 1
    ntok = q.shape[0]
    bs = ntok // qlen
    assert bs * qlen == ntok
    if out is None:
        out = q.new_empty((ntok, num_heads, v_head_dim))
    assert out.is_contiguous()
    if kv_scale_float is None:
        kv_scale_float = (
            1.0
            if isinstance(kv_scale, torch.Tensor) or kv_scale is None
            else float(kv_scale)
        )
    if kv_scale is None:
        kv_scale = _ones(q.device)
    elif not isinstance(kv_scale, torch.Tensor):
        kv_scale = torch.full(
            (1,), float(kv_scale), dtype=torch.float32, device=q.device
        )
    if bs == 0:
        return out
    if thresh is None:
        thresh = auto_threshold_hk(bs)
    thresh = int(min(max(thresh, 0), _INT32_MAX))
    dev = q.device
    rows = num_heads * qlen

    q4 = q.view(bs, qlen, num_heads, D)
    o4 = out.view(bs, qlen, num_heads, v_head_dim)

    # v2 operands / partials
    min_chunk = _SPLIT_MIN_CHUNK
    nsplit = max(2, default_num_splits(bs))
    block_n = 64
    o_part = torch.empty(
        (bs, nsplit, rows, v_head_dim), dtype=torch.bfloat16, device=dev
    )
    lse_part = torch.empty((bs, nsplit, rows), dtype=torch.float32, device=dev)

    # ---------------- 1. Gluon stage 1 with on-device kernel selection
    sel = torch.empty((bs + 2,), dtype=torch.int32, device=dev)
    flag = sel[bs + 1 :]
    g_logits, g_lse = mla_gluon(
        q4[..., :v_head_dim],
        q4[..., v_head_dim:],
        kv,
        o4,
        kv_indices,
        kv_indptr,
        sm_scale,
        kv_pe_offset=v_head_dim,
        use_2d_view=False,
        kv_scale=kv_scale_float,
        sel_buf=sel,
        sel_thresh=thresh,
        skip_reduce=True,
    )
    if g_lse is None:  # one split: Gluon stored the final output directly
        g_logits = o4.view(bs, qlen, num_heads, 1, v_head_dim)
        g_ml = (0, 0, 0, 0)
    else:
        g_ml = g_lse.stride()
    g_ns = g_logits.shape[3]
    g_sl = g_logits.stride()[:4]
    # ---------------- 2. HK / v2 / v3 stage 1 on sel (all-empty in the Gluon regime)
    from sglang.kernels.ops.attention.k3_mla_verify_hk import (
        PART8_MIN_SPLITS,
        hip_reduce_enabled,
        hk_stage1,
    )

    hk_p8 = PART8_MIN_SPLITS if hip_reduce_enabled(nsplit) else 0
    hk_stage1(
        q,
        kv,
        sel,
        kv_indices,
        kv_scale,
        o_part,
        lse_part,
        out,
        nsplit,
        min_chunk,
        sm_scale,
        hk_p8,
    )
    from sglang.kernels.ops.attention.k3_mla_verify_hk import (
        hip_reduce_enabled,
        hk_reduce,
    )

    if hip_reduce_enabled(nsplit):
        hk_reduce(
            o_part,
            lse_part,
            kv_indptr,
            out,
            nsplit,
            min_chunk,
            kv_scale,
            hk_p8,
            regime=flag,
            g_logits=g_logits,
            g_lse=g_lse,
            g_ns=g_ns,
            g_block_n=_GLUON_BLOCK_N,
        )
        return out
    g_lse_arg = g_lse if g_lse is not None else lse_part
    _k3_auto_reduce[(bs, rows, v_head_dim // 128)](
        flag,
        kv_indptr,
        o_part,
        lse_part,
        g_logits,
        g_lse_arg,
        out,
        out.stride(0),
        out.stride(1),
        g_sl[0],
        g_sl[1],
        g_sl[2],
        g_sl[3],
        g_ml[0],
        g_ml[1],
        g_ml[2],
        g_ml[3],
        H=num_heads,
        QLEN=qlen,
        D_V=v_head_dim,
        NSPLIT=nsplit,
        NSPLIT_P2=triton.next_power_of_2(nsplit),
        BLOCK_N=block_n,
        MIN_CHUNK=min_chunk,
        S_BLK=min(64, triton.next_power_of_2(nsplit)),
        G_NSPLIT=g_ns,
        G_S_BLK=min(64, triton.next_power_of_2(g_ns)),
        G_BLOCK_N=_GLUON_BLOCK_N,
        D_BLK=128,
        num_warps=4 if nsplit >= 64 else 2,
    )
    return out
