# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Fused Triton sparse-MLA prefill for DeepSeek Sparse Attention (DSA) on CUDA.

One Triton program per query token: the heads are padded into a 16-row MMA tile
rather than into the grid, the online softmax stays in registers, and ``V`` is
read as ``kv[:, :d_v]``, so there are no partials and no merge pass.
``dsa/triton_sparse_mla.py`` is the gfx950-only FP8 sibling of this bf16 module.

Interface matches ``sgl_kernel.flash_mla.flash_mla_sparse_fwd``::

    q       [T, H, 576] bf16    (absorbed MLA: 512 nope + 64 rope)
    kv      [S, 576]    bf16    (V is kv[:, :512])
    indices [T, topk]   int32   (-1 marks an invalid slot; no upper-bound check)
    out     [T, H, 512] bf16

A row of ``indices`` must not name the same KV position twice. Top-k selection
cannot, so this holds for every DSA caller; it is stated because ``union``
gathers the distinct union of G rows and weights each position once, whereas
the base path would weight a repeat twice.

``union`` (opt-in, 2 or 4): G adjacent query tokens share one gathered union
index set, and a per-row ownership bitmask restores each token's own softmax
support, so the result is the per-token result up to accumulation order. It
pays off because neighbouring tokens' selections overlap heavily on real indexer
output; the measurements are in the PR that added this module.

All tunables are explicit arguments; the module reads no environment variables.
If a tuned tile exceeds the device shared-memory budget, the launcher steps down
through smaller tiles instead of raising ``OutOfResources``.
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Swept on the hardware named; anything else gets _UNTUNED_DEFAULT and a warning.
_PINNED = {
    (9, 0): (64, 8, 2),  # SM90, swept at T=8192
    (12, 0): (64, 4, 2),  # SM120, swept at T=8192
    (12, 1): (64, 4, 2),  # SM121: SM120's tile, not swept separately
}
_UNTUNED_DEFAULT = (64, 8, 3)
_UNTUNED_ARCH_WARNED = set()


@triton.jit
def _sparse_mla_prefill_kernel(
    q_ptr,
    kv_ptr,
    idx_ptr,
    len_ptr,
    o_ptr,
    sm_scale,
    topk,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IDX64: tl.constexpr,
):
    t = tl.program_id(0)
    # Token offsets in int64: at h=128 the q offset t*H*D_QK passes 2^31 from
    # t=29128 and the output offset from t=32768, wrapping to another token.
    t64 = t.to(tl.int64)
    D_TAIL: tl.constexpr = D_QK - D_V

    h = tl.arange(0, BLOCK_H)
    hmask = h < H
    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)

    qb = q_ptr + t64 * H * D_QK
    q_main = tl.load(
        qb + h[:, None] * D_QK + dv[None, :], mask=hmask[:, None], other=0.0
    )
    q_tail = tl.load(
        qb + h[:, None] * D_QK + (D_V + dt)[None, :], mask=hmask[:, None], other=0.0
    )

    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    k_len = tl.load(len_ptr + t)
    for k0 in tl.range(0, k_len, BLOCK_N):
        idx = tl.load(idx_ptr + t64 * topk + k0 + n, mask=(k0 + n) < k_len, other=-1)
        valid = idx >= 0
        if IDX64:
            row = tl.where(valid, idx, 0).to(tl.int64)
        else:
            row = tl.where(valid, idx, 0)
        kb = kv_ptr + row[:, None] * D_QK
        kv_main = tl.load(kb + dv[None, :], mask=valid[:, None], other=0.0)
        kv_tail = tl.load(kb + (D_V + dt)[None, :], mask=valid[:, None], other=0.0)

        qk = tl.dot(q_main, tl.trans(kv_main))
        qk = tl.dot(q_tail, tl.trans(kv_tail), qk) * sm_scale
        qk = tl.where(valid[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv_main.dtype), kv_main)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc * (1.0 / l_safe[:, None])
    tl.store(
        o_ptr + t64 * H * D_V + h[:, None] * D_V + dv[None, :],
        acc.to(o_ptr.dtype.element_ty),
        mask=hmask[:, None],
    )


# ---------------------------------------------------------------------------
# Union path: G adjacent tokens share one gathered KV set. Per group, the G*K
# selected rows are sorted and deduplicated on the device, and an ownership bit
# per unique row lets the kernel below restore each token's own softmax
# support. The cost is O(G*K*log K) per group whatever the KV span, so
# nothing here depends on the index range, reads back to the host, or
# persists across calls.
# ---------------------------------------------------------------------------


@triton.jit
def _union_dedup_kernel(
    sorted_ptr,
    uidx_ptr,
    ubits_ptr,
    ulen_ptr,
    K: tl.constexpr,
    G: tl.constexpr,
    LOG_K: tl.constexpr,
):
    # One program per group of G tokens whose K selected rows are each sorted
    # ascending (-1 pads sort first). Token i emits row v if no earlier token
    # selected it; its ownership bits OR in every token that did. Membership is
    # a vectorised lower_bound over the other token's list, which stays in L1.
    g = tl.program_id(0).to(tl.int64)
    grp = sorted_ptr + g * G * K
    n = tl.arange(0, K)
    cursor = tl.zeros([], tl.int32)
    for i in tl.static_range(G):
        v = tl.load(grp + i * K + n)
        valid = v >= 0
        bits = tl.where(valid, 1 << i, 0).to(tl.int32)
        first = valid
        for j in tl.static_range(G):
            if j != i:
                lst = grp + j * K
                lo = tl.zeros([K], tl.int32)
                hi = tl.full([K], K, tl.int32)
                for _ in tl.static_range(LOG_K):
                    mid = (lo + hi) // 2
                    m = tl.load(lst + mid, mask=mid < K, other=2147483647)
                    lt = m < v
                    lo = tl.where(lt, mid + 1, lo)
                    hi = tl.where(lt, hi, mid)
                found = (tl.load(lst + lo, mask=lo < K, other=-2) == v) & valid
                bits |= tl.where(found, 1 << j, 0).to(tl.int32)
                if j < i:
                    first = first & (found == 0)
        wpos = cursor + tl.cumsum(first.to(tl.int32), axis=0) - first.to(tl.int32)
        tl.store(uidx_ptr + g * (G * K) + wpos, v, mask=first)
        tl.store(ubits_ptr + g * (G * K) + wpos, bits, mask=first)
        cursor += tl.sum(first.to(tl.int32))
    tl.store(ulen_ptr + g, cursor)


# Per-group unique rows and ownership bits for [NG*G, K] indices: uidx [NG, G*K]
# int32 holds each group's distinct rows in its first ulen[g] slots (token-major,
# each token's new rows ascending, the rest unspecified), ubits [NG, G*K] int32
# has bit i set when token i selected the row, ulen [NG] int32. torch sorts each
# K-wide row in shared memory; one Triton pass does the cross-token membership.
def _union_dedup(idx_main, G):
    T_main, K = idx_main.shape
    NG = T_main // G
    srt = torch.sort(idx_main, dim=1)[0]
    uidx = torch.empty(NG, G * K, dtype=torch.int32, device=idx_main.device)
    ubits = torch.empty_like(uidx)
    ulen = torch.empty(NG, dtype=torch.int32, device=idx_main.device)
    _union_dedup_kernel[(NG,)](
        srt,
        uidx,
        ubits,
        ulen,
        K=K,
        G=G,
        # log2(K) + 1 halvings: the last one resolves a one-element interval.
        LOG_K=K.bit_length(),
        num_warps=16 if G == 4 else 8,  # swept on H20 at K=2048
    )
    return uidx, ubits, ulen


@triton.jit
def _sparse_mla_prefill_union_kernel(
    q_ptr,
    kv_ptr,
    uidx_ptr,
    ubits_ptr,
    ulen_ptr,
    o_ptr,
    sm_scale,
    U_STRIDE,
    H: tl.constexpr,
    G: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    g = tl.program_id(0)
    D_TAIL: tl.constexpr = D_QK - D_V
    GH: tl.constexpr = G * H

    r = tl.arange(0, GH)
    tok_of_row = r // H
    dv = tl.arange(0, D_V)
    dt = tl.arange(0, D_TAIL)

    qb = q_ptr + g.to(tl.int64) * GH * D_QK
    q_main = tl.load(qb + r[:, None] * D_QK + dv[None, :])
    q_tail = tl.load(qb + r[:, None] * D_QK + (D_V + dt)[None, :])

    m_i = tl.full([GH], -float("inf"), tl.float32)
    l_i = tl.zeros([GH], tl.float32)
    acc = tl.zeros([GH, D_V], tl.float32)

    n = tl.arange(0, BLOCK_N)
    u_len = tl.load(ulen_ptr + g)
    ub = g.to(tl.int64) * U_STRIDE
    for k0 in tl.range(0, u_len, BLOCK_N):
        inb = (k0 + n) < u_len
        uidx = tl.load(uidx_ptr + ub + k0 + n, mask=inb, other=-1)
        bits = tl.load(ubits_ptr + ub + k0 + n, mask=inb, other=0)
        valid = uidx >= 0
        row = tl.where(valid, uidx, 0).to(tl.int64)
        kb = kv_ptr + row * D_QK
        kv_main = tl.load(kb[:, None] + dv[None, :], mask=valid[:, None], other=0.0)
        kv_tail = tl.load(
            kb[:, None] + (D_V + dt)[None, :], mask=valid[:, None], other=0.0
        )

        qk = tl.dot(q_main, tl.trans(kv_main))
        qk = tl.dot(q_tail, tl.trans(kv_tail), qk) * sm_scale
        sel = ((bits[None, :] >> tok_of_row[:, None]) & 1) != 0
        qk = tl.where(sel & valid[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv_main.dtype), kv_main)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr + g.to(tl.int64) * GH * D_V + r[:, None] * D_V + dv[None, :],
        acc.to(o_ptr.dtype.element_ty),
    )


# Returns True if handled; the T % G tail rows go through the per-token path.
def _union_path(*, q, kv, indices, sm_scale, d_v, out, G, union_config):
    T, h, d_qk = q.shape
    K = indices.shape[-1]
    rows = G * h
    # tl.arange needs a power of two and tl.dot needs M >= 16; the tuned tiles
    # stop at 32 rows. The backend validator rejects the same shapes at startup.
    if G < 2 or rows < 16 or rows > 32 or rows & (rows - 1):
        return False
    if K & (K - 1):  # the dedup kernel tiles each token's list with tl.arange(0, K)
        return False
    T_main = (T // G) * G
    if T_main == 0:
        return False
    NG = T_main // G
    uidx, ubits, ulen = _union_dedup(idx_main=indices[:T_main], G=G)
    U_STRIDE = G * K
    if union_config is not None:
        bn, warps, stages = union_config
    elif torch.cuda.get_device_capability(q.device)[0] >= 12:
        # SM120 sweeps on real indices: (64,4,3) at G=2. At G=4 the 32-row Q
        # tile makes BN=64 exceed the 100 KB budget; (32,4,2) beats every neighbour.
        bn, warps, stages = (64, 4, 3) if G == 2 else (32, 4, 2)
    else:
        bn, warps, stages = (64, 4, 2) if G == 4 else (64, 8, 2)
    # The union Q tile is G*H rows, so its shared-memory footprint grows with
    # the head count (16 heads at G=2 exceeds SM120's 100 KB with the tuned
    # tile). Step down like the per-token launcher rather than fail the request.
    for bn_try, ns_try in _tile_candidates("union", h, G, q.device, bn, stages):
        try:
            _sparse_mla_prefill_union_kernel[(NG,)](
                q[:T_main],
                kv,
                uidx,
                ubits,
                ulen,
                out[:T_main],
                sm_scale,
                U_STRIDE,
                H=h,
                G=G,
                D_QK=d_qk,
                D_V=d_v,
                BLOCK_N=bn_try,
                num_warps=warps,
                num_stages=ns_try,
            )
        except triton.runtime.errors.OutOfResources:
            continue
        _FIT_TILE[("union", h, G, q.device.index)] = (bn_try, ns_try)
        break
    else:
        return False  # no tile fits; caller falls through to the per-token path
    if T_main < T:
        sparse_mla_prefill(
            q[T_main:],
            kv,
            indices[T_main:],
            sm_scale,
            d_v,
            out=out[T_main:],
            union=0,
        )
    return True


def _topk_length(indices, topk):
    valid = indices >= 0
    any_valid = valid.any(dim=-1)
    last = topk - torch.flip(valid, [-1]).int().argmax(dim=-1)
    return torch.where(any_valid, last, torch.zeros_like(last)).to(torch.int32)


# Per-arch tuned (BLOCK_N, num_warps, num_stages); see _PINNED.
def _config(device):
    cap = torch.cuda.get_device_capability(device)
    if cap in _PINNED:
        return _PINNED[cap]
    if cap not in _UNTUNED_ARCH_WARNED:
        _UNTUNED_ARCH_WARNED.add(cap)
        # The kernel is correct on any SM90+ device, but the tile was only swept
        # on the architectures in _PINNED. Say so rather than quietly running a
        # config nobody has measured; add an entry there once swept.
        logger.warning(
            "triton_sparse_mla: no tuned tile for sm_%d%d; falling back to %s. "
            "Sweep and add it to _PINNED for best throughput.",
            cap[0],
            cap[1],
            _UNTUNED_DEFAULT,
        )
    return _UNTUNED_DEFAULT


# Ordered (BLOCK_N, num_stages) candidates: the tuned config first, then
# progressively smaller shared-memory footprints, so one pinned config serves
# head counts and devices whose budget the tuned tile would exceed.
def _smem_fallbacks(bn, stages):
    seen, out = set(), []
    for cand in (
        (bn, stages),
        (bn, 2),
        (bn // 2, stages),
        (bn // 2, 2),
        (bn // 4, 2),
        (16, 2),
    ):
        b, ns = max(16, cand[0]), max(1, cand[1])
        if (b, ns) not in seen:
            seen.add((b, ns))
            out.append((b, ns))
    return out


# The tile that fit last time, per (path, num_heads, group, device index). The
# tuned tile can exceed a device's shared memory (h=32 on SM120's 100 KB); the
# step-down must then run once per shape, not on every layer of every prefill.
_FIT_TILE = {}


# _smem_fallbacks with the tile that fit last time moved to the front.
def _tile_candidates(path, h, G, device, bn, stages):
    cands = _smem_fallbacks(bn, stages)
    fit = _FIT_TILE.get((path, h, G, device.index))
    if fit in cands:
        cands.remove(fit)
        cands.insert(0, fit)
    return cands


def sparse_mla_prefill(
    q,
    kv,
    indices,
    sm_scale,
    d_v=512,
    *,
    topk_length=None,
    out=None,
    union=0,
    config=None,
    union_config=None,
    int64_indexing=None,
):
    """Fused sparse-MLA prefill. Returns ``out`` ``[T, H, d_v]`` bf16.

    Args:
        q: ``[T, H, d_qk]`` bf16 query (absorbed MLA; ``d_qk = d_v + rope``).
        kv: ``[S, d_qk]`` or ``[S, 1, d_qk]`` bf16 latent cache; ``V`` is
            ``kv[:, :d_v]`` (no separate value gather).
        indices: ``[T, topk]`` or ``[T, 1, topk]`` int32 selected slots; ``-1``
            marks an invalid slot and is skipped. Every other value must be a
            row of ``kv``: the kernels make no upper-bound check.
        sm_scale: softmax scale.
        d_v: value head dim (512 for DSA).
        topk_length: optional ``[T]`` int32 per-row valid count. Computed from
            ``indices`` when omitted; pass it to skip that reduction.
        out: optional preallocated ``[T, H, d_v]`` bf16 output.
        union: 0 (off), 2 or 4: share one gathered union index set across ``G``
            adjacent query tokens. Exact, not an approximation: an ownership
            bitmask restores each token's own softmax support.
        config / union_config: optional tile overrides
            ``(BLOCK_N, num_warps, num_stages)``. Defaults are the per-arch
            tuned entries in ``_PINNED``.
    """
    if kv.dim() == 3:  # [S, 1, D] -> [S, D]
        assert kv.shape[1] == 1
        kv = kv.squeeze(1)
    if indices.dim() == 3:  # [T, 1, K] -> [T, K]
        assert indices.shape[1] == 1
        indices = indices.squeeze(1)
    T, h, d_qk = q.shape
    topk = indices.shape[-1]
    q, kv, indices = q.contiguous(), kv.contiguous(), indices.contiguous()
    if kv.dtype != torch.bfloat16:
        # An FP8 pool that reaches this entry (the PAGED transform leaves it
        # packed) would otherwise fail in Triton compile or read fp8 bytes at a
        # 576-element stride.
        raise TypeError(
            f"sparse_mla_prefill needs a bf16 KV cache, got {kv.dtype}; "
            "dequantize the pool first."
        )
    if out is None:
        out = torch.empty(T, h, d_v, dtype=torch.bfloat16, device=q.device)

    if union in (2, 4) and _union_path(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=d_v,
        out=out,
        G=union,
        union_config=union_config,
    ):
        return out

    if topk_length is None:
        topk_length = _topk_length(indices, topk)

    bn, warps, stages = config or _config(q.device)
    # int32 gather addressing keeps the gather loop on IMAD; int64 only when the
    # pool could overflow int32 element offsets (row*d_qk + d_qk-1), i.e. above
    # ~3.7M rows at d_qk=576. No test can allocate that pool, so the mode is
    # overridable to keep the int64 path reachable from a test.
    if int64_indexing is None:
        idx64 = kv.shape[0] > (2**31 - 1 - (d_qk - 1)) // d_qk
    else:
        idx64 = bool(int64_indexing)
    block_h = max(16, triton.next_power_of_2(h))
    for bn_try, ns_try in _tile_candidates("base", h, 0, q.device, bn, stages):
        try:
            _sparse_mla_prefill_kernel[(T,)](
                q,
                kv,
                indices,
                topk_length,
                out,
                sm_scale,
                topk,
                H=h,
                BLOCK_H=block_h,
                D_QK=d_qk,
                D_V=d_v,
                BLOCK_N=bn_try,
                num_warps=warps,
                num_stages=ns_try,
                IDX64=idx64,
            )
        except triton.runtime.errors.OutOfResources:
            # A larger head count (BLOCK_H) or a smaller smem budget can push the
            # pinned tile over the device limit (e.g. h=32 on SM120's 100 KB).
            # Step down the K tile / pipeline depth instead of failing the request.
            continue
        _FIT_TILE[("base", h, 0, q.device.index)] = (bn_try, ns_try)
        return out
    raise triton.runtime.errors.OutOfResources(
        0, 0, "shared memory: no fallback config fits this device/shape"
    )
