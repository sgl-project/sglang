# SPDX-License-Identifier: Apache-2.0
"""Speculative target-verify attention (a causal chain of ``T <= 8`` draft tokens per
request) for the Triton attention backend on gfx1151.

``extend_attention_fwd`` walks the cached prefix once per query block, and the split-KV
verify kernels are gfx95-only. Here one program per (request, kv head, group of ``HG`` q
heads, split) reads its share of the prefix once for all ``T x HG`` query rows of the
request and runs QK and PV on the matrix cores; a second kernel merges the splits. Row
``t`` sees ``seq_lens[b] + t + 1`` tokens: the committed prefix plus block tokens
``0..t``, whose K/V the backend has already written to the cache.

Numerics follow the extend kernel: fp32 online softmax, ``p`` rounded to bf16 for PV. On
gfx1151 with Qwen3.8 shapes (24 q heads, 4 kv heads, head dim 256) this takes 0.35 / 0.8 /
2.6 ms per layer at 1k / 8k / 39k prefixes against 1.3 / 6.6 / 30.7 ms for
``extend_attention_fwd``.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.utils import use_gfx1151_kernels

# (BT, warps, HG) = (32, 4, 6) and 20 splits per kv head, swept on gfx1151 with Qwen3.8
# shapes; HG = 3 avoids the accumulator's register spills but is 2-3x slower
NSPLIT = 20


@triton.jit
def _verify_split_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    part_ptr,
    ml_ptr,
    req_to_token,
    r2t_stride,
    req_pool_indices,
    seq_lens,
    scale,
    T: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    NSPLIT: tl.constexpr,
    BT: tl.constexpr,
    HG: tl.constexpr,
):
    """One program per (request, kv head, group of ``HG`` of its q heads, split); rows are
    (block token, q head of the group), 8 x next_pow2(HG). The groups of a kv head are
    neighbouring programs, so the second read of a token range mostly hits the cache."""
    GQ: tl.constexpr = HQ // HKV
    NG: tl.constexpr = (GQ + HG - 1) // HG
    HP: tl.constexpr = 4 if HG <= 4 else 8
    R: tl.constexpr = 8 * HP
    u = tl.program_id(0)
    g = u % NG
    j = (u // NG) % HKV
    s = (u // (NG * HKV)) % NSPLIT
    b = u // (NG * HKV * NSPLIT)
    prefix = tl.load(seq_lens + b).to(tl.int32)
    rp = tl.load(req_pool_indices + b).to(tl.int64)
    total = prefix + T
    chunk = tl.cdiv(total, NSPLIT)
    t0 = s * chunk
    t1 = tl.minimum(total, t0 + chunk)
    r = tl.arange(0, R)
    m = r // HP
    h = g * HG + r % HP
    valid = (r % HP < HG) & (h < GQ) & (m < T)
    seq_r = prefix + m + 1
    d = tl.arange(0, D)
    qrow = (b * T + m) * HQ + j * GQ + h
    q = tl.load(q_ptr + qrow[:, None] * D + d[None, :], mask=valid[:, None], other=0.0)
    bt = tl.arange(0, BT)
    m_i = tl.full((R,), float("-inf"), tl.float32)
    l_i = tl.zeros((R,), tl.float32)
    acc = tl.zeros((R, D), tl.float32)
    for t in range(t0, t1, BT):
        tt = t + bt
        tok = tl.load(req_to_token + rp * r2t_stride + tl.minimum(tt, t1 - 1))
        # int32 offsets (the caller checks the pool size); 16-byte hints, unmasked loads
        off = tl.multiple_of(tok * (HKV * D) + j * D, 16)[:, None] + d[None, :]
        k = tl.load(tl.max_contiguous(tl.multiple_of(k_ptr + off, [1, 16]), [1, 8]))
        sc = tl.dot(q, tl.trans(k)) * scale
        sc = tl.where(
            (tt[None, :] < seq_r[:, None]) & (tt[None, :] < t1), sc, float("-inf")
        )
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(sc - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(tl.max_contiguous(tl.multiple_of(v_ptr + off, [1, 16]), [1, 8]))
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    # partial row of (request token row, q head, split)
    prow = qrow * NSPLIT + s
    tl.store(part_ptr + prow[:, None] * D + d[None, :], acc, mask=valid[:, None])
    tl.store(ml_ptr + prow * 2, m_i, mask=valid)
    tl.store(ml_ptr + prow * 2 + 1, l_i, mask=valid)


@triton.jit
def _verify_combine_kernel(
    part_ptr, ml_ptr, o_ptr, D: tl.constexpr, NSPLIT: tl.constexpr
):
    """Merge one (token row, q head)'s splits into ``o`` (bf16)."""
    row = tl.program_id(0)
    d = tl.arange(0, D)
    mx = tl.full((8,), float("-inf"), tl.float32)
    for c in tl.static_range(0, NSPLIT, 8):
        s = c + tl.arange(0, 8)
        mx = tl.maximum(
            mx,
            tl.load(
                ml_ptr + (row * NSPLIT + s) * 2, mask=s < NSPLIT, other=float("-inf")
            ),
        )
    mx = tl.max(mx, axis=0)
    num = tl.zeros((D,), tl.float32)
    den = tl.zeros((8,), tl.float32)
    for c in tl.static_range(0, NSPLIT, 8):
        s = c + tl.arange(0, 8)
        smask = s < NSPLIT
        mm = tl.load(ml_ptr + (row * NSPLIT + s) * 2, mask=smask, other=float("-inf"))
        ll = tl.load(ml_ptr + (row * NSPLIT + s) * 2 + 1, mask=smask, other=0.0)
        w = tl.where(mm == float("-inf"), 0.0, tl.exp(mm - mx))
        acc = tl.load(
            part_ptr + (row * NSPLIT + s)[:, None] * D + d[None, :],
            mask=smask[:, None],
            other=0.0,
        )
        num += tl.sum(w[:, None] * acc, axis=0)
        den += w * ll
    tl.store(
        o_ptr + row * D + d, (num / tl.sum(den, axis=0)).to(o_ptr.dtype.element_ty)
    )


def covered(q, k_buffer, num_tokens_per_req, hq, hkv, d) -> bool:
    """Whether ``verify_attention_gfx1151`` takes this layer: gfx1151, bf16, at most 8 draft
    tokens per request and 8 q heads per kv head, int32-addressable contiguous caches."""
    return (
        use_gfx1151_kernels()
        and 0 < num_tokens_per_req <= 8
        and hq % hkv == 0
        and hq // hkv <= 8
        and d in (64, 128, 256)
        and q.dtype == torch.bfloat16
        and k_buffer.dtype == torch.bfloat16
        and k_buffer.is_contiguous()
        and k_buffer.numel() < 2**31
    )


def verify_attention_gfx1151(
    q,
    o,
    k_buffer,
    v_buffer,
    req_to_token,
    req_pool_indices,
    seq_lens,
    num_tokens_per_req,
    scale,
    nsplit=NSPLIT,
    bt=32,
    num_warps=4,
    hg=6,
):
    """``o[b * T + t, h]`` = attention of ``q[b * T + t, h]`` over ``seq_lens[b] + t + 1``
    tokens of request ``b`` (``req_to_token[req_pool_indices[b]]``); ``q`` and ``o`` are
    ``[B * T, HQ, D]``, the caches ``[slots, HKV, D]``. Graph-capturable."""
    rows, hq, d = q.shape
    hkv = k_buffer.shape[1]
    T = num_tokens_per_req
    B = rows // T
    part = torch.empty(rows * hq * nsplit, d, device=q.device, dtype=torch.float32)
    ml = torch.empty(rows * hq * nsplit, 2, device=q.device, dtype=torch.float32)
    ng = -(-(hq // hkv) // hg)
    _verify_split_kernel[(B * hkv * ng * nsplit,)](
        q,
        k_buffer,
        v_buffer,
        part,
        ml,
        req_to_token,
        req_to_token.stride(0),
        req_pool_indices,
        seq_lens,
        scale,
        T=T,
        HQ=hq,
        HKV=hkv,
        D=d,
        NSPLIT=nsplit,
        BT=bt,
        HG=hg,
        num_warps=num_warps,
    )
    _verify_combine_kernel[(rows * hq,)](part, ml, o, D=d, NSPLIT=nsplit, num_warps=4)
    return o
