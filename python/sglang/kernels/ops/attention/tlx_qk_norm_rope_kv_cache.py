# Vendored from facebookexperimental/triton third_party/tlx/ops/kernels/fused/qk_norm_rope_kv_cache/gfx950.py at 87abb3f5a (MIT); see kernels/README.md.
"""Attention input preparation on gfx950: q/k head RMSNorm, RoPE, q quantization and the paged FP8 KV cache write.

One program per token reads the token's ``q``, ``k`` and ``v`` rows (column slices of a fused QKV projection) once and
does what a serving stack does in four launches between the QKV projection and paged attention:

1. weightless RMSNorm of every ``q`` and ``k`` head in FP32, rounded to BF16; ``q`` is then multiplied by ``q_scale``
   and rounded to BF16 again (the result of ``rms_norm(q).to(bf16) * q_scale`` on BF16 tensors);
2. optional RoPE on interleaved pairs ``(x[2i], x[2i + 1])`` over the whole head, with ``cos``/``sin`` read from an
   FP32 ``(max_position, head_dim)`` table (``cos`` in the first half of a row, ``sin`` in the second). With
   ``rope_bf16=True`` the table values are rounded to BF16 and every product and sum is rounded to BF16, as the
   common rotary-embedding kernel that computes in the activation dtype does; otherwise each output is computed in
   FP32 and rounded once;
3. ``k / k_scale`` and ``v / v_scale`` in FP32, converted to ``float8_e4m3fn`` (round to nearest even, saturating)
   and written to ``cache[slot // page, slot % page]`` of ``(num_blocks, page, kv_heads, head_dim)`` caches; tokens
   whose slot is negative (after the optional slot remapping table) write nothing;
4. ``q`` returned in BF16, or in ``float8_e4m3fn`` as ``clamp(q * rcp(q_out_scale), -448, 448)`` rounded to nearest
   even, with the hardware reciprocal (``v_rcp_f32``).

Every program has four waves. The per-head sums of squares have a fixed order, the one a one-program-per-token
Triton norm kernel of four waves compiles to with Triton 3.7, so the normed values match such a kernel bit for bit
at 16 query and 2 KV heads of 64:

- ``q`` (four contiguous elements per lane, a head over 16 lanes): each lane adds ``fma(x0, x0, x2 * x2) +
  fma(x1, x1, x3 * x3)``, written out here because newer Triton adds a lane's elements left to right instead; the
  lane partials are added in a butterfly (lanes ``i`` and ``i ^ 8``, then ``^ 4``, ``^ 2``, ``^ 1``);
- ``k`` (one element per lane, a head per wave): the wave reduction of ``tl.sum``. RoPE needs each element's pair
  partner, which lives in the neighbouring lane; the kernel loads it a second time instead of reshaping into pairs,
  which would put two elements in a lane and change the order.

Every operation is per token, so a token's result does not depend on the batch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_bf16(x, partner, c, s):
    """``x * c + partner * s`` with ``c``, ``s`` and every product and sum rounded to BF16 (FP32 inputs)."""
    c = c.to(tl.bfloat16).to(tl.float32)
    s = s.to(tl.bfloat16).to(tl.float32)
    return ((x * c).to(tl.bfloat16).to(tl.float32) + (partner * s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _rope(x, partner, c, s, ROPE_BF16: tl.constexpr):
    """One RoPE output from BF16 values in FP32: ``x`` the element, ``partner`` the other element of its pair with the
    sign of its term (``-y`` for the first element of a pair, ``+x`` for the second)."""
    if ROPE_BF16:
        out = _rope_bf16(x, partner, c, s)
    else:
        out = (x * c + partner * s).to(tl.bfloat16)
    return out


@triton.jit
def _rcp(x):
    return tl.inline_asm_elementwise("v_rcp_f32 $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _qk_norm_rope_kv_cache_kernel(Q, K, V, POS, COS_SIN, SLOT, SWA_SLOT, K_SCALE, V_SCALE, Q_OUT_SCALE, QO, KC, VC,
                                  stride_q, stride_k, stride_v, stride_cos_sin, stride_block, eps, q_scale,
                                  HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                                  ROPE: tl.constexpr, ROPE_BF16: tl.constexpr, HAS_SWA: tl.constexpr,
                                  KV_SCALE: tl.constexpr, Q_FP8: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    if ROPE:
        cs = COS_SIN + tl.load(POS + t).to(tl.int64) * stride_cos_sin

    # q: four contiguous elements per lane, a head over 16 lanes. Each lane sums its squares as
    # fma(x0, x0, x2 * x2) + fma(x1, x1, x3 * x3), and tl.sum adds the 16 lane partials in a butterfly.
    q = tl.reshape(tl.load(Q + t * stride_q + tl.arange(0, HQ * D)).to(tl.float32), (HQ, D // 4, 2, 2))
    q_even, q_odd = tl.split(q)  # (HQ, D // 4, 2): pair p of a lane, first and second element
    x0, x2 = tl.split(q_even)
    x1, x3 = tl.split(q_odd)
    sq = tl.sum(tl.fma(x0, x0, x2 * x2) + tl.fma(x1, x1, x3 * x3), axis=1)
    rstd_q = tl.rsqrt(sq / D + eps)[:, None, None]
    q_even = ((q_even * rstd_q).to(tl.bfloat16).to(tl.float32) * q_scale).to(tl.bfloat16)
    q_odd = ((q_odd * rstd_q).to(tl.bfloat16).to(tl.float32) * q_scale).to(tl.bfloat16)
    if ROPE:
        pair = (tl.arange(0, D // 4)[:, None] * 2 + tl.arange(0, 2)[None, :])[None, :, :]
        c = tl.load(cs + pair)
        s = tl.load(cs + D // 2 + pair)
        a = q_even.to(tl.float32)
        b = q_odd.to(tl.float32)
        q_even = _rope(a, -b, c, s, ROPE_BF16)
        q_odd = _rope(b, a, c, s, ROPE_BF16)
    offs_qo = t * (HQ * D) + tl.arange(0, HQ * D)
    qo = tl.reshape(tl.join(q_even, q_odd), (HQ * D, ))
    if Q_FP8:
        inv = _rcp(tl.load(Q_OUT_SCALE))
        qf = tl.minimum(tl.maximum(qo.to(tl.float32) * inv, -448.0), 448.0)  # float8_e4m3fn max
        tl.store(QO + offs_qo, qf.to(tl.float8e4nv))
    else:
        tl.store(QO + offs_qo, qo.to(tl.bfloat16))

    # k: the norm sums a head as a one-program norm kernel does (one element per lane, a head per wave). RoPE reads
    # each element's pair partner with a second load, so the norm keeps that layout.
    offs_k = tl.arange(0, HK * D)
    k = tl.load(K + t * stride_k + offs_k).to(tl.float32)
    k2 = tl.reshape(k, (HK, D))
    rstd_k = tl.rsqrt(tl.sum(k2 * k2, axis=1) / D + eps)
    kn = tl.reshape(k2 * rstd_k[:, None], (HK * D, )).to(tl.bfloat16)
    if ROPE:
        partner = tl.load(K + t * stride_k + (offs_k ^ 1)).to(tl.float32)
        pn = tl.reshape(tl.reshape(partner, (HK, D)) * rstd_k[:, None], (HK * D, )).to(tl.bfloat16).to(tl.float32)
        first = (offs_k % 2) == 0
        i = (offs_k % D) // 2
        kn = _rope(kn.to(tl.float32), tl.where(first, -pn, pn), tl.load(cs + i), tl.load(cs + D // 2 + i), ROPE_BF16)
    kf = kn.to(tl.float32)
    vf = tl.load(V + t * stride_v + offs_k).to(tl.float32)
    if KV_SCALE:
        kf = kf / tl.load(K_SCALE)
        vf = vf / tl.load(V_SCALE)

    slot = tl.load(SLOT + t).to(tl.int64)
    if HAS_SWA:
        slot = tl.load(SWA_SLOT + slot, mask=slot >= 0, other=-1).to(tl.int64)
    dst = (slot // PAGE) * stride_block + (slot % PAGE) * (HK * D) + offs_k
    keep = (offs_k < HK * D) & (slot >= 0)
    tl.store(KC + dst, kf.to(tl.float8e4nv), mask=keep)
    tl.store(VC + dst, vf.to(tl.float8e4nv), mask=keep)


def qk_norm_rope_kv_cache(q, k, v, k_cache, v_cache, slot_mapping, eps, q_scale, positions, cos_sin_cache, rope_bf16,
                          swa_slot_mapping, k_scale, v_scale, q_out_scale, q_out):
    T = q.shape[0]
    HK, D = k_cache.shape[2], k_cache.shape[3]
    HQ = q.shape[1] // D
    if q_out is None:
        dtype = torch.float8_e4m3fn if q_out_scale is not None else torch.bfloat16
        q_out = torch.empty(T, HQ * D, device=q.device, dtype=dtype)
    if T == 0:
        return q_out
    rope = positions is not None
    dummy = slot_mapping
    _qk_norm_rope_kv_cache_kernel[(T, )](q, k, v, positions if rope else dummy, cos_sin_cache if rope else dummy,
                                         slot_mapping, swa_slot_mapping if swa_slot_mapping is not None else dummy,
                                         k_scale if k_scale is not None else dummy,
                                         v_scale if v_scale is not None else dummy,
                                         q_out_scale if q_out_scale is not None else dummy, q_out, k_cache, v_cache,
                                         q.stride(0), k.stride(0), v.stride(0), cos_sin_cache.stride(0) if rope else 0,
                                         k_cache.stride(0), float(eps), float(q_scale), HQ, HK, D, k_cache.shape[1],
                                         rope, bool(rope_bf16), swa_slot_mapping is not None, k_scale is not None,
                                         q_out_scale is not None, num_warps=4)
    return q_out
