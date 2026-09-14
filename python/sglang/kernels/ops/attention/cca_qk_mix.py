"""Fused ZAYA1 CCA q/k head-mix + normalize (+ partial-rotary RoPE, + KV store).

One kernel replaces the ~26-launch elementwise tail of the CCA projection --
``_add_grouped_qk_means`` followed by ``_normalize_qk`` in
:class:`CCA <sglang.srt.models.zaya.CCA>`::

    q_out[g,j] = rms(conv_q[g,j] + 0.5*pre_q[g,j] + 0.5*base_k[g])          * sqrt_hd
    k_out[g]   = rms(conv_k[g]   + 0.5*mean_j(pre_q[g,j]) + 0.5*base_k[g])  * sqrt_hd * temp[g]

where ``rms(x) = x * rsqrt(sum(x^2) + eps)`` over the head dim and ``g`` indexes
GQA groups (one k head per group, ``gqa_groups`` q heads inside it).

One program per ``(token, k head)``, holding the whole group as a ``[G, HD]``
tile: the ``G`` q-head RMS sums reduce together along ``axis=1`` and
``mean_j(pre_q)`` along ``axis=0`` of that same tile, so ``pre_q`` is read once
and the group's ``G + 1`` reductions collapse to two.

Rotary
------
``ROT_D > 0`` folds the neox partial rotary in, removing the separate
``sgl_kernel.rotary_embedding`` launch. The shared fused rope path
(``fused_qk_rope_reshape_and_cache``) cannot serve ZAYA1: it asserts
``d_freq in (d // 2, d)``, and ``partial_rotary_factor=0.5`` gives ``d_freq ==
32`` against ``head_dim=128`` -- that kernel has no partial-rotary mode.

The head is loaded as three register tiles -- ``lo = d[0:ROT_D/2]``,
``hi = d[ROT_D/2:ROT_D]``, ``pass = d[ROT_D:HD]`` -- so ``lo`` and ``hi`` sit in
the *same* lane and the neox rotation ``lo' = lo*cos - hi*sin`` /
``hi' = hi*cos + lo*sin`` needs no cross-lane shuffle. Each element is still
loaded once; only the address arithmetic is split.

ORDERING: the RMS sum runs over all ``HD`` dims and the k temperature is applied
*before* the rotation, matching the ``_normalize_qk`` -> ``rotary_emb`` order.

NOT bit-identical to the unfused chain: on ROCm the cos/sin cache is stored in
the model dtype and the separate rotary kernel receives q/k already rounded to
bf16, while this keeps the normalized head in fp32 across the rotation and
rounds once, at the store.

KV store
--------
``HAS_STORE`` also scatters the post-rope ``k`` and the matching ``v`` into the
paged KV buffers at ``out_cache_loc[t]``, so ``RadixAttention`` can be called
with ``save_kv_cache=False`` and the per-layer ``set_kv_buffer`` launch
disappears. Slot resolution mirrors ``rope_cache``'s: ``slot =
out_cache_loc[t]``, then ``slot = full_to_swa[slot]`` on a sliding-window layer
of a hybrid pool, then skip when ``slot < 0`` (the sentinel row batch padding
maps to). A wrong write here corrupts KV silently rather than crashing, so
:func:`store_covered` is deliberately narrower than the kernel could serve.

Follows the structure of ``kda_fused_decode`` (a ``covered()`` predicate gates
supported inputs, everything else falls back to the unfused chain), but is
written in Triton rather than CUDA-JIT so it runs on ROCm.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# The head dim is loaded as one masked block, so it must fit a single Triton
# block. ZAYA1 uses 128; 256 leaves headroom without hurting occupancy.
_MAX_HEAD_DIM = 256

# G bounds register pressure, not just a loop count: a program holds three
# [G, HD] fp32 tiles. Beyond 16 x 256 the torch fallback is the better bet.
_MAX_GQA_GROUPS = 16

_RMS_EPS = 1e-12

_INT_DTYPES = (torch.int32, torch.int64)
_FLOAT_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@triton.jit
def _blend(conv_row, pre_row, half_base_k, off, mask):
    """``conv + 0.5*pre + 0.5*base_k`` in fp32, plus the raw ``pre`` load that the
    k blend reduces over the group. Rows are pre-advanced to the token."""
    pre = tl.load(pre_row + off, mask=mask, other=0.0).to(tl.float32)
    conv = tl.load(conv_row + off, mask=mask, other=0.0).to(tl.float32)
    return conv + 0.5 * pre + half_base_k, pre


@triton.jit
def _kv_slot(loc_ptr, swa_map_ptr, t, HAS_SWA: tl.constexpr):
    """The physical KV slot for token ``t``, or a negative sentinel to skip it.

    ``full_to_swa``'s trailing ``-1`` entry is what makes a padding row map to a
    skip rather than to slot 0.
    """
    slot = tl.load(loc_ptr + t).to(tl.int64)
    if HAS_SWA:
        # Guard the gather: a negative source slot would index before the
        # mapping's base. Costs one select.
        mapped = tl.load(swa_map_ptr + tl.maximum(slot, 0)).to(tl.int64)
        slot = tl.where(slot >= 0, mapped, -1)
    return slot


@triton.jit
def _store_v(value_ptr, v_cache_ptr, t, g, slot, s_v_t, s_v_h, s_vc_t, s_vc_h, off, m):
    """Copy this ``(token, k head)``'s V row into the paged V buffer."""
    v = tl.load(value_ptr + t * s_v_t + g * s_v_h + off, mask=m, other=0.0)
    tl.store(v_cache_ptr + slot * s_vc_t + g * s_vc_h + off, v, mask=m)


@triton.jit
def _cca_qk_mix_kernel(
    conv_qk_ptr,  # [T, latent_q + latent_k] conv output (q segment then k)
    pre_q_ptr,  # [T, latent_q]  raw W_q hidden_states
    base_k_ptr,  # [T, latent_k]  raw W_k hidden_states
    k_scale_ptr,  # [NK] sqrt(head_dim) * per-k-head temperature
    positions_ptr,  # [T] int, or None when ROT_D == 0
    cos_sin_ptr,  # [max_pos, ROT_D] cos half then sin half, or None
    value_ptr,  # [T, NK, HD] V heads, or None when not storing
    k_cache_ptr,  # [rows, NK, HD] NHD paged K buffer, or None
    v_cache_ptr,  # [rows, NK, HD] NHD paged V buffer, or None
    loc_ptr,  # [T] out_cache_loc, or None
    swa_map_ptr,  # [rows+1] full->SWA slot mapping, or None
    q_out_ptr,  # [T, NQ, HD] fp32
    k_out_ptr,  # [T, NK, HD] fp32
    s_conv_t,
    s_pre_t,
    s_base_t,
    s_qout_t,
    s_kout_t,
    s_cos_t,
    s_v_t,
    s_v_h,
    s_kc_t,
    s_kc_h,
    s_vc_t,
    s_vc_h,
    latent_q,
    q_scale,  # sqrt(head_dim)
    eps,
    NK: tl.constexpr,
    G: tl.constexpr,  # gqa_groups: q heads per k head
    HD: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_G: tl.constexpr,  # next_power_of_2(G)
    ROT_D: tl.constexpr,  # rotary_dim, 0 to skip the rotation
    ROT_HALF: tl.constexpr,  # ROT_D // 2
    BLOCK_H: tl.constexpr,  # next_power_of_2(ROT_HALF), 1 when ROT_D == 0
    BLOCK_P: tl.constexpr,  # next_power_of_2(HD - ROT_D), 1 when ROT_D == 0
    HAS_STORE: tl.constexpr = False,
    HAS_SWA: tl.constexpr = False,
):
    pid = tl.program_id(0)
    t = pid // NK
    g = pid % NK

    conv_row = conv_qk_ptr + t * s_conv_t
    pre_row = pre_q_ptr + t * s_pre_t
    base_row = base_k_ptr + t * s_base_t + g * HD
    conv_k_row = conv_row + latent_q + g * HD

    # One [BLOCK_G, ...] tile whose rows are the group's q heads, so the G RMS
    # sums collapse to one axis=1 reduction and mean_j(pre_q) to one axis=0.
    j = tl.arange(0, BLOCK_G)[:, None]
    jmask = j < G
    q_row = (g * G + j) * HD  # [BLOCK_G, 1]

    k_scale = tl.load(k_scale_ptr + g).to(tl.float32)
    inv_g_half = 0.5 / G

    if ROT_D == 0:
        d = tl.arange(0, BLOCK)
        dmask = d < HD

        base_k = tl.load(base_row + d, mask=dmask, other=0.0).to(tl.float32)
        half_base_k = 0.5 * base_k

        off = q_row + d[None, :]
        qmask = jmask & (d[None, :] < HD)
        q, pre_q = _blend(conv_row, pre_row, half_base_k[None, :], off, qmask)
        # Masked lanes carry 0, so they add nothing to their row's sum, and the
        # row itself is never stored.
        inv = tl.rsqrt(tl.sum(q * q, axis=1) + eps) * q_scale
        tl.store(q_out_ptr + t * s_qout_t + off, q * inv[:, None], mask=qmask)

        conv_k = tl.load(conv_k_row + d, mask=dmask, other=0.0).to(tl.float32)
        k = conv_k + inv_g_half * tl.sum(pre_q, axis=0) + half_base_k
        inv_k = tl.rsqrt(tl.sum(k * k, axis=0) + eps) * k_scale
        k = k * inv_k
        tl.store(k_out_ptr + t * s_kout_t + g * HD + d, k, mask=dmask)

        if HAS_STORE:
            slot = _kv_slot(loc_ptr, swa_map_ptr, t, HAS_SWA)
            if slot >= 0:
                tl.store(k_cache_ptr + slot * s_kc_t + g * s_kc_h + d, k, mask=dmask)
                _store_v(
                    value_ptr,
                    v_cache_ptr,
                    t,
                    g,
                    slot,
                    s_v_t,
                    s_v_h,
                    s_vc_t,
                    s_vc_h,
                    d,
                    dmask,
                )
    else:
        # Three tiles: the two rotated halves and the pass-through tail. lo and
        # hi share a lane index, which is what keeps the rotation shuffle-free.
        i = tl.arange(0, BLOCK_H)
        imask = i < ROT_HALF
        p = tl.arange(0, BLOCK_P)
        pmask = p < (HD - ROT_D)

        hb_lo = 0.5 * tl.load(base_row + i, mask=imask, other=0.0).to(tl.float32)
        hb_hi = 0.5 * tl.load(base_row + ROT_HALF + i, mask=imask, other=0.0).to(
            tl.float32
        )
        hb_pa = 0.5 * tl.load(base_row + ROT_D + p, mask=pmask, other=0.0).to(
            tl.float32
        )

        off_lo = q_row + i[None, :]
        off_hi = q_row + ROT_HALF + i[None, :]
        off_pa = q_row + ROT_D + p[None, :]
        qm_h = jmask & imask[None, :]
        qm_p = jmask & pmask[None, :]

        q_lo, pre_lo = _blend(conv_row, pre_row, hb_lo[None, :], off_lo, qm_h)
        q_hi, pre_hi = _blend(conv_row, pre_row, hb_hi[None, :], off_hi, qm_h)
        q_pa, pre_pa = _blend(conv_row, pre_row, hb_pa[None, :], off_pa, qm_p)

        # RMS over the WHOLE head (all three tiles) and before the rotation --
        # the order the unfused chain uses.
        ssq = (
            tl.sum(q_lo * q_lo, axis=1)
            + tl.sum(q_hi * q_hi, axis=1)
            + tl.sum(q_pa * q_pa, axis=1)
        )
        inv = tl.rsqrt(ssq + eps) * q_scale
        q_lo = q_lo * inv[:, None]
        q_hi = q_hi * inv[:, None]
        q_pa = q_pa * inv[:, None]

        pos = tl.load(positions_ptr + t).to(tl.int64)
        cos_row = cos_sin_ptr + pos * s_cos_t
        cos = tl.load(cos_row + i, mask=imask, other=0.0).to(tl.float32)
        sin = tl.load(cos_row + ROT_HALF + i, mask=imask, other=0.0).to(tl.float32)

        q_out_row = q_out_ptr + t * s_qout_t
        cos2 = cos[None, :]
        sin2 = sin[None, :]
        tl.store(q_out_row + off_lo, q_lo * cos2 - q_hi * sin2, mask=qm_h)
        tl.store(q_out_row + off_hi, q_hi * cos2 + q_lo * sin2, mask=qm_h)
        tl.store(q_out_row + off_pa, q_pa, mask=qm_p)

        ck_lo = tl.load(conv_k_row + i, mask=imask, other=0.0).to(tl.float32)
        ck_hi = tl.load(conv_k_row + ROT_HALF + i, mask=imask, other=0.0)
        ck_hi = ck_hi.to(tl.float32)
        ck_pa = tl.load(conv_k_row + ROT_D + p, mask=pmask, other=0.0).to(tl.float32)
        k_lo = ck_lo + inv_g_half * tl.sum(pre_lo, axis=0) + hb_lo
        k_hi = ck_hi + inv_g_half * tl.sum(pre_hi, axis=0) + hb_hi
        k_pa = ck_pa + inv_g_half * tl.sum(pre_pa, axis=0) + hb_pa

        ssq_k = (
            tl.sum(k_lo * k_lo, axis=0)
            + tl.sum(k_hi * k_hi, axis=0)
            + tl.sum(k_pa * k_pa, axis=0)
        )
        inv_k = tl.rsqrt(ssq_k + eps) * k_scale
        k_lo = k_lo * inv_k
        k_hi = k_hi * inv_k
        k_pa = k_pa * inv_k
        k_lo_r = k_lo * cos - k_hi * sin
        k_hi_r = k_hi * cos + k_lo * sin

        k_out_row = k_out_ptr + t * s_kout_t + g * HD
        tl.store(k_out_row + i, k_lo_r, mask=imask)
        tl.store(k_out_row + ROT_HALF + i, k_hi_r, mask=imask)
        tl.store(k_out_row + ROT_D + p, k_pa, mask=pmask)

        if HAS_STORE:
            slot = _kv_slot(loc_ptr, swa_map_ptr, t, HAS_SWA)
            if slot >= 0:
                # The SAME fp32 expression that went to k_out is rounded into the
                # pool, so the fused store is bit-identical to the set_kv_buffer
                # copy it replaces (both round the fp32 result once).
                kc_row = k_cache_ptr + slot * s_kc_t + g * s_kc_h
                tl.store(kc_row + i, k_lo_r, mask=imask)
                tl.store(kc_row + ROT_HALF + i, k_hi_r, mask=imask)
                tl.store(kc_row + ROT_D + p, k_pa, mask=pmask)
                dv = tl.arange(0, BLOCK)
                _store_v(
                    value_ptr,
                    v_cache_ptr,
                    t,
                    g,
                    slot,
                    s_v_t,
                    s_v_h,
                    s_vc_t,
                    s_vc_h,
                    dv,
                    dv < HD,
                )


def covered(
    conv_qk: torch.Tensor,
    pre_q: torch.Tensor,
    base_k: torch.Tensor,
    k_scale: Optional[torch.Tensor],
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
) -> bool:
    """Whether the fused kernel can serve these inputs.

    ``k_scale`` is the folded ``sqrt(head_dim) * temperature`` vector, absent
    until weights load. Says nothing about the rotary or KV-store fusions --
    :func:`rope_covered` and :func:`store_covered` are strictly additional gates,
    so the mix can fuse while either of those falls back.
    """
    if k_scale is None:
        return False
    if not (conv_qk.is_cuda and pre_q.is_cuda and base_k.is_cuda):
        return False
    if head_dim > _MAX_HEAD_DIM or head_dim <= 0:
        return False
    if num_k_heads <= 0 or num_q_heads % num_k_heads != 0:
        return False
    if num_q_heads // num_k_heads > _MAX_GQA_GROUPS:
        return False
    if conv_qk.ndim != 2 or pre_q.ndim != 2 or base_k.ndim != 2:
        return False
    if not (pre_q.shape[0] == base_k.shape[0] == conv_qk.shape[0]):
        return False
    if conv_qk.shape[-1] != (num_q_heads + num_k_heads) * head_dim:
        return False
    if pre_q.shape[-1] != num_q_heads * head_dim:
        return False
    if base_k.shape[-1] != num_k_heads * head_dim:
        return False
    if not (
        conv_qk.stride(-1) == 1 and pre_q.stride(-1) == 1 and base_k.stride(-1) == 1
    ):
        return False
    if k_scale.numel() != num_k_heads or not k_scale.is_contiguous():
        return False
    return all(t.dtype in _FLOAT_DTYPES for t in (conv_qk, pre_q, base_k))


def _same_device(tensor: torch.Tensor, device: torch.device) -> bool:
    """Device equality that reads an un-indexed reference as "any device of this
    type". ``torch.device("cuda") != torch.device("cuda:0")``, and a hand-written
    reference device would otherwise decline every gate silently."""
    got = tensor.device
    if got.type != device.type:
        return False
    return device.index is None or got.index == device.index


def rope_geometry_decline_reason(
    rotary_dim: int, head_dim: int, is_neox_style: bool = True
) -> Optional[str]:
    """Why the head shape is not one the three-tile split can express, or ``None``.

    Requires neox layout (GPT-J interleaves the rotated pair intra-lane, the
    cross-lane case this kernel avoids), ``rotary_dim == head_dim // 2``, and an
    even ``rotary_dim // 2``; anything else is rejected rather than guessed at.
    Split out from :func:`rope_decline_reason` so the geometry can be pinned
    without a GPU.
    """
    if not is_neox_style:
        return "gptj layout (the rotated pair is intra-lane)"
    if rotary_dim <= 0 or head_dim <= 0:
        return f"degenerate dims rotary_dim={rotary_dim} head_dim={head_dim}"
    if rotary_dim != head_dim // 2:
        return f"rotary_dim {rotary_dim} != head_dim//2 {head_dim // 2}"
    if (rotary_dim // 2) % 2 != 0:
        return f"odd rotary half {rotary_dim // 2}"
    return None


def rope_geometry_covered(
    rotary_dim: int, head_dim: int, is_neox_style: bool = True
) -> bool:
    """Whether the head shape fits the three-tile split. See the reason variant."""
    return rope_geometry_decline_reason(rotary_dim, head_dim, is_neox_style) is None


def rope_decline_reason(
    positions: Optional[torch.Tensor],
    cos_sin_cache: Optional[torch.Tensor],
    rotary_dim: int,
    *,
    head_dim: int,
    num_tokens: int,
    is_neox_style: bool = True,
    device: Optional[torch.device] = None,
) -> Optional[str]:
    """Why the neox partial rotary cannot fold into the mix kernel, or ``None``.

    Everything it rejects still gets the fused mix plus a separate rotary launch.
    The reason string, rather than a bare bool, is what keeps such a decline
    visible: it feeds the once-per-outcome fusion log and the tests' assertion
    messages, and is built only on decline.
    """
    if positions is None or cos_sin_cache is None:
        return "no rotary offered"
    geometry = rope_geometry_decline_reason(rotary_dim, head_dim, is_neox_style)
    if geometry is not None:
        return geometry
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[-1] != rotary_dim:
        return (
            f"cos_sin_cache {tuple(cos_sin_cache.shape)} is not "
            f"[max_pos, {rotary_dim}]"
        )
    if cos_sin_cache.stride(-1) != 1:
        return "cos_sin_cache innermost stride != 1"
    if cos_sin_cache.dtype not in _FLOAT_DTYPES:
        return f"cos_sin_cache dtype {cos_sin_cache.dtype}"
    if positions.ndim != 1 or positions.numel() != num_tokens:
        return f"positions {tuple(positions.shape)} is not 1-D of {num_tokens}"
    if positions.dtype not in _INT_DTYPES:
        return f"positions dtype {positions.dtype}"
    if not positions.is_contiguous():
        return "positions not contiguous"
    if not (positions.is_cuda and cos_sin_cache.is_cuda):
        return "positions / cos_sin_cache not on an accelerator"
    if device is not None:
        for name, t in (("positions", positions), ("cos_sin_cache", cos_sin_cache)):
            if not _same_device(t, device):
                return f"{name} on {t.device}, inputs on {device}"
    return None


def rope_covered(
    positions: Optional[torch.Tensor],
    cos_sin_cache: Optional[torch.Tensor],
    rotary_dim: int,
    *,
    head_dim: int,
    num_tokens: int,
    is_neox_style: bool = True,
    device: Optional[torch.device] = None,
) -> bool:
    """Whether the neox partial rotary can fold in. See the reason variant."""
    return (
        rope_decline_reason(
            positions,
            cos_sin_cache,
            rotary_dim,
            head_dim=head_dim,
            num_tokens=num_tokens,
            is_neox_style=is_neox_style,
            device=device,
        )
        is None
    )


def store_decline_reason(
    value: Optional[torch.Tensor],
    k_cache: Optional[torch.Tensor],
    v_cache: Optional[torch.Tensor],
    out_cache_loc: Optional[torch.Tensor],
    full_to_swa: Optional[torch.Tensor],
    *,
    num_k_heads: int,
    head_dim: int,
    num_tokens: int,
    out_dtype: torch.dtype,
    device: torch.device,
) -> Optional[str]:
    """Why the KV scatter cannot fold into the mix kernel, or ``None``.

    A wrong write here does not crash, it corrupts KV, so every check is a hard
    reject rather than a fixup:

    * **3-D ``[rows, heads, dim]`` K/V buffers only** -- the plain NHD layout,
      where the write target is a flat slot row and needs no page arithmetic.
      Requiring 3-D rejects the 5-D SHUFFLE, 4-D HND and page-major strided
      layouts by construction. The caller pins the other half of that invariant
      by checking ``rows == pool.size + pool.page_size``, which makes the flat
      slot index correct for any page size.
    * **matching dtypes** -- an fp8/fp4 pool stores under a different
      ``store_dtype`` and needs scales this kernel does not apply.
    * unit innermost strides; the head axis may be strided (a per-rank slice of
      the replicated V projection is), hence ``s_v_h`` is passed, not assumed.
    * ``full_to_swa`` is indexed by FULL-pool slot id, not by a row of
      ``k_cache``, so its length is the caller's invariant to check.
    """
    for name, t in (
        ("value", value),
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("out_cache_loc", out_cache_loc),
    ):
        if t is None:
            return f"{name} not supplied"
    if k_cache.ndim != 3 or v_cache.ndim != 3:
        return (
            f"kv buffers are not 3-D NHD (k={tuple(k_cache.shape)}, "
            f"v={tuple(v_cache.shape)}); the 5-D SHUFFLE, 4-D HND and "
            "page-major layouts are not served"
        )
    for name, buf in (("k_cache", k_cache), ("v_cache", v_cache)):
        if buf.shape[1] != num_k_heads or buf.shape[2] != head_dim:
            return f"{name} {tuple(buf.shape)} is not [rows, {num_k_heads}, {head_dim}]"
        if buf.dtype != out_dtype:
            return f"{name} dtype {buf.dtype} != out_dtype {out_dtype}"
        if buf.stride(-1) != 1:
            return f"{name} innermost stride != 1"
    if value.ndim != 3 or tuple(value.shape) != (num_tokens, num_k_heads, head_dim):
        return (
            f"value {tuple(value.shape)} is not "
            f"[{num_tokens}, {num_k_heads}, {head_dim}]"
        )
    if value.dtype != out_dtype:
        return f"value dtype {value.dtype} != out_dtype {out_dtype}"
    if value.stride(-1) != 1:
        return "value innermost stride != 1"
    if out_cache_loc.ndim != 1 or out_cache_loc.numel() != num_tokens:
        return f"out_cache_loc {tuple(out_cache_loc.shape)} is not 1-D of {num_tokens}"
    if out_cache_loc.dtype not in _INT_DTYPES:
        return f"out_cache_loc dtype {out_cache_loc.dtype}"
    if not out_cache_loc.is_contiguous():
        return "out_cache_loc not contiguous"
    if full_to_swa is not None:
        if full_to_swa.ndim != 1 or full_to_swa.dtype != torch.int64:
            return (
                f"full_to_swa {tuple(full_to_swa.shape)}/{full_to_swa.dtype} is "
                "not a 1-D int64 mapping"
            )
        if not full_to_swa.is_contiguous() or full_to_swa.numel() == 0:
            return "full_to_swa empty or not contiguous"
    # ``device`` is explicit rather than inferred: ``covered()`` already checked
    # for an accelerator, and passing it lets these branches be tested on CPU.
    for name, t in (
        ("value", value),
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("out_cache_loc", out_cache_loc),
        ("full_to_swa", full_to_swa),
    ):
        if t is not None and not _same_device(t, device):
            return f"{name} on {t.device}, inputs on {device}"
    return None


def store_covered(
    value: Optional[torch.Tensor],
    k_cache: Optional[torch.Tensor],
    v_cache: Optional[torch.Tensor],
    out_cache_loc: Optional[torch.Tensor],
    full_to_swa: Optional[torch.Tensor],
    *,
    num_k_heads: int,
    head_dim: int,
    num_tokens: int,
    out_dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """Whether the KV scatter can fold in. See the reason variant."""
    return (
        store_decline_reason(
            value,
            k_cache,
            v_cache,
            out_cache_loc,
            full_to_swa,
            num_k_heads=num_k_heads,
            head_dim=head_dim,
            num_tokens=num_tokens,
            out_dtype=out_dtype,
            device=device,
        )
        is None
    )


def cca_qk_mix(
    conv_qk: torch.Tensor,
    pre_q: torch.Tensor,
    base_k: torch.Tensor,
    k_scale: torch.Tensor,
    *,
    num_q_heads: int,
    num_k_heads: int,
    head_dim: int,
    q_scale: float,
    eps: float = _RMS_EPS,
    out_dtype: torch.dtype = torch.float32,
    positions: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
    rotary_dim: int = 0,
    value: Optional[torch.Tensor] = None,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    out_cache_loc: Optional[torch.Tensor] = None,
    full_to_swa: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(q, k)`` as ``[T, heads, head_dim]`` in ``out_dtype``.

    Accumulation is always fp32; ``out_dtype`` only picks the store precision.
    ``positions`` / ``cos_sin_cache`` / ``rotary_dim`` also apply the neox partial
    rotary; ``value`` / ``k_cache`` / ``v_cache`` / ``out_cache_loc`` also scatter
    k and v into the paged buffers, with ``full_to_swa`` adding the hybrid-pool
    slot indirection. Caller must have checked :func:`covered`, plus
    :func:`rope_covered` / :func:`store_covered` for the extra arguments.
    """
    num_tokens = conv_qk.shape[0]
    q_out = torch.empty(
        (num_tokens, num_q_heads, head_dim), dtype=out_dtype, device=conv_qk.device
    )
    k_out = torch.empty(
        (num_tokens, num_k_heads, head_dim), dtype=out_dtype, device=conv_qk.device
    )
    if num_tokens == 0:
        return q_out, k_out

    rot_d = int(rotary_dim) if positions is not None else 0
    if rot_d:
        rot_half = rot_d // 2
        block_h = triton.next_power_of_2(rot_half)
        block_p = triton.next_power_of_2(head_dim - rot_d)
        s_cos_t = cos_sin_cache.stride(0)
    else:
        rot_half, block_h, block_p, s_cos_t = 0, 1, 1, 0

    has_store = k_cache is not None
    if has_store:
        s_v_t, s_v_h = value.stride(0), value.stride(1)
        s_kc_t, s_kc_h = k_cache.stride(0), k_cache.stride(1)
        s_vc_t, s_vc_h = v_cache.stride(0), v_cache.stride(1)
    else:
        s_v_t = s_v_h = s_kc_t = s_kc_h = s_vc_t = s_vc_h = 0

    _cca_qk_mix_kernel[(num_tokens * num_k_heads,)](
        conv_qk,
        pre_q,
        base_k,
        k_scale,
        positions if rot_d else None,
        cos_sin_cache if rot_d else None,
        value if has_store else None,
        k_cache,
        v_cache,
        out_cache_loc if has_store else None,
        full_to_swa if has_store else None,
        q_out,
        k_out,
        conv_qk.stride(0),
        pre_q.stride(0),
        base_k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        s_cos_t,
        s_v_t,
        s_v_h,
        s_kc_t,
        s_kc_h,
        s_vc_t,
        s_vc_h,
        num_q_heads * head_dim,
        float(q_scale),
        float(eps),
        NK=num_k_heads,
        G=num_q_heads // num_k_heads,
        HD=head_dim,
        BLOCK=triton.next_power_of_2(head_dim),
        BLOCK_G=triton.next_power_of_2(num_q_heads // num_k_heads),
        ROT_D=rot_d,
        ROT_HALF=rot_half,
        BLOCK_H=block_h,
        BLOCK_P=block_p,
        HAS_STORE=has_store,
        HAS_SWA=(has_store and full_to_swa is not None),
        # One warp keeps each tl.sum inside a single ROCm wavefront; at four the
        # 128-element head dim leaves lanes idle and the reduction goes via LDS.
        # Measured on ZAYA1's head_dim=128; re-sweep if the tile shape changes.
        num_warps=1,
    )
    return q_out, k_out
