# SPDX-License-Identifier: Apache-2.0
"""Varlen flash attention on Intel XPU, shaped for vmoba's LSE merge.

flash-attn has no XPU build. sgl-kernel's XPU varlen kernel returns the same
(out, softmax_lse) pair that MixedAttention reduces over, so vmoba needs only
the argument and return shim below.

Inference only: sgl-kernel exposes no varlen backward. The kernel itself
rejects pre-Battlemage devices on every call, so this module adds no check.
"""

import torch

# Raise ImportError rather than failing later: vmoba.py imports this module to
# discover the XPU path, so a host without an XPU must fall through to its stub.
if not torch.xpu.is_available():
    raise ImportError("vmoba's XPU attention path requires an available Intel XPU")

from sgl_kernel.flash_attn import flash_attn_varlen_func as _xpu_varlen


def flash_attn_varlen_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=False,
    dropout_p=0.0,
):
    """Returns flash-attn's (out, softmax_lse, S_dmask, rng_state) 4-tuple.

    softmax_lse is [num_heads, total_q], the layout vmoba's merge transposes.
    """
    if dropout_p != 0.0:
        raise NotImplementedError("vmoba on XPU does not support dropout")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # K/V arrive as slices of a stacked [tokens, 2, heads, dim] buffer. The
    # kernel honours their row stride since sgl-kernel-xpu#471, so they are
    # passed through; copying them cost ~100 MB each at Wan 1.3B shapes.
    out, softmax_lse = _xpu_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        softmax_scale=softmax_scale,
        causal=causal,
        return_softmax_lse=True,
    )[:2]
    return out, softmax_lse, None, None


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=False,
    dropout_p=0.0,
):
    return flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        dropout_p=dropout_p,
    )[0]


def flash_attn_varlen_backward(*args, **kwargs):
    raise NotImplementedError(
        "vmoba on XPU is forward-only; sgl-kernel exposes no varlen backward."
    )
