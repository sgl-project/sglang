# SPDX-License-Identifier: Apache-2.0
"""Varlen flash attention on Intel XPU, shaped for vmoba's LSE merge.

flash-attn has no XPU build. sgl-kernel's XPU varlen kernel returns the same
(out, softmax_lse) pair that MixedAttention reduces over, so vmoba needs only
the argument and return shim below.

Inference only: sgl-kernel exposes no varlen backward, and the kernel requires
Battlemage (BMG) or later.
"""

import math
from functools import lru_cache

import torch

# Raise ImportError rather than failing later: vmoba.py imports this module to
# discover the XPU path, so a host without an XPU must fall through to its stub.
if not torch.xpu.is_available():
    raise ImportError("vmoba's XPU attention path requires an available Intel XPU")

from sgl_kernel.flash_attn import flash_attn_varlen_func as _xpu_varlen
from sgl_kernel.flash_attn import is_fa3_supported

# flash-attn's signature is wider than what the XPU kernel can honour next to
# return_softmax_lse, which vmoba always needs: window_size raises kernel-side
# as local masking, and softcap is accepted but silently ignored. Neither can
# be forwarded, so only their no-op defaults are tolerated.
_NO_OP_DEFAULTS = {"window_size": (-1, -1), "softcap": 0.0}


def _reject_unsupported(kwargs: dict) -> None:
    """Raise on any option the XPU path cannot apply.

    Absorbing one would change the attention pattern with no error.
    """
    for name, value in kwargs.items():
        if name not in _NO_OP_DEFAULTS or value != _NO_OP_DEFAULTS[name]:
            raise NotImplementedError(
                f"vmoba on XPU does not support flash-attn option {name!r}"
            )


@lru_cache(maxsize=None)
def _assert_kernel_usable(device: torch.device, dtype: torch.dtype) -> None:
    """Reject builds and devices whose softmax_lse is unusable.

    Some XPU builds never populate softmax_lse for multi-query varlen, so the
    caller silently reads zeros and the LSE merge weights every block wrongly.
    __version__ is 0.1.8 on both sides of that fix, so probe the kernel: with
    zeroed q/k/v every logit is 0 and the logsumexp of n_kv keys is log(n_kv).
    Cached per (device, dtype) because the failure is both build- and
    shape-dependent, and a multi-device run must not inherit device 0's verdict.
    """
    if not is_fa3_supported(device):
        raise RuntimeError(
            f"sgl-kernel's XPU flash attention needs Battlemage (BMG) or later; "
            f"{torch.xpu.get_device_name(device)} is unsupported"
        )
    n_kv, heads, head_dim = 8, 4, 64
    zeros = torch.zeros(n_kv, heads, head_dim, dtype=dtype, device=device)
    cu = torch.tensor([0, n_kv], dtype=torch.int32, device=device)
    lse = _xpu_varlen(
        zeros,
        zeros,
        zeros,
        cu,
        cu,
        n_kv,
        n_kv,
        softmax_scale=1.0,
        causal=False,
        return_softmax_lse=True,
    )[1]
    if not torch.allclose(
        lse, torch.full_like(lse, math.log(n_kv)), atol=1e-2, rtol=1e-2
    ):
        raise RuntimeError(
            "This sgl-kernel build does not populate softmax_lse for multi-query "
            "varlen attention, which vmoba needs; upgrade sgl-kernel-xpu."
        )


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
    **kwargs,
):
    """Returns flash-attn's (out, softmax_lse, S_dmask, rng_state) 4-tuple.

    softmax_lse is [num_heads, total_q], the layout vmoba's merge transposes.
    """
    if dropout_p != 0.0:
        raise NotImplementedError("vmoba on XPU does not support dropout")
    if causal:
        # The kernel refuses softmax_lse alongside causal/local masking, and
        # vmoba's merge always needs the LSE. Every vmoba call site is
        # non-causal, so this only guards future ones.
        raise NotImplementedError(
            "vmoba on XPU cannot combine causal masking with softmax_lse"
        )
    _reject_unsupported(kwargs)
    _assert_kernel_usable(q.device, q.dtype)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    # The XPU kernel reads K/V as densely packed rows and returns wrong values
    # for a larger row stride; vmoba's moba branch passes a slice of a stacked
    # [tokens, 2, heads, dim] buffer. Only the last dim is checked kernel-side,
    # so a non-contiguous K/V cannot be caught downstream.
    out, softmax_lse = _xpu_varlen(
        q,
        k.contiguous(),
        v.contiguous(),
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
    **kwargs,
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
        **kwargs,
    )[0]


def flash_attn_varlen_backward(*args, **kwargs):
    raise NotImplementedError(
        "vmoba on XPU is forward-only; sgl-kernel exposes no varlen backward."
    )
