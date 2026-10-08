# SPDX-License-Identifier: Apache-2.0
"""FlashAttention (varlen) dispatch for YuE2.

Both YuE2 attention patterns are ragged varlen and map directly onto SGLang's
shared ``flash_attn_varlen_func`` (FA3 by default, FA2/FA4-selectable) — the AR
decode passes its per-row effective length through ``seqused_k`` instead of a
custom mask.

``SGLANG_YUE2_FA3=1`` (default) uses the shared op; otherwise it falls back to
PyTorch's native variable-length FlashAttention. Imports stay lazy so importing
this never requires ``flash_attn_3``/``sgl_kernel``.
"""
from __future__ import annotations

import os

import torch

_VARLEN = None
_LOADED = False


def _load_varlen():
    """Return SGLang's shared varlen attention, or None when disabled/missing."""
    global _VARLEN, _LOADED
    if not _LOADED:
        _LOADED = True
        if os.environ.get("SGLANG_YUE2_FA3", "1") == "1":
            try:
                from sglang.kernels.ops.attention.flash_attention import (
                    flash_attn_varlen_func,
                )

                _VARLEN = flash_attn_varlen_func
            except Exception:
                _VARLEN = False
        else:
            _VARLEN = False
    return _VARLEN or None


def fa3_available() -> bool:
    """True when the shared (FA3) varlen attention path is active."""
    return _load_varlen() is not None


def varlen_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, max_q, max_k,
                     seqused_k=None):
    """Ragged, non-causal attention (FA3 when available, else native ATen).

    ``q`` is ``[sum(q_len), HQ, HD]`` and ``k``/``v`` are ``[sum(k_len), HKV, HD]``;
    ``seqused_k`` (optional) caps each row's effective key length. Returns
    ``[sum(q_len), HQ, HD]``.
    """
    varlen = _load_varlen()
    if varlen is not None:
        return varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_q, max_k,
                      seqused_k=seqused_k, causal=False, ver=3)
    return torch.ops.aten._flash_attention_forward(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_q, max_k, 0.0, False, False,
        seqused_k=seqused_k)[0]
