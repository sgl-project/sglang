# SPDX-License-Identifier: Apache-2.0
"""Fused paged-decode attention + sigmoid output gate (gfx950, aiter backend).

Qwen3-Next's full-attention layers produce a per-head gate alongside Q and
multiply the attention output by its sigmoid before ``o_proj``. On the AMD path
that decode boundary is three kernel launches:

    KV write  ->  paged decode attention  ->  out * sigmoid(gate)

aiter's ``paged_attention_output_gate_group_fp8_quant`` folds the gate into the
split-K merge the attention already performs, turning those three launches into
two. This module is the in-tree caller.

Scope and non-scope
-------------------
* **Decode only.** Prefill / extend / idle keep the stock path untouched: the
  op is a paged-decode kernel and has nothing to say about ragged prefill.
* **gfx950 + the aiter attention backend only.** Everything else -- including
  every CUDA/NVIDIA deployment -- is inert: :func:`fused_gated_decode_attention`
  returns ``None`` before it touches an aiter symbol, and the caller runs the
  code it runs today. **The CUDA path is not modified by this file.**
* **Fusion only.** The op can additionally emit a group-128 FP8 tensor for the
  following GEMM to consume, but wiring that into ``o_proj`` is a separate and
  larger change (it has to reach into the linear layer's input quantization).
  We therefore ask for ``quant_dtype=None`` and hand ``o_proj`` the same BF16
  tensor it receives today.

Availability
------------
The op is newer than the aiter versions SGLang currently pins, so it is
imported lazily and its absence is a fallback, not an error. The op also
self-describes what it can handle via
``paged_attention_output_gate_supported``; anything it declines takes the stock
path. Both the bind and the decline are logged once, because a fused path that
silently never engages is indistinguishable from the stock one.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch

from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

_is_hip = is_hip()

# The op's FP8 epilogue is deliberately not requested: see "Scope" above.
_QUANT_DTYPE: Optional[torch.dtype] = None

# ``None`` = not probed yet, ``False`` = unavailable in this aiter build.
_OPS: Optional[object] = None

_LOGGED: set = set()


def _log_once(tag: str, message: str) -> None:
    if tag in _LOGGED:
        return
    _LOGGED.add(tag)
    logger.info(message)


def _load_ops() -> Optional[Tuple[Callable, Callable]]:
    """Resolve the aiter op once. Returns ``None`` when it is unavailable."""
    global _OPS
    if _OPS is not None:
        return _OPS or None
    if not _is_hip:
        _OPS = False
        return None
    try:
        from aiter.ops.triton.attention.paged_attention_output_gate import (
            paged_attention_output_gate_group_fp8_quant,
            paged_attention_output_gate_supported,
        )
    except ImportError as exc:
        # Older aiter. Not a problem: the stock three-launch path is correct.
        _log_once(
            "import",
            f"aiter paged_attention_output_gate unavailable ({exc}); "
            "Qwen3-Next full-attention decode keeps the unfused path",
        )
        _OPS = False
        return None
    _OPS = (
        paged_attention_output_gate_group_fp8_quant,
        paged_attention_output_gate_supported,
    )
    return _OPS


def _aiter_backend() -> Optional[object]:
    """The :class:`AiterAttnBackend` serving full attention, or ``None``.

    The active backend lives on the per-forward context, not on ForwardBatch.
    Qwen3-Next runs under ``HybridLinearAttnBackend``, whose full-attention
    half is the one that owns the paged KV pool and the CSR decode metadata.
    """
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        has_forward_context,
    )

    if not has_forward_context():
        return None
    backend = get_attn_backend()
    backend = getattr(backend, "full_attn_backend", backend)
    return backend if isinstance(backend, AiterAttnBackend) else None


def fused_gated_decode_attention(
    layer,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    forward_batch,
) -> Optional[torch.Tensor]:
    """``sigmoid(gate) * attention(q, k, v)`` in one kernel, or ``None``.

    ``None`` means "this call is not eligible" and the caller must run its
    existing path. That is the only failure mode: an unsupported shape, an old
    aiter, a non-gfx950 device and a non-decode batch all return ``None``
    rather than raising.

    On the fused path this writes the KV cache itself (via
    :meth:`AiterAttnBackend.save_kv_cache`), because ``RadixAttention.forward``
    -- which normally does it -- is not called at all.

    Args:
        layer: the ``RadixAttention`` module of this full-attention layer.
        q, k, v: as they would be passed to ``layer(q, k, v, forward_batch)``.
        gate: ``[T, H * head_dim]`` bf16 pre-sigmoid gate.
        forward_batch: the current :class:`ForwardBatch`.
    """
    if not _is_hip or gate is None:
        return None
    if not forward_batch.forward_mode.is_decode():
        return None

    ops = _load_ops()
    if ops is None:
        return None
    fused_op, supported = ops

    backend = _aiter_backend()
    if backend is None:
        # Not --attention-backend aiter, or no forward context. Logged because
        # this is the one decline with no visible cause at the call site.
        _log_once(
            "backend",
            "paged_attention_output_gate: no aiter attention backend on this "
            "forward; keeping the unfused decode path",
        )
        return None
    if backend.forward_metadata is None:
        return None
    # The op indexes a CSR slot list, which is what the aiter backend builds at
    # page_size 1. A larger page size means kv_indices holds pages, not slots.
    if backend.page_size != 1 or backend.use_mla:
        _log_once(
            "layout",
            f"paged_attention_output_gate: page_size={backend.page_size} "
            f"use_mla={backend.use_mla}; keeping the unfused decode path",
        )
        return None
    metadata = backend.forward_metadata
    if metadata.kv_indptr is None or metadata.kv_indices is None:
        return None

    pool = backend.token_to_kv_pool
    key_cache = pool.get_key_buffer(layer.layer_id)
    value_cache = pool.get_value_buffer(layer.layer_id)
    query = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)

    ok, reason = supported(query, key_cache, value_cache, gate, _QUANT_DTYPE)
    if not ok:
        _log_once(
            "unsupported",
            f"paged_attention_output_gate declined ({reason}); "
            "keeping the unfused decode path",
        )
        return None

    k_scale = layer.k_scale if layer.k_scale is not None else backend.k_scale
    v_scale = layer.v_scale if layer.v_scale is not None else backend.v_scale
    if (k_scale is None) != (v_scale is None):
        _log_once(
            "scales",
            "paged_attention_output_gate: only one of k_scale/v_scale is "
            "loaded; keeping the unfused decode path",
        )
        return None

    _log_once(
        "engaged",
        "paged_attention_output_gate engaged for Qwen3-Next full-attention "
        f"decode (heads={query.shape[1]}, head_dim={query.shape[2]}, "
        f"max_context={backend.max_context_len})",
    )

    # 1. What RadixAttention.forward would have done for us.
    backend.save_kv_cache(layer, forward_batch, k, v)

    # 2. Attention + sigmoid gate, fused.
    #    max_context is the deployment's real context limit, not a bucket: the
    #    op uses it to pick the kernel body (>32768 selects the long-context
    #    one) and to size the launch. It never clamps the live length, which
    #    always comes from kv_indptr, so this is a performance input only.
    gated, _quantized, _scales = fused_op(
        query.contiguous(),
        key_cache,
        value_cache,
        metadata.kv_indptr,
        metadata.kv_indices,
        gate.contiguous(),
        scale=layer.scaling,
        max_context=int(backend.max_context_len),
        k_scale=k_scale,
        v_scale=v_scale,
        quant_dtype=_QUANT_DTYPE,
    )
    return gated
