"""Ascend MoE dispatch state for the no-A2A path.

The NPU no-A2A path uses the community ``StandardDispatcher``: token routing
by expert is not a communication concern, so the Ascend work (the routing
permute with its fused activation quantization, and the weighted unpermute)
lives in the MoE runner's ``standard -> ascend`` permute hooks instead of in a
dedicated dispatcher, calling the shared routing entry points in
``hardware_backend.npu.moe.routing``.

This module owns the pieces of state those hooks need, written by the
quantization method once the weights are loaded:

- the dtype the routing op must hand to the grouped matmul
  (``bf16`` / ``int8`` / ``mxfp8``), see :func:`set_dispatch_dtype`;
- whether the finalize output needs a TP all-gather, see
  :func:`set_finalize_all_gather` (GGUF keeps full, non-intermediate-sharded
  expert weights, so the hidden dim must be gathered).

The resolved ``(quant_mode, all-gather)`` tuple is cached on the layer so a
steady-state forward pays no resolution cost.
"""

from __future__ import annotations

from typing import Tuple

from sglang.srt.hardware_backend.npu.moe.routing import (
    INT8_QUANT_MODE,
    MXFP8_QUANT_MODE,
)
from sglang.srt.layers.moe.utils import DispatcherOutputDtype

_DISPATCH_DTYPE_ATTR = "moe_dispatch_dtype"
_FINALIZE_ALL_GATHER_ATTR = "moe_dispatch_finalize_all_gather"
_ROUTING_CACHE_ATTR = "_ascend_routing_cache"

# Quant modes the no-A2A routing can produce (see
# ``hardware_backend.npu.moe.routing``); anything else is a programming error
# in the quant method that set the dtype rather than a user misconfiguration.
_QUANT_MODE = {
    DispatcherOutputDtype.BF16: -1,
    DispatcherOutputDtype.INT8: INT8_QUANT_MODE,
    DispatcherOutputDtype.MXFP8: MXFP8_QUANT_MODE,
}

# ``expert_tokens_num_type=1`` makes the routing emit per-expert token counts
# (the grouped matmul reads ``group_list_type == 1`` as "these are counts, not
# a prefix sum"), so the value sits next to the contract it describes.
GROUP_LIST_TYPE = 1

_SUPPORTED_DTYPES = tuple(_QUANT_MODE)


def set_dispatch_dtype(layer, dtype) -> None:
    """Record the dtype the routing op must produce for this layer."""
    setattr(layer, _DISPATCH_DTYPE_ATTR, DispatcherOutputDtype(dtype).value)


def get_dispatch_dtype(layer) -> DispatcherOutputDtype:
    """The dtype the routing op must produce; BF16 when nothing set it."""
    raw = getattr(layer, _DISPATCH_DTYPE_ATTR, None)
    if raw is None:
        return DispatcherOutputDtype.BF16
    try:
        return DispatcherOutputDtype(raw)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported Ascend dispatch dtype {raw!r}; expected one of "
            f"{[d.value for d in _SUPPORTED_DTYPES]}."
        ) from exc


def set_finalize_all_gather(layer, enabled: bool = True) -> None:
    """Mark that the finalize output needs a TP all-gather (GGUF)."""
    setattr(layer, _FINALIZE_ALL_GATHER_ATTR, bool(enabled))


def quant_mode_of(dtype) -> int:
    """Map a dispatch dtype to the routing op's ``quant_mode``."""
    resolved = DispatcherOutputDtype(dtype)
    if resolved not in _QUANT_MODE:
        raise ValueError(
            f"Ascend MoE routing cannot produce {resolved.value} activations; "
            f"supported: {[d.value for d in _SUPPORTED_DTYPES]}."
        )
    return _QUANT_MODE[resolved]


def get_routing(layer) -> Tuple[int, bool, int]:
    """Return ``(quant_mode, finalize_tp_all_gather, group_list_type)``.

    Call after the weights are loaded: the quantization method writes the
    dtype in ``process_weights_after_loading``, before the first forward.
    """
    dtype = get_dispatch_dtype(layer)
    all_gather = bool(getattr(layer, _FINALIZE_ALL_GATHER_ATTR, False))
    cache = getattr(layer, _ROUTING_CACHE_ATTR, None)
    if cache is not None and cache[0] == (dtype, all_gather):
        return cache[1], cache[2], cache[3]

    resolved = (quant_mode_of(dtype), all_gather, GROUP_LIST_TYPE)
    setattr(layer, _ROUTING_CACHE_ATTR, ((dtype, all_gather), *resolved))
    return resolved
