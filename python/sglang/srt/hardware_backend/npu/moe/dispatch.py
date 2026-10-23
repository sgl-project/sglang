"""Ascend MoE dispatch state and routing kernels for the no-A2A path.

The NPU no-A2A path uses the community ``StandardDispatcher``: token routing
by expert is not a communication concern, so the Ascend work (the routing
permute with its fused activation quantization, and the weighted unpermute)
lives in the MoE runner's ``standard -> ascend`` permute hooks instead of in a
dedicated dispatcher.

This module owns the two pieces of state those hooks need, both written by the
quantization method once the weights are loaded:

- the dtype the routing op must hand to the grouped matmul
  (``bf16`` / ``int8`` / ``mxfp8``), see :func:`set_dispatch_dtype`;
- whether the finalize output needs a TP all-gather, see
  :func:`set_finalize_all_gather` (GGUF keeps full, non-intermediate-sharded
  expert weights, so the hidden dim must be gathered).

The kernels are cached on the layer per (dtype, all-gather) so a steady-state
forward pays no construction cost.
"""

from __future__ import annotations

from typing import Tuple

from sglang.srt.hardware_backend.npu.moe.finalize_routing import (
    AllGatherFinalizeRoutingWrapper,
    BaseFinalizeRouting,
    NPUFinalizeRouting,
)
from sglang.srt.hardware_backend.npu.moe.init_routing import (
    MXFP8_QUANT_MODE,
    BaseInitRouting,
    NPUMoEInitRouting_v2,
)
from sglang.srt.layers.moe.utils import DispatcherOutputDtype
from sglang.srt.runtime_context import get_parallel

_DISPATCH_DTYPE_ATTR = "moe_dispatch_dtype"
_FINALIZE_ALL_GATHER_ATTR = "moe_dispatch_finalize_all_gather"
_ROUTING_CACHE_ATTR = "_ascend_routing_cache"

# ``npu_moe_init_routing_v2``'s ``quant_mode`` argument: -1 keeps the hidden
# states as they are, 1 emits int8 + per-token scale, 3 emits e4m3 + e8m0
# block scales (MXFP8) while fusing the activation quantization into routing.
_QUANT_MODE = {
    DispatcherOutputDtype.BF16: -1,
    DispatcherOutputDtype.INT8: 1,
    DispatcherOutputDtype.MXFP8: MXFP8_QUANT_MODE,
}

# ``expert_tokens_num_type=1`` makes the routing emit per-expert token counts
# (the grouped matmul reads ``group_list_type == 1`` as "these are counts, not
# a prefix sum"), so the value sits next to the contract it describes.
GROUP_LIST_TYPE = 1

# Values ``NPUMoEInitRouting_v2`` accepts; anything else is a programming error
# in the quant method that set the dtype rather than a user misconfiguration.
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
    """Map a dispatch dtype to ``npu_moe_init_routing_v2``'s ``quant_mode``."""
    resolved = DispatcherOutputDtype(dtype)
    if resolved not in _QUANT_MODE:
        raise ValueError(
            f"Ascend MoE routing cannot produce {resolved.value} activations; "
            f"supported: {[d.value for d in _SUPPORTED_DTYPES]}."
        )
    return _QUANT_MODE[resolved]


def get_routing(layer) -> Tuple[BaseInitRouting, BaseFinalizeRouting, int]:
    """Return ``(init, finalize, group_list_type)`` for this layer.

    Call it after the weights are loaded: the quantization method writes the
    dtype in ``process_weights_after_loading``, and the first forward follows.
    """
    dtype = get_dispatch_dtype(layer)
    all_gather = bool(getattr(layer, _FINALIZE_ALL_GATHER_ATTR, False))
    cache = getattr(layer, _ROUTING_CACHE_ATTR, None)
    if cache is not None and cache[0] == (dtype, all_gather):
        return cache[1], cache[2], cache[3]

    init = NPUMoEInitRouting_v2(quant_mode=quant_mode_of(dtype))
    finalize: BaseFinalizeRouting = NPUFinalizeRouting(drop_pad_mode=2)
    if all_gather and get_parallel().tp_size > 1:
        finalize = AllGatherFinalizeRoutingWrapper(finalize, dim=-1)

    cache = ((dtype, all_gather), init, finalize, GROUP_LIST_TYPE)
    setattr(layer, _ROUTING_CACHE_ATTR, cache)
    return init, finalize, GROUP_LIST_TYPE
