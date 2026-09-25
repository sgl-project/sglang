"""Layer-communication helpers for the Nemotron-H model."""

from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.moe.utils import get_moe_a2a_backend

ATTN_LAYERS = (MAMBA, ATTENTION)


def is_attn_layer(layer_type: str) -> bool:
    return layer_type in ATTN_LAYERS


def feeds_mlp_layer(pattern: str, layer_idx: int) -> bool:
    next_idx = layer_idx + 1
    return next_idx < len(pattern) and not is_attn_layer(pattern[next_idx])


def takes_attention_partial(pattern: str, layer_idx: int) -> bool:
    """Whether this layer is an FFN stage right after a mixer, whose attention
    partial sum it completes itself."""
    return (
        layer_idx > 0
        and is_attn_layer(pattern[layer_idx - 1])
        and (feeds_mlp_layer(pattern, layer_idx - 1))
    )


def _build_layer_scatter_modes(
    is_sparse: bool = False, is_last_layer: bool = False
) -> LayerScatterModes:
    scatter_mlp = is_sparse and not get_moe_a2a_backend().is_none()
    mlp_mode = ScatterMode.SCATTERED if scatter_mlp else ScatterMode.FULL
    middle_residual_mode = (
        ScatterMode.SCATTERED if scatter_mlp else ScatterMode.TP_ATTN_FULL
    )
    return LayerScatterModes(
        layer_input_mode=ScatterMode.TP_ATTN_FULL,
        attn_mode=ScatterMode.TP_ATTN_FULL,
        mlp_mode=mlp_mode,
        middle_residual_mode=middle_residual_mode,
        layer_output_mode=ScatterMode.TP_ATTN_FULL,
        is_layer_sparse=is_sparse,
        is_last_layer=is_last_layer,
    )


def make_layer_communicator(
    layer_norm: RMSNorm,
    *,
    for_attn: bool,
    allow_reduce_scatter: bool = False,
    is_sparse: bool = False,
    is_last_layer: bool = False,
    next_takes_attention_partial: bool = False,
    previous_leaves_attention_partial: bool = False,
    allow_deferred_ffn_reduction: bool = True,
) -> LayerCommunicator:
    """The communicator of one stage: a mixer (Mamba / attention) normalizes its
    input with ``layer_norm``, an FFN stage normalizes its own."""
    return LayerCommunicator(
        layer_scatter_modes=_build_layer_scatter_modes(is_sparse, is_last_layer),
        input_layernorm=layer_norm if for_attn else None,
        post_attention_layernorm=None if for_attn else layer_norm,
        # With attention TP > 1, the default gather adds the residual to one
        # rank's partial in bf16 before the cross-rank sum.
        force_layernorm_before_dp_gather=True,
        allow_reduce_scatter=allow_reduce_scatter,
        allow_deferred_ffn_reduction=allow_deferred_ffn_reduction,
        standalone_ffn=not for_attn,
        next_takes_attention_partial=next_takes_attention_partial,
        previous_leaves_attention_partial=previous_leaves_attention_partial,
    )
