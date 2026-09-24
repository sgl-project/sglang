"""Layer-communication helpers for the Nemotron-H model."""

from torch import nn

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


def _build_layer_scatter_modes(is_sparse: bool = False) -> LayerScatterModes:
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
    )


def make_layer_communicator(
    layer_norm: RMSNorm,
    *,
    for_attn: bool,
    allow_reduce_scatter: bool = False,
    is_sparse: bool = False,
    is_last_layer: bool = False,
) -> LayerCommunicator:
    return LayerCommunicator(
        layer_scatter_modes=_build_layer_scatter_modes(is_sparse),
        input_layernorm=layer_norm if for_attn else nn.Identity(),
        post_attention_layernorm=nn.Identity() if for_attn else layer_norm,
        # With attention TP > 1, the default gather adds the residual to one
        # rank's partial in bf16 before the cross-rank sum.
        force_layernorm_before_dp_gather=True,
        allow_reduce_scatter=allow_reduce_scatter,
        is_last_layer=is_last_layer,
    )
