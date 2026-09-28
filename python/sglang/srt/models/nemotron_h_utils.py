"""Layer-communication helpers for the Nemotron-H model."""

from typing import Optional

from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA, MOE
from sglang.srt.layers.communicator import (
    StageKind,
    declare_attn,
    declare_ffn,
    make_stages,
)
from sglang.srt.layers.layernorm import RMSNorm

ATTN_LAYERS = (MAMBA, ATTENTION)


def is_attn_layer(layer_type: str) -> bool:
    return layer_type in ATTN_LAYERS


def _stage_kind(pattern: str, layer_idx: int) -> Optional[StageKind]:
    """Which stage of a decoder layer the stage at ``layer_idx`` stands for: a
    Mamba or attention mixer the attention, an MLP or MoE the FFN; None past
    either end."""
    if not 0 <= layer_idx < len(pattern):
        return None
    return StageKind.ATTENTION if is_attn_layer(pattern[layer_idx]) else StageKind.FFN


def _declaration(pattern: str, layer_idx: int):
    following = _stage_kind(pattern, layer_idx + 1)
    if is_attn_layer(pattern[layer_idx]):
        return declare_attn(
            mixer_exit=True,
            next_kind=following,
            ordinary_only=True,
        )
    return declare_ffn(
        sparse=pattern[layer_idx] == MOE,
        ordinary_only=True,
        return_to_attention=True,
    )


def make_stage_boundary(layer_norm: RMSNorm, *, pattern: str, layer_idx: int):
    from sglang.srt.layers import layernorm_sp
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().attn_cp_size > 1 or layernorm_sp.layernorm_sp_enabled():
        raise NotImplementedError("a Nemotron stage with attention CP or LayerNorm SP")
    previous = _declaration(pattern, layer_idx - 1) if layer_idx > 0 else None
    (boundary,) = make_stages(
        (
            _declaration(pattern, layer_idx),
            layer_norm,
            {"force_layernorm_before_dp_gather": True},
        ),
        previous=previous,
        terminal=layer_idx == len(pattern) - 1,
    )
    return boundary
