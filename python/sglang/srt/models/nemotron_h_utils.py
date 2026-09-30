"""Layer-communication helpers for the Nemotron-H model."""

from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA, MOE
from sglang.srt.layers.layer_boundary import (
    ExitRows,
    ProducerReduction,
    declare_attn,
    declare_ffn,
    make_stages,
)
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NormQuantReadout,
    NormReadout,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.runtime_context import get_parallel

ATTN_LAYERS = (MAMBA, ATTENTION)


def is_attn_layer(layer_type: str) -> bool:
    return layer_type in ATTN_LAYERS


def _declaration(pattern: str, layer_idx: int):
    if is_attn_layer(pattern[layer_idx]):
        return declare_attn(
            read=NormQuantReadout(reads_before_dp_gather=True),
            reduction=ProducerReduction.EXIT_SCOPED,
            gathers_attn_tp_input=False,
        )
    return declare_ffn(
        sparse=pattern[layer_idx] == MOE,
        read=NormReadout(reads_before_dp_gather=True),
        dense_tp_size=get_parallel().tp_size if pattern[layer_idx] != MOE else None,
        exit_rows=ExitRows.ATTENTION,
    )


def make_stage_boundary(layer_norm: RMSNorm, *, pattern: str, layer_idx: int):
    from sglang.srt.layers import layernorm_sp

    if get_parallel().attn_cp_size > 1 or layernorm_sp.layernorm_sp_enabled():
        raise NotImplementedError("a Nemotron stage with attention CP or LayerNorm SP")
    previous = _declaration(pattern, layer_idx - 1) if layer_idx > 0 else None
    (boundary,) = make_stages(
        (
            _declaration(pattern, layer_idx),
            layer_norm,
        ),
        previous=previous,
        following=_declaration(pattern, layer_idx + 1)
        if layer_idx + 1 < len(pattern)
        else None,
        terminal=layer_idx == len(pattern) - 1,
    )
    return boundary
