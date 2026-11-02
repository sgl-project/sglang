"""Layer-communication helpers for the Nemotron-H model."""

from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA, MOE, NemotronHConfig
from sglang.srt.layers.layer_boundary import (
    ExitRows,
    ProducerReduction,
    append_stages,
    declare_attn,
    declare_ffn,
)
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NormQuantReadout,
    NormReadout,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.moe.latent_moe_comm import use_latent_moe_comm
from sglang.srt.runtime_context import get_parallel

ATTN_LAYERS = (MAMBA, ATTENTION)


def is_attn_layer(layer_type: str) -> bool:
    return layer_type in ATTN_LAYERS


def uses_latent_moe_comm(config: NemotronHConfig, layer_idx: int) -> bool:
    """Whether a MoE layer exchanges latent rows across EP instead of
    hidden-width rows (SGLANG_OPT_LATENT_MOE_COMM)."""
    return config.hybrid_override_pattern[layer_idx] == MOE and use_latent_moe_comm(
        is_latent_moe=getattr(
            config.get_nemotron_h_config_for_layer(layer_idx), "moe_latent_size", None
        )
        is not None
    )


def _declaration(pattern: str, layer_idx: int, latent_moe_comm: bool):
    if is_attn_layer(pattern[layer_idx]):
        return declare_attn(
            read=NormQuantReadout(reads_before_dp_gather=True),
            reduction=ProducerReduction.EXIT_SCOPED,
            gathers_attn_tp_input=False,
        )
    is_moe = pattern[layer_idx] == MOE
    # A latent-exchange MoE runs on its DP group's rows like an attention-TP
    # dense FFN, owns the exchange across DP ranks, and completes its sum.
    if latent_moe_comm:
        dense_tp_size = get_parallel().attn_tp_size
    else:
        dense_tp_size = None if is_moe else get_parallel().tp_size
    return declare_ffn(
        sparse=is_moe and not latent_moe_comm,
        read=NormReadout(reads_before_dp_gather=True),
        dense_tp_size=dense_tp_size,
        exit_rows=ExitRows.ATTENTION,
        output_complete=latent_moe_comm,
    )


def stage_facts(
    pattern: str, layer_idx: int, *, latent_moe_layers: frozenset[int] = frozenset()
):
    """The one stage each Nemotron-H layer declares, from the pattern and the
    layers that exchange latent MoE rows: the model's shared declaration
    function, which its layers declare with too (see make_layers)."""
    return (_declaration(pattern, layer_idx, layer_idx in latent_moe_layers),)


def make_stage_boundary(
    layer_norm: RMSNorm,
    *,
    pattern: str,
    layer_idx: int,
    latent_moe_comm: bool = False,
):
    from sglang.srt.layers import layernorm_sp

    if get_parallel().attn_cp_size > 1 or layernorm_sp.layernorm_sp_enabled():
        raise NotImplementedError("a Nemotron stage with attention CP or LayerNorm SP")
    (boundary,) = append_stages(
        (_declaration(pattern, layer_idx, latent_moe_comm), layer_norm)
    )
    return boundary
