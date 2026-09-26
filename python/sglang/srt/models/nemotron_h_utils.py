"""Layer-communication helpers for the Nemotron-H model."""

from typing import Optional

from sglang.srt.configs.nemotron_h import ATTENTION, MAMBA, MOE
from sglang.srt.layers.boundary_layout import (
    Layout,
    StageDecl,
    StageInput,
    StageOutput,
    SumGroup,
    TokenAxis,
    stage_edges,
)
from sglang.srt.layers.communicator import (
    InputRead,
    LayerCommunicator,
    LayerScatterModes,
    LayerStage,
    ScatterMode,
    token_axis_sizes,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.moe.utils import get_moe_a2a_backend

ATTN_LAYERS = (MAMBA, ATTENTION)


def is_attn_layer(layer_type: str) -> bool:
    return layer_type in ATTN_LAYERS


def _stage_kind(pattern: str, layer_idx: int) -> Optional[InputRead]:
    """How the stage at ``layer_idx`` reads its input: a Mamba or attention
    mixer like an attention, an MLP or MoE like an FFN; None past either end."""
    if not 0 <= layer_idx < len(pattern):
        return None
    return InputRead.ATTENTION if is_attn_layer(pattern[layer_idx]) else InputRead.FFN


def _stage_decl(pattern: str, layer_idx: int) -> StageDecl:
    """What the stage at ``layer_idx`` declares. A mixer computes on the
    attention's rows and leaves its attention-TP sum to an FFN stage after it,
    or, when a fused kernel takes it, to a mixer after it. An FFN computes on
    the TP group's rows, a MoE dispatched by an a2a backend on this rank's own,
    and may leave its sum to a mixer after it."""
    axis_sizes = token_axis_sizes()
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axis_sizes
    )
    following = _stage_kind(pattern, layer_idx + 1)
    if is_attn_layer(pattern[layer_idx]):
        owes = axis_sizes[TokenAxis.ATTN_TP_SCATTER] > 1
        return StageDecl(
            StageInput(attention),
            StageOutput(
                attention,
                group=SumGroup.ATTN_TP if owes else None,
                always_leaves=owes and following is InputRead.FFN,
                leaves_for_next_layer=owes and following is InputRead.ATTENTION,
            ),
        )
    sparse = pattern[layer_idx] == MOE
    if sparse and not get_moe_a2a_backend().is_none():
        local = Layout.sharded_over(
            TokenAxis.ATTN_DP,
            TokenAxis.ATTN_CP,
            TokenAxis.ATTN_TP_SCATTER,
            axis_sizes=axis_sizes,
        )
        return StageDecl(StageInput(local), StageOutput(local))
    full = Layout.sharded_over(axis_sizes=axis_sizes)
    return StageDecl(
        StageInput(full),
        StageOutput(
            full,
            group=SumGroup.MOE_OUTPUT if sparse else SumGroup.TP,
            leaves_for_next_layer=following is InputRead.ATTENTION,
            leaves_for_reduce_scatter=True,
            leaves_for_reduce_scatterv=True,
        ),
    )


def layer_stage(pattern: str, layer_idx: int) -> LayerStage:
    """The layer's stage and its two boundaries, from its own declaration and
    the previous stage's."""
    rows = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=token_axis_sizes()
    )
    return LayerStage(
        reads=_stage_kind(pattern, layer_idx),
        edges=stage_edges(
            previous=(
                _stage_decl(pattern, layer_idx - 1).output if layer_idx > 0 else None
            ),
            stage=_stage_decl(pattern, layer_idx),
            rows=rows,
        ),
        enters_stack=layer_idx == 0,
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
    layer_norm: RMSNorm, *, pattern: str, layer_idx: int
) -> LayerCommunicator:
    """The communicator of a layer that is one stage: only its own norm, and
    boundaries built from the stages next to it in the pattern."""
    stage = layer_stage(pattern, layer_idx)
    for_attn = stage.reads is InputRead.ATTENTION
    return LayerCommunicator(
        layer_scatter_modes=_build_layer_scatter_modes(
            pattern[layer_idx] == MOE, is_last_layer=layer_idx == len(pattern) - 1
        ),
        input_layernorm=layer_norm if for_attn else None,
        post_attention_layernorm=None if for_attn else layer_norm,
        # With attention TP > 1, the default gather adds the residual to one
        # rank's partial in bf16 before the cross-rank sum.
        force_layernorm_before_dp_gather=True,
        allow_reduce_scatter=not for_attn,
        stage=stage,
    )
