# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Token layouts of the tensors handed across layer communication boundaries."""

from enum import Enum, auto
from typing import FrozenSet, Mapping, Optional

import msgspec


class TokenAxis(Enum):
    """A parallel dimension across which ranks hold different tokens."""

    ATTN_DP = auto()
    ATTN_CP = auto()
    # Each attention-TP rank holds a slice of its group's tokens.
    ATTN_TP_SCATTER = auto()


class Layout(msgspec.Struct, frozen=True):
    """The token axes a rank's rows are sharded over. Single-rank axes are left
    out, so two layouts are equal exactly when every rank holds the same tokens."""

    sharded: FrozenSet[TokenAxis]

    @classmethod
    def sharded_over(
        cls, *axes: TokenAxis, axis_sizes: Mapping[TokenAxis, int]
    ) -> "Layout":
        return cls(frozenset(axis for axis in axes if axis_sizes[axis] > 1))


class SumGroup(Enum):
    """The group a stage's output is summed over, by name: the handle comes
    from get_parallel() when the sum runs."""

    ATTN_TP = auto()
    TP = auto()
    # The group one all-reduce of a MoE output runs over.
    MOE_OUTPUT = auto()


class StageInput(msgspec.Struct, frozen=True):
    """The rows a stage's consumer needs: sharded over the token axes its
    compute group does not span."""

    layout: Layout
    # Token axes the consumer gathers over itself when its input arrives
    # sharded over them.
    gathers_itself: FrozenSet[TokenAxis] = frozenset()


class StageOutput(msgspec.Struct, frozen=True):
    """What a stage's producer hands the boundary after it, fixed at
    construction: its rows, the group its output is summed over, and when it
    leaves that sum to the boundary instead of completing it.

    What one output actually owes is carried with the value: a plain
    tensor handed across layers is complete, an ``UnreducedOutput`` names what
    is left. A raw compute output entering a boundary is read with this
    declaration and the boundary's decision for the batch."""

    layout: Layout
    # None when there is nothing to sum.
    group: Optional[SumGroup] = None
    # The producer never reduces its output, e.g. a row-parallel projection
    # built with reduce_results=False.
    always_leaves: bool = False
    # Otherwise it leaves the sum only when the boundary publishes the flag for
    # it: fuse_mlp_allreduce, or mlp_reduce_scatter.
    leaves_for_next_layer: bool = False
    leaves_for_reduce_scatter: bool = False
    # Whether it leaves the sum to the attention-DP reduce_scatterv whenever that
    # combine applies, as a MoE block does without any flag.
    leaves_for_reduce_scatterv: bool = False


class DecoderLayerSides(msgspec.Struct, frozen=True):
    """The declarations both sides of a decoder layer's boundaries are chosen
    from."""

    # The rows the layer's input and residual arrive on.
    input_rows: Layout
    attention: StageInput
    attention_output: StageOutput
    ffn: StageInput
    ffn_output: StageOutput
    # The rows the residual is on while the FFN runs.
    ffn_residual_rows: Layout
    # The rows the layer hands on.
    output_rows: Layout
    # The group the layer's input arrives as a partial sum over, if any.
    input_owes: Optional[SumGroup] = None
    # Whether the residual is added into one rank's share of the attention
    # output's sum before that sum completes, instead of after it.
    residual_joins_attention_sum: bool = False


def sequence_parallel_layer_sides(
    *, axis_sizes: Mapping[TokenAxis, int]
) -> DecoderLayerSides:
    """A decoder layer while a LayerNorm SP region is active. Its linears
    all-gather their input and reduce-scatter their output themselves, so the
    layer's rows, its residual and both outputs stay on each TP rank's slice."""
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axis_sizes
    )
    local = Layout.sharded_over(
        TokenAxis.ATTN_DP,
        TokenAxis.ATTN_CP,
        TokenAxis.ATTN_TP_SCATTER,
        axis_sizes=axis_sizes,
    )
    gathers = frozenset({TokenAxis.ATTN_TP_SCATTER})
    return DecoderLayerSides(
        input_rows=local,
        # qkv and gate_up gather the slices into their GEMM.
        attention=StageInput(attention, gathers_itself=gathers),
        # o_proj and down reduce-scatter out of theirs, whether fused or not.
        attention_output=StageOutput(local),
        ffn=StageInput(
            Layout.sharded_over(axis_sizes=axis_sizes), gathers_itself=gathers
        ),
        ffn_output=StageOutput(local),
        ffn_residual_rows=local,
        output_rows=local,
    )


def input_scattered_layer_sides(
    *,
    axis_sizes: Mapping[TokenAxis, int],
    ffn_group: SumGroup,
    hands_on_partial: bool,
) -> DecoderLayerSides:
    """A decoder layer on a batch whose attention input is scattered over
    attention TP. Each layer's input is a partial sum over TP on the full rows:
    the embedding and every FFN before it leave that sum, and a reduce-scatter
    completes it onto each rank's slice, which the attention gathers itself.
    The residual comes back to the full rows inside the attention output's sum."""
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axis_sizes
    )
    return DecoderLayerSides(
        input_rows=attention,
        attention=StageInput(
            attention, gathers_itself=frozenset({TokenAxis.ATTN_TP_SCATTER})
        ),
        attention_output=StageOutput(
            attention, group=SumGroup.ATTN_TP, always_leaves=True
        ),
        ffn=StageInput(Layout.sharded_over(axis_sizes=axis_sizes)),
        # The next layer's input completes the sum; the last layer's FFN does.
        ffn_output=StageOutput(
            attention, group=ffn_group, leaves_for_reduce_scatter=hands_on_partial
        ),
        ffn_residual_rows=attention,
        output_rows=attention,
        input_owes=SumGroup.TP,
        residual_joins_attention_sum=True,
    )


def decoder_layer_sides(
    *,
    axis_sizes: Mapping[TokenAxis, int],
    ffn_on_local_rows: bool,
    previous_on_local_rows: bool,
    is_last_layer: bool,
    attention_gathers_local_rows: bool,
    ffn_group: SumGroup,
    leaves_for_next_layer: bool,
    leaves_for_reduce_scatter: bool,
    leaves_for_reduce_scatterv: bool,
) -> DecoderLayerSides:
    """An attention followed by an FFN, derived from the groups each computes
    over. The FFN runs either on the TP group (a dense MLP, or a MoE not
    dispatched per DP shard) or on this rank's local rows (a MoE dispatched per
    DP shard, which completes its own combine, or a dense MLP on every rank)."""
    # Attention computes over the attention-TP ranks of one DP (and CP) shard.
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axis_sizes
    )
    # Each attention-TP rank's own slice of those rows.
    local = Layout.sharded_over(
        TokenAxis.ATTN_DP,
        TokenAxis.ATTN_CP,
        TokenAxis.ATTN_TP_SCATTER,
        axis_sizes=axis_sizes,
    )
    # The TP group spans every token axis, so an FFN on it needs every row.
    ffn = local if ffn_on_local_rows else Layout.sharded_over(axis_sizes=axis_sizes)
    attention_tp = axis_sizes[TokenAxis.ATTN_TP_SCATTER] > 1
    return DecoderLayerSides(
        input_rows=local if previous_on_local_rows else attention,
        attention=StageInput(
            attention,
            gathers_itself=(
                frozenset({TokenAxis.ATTN_TP_SCATTER})
                if attention_gathers_local_rows
                else frozenset()
            ),
        ),
        # The output projection leaves the attention-TP sum to prepare_mlp.
        attention_output=StageOutput(
            attention,
            group=SumGroup.ATTN_TP if attention_tp else None,
            always_leaves=attention_tp,
        ),
        ffn=StageInput(ffn),
        ffn_output=(
            StageOutput(ffn)
            if ffn_on_local_rows
            else StageOutput(
                ffn,
                group=ffn_group,
                leaves_for_next_layer=leaves_for_next_layer,
                leaves_for_reduce_scatter=leaves_for_reduce_scatter,
                leaves_for_reduce_scatterv=leaves_for_reduce_scatterv,
            )
        ),
        ffn_residual_rows=local if ffn_on_local_rows else attention,
        output_rows=local if ffn_on_local_rows and not is_last_layer else attention,
    )
