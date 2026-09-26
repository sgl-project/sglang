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

    # The rows the layer takes and hands on; the residual stays on them.
    layer_rows: Layout
    attention: StageInput
    attention_output: StageOutput
    ffn: StageInput
    ffn_output: StageOutput


def dense_decoder_layer_sides(
    *,
    axis_sizes: Mapping[TokenAxis, int],
    leaves_for_next_layer: bool,
    leaves_for_reduce_scatter: bool,
) -> DecoderLayerSides:
    """An attention followed by a dense MLP on the TP group, derived from the
    groups each computes over."""
    # Attention computes over the attention-TP ranks of one DP (and CP) shard.
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axis_sizes
    )
    # The TP group spans every token axis, so its MLP needs every row.
    ffn = Layout.sharded_over(axis_sizes=axis_sizes)
    attention_tp = axis_sizes[TokenAxis.ATTN_TP_SCATTER] > 1
    return DecoderLayerSides(
        layer_rows=attention,
        attention=StageInput(attention),
        # The output projection leaves the attention-TP sum to prepare_mlp.
        attention_output=StageOutput(
            attention,
            group=SumGroup.ATTN_TP if attention_tp else None,
            always_leaves=attention_tp,
        ),
        ffn=StageInput(ffn),
        ffn_output=StageOutput(
            ffn,
            group=SumGroup.TP,
            leaves_for_next_layer=leaves_for_next_layer,
            leaves_for_reduce_scatter=leaves_for_reduce_scatter,
            leaves_for_reduce_scatterv=leaves_for_reduce_scatter,
        ),
    )
