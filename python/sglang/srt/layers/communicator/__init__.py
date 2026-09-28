# Copyright 2023-2024 SGLang Team
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
"""Communication between a decoder layer's stages."""

from sglang.srt.layers.communicator.adapters.attention import (
    AttentionInputs,
    get_attn_tp_context,
)
from sglang.srt.layers.communicator.boundary import (
    BoundarySteps,
    DecoderLayerSides,
    EdgeDecl,
    FusedMlpInput,
    LayerStage,
    StageDecl,
    StageEntry,
    StageInput,
    StageKind,
    StageOutput,
    decoder_layer_edges,
    decoder_layer_sides,
    input_scattered_layer_sides,
    make_boundary,
    make_output_boundary,
    sequence_parallel_layer_sides,
    stage_edges,
    tbo_split_moves,
)
from sglang.srt.layers.communicator.layer import (
    FfnCompletion,
    FfnExit,
    FfnExitFusion,
    LayerCommunicator,
    LayerFacts,
    MHCLayerCommunicator,
    MixerExit,
)
from sglang.srt.layers.communicator.layout import (
    CommunicateContext,
    Layout,
    SumGroup,
    TokenAxis,
    enable_moe_dense_fully_dp,
    moe_cp_gathers_sparse_moe_input,
    sparse_moe_gathers_over_moe_cp,
    token_axis_sizes,
)
from sglang.srt.layers.communicator.ops import (
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    layer_input_buffer,
    move_rows,
    tp_reduce_scatter,
)
from sglang.srt.layers.communicator.output import (
    HandoffOutput,
    UnreducedOutput,
    reduce_output,
)
from sglang.srt.layers.communicator.residual import LayerResidual
from sglang.srt.layers.communicator.residual.add_norm import (
    ADD,
    FUSE_ALLREDUCE_MAX_BATCH_SIZE,
    NORM_QUANT_READ,
    NORM_READ,
    PLAIN_RESIDUAL,
)
from sglang.srt.layers.communicator.residual.mhc import (
    MHCState,
)

__all__ = [
    "ADD",
    "AttentionInputs",
    "BoundarySteps",
    "CommunicateContext",
    "CommunicateSimpleFn",
    "CommunicateSummableTensorPairFn",
    "DecoderLayerSides",
    "EdgeDecl",
    "FUSE_ALLREDUCE_MAX_BATCH_SIZE",
    "FfnCompletion",
    "FfnExit",
    "FfnExitFusion",
    "FusedMlpInput",
    "HandoffOutput",
    "LayerCommunicator",
    "LayerResidual",
    "LayerFacts",
    "LayerStage",
    "Layout",
    "MHCLayerCommunicator",
    "MHCState",
    "MixerExit",
    "NORM_QUANT_READ",
    "NORM_READ",
    "PLAIN_RESIDUAL",
    "StageDecl",
    "StageEntry",
    "StageInput",
    "StageKind",
    "StageOutput",
    "SumGroup",
    "TokenAxis",
    "UnreducedOutput",
    "decoder_layer_edges",
    "decoder_layer_sides",
    "enable_moe_dense_fully_dp",
    "get_attn_tp_context",
    "input_scattered_layer_sides",
    "layer_input_buffer",
    "make_boundary",
    "make_output_boundary",
    "moe_cp_gathers_sparse_moe_input",
    "move_rows",
    "reduce_output",
    "sequence_parallel_layer_sides",
    "sparse_moe_gathers_over_moe_cp",
    "stage_edges",
    "tbo_split_moves",
    "token_axis_sizes",
    "tp_reduce_scatter",
]
