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
"""Stage boundaries: layout transport, residual updates, reads, and fusion."""

from sglang.srt.layers.layer_boundary.adapters.attention import (
    AttentionInputs,
    get_attn_tp_context,
)
from sglang.srt.layers.layer_boundary.boundary import (
    make_boundary,
    make_output_boundary,
    tbo_split_moves,
)
from sglang.srt.layers.layer_boundary.construction import (
    BatchVariant,
    StageEdges,
)
from sglang.srt.layers.layer_boundary.contracts import (
    EdgeDecl,
    FusedMlpInput,
    HandoffRows,
    ProducerReduction,
    StageDecl,
    StageEntry,
    StageInput,
    StageKind,
    StageOutput,
    StageSteps,
)
from sglang.srt.layers.layer_boundary.exit import FfnCompletion, FfnExit, MixerExit
from sglang.srt.layers.layer_boundary.factories import (
    declare_attn,
    declare_ffn,
    make_attn_stage,
    make_ffn_stage,
    make_stages,
)
from sglang.srt.layers.layer_boundary.layout import (
    Layout,
    SumGroup,
    TokenAxis,
    enable_moe_dense_fully_dp,
    moe_cp_gathers_sparse_moe_input,
    sparse_moe_gathers_over_moe_cp,
    token_axis_sizes,
)
from sglang.srt.layers.layer_boundary.ops import (
    move_rows,
    tp_reduce_scatter,
)
from sglang.srt.layers.layer_boundary.output import (
    HandoffOutput,
    UnreducedOutput,
    reduce_output,
)
from sglang.srt.layers.layer_boundary.residual import LayerResidual
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    ADD,
    FUSE_ALLREDUCE_MAX_BATCH_SIZE,
    NORM_QUANT_READ,
    NORM_READ,
    PLAIN_RESIDUAL,
)
from sglang.srt.layers.layer_boundary.residual.mhc import (
    MHCState,
)

__all__ = [
    "declare_attn",
    "declare_ffn",
    "make_attn_stage",
    "make_ffn_stage",
    "make_stages",
    "ADD",
    "AttentionInputs",
    "StageSteps",
    "EdgeDecl",
    "HandoffRows",
    "ProducerReduction",
    "FUSE_ALLREDUCE_MAX_BATCH_SIZE",
    "FfnCompletion",
    "FfnExit",
    "FusedMlpInput",
    "HandoffOutput",
    "LayerResidual",
    "Layout",
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
    "enable_moe_dense_fully_dp",
    "get_attn_tp_context",
    "make_boundary",
    "make_output_boundary",
    "moe_cp_gathers_sparse_moe_input",
    "move_rows",
    "reduce_output",
    "sparse_moe_gathers_over_moe_cp",
    "tbo_split_moves",
    "token_axis_sizes",
    "tp_reduce_scatter",
]
