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
    bind_entry,
    bind_exit,
    tbo_split_moves,
)
from sglang.srt.layers.layer_boundary.construction import (
    BatchVariant,
)
from sglang.srt.layers.layer_boundary.contracts import (
    EdgeContract,
    EntryPath,
    ExitRows,
    FfnInputFusion,
    InputContract,
    OutputContract,
    ProducerReduction,
    ReadoutFusion,
    StageKind,
    StagePath,
)
from sglang.srt.layers.layer_boundary.exit import MixerExit
from sglang.srt.layers.layer_boundary.factories import (
    append_stages,
    declare_attn,
    declare_ffn,
    layer_stack,
)
from sglang.srt.layers.layer_boundary.layout import (
    Layout,
    SumGroup,
    TokenAxis,
    batch_gathers_over_moe_cp,
    is_dense_ffn_fully_dp,
)
from sglang.srt.layers.layer_boundary.ops import (
    move_rows,
    tp_reduce_scatter,
)
from sglang.srt.layers.layer_boundary.output import (
    DeferredFinalize,
    UnreducedOutput,
    complete_owed,
)
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    FUSE_ALLREDUCE_MAX_BATCH_SIZE,
    NORM_QUANT_READOUT,
    PLAIN_ADD,
    PLAIN_RESIDUAL_OPS,
)
from sglang.srt.layers.layer_boundary.residual.gated import (
    GatedResidualState,
)
from sglang.srt.layers.layer_boundary.residual.ihc import (
    IHCState,
)
from sglang.srt.layers.layer_boundary.residual.mhc import (
    MHCState,
)

__all__ = [
    "declare_attn",
    "declare_ffn",
    "append_stages",
    "layer_stack",
    "PLAIN_ADD",
    "AttentionInputs",
    "StagePath",
    "EdgeContract",
    "ExitRows",
    "ProducerReduction",
    "FUSE_ALLREDUCE_MAX_BATCH_SIZE",
    "FfnInputFusion",
    "ReadoutFusion",
    "DeferredFinalize",
    "GatedResidualState",
    "IHCState",
    "Layout",
    "MHCState",
    "MixerExit",
    "NORM_QUANT_READOUT",
    "PLAIN_RESIDUAL_OPS",
    "EntryPath",
    "InputContract",
    "StageKind",
    "OutputContract",
    "SumGroup",
    "TokenAxis",
    "UnreducedOutput",
    "is_dense_ffn_fully_dp",
    "get_attn_tp_context",
    "bind_entry",
    "bind_exit",
    "batch_gathers_over_moe_cp",
    "move_rows",
    "complete_owed",
    "tbo_split_moves",
    "tp_reduce_scatter",
]
