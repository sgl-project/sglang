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
"""Gated hyper-connections expressed as stage reads and residual updates."""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.layer_boundary.adapters.attention import (
    _redistribute_to_attn_tp_shards,
)
from sglang.srt.layers.layer_boundary.ops import gather_attention_tp
from sglang.srt.layers.layer_boundary.residual import LayerResidual

if TYPE_CHECKING:
    from sglang.srt.layers.hyperconnection import GatedResidual


class _BranchState:
    def __init__(self, connection: "GatedResidual"):
        self.connection = connection
        self.normalized: Optional[torch.Tensor] = None

    def mix(self, residual):
        hidden_states, (residual, self.normalized) = self.connection.mix(residual)
        return hidden_states, residual

    def combine(self, hidden_states, residual):
        if self.normalized is None:
            raise RuntimeError("a gated residual update requires its stage's read")
        output = self.connection.combine(hidden_states, (residual, self.normalized))
        self.normalized = None
        return output

    def to_attn_tp_shard(self, residual):
        if self.normalized is None:
            raise RuntimeError("a gated residual shard requires its stage's read")
        self.normalized = _redistribute_to_attn_tp_shards(self.normalized)
        return _redistribute_to_attn_tp_shards(residual)

    def from_attn_tp_shards(self, residual):
        if self.normalized is None:
            raise RuntimeError("a gated residual gather requires its stage's read")
        # The tuple path allocates separate outputs. Gathering both through the
        # shared DP scratch buffer would overwrite the first gathered tensor.
        residual, self.normalized = gather_attention_tp(
            (residual, self.normalized), None
        )
        return residual


class _GatedRead:
    norms_plainly = False
    before_gather = True

    def __init__(self, state: _BranchState, *, enters_stack: bool):
        self.state = state
        self.enters_stack = enters_stack

    def enter(self, hidden_states):
        if not self.enters_stack:
            raise RuntimeError("a gated FFN does not enter the layer stack")
        connection = self.state.connection
        width = hidden_states.shape[-1]
        if width == connection.hc_count * connection.hidden_size:
            return hidden_states
        if width != connection.hidden_size:
            raise ValueError(f"unexpected gated residual input width: {width}")
        return torch.cat([hidden_states] * connection.hc_count, dim=-1)

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        if norm is not None or quant_format:
            raise NotImplementedError("gated residual mixing owns its normalization")
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        return self.state.mix(residual)

    def update_and_read(
        self,
        update,
        hidden_states,
        residual,
        norm,
        quant_format="",
        post_residual_addition=None,
    ):
        return self.read(
            update.update(hidden_states, residual),
            norm,
            quant_format,
            post_residual_addition,
        )


class _GatedUpdate:
    adds_plainly = False
    can_defer_across_layers = False
    # MoE CP gathers only the compute input. The boundary returns the output
    # to the local residual rows before applying this update.
    supports_moe_cp_gather = True

    def __init__(self, state: _BranchState, *, at_producer: bool):
        self.state = state
        self.at_producer = at_producer

    def update(self, hidden_states, residual):
        return self.state.combine(hidden_states, residual)

    def residual_to_attn_tp_shard(self, residual):
        return self.state.to_attn_tp_shard(residual)

    def residual_from_attn_tp_shards(self, residual):
        return self.state.from_attn_tp_shards(residual)


class GatedResidualState:
    """Adapt attention/FFN GatedResidual modules to the layer boundaries.

    The batch's residual stream retains the unnormalized hyper-input tensor.
    Each branch retains the normalized input needed by its next combine, and
    moves that auxiliary tensor whenever its residual changes token rows.
    FFN updates run at the producer so no layer carries another layer's state.
    The owning model continues to register and load the connection modules.
    """

    def __init__(self, attention: "GatedResidual", ffn: "GatedResidual"):
        attention_state = _BranchState(attention)
        ffn_state = _BranchState(ffn)
        self._residual = LayerResidual(
            attention_read=_GatedRead(attention_state, enters_stack=True),
            attention_update=_GatedUpdate(attention_state, at_producer=False),
            ffn_read=_GatedRead(ffn_state, enters_stack=False),
            ffn_update=_GatedUpdate(ffn_state, at_producer=True),
        )

    def layer_residual(self) -> LayerResidual:
        return self._residual
