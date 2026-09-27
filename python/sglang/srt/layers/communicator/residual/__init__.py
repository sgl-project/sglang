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
"""How a stage writes its output into the residual (its update) and how a
stage reads its input back from it (its read). The producer of a boundary
declares the update, the consumer the read; the boundary steps complete sums
and move tokens around the two."""

from typing import NamedTuple, Optional, Protocol, Tuple

import torch


class StageUpdate(Protocol):
    """How a stage's output is written into the residual."""

    # A plain add, which one rank may run before the sum it adds into
    # completes (the DP partial order, a residual that joins the attention
    # output's sum), and which a fused add + norm kernel may run.
    adds_plainly: bool
    # The stage writes its output into the residual itself at its end, instead
    # of leaving that to the next stage's read.
    at_producer: bool

    def update(self, hidden_states, residual) -> torch.Tensor:
        """The residual with the output written into it."""

    def residual_to_attn_tp_shard(self, residual, context) -> torch.Tensor:
        """This attention-TP rank's slice of the residual, with whatever the
        update reads along with it."""

    def residual_from_attn_tp_shards(self, residual) -> torch.Tensor:
        """The residual gathered from every attention-TP rank's slice."""


class StageRead(Protocol):
    """How a stage reads its input from the residual."""

    # The input is the residual's norm, in the quantization the call asks for,
    # and the residual is left as it is: what a fused add + norm kernel computes,
    # and what may run on the rows a sum completes onto.
    norms_plainly: bool

    def enter(self, hidden_states) -> torch.Tensor:
        """The residual the layer stack starts from, given its input."""

    def read(
        self,
        residual,
        norm,
        quant_format: str = "",
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """The input and the residual, from a residual that already holds the
        previous stage's output."""

    def update_and_read(
        self,
        update: StageUpdate,
        hidden_states,
        residual,
        norm,
        quant_format: str = "",
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Write the previous stage's output into the residual with its
        producer's ``update``, then read the input."""


class LayerResidual(NamedTuple):
    """The reads and updates a decoder layer declares for its attention and
    its FFN."""

    attention_read: StageRead
    attention_update: StageUpdate
    ffn_read: StageRead
    ffn_update: StageUpdate
