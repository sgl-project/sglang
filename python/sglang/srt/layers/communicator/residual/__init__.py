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
"""How a layer writes its stages' outputs into the residual and reads the next
stage's input back."""

from typing import Protocol, Tuple

import torch


class ResidualOps(Protocol):
    """How a layer writes each stage's output into its residual and reads the
    next stage's input from it. The boundary steps complete sums and move
    tokens around these operations. AddAndNorm adds and normalizes; MHC's
    hyper-connection streams implement the same operations (residual.mhc)."""

    # The write-back is a plain add, which one rank may run before the sum it
    # adds into completes: the DP partial order, and a residual that joins the
    # attention output's sum.
    adds_plainly: bool
    # The layer writes its FFN output into the residual itself instead of
    # leaving that to the next layer's input.
    updates_residual_after_ffn: bool

    def enter(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The residual the layer stack starts from, given its input."""

    def read_attention_input(self, residual, norm, quant_format: str) -> Tuple:
        """The attention input and the residual, from a residual that already
        holds the previous layer's output."""

    def update_and_read_attention_input(
        self, hidden_states, residual, norm, quant_format: str, post_residual_addition
    ) -> Tuple:
        """Write the previous layer's output into the residual, then read the
        attention input from it."""

    def update_and_read_ffn_input(self, hidden_states, residual, norm) -> Tuple:
        """Write the attention output into the residual, then read the FFN
        input from it."""

    def update_residual(self, hidden_states, residual) -> torch.Tensor:
        """The residual with the FFN output written into it."""

    def residual_to_attn_tp_shard(self, residual, context) -> torch.Tensor:
        """This attention-TP rank's slice of the residual."""

    def residual_from_attn_tp_shards(self, residual) -> torch.Tensor:
        """The residual gathered from every attention-TP rank's slice."""
