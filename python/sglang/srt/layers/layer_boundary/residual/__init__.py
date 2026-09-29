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
    """Producer-owned operation writing a contribution into the residual.

    Fields:
        adds_plainly: Whether update is plain addition, permitting compatible
            add+norm fusion and adding the residual on one rank before a sum.
        at_producer: Whether the producer writes the residual at its exit rather
            than leaving the operation for the next prepare.
        can_defer_across_layers: Whether parameters/state remain valid after
            leaving the producer layer, including any offload or state reuse.

    The shard conversion methods must move any update-associated state together
    with the residual. Nonlinear updates must not claim adds_plainly.
    """

    # A plain add, which one rank may run before the sum it adds into
    # completes (the DP partial order, a residual that joins the attention
    # output's sum), and which a fused add + norm kernel may run.
    adds_plainly: bool
    # The stage writes its output into the residual itself at its end, instead
    # of leaving that to the next stage's read.
    at_producer: bool
    # Its parameters and state remain valid after leaving the producer layer.
    # Stateful or offloaded implementations must not opt in without that guarantee.
    can_defer_across_layers: bool

    def update(self, hidden_states, residual) -> torch.Tensor:
        """Write hidden_states, the producer contribution, into residual.

        Implementations define their residual representation and return the updated
        tensor. Boundary ordering may put a plain add before an eligible sum.
        This operation does not perform the consumer's normalization.
        """

    def residual_to_attn_tp_shard(self, residual) -> torch.Tensor:
        """This attention-TP rank's slice of the residual, with whatever the
        update reads along with it."""

    def residual_from_attn_tp_shards(self, residual) -> torch.Tensor:
        """The residual gathered from every attention-TP rank's slice."""


class StageRead(Protocol):
    """Consumer-owned operation deriving compute input from the residual.

    Fields:
        norms_plainly: Read is normalization/optional quantization without changing
            the residual, allowing compatible fused add+norm implementations.
        before_gather: Preserve this read on source rows before a DP gather.

    enter initializes the stack residual. read consumes an already-written
    residual; update_and_read first applies the actual producer's update.
    Both reads return (compute_input, residual), preserving their kernel's
    rounding order rather than normalizing a separately rounded snapshot.
    """

    # The input is the residual's norm, in the quantization the call asks for,
    # and the residual is left as it is: what a fused add + norm kernel computes,
    # and what may run on the rows a sum completes onto.
    norms_plainly: bool
    # Preserve the read on the source rows before a DP gather.
    before_gather: bool

    def enter(self, hidden_states) -> torch.Tensor:
        """The residual the layer stack starts from, given its input."""

    def read(
        self,
        residual,
        norm,
        quant_format: str = "",
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Read input from a residual that already includes the producer update.

        Args:
            residual: Updated residual tensor in this implementation's representation.
            norm: Consumer normalization module.
            quant_format: Requested input quantization format, or empty for the
                read's default format; supported values belong to the implementation.
            post_residual_addition: Kept for signature parity with update_and_read;
                current read implementations ignore or reject it.

        Returns:
            (compute_input, residual), potentially with a quantized input format.
        """

    def update_and_read(
        self,
        update: StageUpdate,
        hidden_states,
        residual,
        norm,
        quant_format: str = "",
        post_residual_addition: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """Apply a producer update and read input while preserving kernel ordering.

        Args:
            update: Actual producer's StageUpdate, not a consumer-inferred operation.
            hidden_states: Producer contribution after required communication.
            residual: Previous residual, or None at entry where supported.
            norm: Consumer normalization module.
            quant_format: Requested input quantization format, as in read().
            post_residual_addition: Optional extra added after update, before norm.

        Returns:
            (compute_input, residual). A fused implementation can preserve FP32
            accumulation through norm without materializing a rounded intermediate.
        """


class LayerResidual(NamedTuple):
    """The reads and updates a decoder layer declares for its attention and
    its FFN."""

    attention_read: StageRead
    attention_update: StageUpdate
    ffn_read: StageRead
    ffn_update: StageUpdate
