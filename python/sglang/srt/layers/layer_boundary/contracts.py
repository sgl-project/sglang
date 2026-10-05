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
"""Immutable stage declarations and bound input/output paths."""

from enum import Enum, auto
from typing import Callable, FrozenSet, Optional, Tuple

import msgspec
import torch

from sglang.srt.distributed import GroupCoordinator
from sglang.srt.layers.layer_boundary.layout import Layout, SumGroup, TokenAxis
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.residual import ResidualReadout, ResidualUpdate
from sglang.srt.layers.layer_boundary.residual.add_norm import NORM_READOUT, PLAIN_ADD


class ProducerReduction(Enum):
    """Who completes the sum a stage's output owes; compute never does.

    ALWAYS_PARTIAL is attention-only: finish() hands the partial sum to the
    next stage's input as its declared sum. EXIT_SCOPED, for an FFN or a
    single-stage mixer: the exit completes the sum, or carries it to the next
    stage's input.
    """

    ALWAYS_PARTIAL = auto()
    EXIT_SCOPED = auto()


class ExitRows(Enum):
    """The rows a layer or branch must leave its output on, independent of
    compute's input rows."""

    ATTENTION = auto()
    TBO_SPLIT = auto()


class BatchVariant(Enum):
    ORDINARY = auto()
    CONTEXT_PARALLEL = auto()
    INPUT_SCATTERED = auto()
    SEQUENCE_PARALLEL = auto()


class InputContract(msgspec.Struct, frozen=True):
    """Declare the consumer's input rows and read operation.

    Fields:
        layout: Token sharding required by compute.
        gathered_by_compute: Token axes compute can gather internally; the boundary
            may hand it input still sharded over those axes.
        read: Operation that derives input from the updated residual.
    """

    layout: Layout
    # Token axes the consumer gathers over itself when its input arrives
    # sharded over them.
    gathered_by_compute: FrozenSet[TokenAxis] = frozenset()
    # How the consumer reads its input from the residual.
    read: ResidualReadout = NORM_READOUT


class OutputContract(msgspec.Struct, frozen=True):
    """Declare producer rows and permitted reduction handoffs at construction.

    Fields:
        layout: Token sharding of the producer contribution.
        group: The group the output owes its sum over, or None when it is
            complete.
        always_partial: The sum is handed to the next stage's input as its
            declared sum.
        may_defer_to_next: The exit may carry the sum to the following
            stage's input instead of completing it.
        may_reduce_scatter: A fixed-size reduce-scatter selected by the
            boundary may complete the sum.
        may_reduce_scatterv: The selected variable-size attention-DP combine
            may complete the sum.
        update: Producer residual operation; None on an arrival contract that
            declares capabilities and obtains the actual update from the stream.
        transform: Optional operation on the contribution before residual update.

    These are permissions, not a record of a particular output. The exit
    decision and ResidualStream record what that output actually owes.
    """

    layout: Layout
    # None when there is nothing to sum.
    group: Optional[SumGroup] = None
    # The sum is always handed to the next stage's input as a declared sum.
    always_partial: bool = False
    # Otherwise the exit completes it, or may carry it to the next input.
    may_defer_to_next: bool = False
    may_reduce_scatter: bool = False
    # Whether the selected attention-DP combine may complete this sum.
    may_reduce_scatterv: bool = False
    # How the producer's output is written into the residual. An arrival
    # description has no producer object; its edge declares capabilities only.
    update: Optional[ResidualUpdate] = PLAIN_ADD
    transform: Optional[OutputTransform] = None


class StageContract(msgspec.Struct, frozen=True):
    """Resolved input and output contracts for one stage and batch variant.

    Fields:
        input: Consumer row requirement and read operation.
        output: Producer row, reduction and residual-update contract.
    """

    input: InputContract
    output: OutputContract


class EdgeContract(msgspec.Struct, frozen=True):
    """Describe both sides of a boundary and the residual's row movement.

    Fields:
        produced: Producer output contract as seen by this side of the edge.
        need: Consumer input contract.
        residual: Residual layout before the boundary.
        residual_to: Residual layout after the boundary.
        residual_joins_sum: Whether one rank may add the residual into a partial
            before reduction; valid only for an eligible plain-add update.
        arriving_plain_add: Allowed values of ResidualUpdate.is_plain_add for
            arriving contributions. Empty means use produced.update's capability.
            The actual update object travels with the residual stream.
    """

    produced: OutputContract
    need: InputContract
    residual: Layout
    residual_to: Layout
    # Whether the residual is added into one rank's share of the produced sum
    # before that sum completes, instead of after it.
    residual_joins_sum: bool = False
    # Capabilities allowed to arrive from another layer, not its update object.
    arriving_plain_add: Tuple[bool, ...] = ()


class FfnInputFusion(msgspec.Struct, frozen=True):
    """A kernel that completes the sum the attention output owes together with
    the residual add and the post-attention norm, in that order.

    ``run(hidden_states, residual, forward_batch)`` returns the FFN's input and
    the residual, or None when it does not take the batch; it returns None only
    before touching its inputs or starting a collective."""

    # The group whose sum it completes.
    completes: SumGroup
    run: Callable[..., Optional[Tuple[torch.Tensor, torch.Tensor]]]
    # Callable(residual, forward_batch): True guarantees this candidate is
    # selected and does not mutate residual, including backend fallback.
    preserves_residual: Optional[Callable] = None


class CpMoves(msgspec.Struct, frozen=True):
    """How a CP extend's rows reach an FFN that needs all of them and come back,
    chosen once for the kind of prefill CP: ``gather`` gathers the FFN input
    after each rank has completed its own block, ``take_back`` returns this
    rank's block of a complete output, and ``reduce_scatter``, where there is
    one, completes a sum left over the ranks of ``reduce_scatter_group()`` and
    returns the block in the same collective."""

    gather: Callable
    take_back: Callable
    reduce_scatter: Optional[Callable] = None
    reduce_scatter_group: Optional[Callable[[], GroupCoordinator]] = None


class EntryPath(msgspec.Struct, frozen=True):
    """Bound consumer operations for one batch variant.

    Fields:
        prepare: Callable(hidden_states, residual, forward_batch, norm, **call)
            returning compute input and updated residual; finishes owed work,
            applies the producer update and performs the consumer read.
        input_rows: Layout handed to compute after preparation and input_move.
        input_move: Optional movement after prepare, before compute takes the input.
        attn_input_adapter: Optional callable(input, forward_batch, qkv_latent_func) that
            adapts already-placed input for attention.
        capture_move: Optional movement of the updated residual back onto the
            producer's rows for auxiliary capture.
        capture_move_allocates: Whether capture_move returns fresh storage (a
            gather) rather than a view (a cut); capture then retains it
            without copying.
        declared_sum: Statically owed sum group for an otherwise raw input tensor.
        preserves_residual: Optional (residual, batch) predicate certifying this
            input path leaves residual untouched, including backend fallback.
        capture_preserves_residual: Same guarantee for the local following stage,
            copied at construction so auxiliary capture can retain its input.
    """

    # Completes what the input owes, writes the previous stage's output into
    # the residual and reads this stage's input:
    # (hidden_states, residual, forward_batch, norm, **call).
    prepare: Callable
    # The rows prepare hands the stage.
    input_rows: Layout
    # Moves the input onto the stage's rows after prepare, when prepare does
    # not: (hidden_states, forward_batch) -> hidden_states.
    input_move: Optional[Callable] = None
    # Hands the stage its input once it is on its rows:
    # (hidden_states, forward_batch, qkv_latent_func) -> hidden_states.
    attn_input_adapter: Optional[Callable] = None
    # Return the updated residual to the producer's rows for aux capture.
    capture_move: Optional[Callable] = None
    capture_move_allocates: bool = False
    declared_sum: Optional[SumGroup] = None
    preserves_residual: Optional[Callable] = None
    # Bound from a local successor at construction; no runtime plan lookup.
    capture_preserves_residual: Optional[Callable] = None


class StagePath(msgspec.Struct, frozen=True):
    """Precomputed entry and exit work for one stage and batch variant.

    Fields:
        entry: Bound consumer half: preparation, input move and input adapter.
        output: Producer contract used by the exit decision.
        output_move: Fixed output transport, or None when absent or chosen per
            batch by the attention-DP exit path.
        output_move_completes_sum: Whether that move also reduces the output.
        returns_over_dp: Whether output uses batch-dependent attention-DP transport.
        writes_at_handoff: Whether the exit completes the output and writes it
            into the residual, for an FFN that hands off to another pipeline rank.
    """

    entry: EntryPath
    output: OutputContract
    # None means no fixed move. returns_over_dp selects batch-dependent DP transport.
    output_move: Optional[Callable]
    output_move_completes_sum: bool = False

    returns_over_dp: bool = False
    writes_at_handoff: bool = False


class StageKind(Enum):
    """Select attention/mixer or FFN input/output adapters.

    This is a boundary role, not a compute implementation or a requirement
    that layers contain exactly two alternating stages.
    """

    ATTENTION = auto()
    FFN = auto()
