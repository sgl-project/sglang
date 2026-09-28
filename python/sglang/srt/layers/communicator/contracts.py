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
from sglang.srt.layers.communicator.layout import Layout, SumGroup, TokenAxis
from sglang.srt.layers.communicator.output import OutputTransform
from sglang.srt.layers.communicator.residual import StageRead, StageUpdate
from sglang.srt.layers.communicator.residual.add_norm import ADD, NORM_READ


class ProducerReduction(Enum):
    """Whether compute always leaves its sum or follows the boundary scope."""

    PARTIAL = auto()
    SCOPED = auto()
    # A replicated component is added after the sum inside compute.
    LOCAL_TAIL = auto()


class HandoffRows(Enum):
    """A required layer/branch handoff, independent of compute's input rows."""

    ATTENTION = auto()
    TBO_SPLIT = auto()


class BatchVariant(Enum):
    ORDINARY = auto()
    CONTEXT_PARALLEL = auto()
    INPUT_SCATTERED = auto()
    SEQUENCE_PARALLEL = auto()


class StageInput(msgspec.Struct, frozen=True):
    """The rows a stage's consumer needs: sharded over the token axes its
    compute group does not span."""

    layout: Layout
    # Token axes the consumer gathers over itself when its input arrives
    # sharded over them.
    gathers_itself: FrozenSet[TokenAxis] = frozenset()
    # How the consumer reads its input from the residual.
    read: StageRead = NORM_READ


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
    # How the producer's output is written into the residual. An arrival
    # description has no producer object; its edge declares capabilities only.
    update: Optional[StageUpdate] = ADD
    transform: Optional[OutputTransform] = None


class StageDecl(msgspec.Struct, frozen=True):
    """A computing stage's two sides: the rows its input must be on, and what
    its output is."""

    input: StageInput
    output: StageOutput


class EdgeDecl(msgspec.Struct, frozen=True):
    """One boundary between two stages, as the layer that runs one side of it
    sees it: what arrives from the producer, what the consumer needs, and the
    rows the residual is on before and after the boundary."""

    produced: StageOutput
    need: StageInput
    residual: Layout
    residual_to: Layout
    # Whether the residual is added into one rank's share of the produced sum
    # before that sum completes, instead of after it.
    residual_joins_sum: bool = False
    # Capabilities allowed to arrive from another layer, not its update object.
    update_capabilities: Tuple[bool, ...] = ()


class FusedMlpInput(msgspec.Struct, frozen=True):
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


class StageEntry(msgspec.Struct, frozen=True):
    """The boundary into one of a layer's stages, as the layer runs it."""

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
    handoff: Optional[Callable] = None
    # Return the updated residual to the producer's rows for aux capture.
    capture_move: Optional[Callable] = None
    capture_move_allocates: bool = False
    input_sum: Optional[SumGroup] = None
    preserves_residual: Optional[Callable] = None
    # Bound from a local successor at construction; no runtime plan lookup.
    capture_preserves_residual: Optional[Callable] = None


class StageSteps(msgspec.Struct, frozen=True):
    """One batch variant of one stage: its input and output boundary paths."""

    entry: StageEntry
    output: StageOutput
    # None means no fixed move. returns_over_dp selects batch-dependent DP transport.
    output_move: Optional[Callable]
    output_move_completes_sum: bool = False

    returns_over_dp: bool = False


class StageKind(Enum):
    """Which of a decoder layer's two stages a layer that is one stage takes
    the place of: its norm, its read and update, its fused kernels and its
    entry in the layer's steps."""

    ATTENTION = auto()
    FFN = auto()
