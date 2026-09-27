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
"""What each stage produces and needs, and the boundary steps chosen from those
declarations."""

from enum import Enum, auto
from functools import partial
from typing import Callable, FrozenSet, Mapping, Optional, Tuple, Union

import msgspec
import torch

from sglang.srt.distributed import GroupCoordinator
from sglang.srt.layers.communicator.layout import (
    CommunicateContext,
    Layout,
    SumGroup,
    TokenAxis,
    _gathers_over_attention_cp,
    _same_ranks,
    _sum_group,
    token_axis_sizes,
)
from sglang.srt.layers.communicator.ops import (
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    _hand_qkv_hook_its_input,
    _mlp_input_completing_owed,
    _mlp_input_dp_partial,
    _mlp_input_dp_replicate,
    _mlp_input_gather_attention_cp,
    _mlp_input_gather_moe_cp,
    _mlp_input_norm,
    _mlp_input_on_residual_shard,
    _mlp_input_residual_into_sum,
    _mlp_input_scatter,
    _mlp_input_without_dp,
    tp_reduce_scatter,
)
from sglang.srt.layers.communicator.output import (
    HandoffOutput,
    UnreducedOutput,
    reduce_output,
)
from sglang.srt.layers.communicator.residual import ResidualOps
from sglang.srt.layers.communicator.residual.add_norm import ADD_AND_NORM
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel


class StageInput(msgspec.Struct, frozen=True):
    """The rows a stage's consumer needs: sharded over the token axes its
    compute group does not span."""

    layout: Layout
    # Token axes the consumer gathers over itself when its input arrives
    # sharded over them.
    gathers_itself: FrozenSet[TokenAxis] = frozenset()


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
    """What a decoder layer declares about its attention, its FFN and the rows
    between them; ``decoder_layer_edges`` turns it into the layer's
    boundaries."""

    # The rows the layer's input and residual arrive on.
    input_rows: Layout
    attention: StageInput
    attention_output: StageOutput
    ffn: StageInput
    ffn_output: StageOutput
    # The rows the residual is on while the FFN runs.
    ffn_residual_rows: Layout
    # The rows the layer hands on.
    output_rows: Layout
    # The group the layer's input arrives as a partial sum over, if any.
    input_owes: Optional[SumGroup] = None
    # Whether the residual is added into one rank's share of the attention
    # output's sum before that sum completes, instead of after it.
    residual_joins_attention_sum: bool = False


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


class DecoderLayerEdges(msgspec.Struct, frozen=True):
    """A decoder layer's three boundaries: from the rows the previous layer
    handed on into the attention, from the attention into the FFN, and from
    the FFN onto the rows the layer hands on. The next layer runs the rest of
    the last one."""

    into_attention: EdgeDecl
    into_ffn: EdgeDecl
    out_of_ffn: EdgeDecl


def decoder_layer_edges(sides: DecoderLayerSides) -> DecoderLayerEdges:
    """The boundaries of a decoder layer, each with the declarations of its two
    sides. What arrives at the layer is complete unless it owes a sum by
    construction (``input_owes``); a sum the previous layer leaves for a batch
    comes with the value."""
    owes = sides.input_owes is not None
    arrived = StageOutput(sides.input_rows, group=sides.input_owes, always_leaves=owes)
    # Completing what the input owes leaves it and the residual on each
    # attention-TP rank's slice.
    attention_rows = (
        Layout(sides.input_rows.sharded | {TokenAxis.ATTN_TP_SCATTER})
        if owes
        else sides.input_rows
    )
    return DecoderLayerEdges(
        into_attention=EdgeDecl(
            produced=arrived,
            need=sides.attention,
            residual=sides.input_rows,
            residual_to=attention_rows,
        ),
        into_ffn=EdgeDecl(
            produced=sides.attention_output,
            need=sides.ffn,
            residual=attention_rows,
            residual_to=sides.ffn_residual_rows,
            residual_joins_sum=sides.residual_joins_attention_sum,
        ),
        out_of_ffn=EdgeDecl(
            produced=sides.ffn_output,
            need=StageInput(sides.output_rows),
            residual=sides.ffn_residual_rows,
            residual_to=sides.output_rows,
        ),
    )


def stage_edges(
    *, previous: Optional[StageOutput], stage: StageDecl, rows: Layout
) -> Tuple[EdgeDecl, EdgeDecl]:
    """The two boundaries of a layer that is one stage of a sequence of stages:
    into it from the previous stage's output (None at the start of the layer
    stack) and out of it onto ``rows``, the rows every layer hands on and the
    residual is on between stages. The previous output arrives on those rows: a
    stage whose output is elsewhere (an FFN on the TP group) moves it back there
    first, and what it may leave of its sum comes with the value; otherwise the
    value is complete. The residual follows the input onto a finer slice and
    stays where it is when the input is gathered."""
    arrived = (
        StageOutput(
            rows,
            group=previous.group,
            always_leaves=previous.always_leaves,
            leaves_for_next_layer=previous.leaves_for_next_layer,
        )
        if previous is not None
        and (previous.always_leaves or previous.leaves_for_next_layer)
        else StageOutput(rows)
    )
    during = stage.input.layout if rows.sharded <= stage.input.layout.sharded else rows
    return (
        EdgeDecl(produced=arrived, need=stage.input, residual=rows, residual_to=during),
        EdgeDecl(
            produced=stage.output,
            need=StageInput(rows),
            residual=during,
            residual_to=rows,
        ),
    )


def sequence_parallel_layer_sides(
    *, axis_sizes: Mapping[TokenAxis, int]
) -> DecoderLayerSides:
    """A decoder layer while a LayerNorm SP region is active. Its linears
    all-gather their input and reduce-scatter their output themselves, so the
    layer's rows, its residual and both outputs stay on each TP rank's slice."""
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axis_sizes
    )
    local = Layout.sharded_over(
        TokenAxis.ATTN_DP,
        TokenAxis.ATTN_CP,
        TokenAxis.ATTN_TP_SCATTER,
        axis_sizes=axis_sizes,
    )
    gathers = frozenset({TokenAxis.ATTN_TP_SCATTER})
    return DecoderLayerSides(
        input_rows=local,
        # qkv and gate_up gather the slices into their GEMM.
        attention=StageInput(attention, gathers_itself=gathers),
        # o_proj and down reduce-scatter out of theirs, whether fused or not.
        attention_output=StageOutput(local),
        ffn=StageInput(
            Layout.sharded_over(axis_sizes=axis_sizes), gathers_itself=gathers
        ),
        ffn_output=StageOutput(local),
        ffn_residual_rows=local,
        output_rows=local,
    )


def scattered_residual_layer_sides(
    *,
    axis_sizes: Mapping[TokenAxis, int],
    ffn_group: SumGroup,
    is_first_layer: bool,
    is_last_layer: bool,
    leaves_for_reduce_scatter: bool,
) -> DecoderLayerSides:
    """A decoder layer on an input-scattered batch whose residual stays on
    each attention-TP rank's slice (MHC). The attention and the FFN compute on
    the full rows. The attention takes the slice and gathers it itself (its
    QKV hook, or the handoff before an attention that needs the rows); the
    boundary brings the FFN its rows and moves both outputs back onto the
    slice: the attention output's TP sum by a reduce-scatter, the FFN's too
    when it leaves it (``leaves_for_reduce_scatter``). The first layer's input
    is the embedding's partial sum on the full rows; the last layer hands on
    the full rows again."""
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axis_sizes
    )
    local = Layout.sharded_over(
        TokenAxis.ATTN_DP,
        TokenAxis.ATTN_CP,
        TokenAxis.ATTN_TP_SCATTER,
        axis_sizes=axis_sizes,
    )
    return DecoderLayerSides(
        input_rows=attention if is_first_layer else local,
        attention=StageInput(
            attention, gathers_itself=frozenset({TokenAxis.ATTN_TP_SCATTER})
        ),
        attention_output=StageOutput(
            attention, group=SumGroup.ATTN_TP, always_leaves=True
        ),
        ffn=StageInput(Layout.sharded_over(axis_sizes=axis_sizes)),
        ffn_output=StageOutput(
            attention,
            group=ffn_group,
            leaves_for_reduce_scatter=leaves_for_reduce_scatter,
        ),
        ffn_residual_rows=local,
        output_rows=attention if is_last_layer else local,
        input_owes=SumGroup.TP if is_first_layer else None,
    )


def input_scattered_layer_sides(
    *,
    axis_sizes: Mapping[TokenAxis, int],
    ffn_group: SumGroup,
    hands_on_partial: bool,
) -> DecoderLayerSides:
    """A decoder layer on a batch whose attention input is scattered over
    attention TP. Each layer's input is a partial sum over TP on the full rows:
    the embedding and every FFN before it leave that sum, and a reduce-scatter
    completes it onto each rank's slice, which the attention gathers itself.
    The residual comes back to the full rows inside the attention output's sum."""
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axis_sizes
    )
    return DecoderLayerSides(
        input_rows=attention,
        attention=StageInput(
            attention, gathers_itself=frozenset({TokenAxis.ATTN_TP_SCATTER})
        ),
        attention_output=StageOutput(
            attention, group=SumGroup.ATTN_TP, always_leaves=True
        ),
        ffn=StageInput(Layout.sharded_over(axis_sizes=axis_sizes)),
        # The next layer's input completes the sum; the last layer's FFN does.
        ffn_output=StageOutput(
            attention, group=ffn_group, leaves_for_reduce_scatter=hands_on_partial
        ),
        ffn_residual_rows=attention,
        output_rows=attention,
        input_owes=SumGroup.TP,
        residual_joins_attention_sum=True,
    )


def decoder_layer_sides(
    *,
    axis_sizes: Mapping[TokenAxis, int],
    ffn_on_local_rows: bool,
    previous_on_local_rows: bool,
    is_last_layer: bool,
    attention_gathers_local_rows: bool,
    ffn_group: SumGroup,
    leaves_for_next_layer: bool,
    leaves_for_reduce_scatter: bool,
    leaves_for_reduce_scatterv: bool,
    hands_on_attention_rows: bool = False,
    ffn_shards_over_cp: bool = False,
) -> DecoderLayerSides:
    """An attention followed by an FFN, derived from the groups each computes
    over. The FFN runs either on the TP group (a dense MLP, or a MoE not
    dispatched per DP shard) or on this rank's local rows (a MoE dispatched per
    DP shard, which completes its own combine, or a dense MLP on every rank).
    A MoE whose data-parallel groups are the CP ranks (``ffn_shards_over_cp``)
    computes each CP shard on its own ranks: its rows stay sharded over CP.
    A layer on local rows hands on the attention's rows when it is the last, or
    with ``hands_on_attention_rows``."""
    # Attention computes over the attention-TP ranks of one DP (and CP) shard.
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axis_sizes
    )
    # Each attention-TP rank's own slice of those rows.
    local = Layout.sharded_over(
        TokenAxis.ATTN_DP,
        TokenAxis.ATTN_CP,
        TokenAxis.ATTN_TP_SCATTER,
        axis_sizes=axis_sizes,
    )
    # The TP group spans every token axis, so an FFN on it needs every row.
    ffn = (
        local
        if ffn_on_local_rows
        else Layout.sharded_over(
            *((TokenAxis.ATTN_CP,) if ffn_shards_over_cp else ()),
            axis_sizes=axis_sizes,
        )
    )
    attention_tp = axis_sizes[TokenAxis.ATTN_TP_SCATTER] > 1
    return DecoderLayerSides(
        input_rows=local if previous_on_local_rows else attention,
        attention=StageInput(
            attention,
            gathers_itself=(
                frozenset({TokenAxis.ATTN_TP_SCATTER})
                if attention_gathers_local_rows
                else frozenset()
            ),
        ),
        # The output projection leaves the attention-TP sum to prepare_mlp.
        attention_output=StageOutput(
            attention,
            group=SumGroup.ATTN_TP if attention_tp else None,
            always_leaves=attention_tp,
        ),
        ffn=StageInput(ffn),
        ffn_output=(
            StageOutput(ffn)
            if ffn_on_local_rows
            else StageOutput(
                ffn,
                group=ffn_group,
                leaves_for_next_layer=leaves_for_next_layer,
                leaves_for_reduce_scatter=leaves_for_reduce_scatter,
                leaves_for_reduce_scatterv=leaves_for_reduce_scatterv,
            )
        ),
        ffn_residual_rows=local if ffn_on_local_rows else attention,
        output_rows=(
            local
            if ffn_on_local_rows and not (is_last_layer or hands_on_attention_rows)
            else attention
        ),
    )


class FusedMlpInput(msgspec.Struct, frozen=True):
    """A kernel that completes the sum the attention output owes together with
    the residual add and the post-attention norm, in that order.

    ``run(hidden_states, residual, forward_batch)`` returns the FFN's input and
    the residual, or None when it does not take the batch; it returns None only
    before touching its inputs or starting a collective."""

    # The group whose sum it completes.
    completes: SumGroup
    run: Callable[..., Optional[Tuple[torch.Tensor, torch.Tensor]]]
    # It may hand back a new residual and leave the one it took unchanged.
    may_return_new_residual: bool


def tbo_split_moves(layer_input_rows: Layout) -> Tuple[Callable, Callable]:
    """The moves around the two-batch-overlap split, which cuts the attention's
    rows: from the rows the first overlapped layer takes to the attention's,
    and back again for each half."""
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=token_axis_sizes()
    )
    pair = CommunicateSummableTensorPairFn
    if layer_input_rows == attention:
        return pair._trivial, pair._trivial
    if layer_input_rows.sharded - attention.sharded == {TokenAxis.ATTN_TP_SCATTER}:
        # Each rank's slice: write the residual in and gather over attention
        # TP, then take the slice of each half.
        return pair._gather, pair._scatter
    raise NotImplementedError(f"{layer_input_rows=}")


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


def _cp_moves() -> CpMoves:
    """DSA and MLA CP gather equal shards over the attention-CP group and can
    complete a sum over it. GQA prefill CP gathers blocks padded to the longest
    over the MoE-CP group and takes back only a complete output."""
    if _gathers_over_attention_cp():
        return CpMoves(
            gather=_mlp_input_gather_attention_cp,
            take_back=CommunicateSummableTensorPairFn._take_back_attention_cp_shard,
            reduce_scatter=CommunicateSummableTensorPairFn._reduce_scatter_over_cp,
            reduce_scatter_group=lambda: get_parallel().attn_cp_group,
        )
    return CpMoves(
        gather=_mlp_input_gather_moe_cp,
        take_back=CommunicateSummableTensorPairFn._scatter_hidden_states_moe,
    )


class BoundarySteps(msgspec.Struct, frozen=True):
    """The steps a batch runs at a layer's boundaries: into the attention,
    from the attention output to the FFN input, and the FFN output on to the
    rows the layer hands on."""

    # The half into the attention: completes what the input owes, writes the
    # previous output into the residual and reads the attention input
    # (_attention_input_step); attention_input then moves it.
    attention_prepare: Callable
    attention_input: Callable
    ffn_input: Callable
    # The rows ffn_input hands the FFN.
    ffn_input_rows: Layout
    # What the FFN exit reads: the FFN output's group and what it may leave.
    ffn_output: StageOutput
    # The postprocess that moves the FFN output on; None when it goes back over
    # attention DP, whose step the FFN exit and postprocess choose per batch.
    ffn_output_move: Optional[Callable]
    # Whether the next layer's input can take the FFN's sum.
    ffn_sum_is_movable: bool
    # Whether ffn_output_move also completes the sum the FFN leaves.
    ffn_output_move_completes_sum: bool = False
    # The fused kernels ffn_input tries first.
    fused: Tuple["FusedMlpInput", ...] = ()
    # Hands the attention its input once attention_input has moved it:
    # (hidden_states, forward_batch, qkv_latent_func) -> hidden_states.
    attention_handoff: Callable = _hand_qkv_hook_its_input

    @property
    def returns_over_dp(self) -> bool:
        return self.ffn_output_move is None


def _attention_input_step(
    hidden_states: Union[torch.Tensor, "UnreducedOutput"],
    residual: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
    norm: torch.nn.Module,
    context: "CommunicateContext",
    *,
    quant_format: str,
    post_residual_addition: Optional[torch.Tensor],
    layer_input: Optional[Callable],
    fusions: Tuple[Callable, ...],
    enters_stack: bool,
    residual_ops: ResidualOps,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A boundary's half into an attention: complete what the previous layer
    left (a value that owes a sum, or a producer's handoff one of ``fusions``
    consumes with the add and norm), what the input owes by construction
    (``layer_input``), then write the previous output into the residual and read
    the attention input with ``norm``. The layer stack's first layer
    (``enters_stack``) starts its residual from its input."""
    enters = residual is None and enters_stack
    owed = None if isinstance(hidden_states, torch.Tensor) else hidden_states
    if owed is not None and residual is None:
        raise RuntimeError(f"{type(owed).__name__} requires residual input")
    if isinstance(owed, UnreducedOutput) and owed.reduce_and_redistribute is not None:
        # No fused kernel runs under attention DP: the reduce-scatter back to
        # this rank's tokens comes first.
        hidden_states, owed = reduce_output(owed), None
    if owed is not None:
        for fused in fusions:
            result = fused(owed, residual, forward_batch, post_residual_addition)
            if result is not None:
                return result
        if isinstance(owed, HandoffOutput):
            # No fused kernel took the handoff: its producer completes it.
            hidden_states, owed = reduce_output(owed), None
        else:
            hidden_states = owed.partial
    if layer_input is not None:
        hidden_states, residual = layer_input(hidden_states, residual, context)
    if enters:
        hidden_states, residual = residual_ops.enter(hidden_states), None
    if owed is not None and hidden_states.shape[0] != 0:
        hidden_states = reduce_output(owed)
    if residual is None:
        # The previous layer already wrote its output into the residual.
        return residual_ops.read_attention_input(hidden_states, norm, quant_format)
    return residual_ops.update_and_read_attention_input(
        hidden_states, residual, norm, quant_format, post_residual_addition
    )


def _another_stage(*args, **kwargs):
    """The steps of a stage a single-stage layer does not have."""
    raise RuntimeError("this layer is one stage and does not have the other")


class InputRead(Enum):
    """How a boundary's consumer reads its input from the residual: with the
    attention input norm (prepare_attn) or with the FFN input norm and its
    fused kernels (prepare_mlp)."""

    ATTENTION = auto()
    FFN = auto()


class LayerStage(msgspec.Struct, frozen=True):
    """A layer that is one stage of a sequence of stages, each an
    attention-like mixer or an FFN: how it reads its input, its two boundaries
    (into it, and out of it onto the rows every layer hands on, as
    ``stage_edges`` gives them), and whether the layer stack
    starts at it."""

    reads: InputRead
    edges: Tuple[EdgeDecl, EdgeDecl]
    enters_stack: bool = False


class Boundary(msgspec.Struct, frozen=True):
    """The steps one layer runs at one boundary, chosen from both sides'
    declarations. A layer runs the consumer's half of a boundary into one of
    its stages, and the producer's half of the boundary after its last stage;
    the neighbouring layer runs the other half of that one."""

    edge: EdgeDecl
    # The consumer's half: completing what the input owes, the add and the
    # norm; into the FFN also the move onto the rows it needs.
    prepare: Optional[Callable] = None
    # Into an attention, the move onto its rows after prepare.
    input_move: Optional[Callable] = None
    # The fused kernels an FFN's prepare tries first.
    fused: Tuple["FusedMlpInput", ...] = ()
    # The producer's half: the postprocess that moves the output onto the rows
    # the layer hands on; None when it goes back over attention DP, whose step
    # the FFN exit and postprocess choose per batch.
    output_move: Optional[Callable] = None
    # Whether output_move also completes the sum the producer leaves.
    output_move_completes_sum: bool = False

    @property
    def input_rows(self) -> Layout:
        """The rows the consumer is handed: what it needs, still sharded over
        the axes it gathers itself."""
        need = self.edge.need
        return Layout(
            need.layout.sharded
            | (self.edge.produced.layout.sharded & need.gathers_itself)
        )


def make_boundary(
    edge: EdgeDecl,
    *,
    reads: Optional[InputRead],
    fusions: Tuple = (),
    force_layernorm_before_gather: bool = False,
    cp_moves: Optional[CpMoves] = None,
    residual_ops: ResidualOps = ADD_AND_NORM,
    enters_stack: bool = False,
) -> Boundary:
    """The steps a layer runs at ``edge``, around the residual operations;
    ``cp_moves`` for an edge that gathers over or returns across attention CP.
    ``reads`` is how the consumer reads its input, or None when the consumer
    runs in the next layer and ``edge.need`` is the rows this layer hands on:
    then only the producer's half runs here. ``fusions`` are the fused kernels
    the consumer tries first (FusedMlpInput into an FFN, the attention input's
    entries into an attention); ``enters_stack`` for the edge into the layer
    stack's first attention. The consumer's half reads only this edge's
    declarations, never what the producer chose for a batch; a sum left for a
    batch arrives with the value."""
    if reads is None:
        if edge.need.layout != edge.residual_to:
            raise NotImplementedError(f"{edge=}")
        returns_over_dp, output_move, completes_sum = _select_ffn_output_move(
            edge.produced,
            residual=edge.residual,
            to=edge.residual_to,
            cp_moves=cp_moves,
            residual_ops=residual_ops,
        )
        return Boundary(
            edge,
            output_move=None if returns_over_dp else output_move,
            output_move_completes_sum=completes_sum,
        )
    if reads is InputRead.FFN:
        input_step, fused = _select_ffn_input(
            edge.produced,
            residual=edge.residual,
            residual_to=edge.residual_to,
            need=edge.need,
            force_layernorm_before_gather=force_layernorm_before_gather,
            fusions=fusions,
            residual_joins_sum=edge.residual_joins_sum,
            cp_moves=cp_moves,
            residual_ops=residual_ops,
        )
        return Boundary(edge, prepare=input_step, fused=fused)
    layer_input = None
    if edge.produced.always_leaves:
        # A reduce-scatter completes the TP sum onto each rank's slice, which the
        # attention takes: its group is the TP group without attention DP or CP.
        if (
            edge.produced.group is not SumGroup.TP
            or edge.residual.sharded
            or TokenAxis.ATTN_TP_SCATTER not in edge.need.gathers_itself
            or edge.residual_to.sharded != {TokenAxis.ATTN_TP_SCATTER}
        ):
            raise NotImplementedError(f"{edge=}")
        layer_input = tp_reduce_scatter
    return Boundary(
        edge,
        prepare=partial(
            _attention_input_step,
            layer_input=layer_input,
            fusions=fusions,
            enters_stack=enters_stack,
            residual_ops=residual_ops,
        ),
        input_move=_select_attention_input_move(edge.residual_to, edge.need),
    )


def _select_boundary_steps(
    sides: DecoderLayerSides,
    *,
    fusions: Tuple["FusedMlpInput", ...] = (),
    force_layernorm_before_gather: bool = False,
    cp_moves: Optional[CpMoves] = None,
    residual_ops: ResidualOps = ADD_AND_NORM,
    attention_handoff: Callable = _hand_qkv_hook_its_input,
    attention_fusions: Tuple[Callable, ...] = (),
    enters_stack: bool = False,
) -> BoundarySteps:
    """The steps of a decoder layer: its three boundaries, each chosen from the
    declarations of its two sides, around the layer's residual operations. Both
    edges that cross attention CP take the same ``cp_moves``; the attention's
    input tries ``attention_fusions``, and the layer stack's first layer
    (``enters_stack``) starts its residual there."""
    edges = decoder_layer_edges(sides)
    out_of_ffn = make_boundary(
        edges.out_of_ffn, reads=None, cp_moves=cp_moves, residual_ops=residual_ops
    )
    into_ffn = make_boundary(
        edges.into_ffn,
        reads=InputRead.FFN,
        fusions=fusions,
        force_layernorm_before_gather=force_layernorm_before_gather,
        cp_moves=cp_moves,
        residual_ops=residual_ops,
    )
    into_attention = make_boundary(
        edges.into_attention,
        reads=InputRead.ATTENTION,
        fusions=attention_fusions,
        residual_ops=residual_ops,
        enters_stack=enters_stack,
    )
    return BoundarySteps(
        attention_prepare=into_attention.prepare,
        attention_input=into_attention.input_move,
        ffn_input=into_ffn.prepare,
        ffn_input_rows=into_ffn.input_rows,
        ffn_output=edges.out_of_ffn.produced,
        ffn_output_move=out_of_ffn.output_move,
        ffn_output_move_completes_sum=out_of_ffn.output_move_completes_sum,
        ffn_sum_is_movable=edges.out_of_ffn.produced.group is not None,
        fused=into_ffn.fused,
        attention_handoff=attention_handoff,
    )


def _select_attention_input_move(rows: Layout, need: StageInput) -> Callable:
    """How the rows a layer takes become its attention's input: as they are,
    or gathered over attention TP from each rank's slice, unless the attention
    gathers them itself."""
    gathered = rows.sharded - need.layout.sharded - need.gathers_itself
    if not need.layout.sharded <= rows.sharded or gathered not in (
        frozenset(),
        {TokenAxis.ATTN_TP_SCATTER},
    ):
        raise NotImplementedError(f"{rows=} {need=}")
    if gathered:
        return CommunicateSimpleFn._scattered_to_tp_attn_full
    return CommunicateSimpleFn._trivial


def _select_ffn_input(
    produced: StageOutput,
    *,
    residual: Layout,
    residual_to: Layout,
    need: StageInput,
    force_layernorm_before_gather: bool,
    fusions: Tuple[FusedMlpInput, ...],
    residual_joins_sum: bool = False,
    cp_moves: Optional[CpMoves] = None,
    residual_ops: ResidualOps = ADD_AND_NORM,
) -> Tuple[Callable, Tuple[FusedMlpInput, ...]]:
    """The steps from the attention output to the FFN input, and the fused
    kernels they try first: complete the attention-TP sum, move the residual to
    the rows it has while the FFN runs, write the output into it and read the
    FFN input, and bring the rows to what the FFN's group needs: a gather over
    attention DP, or each rank's own slice. A kernel in ``fusions`` is tried
    only when nothing is gathered or sliced, and only if it completes the sum
    the attention output owes. A write-back that is not a plain add runs only
    after the sum completes."""
    if produced.leaves_for_next_layer and produced.group not in (
        None,
        SumGroup.ATTN_TP,
    ):
        # A producer that leaves its sum only for some batches (an FFN before
        # this one) hands that sum on with the value: complete it, then the
        # input is a complete output.
        step, fused = _select_ffn_input(
            StageOutput(produced.layout),
            residual=residual,
            residual_to=residual_to,
            need=need,
            force_layernorm_before_gather=force_layernorm_before_gather,
            fusions=fusions,
            residual_joins_sum=residual_joins_sum,
            cp_moves=cp_moves,
            residual_ops=residual_ops,
        )
        return partial(_mlp_input_completing_owed, step=step), fused
    # What the attention output owes decides the steps: the attention-TP sum,
    # always left by the output projection, or nothing.
    owes_attention_tp = produced.group is SumGroup.ATTN_TP
    if owes_attention_tp != produced.always_leaves or produced.group not in (
        None,
        SumGroup.ATTN_TP,
    ):
        raise NotImplementedError(f"{produced=}")
    gathered = produced.layout.sharded - need.layout.sharded - need.gathers_itself
    sliced = need.layout.sharded - produced.layout.sharded
    if sliced:
        # Each attention-TP rank takes its own slice: the reduce-scatter
        # completes the attention-TP sum and slices in one collective.
        if (
            sliced != {TokenAxis.ATTN_TP_SCATTER}
            or gathered
            or not owes_attention_tp
            or residual_to != need.layout
            or residual not in (produced.layout, need.layout)
        ):
            raise NotImplementedError(f"{produced=} {residual=} {need=}")
        return (
            partial(
                _mlp_input_scatter,
                scatters_residual=residual != residual_to,
                residual_ops=residual_ops,
            ),
            (),
        )
    if residual_to.sharded - produced.layout.sharded == {TokenAxis.ATTN_TP_SCATTER}:
        # The residual stays on each rank's slice while the FFN takes the
        # attention's rows (MHC on an input-scattered batch).
        if gathered or not owes_attention_tp or residual != residual_to:
            raise NotImplementedError(f"{produced=} {residual=} {need=}")
        return partial(_mlp_input_on_residual_shard, residual_ops=residual_ops), ()
    if gathered == {TokenAxis.ATTN_CP}:
        # Each CP rank completes its own chunk, then the CP moves gather them.
        if cp_moves is None:
            raise NotImplementedError(f"{produced=} {need=}")
        on_chunk, fused = _select_ffn_input(
            produced,
            residual=residual,
            residual_to=residual_to,
            need=StageInput(produced.layout),
            force_layernorm_before_gather=force_layernorm_before_gather,
            fusions=fusions,
            residual_joins_sum=residual_joins_sum,
            residual_ops=residual_ops,
        )
        return partial(cp_moves.gather, gather=on_chunk), fused
    if (
        residual_to != produced.layout
        or gathered
        not in (
            frozenset(),
            {TokenAxis.ATTN_DP},
            {TokenAxis.ATTN_DP, TokenAxis.ATTN_CP},
        )
        or residual.sharded - residual_to.sharded
        not in (frozenset(), {TokenAxis.ATTN_TP_SCATTER})
    ):
        raise NotImplementedError(f"{produced=} {residual=} {need=}")
    # A residual arriving on each rank's slice is gathered back first.
    gathers_residual = residual != residual_to
    if not gathered:
        if not owes_attention_tp:
            return partial(_mlp_input_norm, residual_ops=residual_ops), ()
        if gathers_residual and residual_joins_sum:
            # Each rank adds its slice of the residual into its share of the
            # sum, so the all-reduce also brings the residual back to every row.
            if not residual_ops.adds_plainly:
                raise NotImplementedError(f"{produced=} {residual=} {need=}")
            return _mlp_input_residual_into_sum, ()
        fused = tuple(f for f in fusions if f.completes is produced.group)
        return (
            partial(
                _mlp_input_without_dp,
                gathers_residual=gathers_residual,
                fusions=tuple(f.run for f in fused),
                residual_ops=residual_ops,
            ),
            fused,
        )
    # The partial order adds the residual on attention-TP rank 0 before the DP
    # gather's collective completes that sum, which only a plain residual add
    # allows.
    # Over attention DP and CP, the DP gather puts each CP rank's shard in its
    # DP group's slot, so the one DP sum gathers both axes.
    places_cp_shards = TokenAxis.ATTN_CP in gathered
    if (
        owes_attention_tp
        and not force_layernorm_before_gather
        and residual_ops.adds_plainly
    ):
        return (
            partial(
                _mlp_input_dp_partial,
                gathers_residual=gathers_residual,
                places_cp_shards=places_cp_shards,
            ),
            (),
        )
    return (
        partial(
            _mlp_input_dp_replicate,
            gathers_residual=gathers_residual,
            reduces_attention_tp=owes_attention_tp,
            places_cp_shards=places_cp_shards,
            residual_ops=residual_ops,
        ),
        (),
    )


def _select_ffn_output_move(
    produced: StageOutput,
    *,
    residual: Layout,
    to: Layout,
    cp_moves: Optional[CpMoves] = None,
    residual_ops: ResidualOps = ADD_AND_NORM,
) -> Tuple[bool, Optional[Callable], bool]:
    """How the FFN output reaches the rows the layer hands on: whether it goes
    back by undoing the attention-DP gather (the FFN exit and postprocess run
    that step), or else the postprocess that moves it, None when there is none
    to choose here; and whether that move also completes the sum the FFN
    leaves."""
    pair = CommunicateSummableTensorPairFn
    if produced.layout == residual:
        if to == residual:
            return False, pair._trivial, False
        if to.sharded == residual.sharded - {TokenAxis.ATTN_TP_SCATTER}:
            # Each rank's slice back to the attention's rows: write the output
            # into the residual, then gather over attention TP.
            return False, partial(pair._gather, residual_ops=residual_ops), False
        raise NotImplementedError(f"{produced=} {residual=} {to=}")
    returned = residual.sharded - produced.layout.sharded
    if returned == {TokenAxis.ATTN_TP_SCATTER} and to in (residual, produced.layout):
        # The residual stays on each rank's slice (MHC on an input-scattered
        # batch): a reduce-scatter onto the slice completes the sum the FFN
        # leaves; a complete output is only sliced.
        sums = produced.leaves_for_reduce_scatter
        return (
            False,
            partial(
                pair._onto_residual_shard,
                sums=sums,
                gathers_back=to != residual,
                residual_ops=residual_ops,
            ),
            sums,
        )
    if to != residual or not produced.layout.sharded <= residual.sharded:
        raise NotImplementedError(f"{produced=} {residual=} {to=}")
    if returned == {TokenAxis.ATTN_CP}:
        if cp_moves is None:
            raise NotImplementedError(f"{produced=} {residual=} {to=}")
        if not produced.leaves_for_reduce_scatter:
            # A complete output: this rank's block of it, nothing summed.
            return False, cp_moves.take_back, False
        # The FFN leaves its sum: only a take-back that sums over the same
        # ranks completes it.
        if cp_moves.reduce_scatter is None or not _same_ranks(
            _sum_group(produced.group), cp_moves.reduce_scatter_group()
        ):
            raise NotImplementedError(f"{produced=} {residual=} {to=}")
        return False, cp_moves.reduce_scatter, True
    if returned == {TokenAxis.ATTN_DP, TokenAxis.ATTN_CP}:
        # This rank's CP shard, from where the DP gather put it.
        return False, CommunicateSummableTensorPairFn._take_back_cp_shard, False
    if returned != {TokenAxis.ATTN_DP}:
        raise NotImplementedError(f"{produced=} {residual=} {to=}")
    return True, None, False
