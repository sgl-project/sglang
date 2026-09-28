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
from typing import Callable, FrozenSet, Mapping, Optional, Tuple

import msgspec
import torch

from sglang.srt.distributed import GroupCoordinator
from sglang.srt.layers.communicator.layout import (
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
    _consumer_step,
    _hand_qkv_hook_its_input,
    _mlp_input_dp_partial,
    _mlp_input_dp_replicate,
    _mlp_input_gather_attention_cp,
    _mlp_input_gather_moe_cp,
    _mlp_input_on_residual_shard,
    _mlp_input_residual_into_sum,
    _mlp_input_scatter,
    _mlp_input_slice,
    _mlp_input_without_dp,
    _read_input,
    tp_reduce_scatter,
)
from sglang.srt.layers.communicator.residual import (
    LayerResidual,
    StageRead,
    StageUpdate,
)
from sglang.srt.layers.communicator.residual.add_norm import (
    ADD,
    NORM_QUANT_READ,
    NORM_READ,
)
from sglang.srt.runtime_context import get_parallel


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
    # How the producer's output is written into the residual.
    update: StageUpdate = ADD


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
    # The previous layer's FFN output, written in as this layer's FFN writes its
    # own.
    arrived = StageOutput(
        sides.input_rows,
        group=sides.input_owes,
        always_leaves=owes,
        update=sides.ffn_output.update,
    )
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


def with_residual(
    sides: DecoderLayerSides, residual: LayerResidual
) -> DecoderLayerSides:
    """``sides`` with the reads and updates the layer declares for its
    attention and its FFN."""
    replace = msgspec.structs.replace
    return replace(
        sides,
        attention=replace(sides.attention, read=residual.attention_read),
        attention_output=replace(
            sides.attention_output, update=residual.attention_update
        ),
        ffn=replace(sides.ffn, read=residual.ffn_read),
        ffn_output=replace(sides.ffn_output, update=residual.ffn_update),
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
    update = previous.update if previous is not None else ADD
    arrived = (
        StageOutput(
            rows,
            group=previous.group,
            always_leaves=previous.always_leaves,
            leaves_for_next_layer=previous.leaves_for_next_layer,
            update=update,
        )
        if previous is not None
        and (previous.always_leaves or previous.leaves_for_next_layer)
        else StageOutput(rows, update=update)
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
        attention=StageInput(attention, gathers_itself=gathers, read=NORM_QUANT_READ),
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
            attention,
            gathers_itself=frozenset({TokenAxis.ATTN_TP_SCATTER}),
            read=NORM_QUANT_READ,
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
            attention,
            gathers_itself=frozenset({TokenAxis.ATTN_TP_SCATTER}),
            read=NORM_QUANT_READ,
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
            read=NORM_QUANT_READ,
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


class StageEntry(msgspec.Struct, frozen=True):
    """The boundary into one of a layer's stages, as the layer runs it."""

    # Completes what the input owes, writes the previous stage's output into
    # the residual and reads this stage's input:
    # (hidden_states, residual, forward_batch, norm, context, **call).
    prepare: Callable
    # The rows prepare hands the stage.
    input_rows: Layout
    # Moves the input onto the stage's rows after prepare, when prepare does
    # not: (hidden_states, forward_batch, context) -> hidden_states.
    input_move: Optional[Callable] = None
    # Hands the stage its input once it is on its rows:
    # (hidden_states, forward_batch, qkv_latent_func) -> hidden_states.
    handoff: Optional[Callable] = None
    # The fused kernels prepare tries first.
    fused: Tuple["FusedMlpInput", ...] = ()


class BoundarySteps(msgspec.Struct, frozen=True):
    """The steps a batch runs at a layer's boundaries: into each of its
    stages, and the last stage's output on to the rows the layer hands on."""

    # The boundary into the layer's attention and into its FFN; None for a
    # stage a layer that is one stage does not have.
    attention: Optional[StageEntry]
    ffn: Optional[StageEntry]
    # What the FFN exit reads: the FFN output's group and what it may leave.
    ffn_output: StageOutput
    # The postprocess that moves the FFN output on; None when it goes back over
    # attention DP, whose step the FFN exit and postprocess choose per batch.
    ffn_output_move: Optional[Callable]
    # Whether the next layer's input can take the FFN's sum.
    ffn_sum_is_movable: bool
    # Whether ffn_output_move also completes the sum the FFN leaves.
    ffn_output_move_completes_sum: bool = False

    @property
    def returns_over_dp(self) -> bool:
        return self.ffn_output_move is None


class StageKind(Enum):
    """Which of a decoder layer's two stages a layer that is one stage takes
    the place of: its norm, its read and update, its fused kernels and its
    entry in the layer's steps."""

    ATTENTION = auto()
    FFN = auto()


class LayerStage(msgspec.Struct, frozen=True):
    """A layer that is one stage of a sequence of stages, each an
    attention-like mixer or an FFN: which of the two it is, its two boundaries
    (into it, and out of it onto the rows every layer hands on, as
    ``stage_edges`` gives them), and whether the layer stack
    starts at it."""

    kind: StageKind
    edges: Tuple[EdgeDecl, EdgeDecl]
    enters_stack: bool = False


class Boundary(msgspec.Struct, frozen=True):
    """The steps one layer runs at one boundary, chosen from both sides'
    declarations. A layer runs the consumer's half of a boundary into one of
    its stages, and the producer's half of the boundary after its last stage;
    the neighbouring layer runs the other half of that one."""

    edge: EdgeDecl
    # The consumer's half: completing what the input owes, the add and the
    # norm, and the moves onto the rows it needs that come before the read.
    prepare: Optional[Callable] = None
    # The move onto the consumer's rows after prepare; None when there is none.
    input_move: Optional[Callable] = None
    # The fused kernels prepare tries first on a sum the input always owes.
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
        the axes it gathers itself as the rows its input is read on are."""
        need = self.edge.need
        return Layout(
            need.layout.sharded | (self.edge.residual_to.sharded & need.gathers_itself)
        )


def make_boundary(
    edge: EdgeDecl,
    *,
    fusions: Tuple["FusedMlpInput", ...] = (),
    carried_fusions: Tuple[Callable, ...] = (),
    force_layernorm_before_gather: bool = False,
    cp_moves: Optional[CpMoves] = None,
    enters_stack: bool = False,
) -> Boundary:
    """The consumer's half of ``edge``, chosen from the edge's declarations and
    the producer's update: what a value carries for a batch (a sum or a handoff
    its producer left) is completed first, trying ``carried_fusions``; a sum
    the producer always leaves as the layouts say, trying ``fusions``. Fused
    kernels run only for a plain add and a read that is the residual's norm
    and leaves the residual as it is; orders that add the residual before the
    sum completes only for a plain add. ``cp_moves`` for an edge that gathers
    over attention CP; ``enters_stack`` for the edge into the layer stack's
    first stage. The steps read only this edge's declarations, never what the
    producer chose for a batch."""
    update = edge.produced.update
    # A fused kernel runs the add and the norm itself.
    plain = update.adds_plainly and edge.need.read.norms_plainly
    step, fused, input_move = _select_input_steps(
        edge.produced,
        residual=edge.residual,
        residual_to=edge.residual_to,
        need=edge.need,
        update=update,
        fusions=fusions if plain else (),
        force_layernorm_before_gather=force_layernorm_before_gather,
        residual_joins_sum=edge.residual_joins_sum,
        cp_moves=cp_moves,
        enters_stack=enters_stack,
    )
    return Boundary(
        edge,
        prepare=partial(
            _consumer_step,
            step=step,
            carried_fusions=carried_fusions if plain else (),
            owes_by_construction=edge.produced.always_leaves,
        ),
        input_move=input_move,
        fused=fused,
    )


def make_output_boundary(
    edge: EdgeDecl, *, cp_moves: Optional[CpMoves] = None
) -> Boundary:
    """The producer's half of ``edge``, out of a layer's last stage onto the
    rows the layer hands on (``edge.need``), whose consumer runs in the next
    layer: the postprocess that moves the output there. ``cp_moves`` for an
    edge that returns across attention CP."""
    if edge.need.layout != edge.residual_to:
        raise NotImplementedError(f"{edge=}")
    returns_over_dp, output_move, completes_sum = _select_ffn_output_move(
        edge.produced,
        residual=edge.residual,
        to=edge.residual_to,
        cp_moves=cp_moves,
        update=edge.produced.update,
    )
    return Boundary(
        edge,
        output_move=None if returns_over_dp else output_move,
        output_move_completes_sum=completes_sum,
    )


def _select_boundary_steps(
    sides: DecoderLayerSides,
    *,
    fusions: Tuple["FusedMlpInput", ...] = (),
    force_layernorm_before_gather: bool = False,
    cp_moves: Optional[CpMoves] = None,
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
    out_of_ffn = make_output_boundary(edges.out_of_ffn, cp_moves=cp_moves)
    into_ffn = make_boundary(
        edges.into_ffn,
        fusions=fusions,
        force_layernorm_before_gather=force_layernorm_before_gather,
        cp_moves=cp_moves,
    )
    into_attention = make_boundary(
        edges.into_attention,
        carried_fusions=attention_fusions,
        force_layernorm_before_gather=force_layernorm_before_gather,
        cp_moves=cp_moves,
        enters_stack=enters_stack,
    )
    return BoundarySteps(
        attention=StageEntry(
            prepare=into_attention.prepare,
            input_rows=into_attention.input_rows,
            input_move=into_attention.input_move,
            handoff=attention_handoff,
        ),
        ffn=StageEntry(
            prepare=into_ffn.prepare,
            input_rows=into_ffn.input_rows,
            input_move=into_ffn.input_move,
            fused=into_ffn.fused,
        ),
        ffn_output=edges.out_of_ffn.produced,
        ffn_output_move=out_of_ffn.output_move,
        ffn_output_move_completes_sum=out_of_ffn.output_move_completes_sum,
        ffn_sum_is_movable=edges.out_of_ffn.produced.group is not None,
    )


def _select_input_steps(
    produced: StageOutput,
    *,
    residual: Layout,
    residual_to: Layout,
    need: StageInput,
    update: StageUpdate,
    fusions: Tuple[FusedMlpInput, ...],
    force_layernorm_before_gather: bool,
    residual_joins_sum: bool,
    cp_moves: Optional[CpMoves],
    enters_stack: bool,
) -> Tuple[Callable, Tuple[FusedMlpInput, ...], Optional[Callable]]:
    """The steps into a stage for a value that is complete or owes the sum its
    producer always leaves, the fused kernels they try first, and the move
    after them: complete that sum, move the residual to the rows it has while
    the stage runs, write the output into it with ``update`` and read the input,
    and bring the input onto the rows the stage needs: a gather over attention
    DP or CP, each rank's own slice, or after the read a gather over attention
    TP. A kernel in ``fusions`` is tried only when nothing is gathered or
    sliced, and only if it completes the sum the input owes. An update that is
    not a plain add runs only after the sum completes."""
    read = need.read
    # What the input owes by construction; a sum the producer leaves only for
    # some batches comes with the value and is completed before these steps.
    owes = produced.group if produced.always_leaves else None
    if produced.always_leaves and owes is None:
        raise NotImplementedError(f"{produced=}")
    gathered = produced.layout.sharded - need.layout.sharded - need.gathers_itself
    sliced = need.layout.sharded - produced.layout.sharded
    if sliced:
        # Each attention-TP rank takes its own slice: the reduce-scatter
        # completes the attention-TP sum and slices in one collective; a
        # complete value is only sliced.
        if (
            sliced != {TokenAxis.ATTN_TP_SCATTER}
            or gathered
            or owes not in (None, SumGroup.ATTN_TP)
            or residual_to != need.layout
            or residual not in (produced.layout, need.layout)
        ):
            raise NotImplementedError(f"{produced=} {residual=} {need=}")
        return (
            partial(
                _mlp_input_scatter if owes is SumGroup.ATTN_TP else _mlp_input_slice,
                scatters_residual=residual != residual_to,
                read=read,
                update=update,
            ),
            (),
            None,
        )
    if residual_to.sharded - produced.layout.sharded == {TokenAxis.ATTN_TP_SCATTER}:
        if residual == residual_to:
            # The residual stays on each rank's slice while the stage takes the
            # rows around it (MHC on an input-scattered batch).
            if gathered or owes is not SumGroup.ATTN_TP:
                raise NotImplementedError(f"{produced=} {residual=} {need=}")
            return (
                partial(_mlp_input_on_residual_shard, read=read, update=update),
                (),
                None,
            )
        # A reduce-scatter completes the TP sum onto each rank's slice, which
        # the stage takes: its group is the TP group without attention DP or CP.
        if (
            owes is not SumGroup.TP
            or residual.sharded
            or TokenAxis.ATTN_TP_SCATTER not in need.gathers_itself
            or residual_to.sharded != {TokenAxis.ATTN_TP_SCATTER}
        ):
            raise NotImplementedError(f"{produced=} {residual=} {need=}")
        return (
            partial(
                _read_input,
                layer_input=tp_reduce_scatter,
                enters_stack=enters_stack,
                read=read,
                update=update,
            ),
            (),
            None,
        )
    if gathered == {TokenAxis.ATTN_CP}:
        # Each CP rank completes its own chunk, then the CP moves gather them.
        if cp_moves is None:
            raise NotImplementedError(f"{produced=} {need=}")
        on_chunk, fused, _ = _select_input_steps(
            produced,
            residual=residual,
            residual_to=residual_to,
            need=StageInput(produced.layout, read=read),
            update=update,
            fusions=fusions,
            force_layernorm_before_gather=force_layernorm_before_gather,
            residual_joins_sum=residual_joins_sum,
            cp_moves=cp_moves,
            enters_stack=enters_stack,
        )
        return partial(cp_moves.gather, gather=on_chunk), fused, None
    if gathered == {TokenAxis.ATTN_TP_SCATTER}:
        # A complete input on each rank's slice, gathered over attention TP
        # once it is read.
        if (
            owes is not None
            or residual != residual_to
            or residual_to != produced.layout
        ):
            raise NotImplementedError(f"{produced=} {residual=} {need=}")
        return (
            partial(
                _read_input,
                layer_input=None,
                enters_stack=enters_stack,
                read=read,
                update=update,
            ),
            (),
            CommunicateSimpleFn._scattered_to_tp_attn_full,
        )
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
        if owes is None:
            if gathers_residual:
                raise NotImplementedError(f"{produced=} {residual=} {need=}")
            return (
                partial(
                    _read_input,
                    layer_input=None,
                    enters_stack=enters_stack,
                    read=read,
                    update=update,
                ),
                (),
                None,
            )
        if gathers_residual and residual_joins_sum:
            # Each rank adds its slice of the residual into its share of the
            # sum, so the all-reduce also brings the residual back to every row.
            if owes is not SumGroup.ATTN_TP or not update.adds_plainly:
                raise NotImplementedError(f"{produced=} {residual=} {need=}")
            return partial(_mlp_input_residual_into_sum, read=read), (), None
        # A sum over TP completes only on rows every TP rank holds: the TP group
        # spans attention DP and CP.
        if owes is not SumGroup.ATTN_TP and (
            owes is not SumGroup.TP or produced.layout.sharded
        ):
            raise NotImplementedError(f"{produced=} {need=}")
        fused = tuple(f for f in fusions if f.completes is owes)
        return (
            partial(
                _mlp_input_without_dp,
                gathers_residual=gathers_residual,
                fusions=tuple(f.run for f in fused),
                group=owes,
                read=read,
                update=update,
            ),
            fused,
            None,
        )
    if owes not in (None, SumGroup.ATTN_TP):
        raise NotImplementedError(f"{produced=} {need=}")
    owes_attention_tp = owes is SumGroup.ATTN_TP
    # The partial order adds the residual on attention-TP rank 0 before the DP
    # gather's collective completes that sum, which only a plain residual add
    # allows.
    # Over attention DP and CP, the DP gather puts each CP rank's shard in its
    # DP group's slot, so the one DP sum gathers both axes.
    places_cp_shards = TokenAxis.ATTN_CP in gathered
    if (
        owes_attention_tp
        and not force_layernorm_before_gather
        and update.adds_plainly
        and read.norms_plainly
    ):
        return (
            partial(
                _mlp_input_dp_partial,
                gathers_residual=gathers_residual,
                places_cp_shards=places_cp_shards,
                read=read,
            ),
            (),
            None,
        )
    return (
        partial(
            _mlp_input_dp_replicate,
            gathers_residual=gathers_residual,
            reduces_attention_tp=owes_attention_tp,
            places_cp_shards=places_cp_shards,
            read=read,
            update=update,
        ),
        (),
        None,
    )


def _select_ffn_output_move(
    produced: StageOutput,
    *,
    residual: Layout,
    to: Layout,
    cp_moves: Optional[CpMoves] = None,
    update: StageUpdate = ADD,
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
            return False, partial(pair._gather, update=update), False
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
                update=update,
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
