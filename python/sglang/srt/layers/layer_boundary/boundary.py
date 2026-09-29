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

from functools import partial
from typing import Callable, Optional, Tuple

import msgspec

from sglang.srt.layers.layer_boundary.contracts import (
    CpMoves,
    EdgeDecl,
    FusedMlpInput,
    StageInput,
    StageOutput,
)
from sglang.srt.layers.layer_boundary.layout import (
    Layout,
    SumGroup,
    TokenAxis,
    _gathers_over_attention_cp,
    _same_ranks,
    _sum_group,
    token_axis_sizes,
)
from sglang.srt.layers.layer_boundary.ops import (
    gather_attention_tp,
    identity_output,
    move_rows,
    output_on_residual_shard,
    reduce_scatter_over_cp,
    scatter_moe_cp_output,
    scatter_output,
    take_back_attention_cp_shard,
    take_back_cp_shard,
    tp_reduce_scatter,
    tp_slice,
    update_and_gather,
)
from sglang.srt.layers.layer_boundary.prepare import (
    _consumer_step,
    _dispatch_consumer,
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
)
from sglang.srt.runtime_context import get_parallel


def tbo_split_moves(layer_input_rows: Layout) -> Tuple[Callable, Callable]:
    """The moves around the two-batch-overlap split, which cuts the attention's
    rows: from the rows the first overlapped layer takes to the attention's,
    and back again for each half."""
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=token_axis_sizes()
    )

    if layer_input_rows == attention:
        return identity_output, identity_output
    if layer_input_rows.sharded - attention.sharded == {TokenAxis.ATTN_TP_SCATTER}:
        # Each rank's slice: write the residual in and gather over attention
        # TP, then take the slice of each half.
        return update_and_gather, scatter_output
    raise NotImplementedError(f"{layer_input_rows=}")


def _cp_moves() -> CpMoves:
    """DSA and MLA CP gather equal shards over the attention-CP group and can
    complete a sum over it. GQA prefill CP gathers blocks padded to the longest
    over the MoE-CP group and takes back only a complete output."""
    if _gathers_over_attention_cp():
        return CpMoves(
            gather=_mlp_input_gather_attention_cp,
            take_back=take_back_attention_cp_shard,
            reduce_scatter=reduce_scatter_over_cp,
            reduce_scatter_group=lambda: get_parallel().attn_cp_group,
        )
    return CpMoves(
        gather=_mlp_input_gather_moe_cp,
        take_back=scatter_moe_cp_output,
    )


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
    # The producer's half: the postprocess that moves the output onto the rows
    # the layer hands on; None when it goes back over attention DP, whose step
    # the FFN exit and postprocess choose per batch.
    output_move: Optional[Callable] = None
    # Whether output_move also completes the sum the producer leaves.
    output_move_completes_sum: bool = False
    returns_over_dp: bool = False
    preserves_residual: Optional[Callable] = None

    @property
    def input_rows(self) -> Layout:
        """The rows the consumer is handed: what it needs, still sharded over
        the axes it gathers itself as the rows its input is read on are."""
        return input_rows(self.edge)

    @property
    def capture_move_allocates(self) -> bool:
        """Returning through a gather allocates; returning through a cut aliases."""
        return bool(self.edge.residual_to.sharded - self.edge.produced.layout.sharded)

    @property
    def capture_move(self) -> Optional[Callable]:
        if self.edge.residual_to == self.edge.produced.layout:
            return None
        return partial(
            move_rows, rows=self.edge.residual_to, to=self.edge.produced.layout
        )


def input_rows(edge: EdgeDecl) -> Layout:
    """Consumer rows, retaining the axes it gathers inside its computation."""
    return Layout(
        edge.need.layout.sharded | (edge.residual_to.sharded & edge.need.gathers_itself)
    )


def make_boundary(
    edge: EdgeDecl,
    *,
    fusions: Tuple["FusedMlpInput", ...] = (),
    carried_fusions: Tuple[Callable, ...] = (),
    cp_moves: Optional[CpMoves] = None,
    enters_stack: bool = False,
) -> Boundary:
    """Bind the consumer half of an edge at construction time.

    Args:
        edge: Producer/consumer contracts, residual rows and update capabilities.
        fusions: Ordered candidates for completing a declared sum with the read.
        carried_fusions: Ordered candidates accepting dynamically carried work.
        cp_moves: Context-parallel strategy operations for this edge, if needed.
        enters_stack: Whether the read must initialize the stack's residual.

    Returns:
        A Boundary whose prepare accepts the actual update from the stream.
        Multiple update capabilities select among preconstructed paths; a single
        capability needs no runtime dispatcher.
    """
    capabilities = edge.update_capabilities
    if not capabilities:
        if edge.produced.update is None:
            raise ValueError("an arrival must declare its update capabilities")
        capabilities = (edge.produced.update.adds_plainly,)
    paths = {
        capability: _bind_consumer(
            edge,
            adds_plainly=capability,
            fusions=fusions,
            carried_fusions=carried_fusions,
            cp_moves=cp_moves,
            enters_stack=enters_stack,
        )
        for capability in capabilities
    }
    first = next(iter(paths.values()))
    if len(paths) == 1:
        return first
    if any(path.input_move != first.input_move for path in paths.values()):
        raise NotImplementedError("update capabilities require different input moves")
    return msgspec.structs.replace(
        first,
        preserves_residual=None,
        prepare=partial(
            _dispatch_consumer,
            paths={capability: path.prepare for capability, path in paths.items()},
        ),
    )


def _bind_consumer(
    edge: EdgeDecl,
    *,
    adds_plainly: bool,
    fusions: Tuple["FusedMlpInput", ...] = (),
    carried_fusions: Tuple[Callable, ...] = (),
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
    # A fused kernel runs the add and the norm itself.
    plain = adds_plainly and edge.need.read.norms_plainly
    step, input_move = _select_input_steps(
        edge.produced,
        residual=edge.residual,
        residual_to=edge.residual_to,
        need=edge.need,
        adds_plainly=adds_plainly,
        fusions=fusions if plain else (),
        residual_joins_sum=edge.residual_joins_sum,
        cp_moves=cp_moves,
        enters_stack=enters_stack,
    )
    completed_step = None
    if edge.produced.always_leaves:
        completed_step, _ = _select_input_steps(
            msgspec.structs.replace(edge.produced, always_leaves=False),
            residual=edge.residual,
            residual_to=edge.residual_to,
            need=edge.need,
            adds_plainly=adds_plainly,
            fusions=(),
            residual_joins_sum=False,
            cp_moves=cp_moves,
            enters_stack=enters_stack,
        )
    # A written stream entering the first physical layer must not run enter
    # again (e.g. MHC expansion). Both alternatives are bound at construction.
    written_step = None
    if enters_stack:
        written_step, _ = _select_input_steps(
            edge.produced,
            residual=edge.residual,
            residual_to=edge.residual_to,
            need=edge.need,
            adds_plainly=adds_plainly,
            fusions=fusions if plain else (),
            residual_joins_sum=edge.residual_joins_sum,
            cp_moves=cp_moves,
            enters_stack=False,
        )
    preserves_residual = None
    if (
        edge.produced.always_leaves
        and edge.produced.group is SumGroup.ATTN_TP
        and isinstance(step, partial)
        and step.func is _mlp_input_without_dp
        and not step.keywords["gathers_residual"]
    ):
        selected = step.keywords["fusions"]
        if selected:
            # Only the first reachable candidate can certify ownership; an
            # earlier custom candidate could otherwise mutate the residual.
            preserves_residual = next(
                f.preserves_residual for f in fusions if f.run == selected[0]
            )
    return Boundary(
        edge,
        preserves_residual=preserves_residual,
        prepare=partial(
            _consumer_step,
            step=step,
            adds_plainly=adds_plainly,
            carried_fusions=carried_fusions if plain else (),
            expected_sum=edge.produced.group if edge.produced.always_leaves else None,
            completed_step=completed_step,
            written_step=written_step,
        ),
        input_move=input_move,
    )


def make_output_boundary(
    edge: EdgeDecl, *, cp_moves: Optional[CpMoves] = None
) -> Boundary:
    """Bind the producer half of a layer/branch handoff.

    Args:
        edge: Output contract and destination/residual rows. Deferred updates
            must outlive the producer; pipeline handoffs require a plain add
            or a residual already written by the producer.
        cp_moves: Context-parallel return operations, when the edge needs them.

    Returns:
        A Boundary with fixed transport or a marker for batch-dependent DP
        transport. The receiver binds its read independently.
    """
    update = edge.produced.update
    if not getattr(update, "at_producer", False):
        if not getattr(update, "can_defer_across_layers", False):
            raise NotImplementedError(
                "a deferred update must guarantee its lifetime across layers"
            )
        if not update.adds_plainly and get_parallel().pp_size > 1:
            raise NotImplementedError(
                "pipeline boundaries require a plain add or a producer-written residual"
            )
    if edge.need.layout != edge.residual_to:
        raise NotImplementedError(f"{edge=}")
    returns_over_dp, output_move, completes_sum = _select_ffn_output_move(
        edge.produced,
        residual=edge.residual,
        to=edge.residual_to,
        cp_moves=cp_moves,
    )
    return Boundary(
        edge,
        output_move=output_move,
        returns_over_dp=returns_over_dp,
        output_move_completes_sum=completes_sum,
    )


def _select_input_steps(
    produced: StageOutput,
    *,
    residual: Layout,
    residual_to: Layout,
    need: StageInput,
    adds_plainly: bool,
    fusions: Tuple[FusedMlpInput, ...],
    residual_joins_sum: bool,
    cp_moves: Optional[CpMoves],
    enters_stack: bool,
) -> Tuple[Callable, Optional[Callable]]:
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
            ),
            None,
        )
    if residual_to.sharded - produced.layout.sharded == {TokenAxis.ATTN_TP_SCATTER}:
        if residual == residual_to:
            # The residual stays on each rank's slice while the stage takes the
            # rows around it (MHC on an input-scattered batch).
            if gathered or owes not in (None, SumGroup.ATTN_TP):
                raise NotImplementedError(f"{produced=} {residual=} {need=}")
            return (
                partial(
                    _mlp_input_on_residual_shard, read=read, reduces=owes is not None
                ),
                None,
            )
        # A reduce-scatter completes the TP sum onto each rank's slice, which
        # the stage takes: its group is the TP group without attention DP or CP.
        if (
            owes not in (None, SumGroup.TP)
            or residual.sharded
            or TokenAxis.ATTN_TP_SCATTER not in need.gathers_itself
            or residual_to.sharded != {TokenAxis.ATTN_TP_SCATTER}
        ):
            raise NotImplementedError(f"{produced=} {residual=} {need=}")
        return (
            partial(
                _read_input,
                layer_input=tp_reduce_scatter if owes is not None else tp_slice,
                enters_stack=enters_stack,
                read=read,
            ),
            None,
        )
    if gathered == {TokenAxis.ATTN_CP}:
        # Each CP rank completes its own chunk, then the CP moves gather them.
        if cp_moves is None:
            raise NotImplementedError(f"{produced=} {need=}")
        on_chunk, _ = _select_input_steps(
            produced,
            residual=residual,
            residual_to=residual_to,
            need=StageInput(produced.layout, read=read),
            adds_plainly=adds_plainly,
            fusions=fusions,
            residual_joins_sum=residual_joins_sum,
            cp_moves=cp_moves,
            enters_stack=enters_stack,
        )
        return (partial(cp_moves.gather, gather=on_chunk), None)
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
            ),
            gather_attention_tp,
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
                return (
                    partial(
                        _mlp_input_without_dp,
                        gathers_residual=True,
                        fusions=(),
                        group=None,
                        read=read,
                    ),
                    None,
                )
            return (
                partial(
                    _read_input,
                    layer_input=None,
                    enters_stack=enters_stack,
                    read=read,
                ),
                None,
            )
        if gathers_residual and residual_joins_sum:
            # Each rank adds its slice of the residual into its share of the
            # sum, so the all-reduce also brings the residual back to every row.
            if owes is not SumGroup.ATTN_TP or not adds_plainly:
                raise NotImplementedError(f"{produced=} {residual=} {need=}")
            return (partial(_mlp_input_residual_into_sum, read=read), None)
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
            ),
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
        and not read.before_gather
        and adds_plainly
        and read.norms_plainly
    ):
        return (
            partial(
                _mlp_input_dp_partial,
                gathers_residual=gathers_residual,
                places_cp_shards=places_cp_shards,
                read=read,
            ),
            None,
        )
    return (
        partial(
            _mlp_input_dp_replicate,
            gathers_residual=gathers_residual,
            reduces_attention_tp=owes_attention_tp,
            places_cp_shards=places_cp_shards,
            read=read,
        ),
        None,
    )


def _select_ffn_output_move(
    produced: StageOutput,
    *,
    residual: Layout,
    to: Layout,
    cp_moves: Optional[CpMoves] = None,
) -> Tuple[bool, Optional[Callable], bool]:
    """How the FFN output reaches the rows the layer hands on: whether it goes
    back by undoing the attention-DP gather (the FFN exit and postprocess run
    that step), or else the postprocess that moves it, None when there is none
    to choose here; and whether that move also completes the sum the FFN
    leaves."""

    update = produced.update
    if produced.layout == residual:
        if to == residual:
            return False, identity_output, False
        if to.sharded == residual.sharded - {TokenAxis.ATTN_TP_SCATTER}:
            # Each rank's slice back to the attention's rows: write the output
            # into the residual, then gather over attention TP.
            return False, partial(update_and_gather, update=update), False
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
                output_on_residual_shard,
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
        return False, take_back_cp_shard, False
    if returned != {TokenAxis.ATTN_DP}:
        raise NotImplementedError(f"{produced=} {residual=} {to=}")
    return True, None, False
