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

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import msgspec

from sglang.srt.layers.layer_boundary.contracts import (
    CpMoves,
    EdgeContract,
    EntryPath,
    FfnInputFusion,
    InputContract,
    OutputContract,
)
from sglang.srt.layers.layer_boundary.layout import (
    Layout,
    SumGroup,
    TokenAxis,
    _cp_gathers_over_attn_cp,
    _same_ranks,
    _sum_group,
    token_axis_sizes,
)
from sglang.srt.layers.layer_boundary.ops import (
    attn_cp_reduce_scatter_output,
    attn_cp_take_back_output,
    attn_tp_gather_input,
    attn_tp_slice_output,
    dp_cp_take_back_output,
    keep_output,
    moe_cp_take_back_output,
    move_rows,
    residual_slice_output,
    tp_reduce_scatter,
    tp_slice,
    update_attn_tp_gather_output,
)
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.prepare import (
    _attn_tp_reduce_scatter_update_read,
    _attn_tp_slice_update_read,
    _dispatch_by_update,
    _dp_gather_sum_read,
    _move_before_read,
    _reduce_update_read,
    _reduce_update_read_dp_gather,
    _run_entry,
    _then_attn_cp_gather,
    _then_moe_cp_gather,
    _tp_reduce_scatter_update_read_gather,
    _tp_sum_with_residual_read,
    _update_read,
)
from sglang.srt.layers.layer_boundary.residual import ResidualReadout
from sglang.srt.runtime_context import get_parallel


def tbo_split_moves(layer_input_rows: Layout) -> Tuple[Callable, Callable]:
    """The moves around the two-batch-overlap split, which cuts the attention's
    rows: from the rows the first overlapped layer takes to the attention's,
    and back again for each half."""
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=token_axis_sizes()
    )

    if layer_input_rows == attention:
        return keep_output, keep_output
    if layer_input_rows.sharded - attention.sharded == {TokenAxis.ATTN_TP}:
        # Each rank's slice: write the residual in and gather over attention
        # TP, then take the slice of each half.
        return update_attn_tp_gather_output, attn_tp_slice_output
    raise NotImplementedError(f"{layer_input_rows=}")


def _cp_moves() -> CpMoves:
    """DSA and MLA CP gather equal shards over the attention-CP group and can
    complete a sum over it. GQA prefill CP gathers blocks padded to the longest
    over the MoE-CP group and takes back only a complete output."""
    if _cp_gathers_over_attn_cp():
        return CpMoves(
            gather=_then_attn_cp_gather,
            take_back=attn_cp_take_back_output,
            reduce_scatter=attn_cp_reduce_scatter_output,
            reduce_scatter_group=lambda: get_parallel().attn_cp_group,
        )
    return CpMoves(
        gather=_then_moe_cp_gather,
        take_back=moe_cp_take_back_output,
    )


class ExitMove(NamedTuple):
    """The producer's half of a boundary, which a layer runs after its last
    stage; the neighbouring layer runs the consumer's half."""

    # Moves the output onto the rows the layer hands on; None when it goes
    # back over attention DP, whose step the FFN exit and finish choose per
    # batch.
    output_move: Optional[Callable] = None
    # Whether output_move also completes the sum the producer leaves.
    output_move_completes_sum: bool = False
    returns_over_dp: bool = False


def input_rows(edge: EdgeContract) -> Layout:
    """The rows the consumer is handed: what it needs, still sharded over the
    axes it gathers itself as the rows its input is read on are."""
    return Layout(
        edge.need.layout.sharded
        | (edge.residual_to.sharded & edge.need.gathered_by_compute)
    )


def _capture_move(edge: EdgeContract) -> Tuple[Optional[Callable], bool]:
    """The move of the updated residual back onto the producer's rows for aux
    capture, and whether it allocates: returning through a gather allocates;
    returning through a cut aliases."""
    if edge.residual_to == edge.produced.layout:
        return None, False
    move = partial(move_rows, rows=edge.residual_to, to=edge.produced.layout)
    return move, bool(edge.residual_to.sharded - edge.produced.layout.sharded)


@dataclass(frozen=True)
class _TransformedRead:
    """The read of an output its producer declared an OutputTransform on: the
    transform runs on the complete sum, then the update and ``inner``. Not a
    plain norm, so no fused add + norm and no residual-first order takes it."""

    inner: ResidualReadout
    transform: OutputTransform

    is_plain_norm = False

    @property
    def reads_before_dp_gather(self):
        return self.inner.reads_before_dp_gather

    def init_residual(self, hidden_states):
        return self.inner.init_residual(hidden_states)

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        return self.inner.read(residual, norm, quant_format, post_residual_addition)

    def update_and_read(
        self,
        update,
        hidden_states,
        residual,
        norm,
        quant_format="",
        post_residual_addition=None,
    ):
        if hidden_states.shape[0] != 0:
            hidden_states = self.transform.apply(hidden_states)
        return self.inner.update_and_read(
            update,
            hidden_states,
            residual,
            norm,
            quant_format=quant_format,
            post_residual_addition=post_residual_addition,
        )


def bind_entry(
    edge: EdgeContract,
    *,
    fusions: Tuple["FfnInputFusion", ...] = (),
    carried_fusions: Tuple[Callable, ...] = (),
    cp_moves: Optional[CpMoves] = None,
    enters_stack: bool = False,
    attn_input_adapter: Optional[Callable] = None,
) -> EntryPath:
    """Bind the consumer half of an edge at construction time.

    Args:
        edge: Producer/consumer contracts, residual rows and update capabilities.
        fusions: Ordered candidates for completing a declared sum with the read.
        carried_fusions: Ordered candidates accepting dynamically carried work.
        cp_moves: Context-parallel strategy operations for this edge, if needed.
        enters_stack: Whether the read must initialize the stack's residual.
        attn_input_adapter: Hands an attention its input once it is on its rows.

    Returns:
        An EntryPath whose prepare accepts the actual update from the stream.
        Multiple update capabilities select among preconstructed paths; a single
        capability needs no runtime dispatcher.
    """
    if edge.produced.transform is not None:
        # An attention's transform, run once its sum is complete.
        edge = msgspec.structs.replace(
            edge,
            need=msgspec.structs.replace(
                edge.need,
                read=_TransformedRead(edge.need.read, edge.produced.transform),
            ),
            residual_joins_sum=False,
        )
    capabilities = edge.arriving_plain_add
    if not capabilities:
        if edge.produced.update is None:
            raise ValueError("an arrival must declare its update capabilities")
        capabilities = (edge.produced.update.is_plain_add,)
    paths = {
        capability: _bind_entry_path(
            edge,
            is_plain_add=capability,
            fusions=fusions,
            carried_fusions=carried_fusions,
            cp_moves=cp_moves,
            enters_stack=enters_stack,
            attn_input_adapter=attn_input_adapter,
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
            _dispatch_by_update,
            paths={capability: path.prepare for capability, path in paths.items()},
        ),
    )


def _bind_entry_path(
    edge: EdgeContract,
    *,
    is_plain_add: bool,
    fusions: Tuple["FfnInputFusion", ...] = (),
    carried_fusions: Tuple[Callable, ...] = (),
    cp_moves: Optional[CpMoves] = None,
    enters_stack: bool = False,
    attn_input_adapter: Optional[Callable] = None,
) -> EntryPath:
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
    plain = is_plain_add and edge.need.read.is_plain_norm
    step, input_move = _select_entry_step(
        edge.produced,
        residual=edge.residual,
        residual_to=edge.residual_to,
        need=edge.need,
        is_plain_add=is_plain_add,
        fusions=fusions if plain else (),
        residual_joins_sum=edge.residual_joins_sum,
        cp_moves=cp_moves,
        enters_stack=enters_stack,
    )
    completed_step = None
    if edge.produced.always_partial:
        completed_step, _ = _select_entry_step(
            msgspec.structs.replace(edge.produced, always_partial=False),
            residual=edge.residual,
            residual_to=edge.residual_to,
            need=edge.need,
            is_plain_add=is_plain_add,
            fusions=(),
            residual_joins_sum=False,
            cp_moves=cp_moves,
            enters_stack=enters_stack,
        )
    # A written stream entering the first physical layer must not run enter
    # again (e.g. MHC expansion). Both alternatives are bound at construction.
    written_step = None
    if enters_stack:
        written_step, _ = _select_entry_step(
            edge.produced,
            residual=edge.residual,
            residual_to=edge.residual_to,
            need=edge.need,
            is_plain_add=is_plain_add,
            fusions=fusions if plain else (),
            residual_joins_sum=edge.residual_joins_sum,
            cp_moves=cp_moves,
            enters_stack=False,
        )
    preserves_residual = None
    if (
        edge.produced.always_partial
        and edge.produced.group is SumGroup.ATTN_TP
        and isinstance(step, partial)
        and step.func is _reduce_update_read
        and not step.keywords["gathers_residual"]
    ):
        selected = step.keywords["fusions"]
        if selected:
            # Only the first reachable candidate can certify ownership; an
            # earlier custom candidate could otherwise mutate the residual.
            preserves_residual = next(
                f.preserves_residual for f in fusions if f.run == selected[0]
            )
    declared_sum = edge.produced.group if edge.produced.always_partial else None
    capture_move, capture_move_allocates = _capture_move(edge)
    return EntryPath(
        prepare=partial(
            _run_entry,
            step=step,
            is_plain_add=is_plain_add,
            carried_fusions=carried_fusions if plain else (),
            expected_sum=declared_sum,
            completed_step=completed_step,
            written_step=written_step,
        ),
        input_rows=input_rows(edge),
        input_move=input_move,
        attn_input_adapter=attn_input_adapter,
        capture_move=capture_move,
        capture_move_allocates=capture_move_allocates,
        declared_sum=declared_sum,
        preserves_residual=preserves_residual,
    )


def bind_exit(
    edge: EdgeContract,
    *,
    cp_moves: Optional[CpMoves] = None,
    attn_tp_gather: Optional[Callable] = None,
) -> ExitMove:
    """Bind the producer half of a layer or branch exit.

    Args:
        edge: Output contract and destination/residual rows. Deferred updates
            must outlive the producer; pipeline handoffs require a plain add
            or a residual already written by the producer.
        cp_moves: Context-parallel return operations, when the edge needs them.
        attn_tp_gather: The stage's implementation of a gather over attention
            TP, tried before the default one.

    Returns:
        An ExitMove with fixed transport or a marker for batch-dependent DP
        transport. The receiver binds its read independently.
    """
    update = edge.produced.update
    if not getattr(update, "applied_at_exit", False):
        if not getattr(update, "outlives_layer", False):
            raise NotImplementedError(
                "a deferred update must guarantee its lifetime across layers"
            )
        if not update.is_plain_add and get_parallel().pp_size > 1:
            raise NotImplementedError(
                "pipeline boundaries require a plain add or a producer-written residual"
            )
    if edge.need.layout != edge.residual_to:
        raise NotImplementedError(f"{edge=}")
    return _select_exit_move(
        edge.produced,
        residual=edge.residual,
        to=edge.residual_to,
        cp_moves=cp_moves,
        attn_tp_gather=attn_tp_gather,
    )


def _read_fusions(read, owes, is_plain_add, *, scatters):
    """The kernels a read supplies that complete ``owes`` on the rows this
    entry completes it on. They take the residual add themselves, so only a
    plain one."""
    if not is_plain_add:
        return ()
    return tuple(
        f.run
        for f in getattr(read, "completing_fusions", ())
        if f.completes is owes and f.scatters == scatters
    )


def _select_entry_step(
    produced: OutputContract,
    *,
    residual: Layout,
    residual_to: Layout,
    need: InputContract,
    is_plain_add: bool,
    fusions: Tuple[FfnInputFusion, ...],
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
    owes = produced.group if produced.always_partial else None
    if produced.always_partial and owes is None:
        raise NotImplementedError(f"{produced=}")
    gathered = produced.layout.sharded - need.layout.sharded - need.gathered_by_compute
    sliced = need.layout.sharded - produced.layout.sharded
    if residual_to.sharded - produced.layout.sharded == {TokenAxis.ATTN_CP}:
        # A head-parallel mixer consumes full context, but its residual and
        # readout coefficients remain on the CP shard. Complete the producer's
        # sum onto that shard before the (possibly nonlinear) update/read.
        if (
            residual != residual_to
            or cp_moves is None
            or owes not in (None, SumGroup.ATTN_CP)
            or (owes is not None and cp_moves.reduce_scatter is None)
        ):
            raise NotImplementedError(f"{produced=} {residual=} {need=}")
        local = msgspec.structs.replace(
            produced, layout=residual_to, group=None, always_partial=False
        )
        read_local, input_move = _select_entry_step(
            local,
            residual=residual,
            residual_to=residual_to,
            need=need,
            is_plain_add=is_plain_add,
            fusions=(),
            residual_joins_sum=False,
            cp_moves=cp_moves,
            enters_stack=enters_stack,
        )
        return partial(
            _move_before_read,
            move=cp_moves.reduce_scatter if owes is not None else cp_moves.take_back,
            read=read_local,
        ), input_move
    if sliced:
        # Each attention-TP rank takes its own slice: the reduce-scatter
        # completes the attention-TP sum and slices in one collective; a
        # complete value is only sliced.
        if (
            sliced != {TokenAxis.ATTN_TP}
            or gathered
            or owes not in (None, SumGroup.ATTN_TP)
            or residual_to != need.layout
            or residual not in (produced.layout, need.layout)
        ):
            raise NotImplementedError(f"{produced=} {residual=} {need=}")
        if owes is SumGroup.ATTN_TP:
            step = partial(
                _attn_tp_reduce_scatter_update_read,
                read_fusions=_read_fusions(read, owes, is_plain_add, scatters=True),
            )
        else:
            step = _attn_tp_slice_update_read
        return (
            partial(step, scatters_residual=residual != residual_to, read=read),
            None,
        )
    if residual_to.sharded - produced.layout.sharded == {TokenAxis.ATTN_TP}:
        if residual == residual_to:
            # The residual stays on each rank's slice while the stage takes the
            # rows around it (MHC on an input-scattered batch).
            if gathered or owes not in (None, SumGroup.ATTN_TP):
                raise NotImplementedError(f"{produced=} {residual=} {need=}")
            return (
                partial(
                    _tp_reduce_scatter_update_read_gather,
                    read=read,
                    reduces=owes is not None,
                ),
                None,
            )
        # A reduce-scatter completes the TP sum onto each rank's slice, which
        # the stage takes: its group is the TP group without attention DP or CP.
        if (
            owes not in (None, SumGroup.TP)
            or residual.sharded
            or TokenAxis.ATTN_TP not in need.gathered_by_compute
            or residual_to.sharded != {TokenAxis.ATTN_TP}
        ):
            raise NotImplementedError(f"{produced=} {residual=} {need=}")
        return (
            partial(
                _update_read,
                pre_move=tp_reduce_scatter if owes is not None else tp_slice,
                enters_stack=enters_stack,
                read=read,
            ),
            None,
        )
    if gathered == {TokenAxis.ATTN_CP}:
        # Each CP rank completes its own chunk, then the CP moves gather them.
        if cp_moves is None:
            raise NotImplementedError(f"{produced=} {need=}")
        on_chunk, _ = _select_entry_step(
            produced,
            residual=residual,
            residual_to=residual_to,
            need=InputContract(produced.layout, read=read),
            is_plain_add=is_plain_add,
            fusions=fusions,
            residual_joins_sum=residual_joins_sum,
            cp_moves=cp_moves,
            enters_stack=enters_stack,
        )
        return (partial(cp_moves.gather, gather=on_chunk), None)
    if gathered == {TokenAxis.ATTN_TP}:
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
                _update_read,
                pre_move=None,
                enters_stack=enters_stack,
                read=read,
            ),
            attn_tp_gather_input,
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
        not in (frozenset(), {TokenAxis.ATTN_TP})
    ):
        raise NotImplementedError(f"{produced=} {residual=} {need=}")
    # A residual arriving on each rank's slice is gathered back first.
    gathers_residual = residual != residual_to
    if not gathered:
        if owes is None:
            if gathers_residual:
                return (
                    partial(
                        _reduce_update_read,
                        gathers_residual=True,
                        fusions=(),
                        group=None,
                        read=read,
                    ),
                    None,
                )
            return (
                partial(
                    _update_read,
                    pre_move=None,
                    enters_stack=enters_stack,
                    read=read,
                ),
                None,
            )
        if gathers_residual and residual_joins_sum:
            # Each rank adds its slice of the residual into its share of the
            # sum, so the all-reduce also brings the residual back to every row.
            if owes is not SumGroup.ATTN_TP or not is_plain_add:
                raise NotImplementedError(f"{produced=} {residual=} {need=}")
            return (partial(_tp_sum_with_residual_read, read=read), None)
        # A sum over TP completes only on rows every TP rank holds: the TP group
        # spans attention DP and CP.
        if owes not in (SumGroup.ATTN_TP, SumGroup.ATTN_CP) and (
            owes is not SumGroup.TP or produced.layout.sharded
        ):
            raise NotImplementedError(f"{produced=} {need=}")
        fused = tuple(f for f in fusions if f.completes is owes)
        read_fused = _read_fusions(read, owes, is_plain_add, scatters=False)
        return (
            partial(
                _reduce_update_read,
                gathers_residual=gathers_residual,
                fusions=tuple(f.run for f in fused),
                read_fusions=read_fused,
                group=owes,
                read=read,
            ),
            None,
        )
    if owes not in (None, SumGroup.ATTN_TP, SumGroup.ATTN_CP):
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
        and not read.reads_before_dp_gather
        and is_plain_add
        and read.is_plain_norm
    ):
        return (
            partial(
                _dp_gather_sum_read,
                gathers_residual=gathers_residual,
                places_cp_shards=places_cp_shards,
                read=read,
            ),
            None,
        )
    return (
        partial(
            _reduce_update_read_dp_gather,
            gathers_residual=gathers_residual,
            reduces_attention_tp=owes is not None,
            group=owes,
            places_cp_shards=places_cp_shards,
            read=read,
        ),
        None,
    )


def _select_exit_move(
    produced: OutputContract,
    *,
    residual: Layout,
    to: Layout,
    cp_moves: Optional[CpMoves] = None,
    attn_tp_gather: Optional[Callable] = None,
) -> ExitMove:
    """How the FFN output reaches the rows the layer hands on: by undoing the
    attention-DP gather (the FFN exit and finish run that step), or else the
    move that takes it there, None when there is none to choose here; and
    whether that move also completes the sum the FFN leaves."""

    update = produced.update
    if produced.layout == residual:
        if to == residual:
            return ExitMove(keep_output)
        if to.sharded == residual.sharded - {TokenAxis.ATTN_TP}:
            # Each rank's slice back to the attention's rows: write the output
            # into the residual, then gather over attention TP.
            return ExitMove(
                partial(
                    update_attn_tp_gather_output,
                    update=update,
                    gather=attn_tp_gather,
                )
            )
        raise NotImplementedError(f"{produced=} {residual=} {to=}")
    returned = residual.sharded - produced.layout.sharded
    if returned == {TokenAxis.ATTN_TP} and to in (residual, produced.layout):
        # The residual stays on each rank's slice (MHC on an input-scattered
        # batch): a reduce-scatter onto the slice completes the sum the FFN
        # leaves; a complete output is only sliced.
        if not produced.may_reduce_scatter:
            return ExitMove(
                partial(
                    residual_slice_output,
                    sums=False,
                    gathers_back=to != residual,
                    update=update,
                )
            )
        return ExitMove(
            partial(
                residual_slice_output,
                sums=True,
                gathers_back=to != residual,
                update=update,
            ),
            output_move_completes_sum=True,
        )
    if to != residual or not produced.layout.sharded <= residual.sharded:
        raise NotImplementedError(f"{produced=} {residual=} {to=}")
    if returned == {TokenAxis.ATTN_CP}:
        if cp_moves is None:
            raise NotImplementedError(f"{produced=} {residual=} {to=}")
        if not produced.may_reduce_scatter:
            # A complete output: this rank's block of it, nothing summed.
            return ExitMove(cp_moves.take_back)
        if cp_moves.reduce_scatter is None:
            raise NotImplementedError(f"{produced=} {residual=} {to=}")
        # Only a take-back over the FFN's sum group can complete its reduction.
        # With attention TP > 1, the full TP sum spans more ranks than CP:
        # let the exit complete that sum, then take this rank's CP rows.
        if not _same_ranks(_sum_group(produced.group), cp_moves.reduce_scatter_group()):
            if produced.always_partial:
                raise NotImplementedError(f"{produced=} {residual=} {to=}")
            return ExitMove(cp_moves.take_back)
        return ExitMove(cp_moves.reduce_scatter, output_move_completes_sum=True)
    if returned == {TokenAxis.ATTN_DP, TokenAxis.ATTN_CP}:
        # This rank's CP shard, from where the DP gather put it.
        return ExitMove(dp_cp_take_back_output)
    if returned != {TokenAxis.ATTN_DP}:
        raise NotImplementedError(f"{produced=} {residual=} {to=}")
    return ExitMove(returns_over_dp=True)
