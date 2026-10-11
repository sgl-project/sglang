"""Construct one computing stage and its transport from a predecessor contract."""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Callable, Mapping, NamedTuple, Optional

from sglang.srt.environ import envs
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.layer_boundary.adapters.overlap import (
    resolve_exit_rows,
    tbo_exit_rows,
)
from sglang.srt.layers.layer_boundary.boundary import _cp_moves
from sglang.srt.layers.layer_boundary.construction import (
    BatchVariant,
    _bind_stage,
)
from sglang.srt.layers.layer_boundary.contracts import (
    EdgeContract,
    ExitRows,
    InputContract,
    OutputContract,
    ProducerReduction,
    StageContract,
    StageKind,
)
from sglang.srt.layers.layer_boundary.facts import facts_of
from sglang.srt.layers.layer_boundary.layout import (
    Layout,
    SumGroup,
    TokenAxis,
    _cp_gathers_over_attn_cp,
    _prefill_cp_shards_tokens,
    batches_are_unpadded,
    input_scattered_configured,
    is_dense_ffn_fully_dp,
    moe_gathers_over_moe_cp,
    token_axis_sizes,
)
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.residual import (
    ResidualReadout,
    ResidualUpdate,
)
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NORM_QUANT_READOUT,
    NORM_READOUT,
    PLAIN_ADD,
)
from sglang.srt.layers.layer_boundary.stage import StageBoundary
from sglang.srt.layers.moe import is_moe_input_scattered_across_dp_ranks
from sglang.srt.runtime_context import get_exec, get_parallel

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()


def _reject_unsupported_cp_moe(moe_on_local_rows: bool, cp_shards: bool) -> None:
    """A MoE layer under attention CP whose tokens the steps cannot bring
    to it: under a prefill CP, one dispatched per DP shard under attention DP
    and GQA CP, and one on the TP group whose data-parallel groups are the CP
    ranks under DSA or MLA CP; and under attention DP, one on the TP group
    whose data-parallel groups are the CP ranks."""
    parallel = get_parallel()
    gqa = not _cp_gathers_over_attn_cp()
    if moe_on_local_rows:
        if cp_shards and gqa and parallel.attn_dp_size > 1:
            raise NotImplementedError(
                "a MoE dispatched per DP shard under attention DP and GQA prefill CP"
            )
    elif parallel.moe_dp_size == parallel.attn_cp_size:
        if cp_shards and not gqa:
            raise NotImplementedError(
                "a MoE on the TP group with moe_dp_size == attn_cp_size under "
                "DSA or MLA prefill CP"
            )
        if parallel.attn_dp_size > 1:
            raise NotImplementedError(
                "a MoE on the TP group with moe_dp_size == attn_cp_size under "
                "attention DP and attention CP"
            )


def _unpadded_possible() -> bool:
    """Whether a batch whose rows do not divide over attention TP may reach an
    FFN that would run on this rank's attention-TP slice, so that FFN needs a
    variant that stays on the attention's rows."""
    return batches_are_unpadded() and (
        is_moe_input_scattered_across_dp_ranks() or is_dense_ffn_fully_dp()
    )


def _active_variants():
    yield BatchVariant.ORDINARY
    if _prefill_cp_shards_tokens():
        yield BatchVariant.CONTEXT_PARALLEL
    if input_scattered_configured():
        yield BatchVariant.INPUT_SCATTERED
    if layernorm_sp.layernorm_sp_enabled():
        yield BatchVariant.SEQUENCE_PARALLEL
    if _unpadded_possible():
        yield BatchVariant.UNPADDED


def _row_layouts(variant):
    axes = token_axis_sizes(cp_active=variant is BatchVariant.CONTEXT_PARALLEL)
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axes
    )
    local = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, TokenAxis.ATTN_TP, axis_sizes=axes
    )
    return axes, attention, local, Layout.sharded_over(axis_sizes=axes)


def _ffn_on_rank_rows(sparse, dense_tp_size) -> bool:
    """Whether an FFN computes on this rank's own rows (its attention-TP slice
    of them, or an unpadded batch's rows) rather than on the attention's."""
    if sparse:
        return is_moe_input_scattered_across_dp_ranks()
    if dense_tp_size is None:
        return is_dense_ffn_fully_dp()
    return dense_tp_size == 1


def _resolve_ffn(
    variant,
    *,
    sparse,
    terminal=False,
    output_transform=None,
    read=NORM_READOUT,
    update=PLAIN_ADD,
    dense_tp_size=None,
    output_complete=False,
):
    parallel = get_parallel()
    if dense_tp_size not in (None, 1, parallel.attn_tp_size, parallel.tp_size):
        raise ValueError(
            "dense FFN rows support local compute, the attention TP group or the "
            "full TP group"
        )
    strategy = get_exec().comm.boundary_reduction
    if strategy not in ("ar", "rs", "rsv", "rs+rsv"):
        raise ValueError(
            "boundary_reduction must be resolved before model construction"
        )
    can_move_output = output_transform is None or output_transform.before_reduce_scatter
    use_reduce_scatter = strategy in ("rs", "rs+rsv") and can_move_output
    use_reduce_scatterv = strategy in ("rsv", "rs+rsv") and can_move_output
    cp_shards = _prefill_cp_shards_tokens()
    axes, attention, local, full = _row_layouts(variant)
    on_rank_rows = _ffn_on_rank_rows(sparse, dense_tp_size)
    if variant is BatchVariant.UNPADDED and on_rank_rows:
        # Rows that do not divide over attention TP stay whole: the FFN runs on
        # the attention's rows (an a2a MoE dispatches them from every
        # attention-TP rank) and its output is complete, as a2a's combine or
        # local compute leaves it.
        return (
            StageContract(
                InputContract(attention, read=read),
                OutputContract(attention, update=update, transform=output_transform),
            ),
            attention,
            attention,
        )
    on_cp_shards = (
        sparse
        and parallel.attn_cp_size > 1
        and parallel.moe_dp_size == parallel.attn_cp_size
        and not _cp_gathers_over_attn_cp()
    )
    # Over attention TP, short of the full TP group: the FFN owes only the
    # attention-TP sum, and nothing moves over DP.
    on_attention_rows = (
        not sparse
        and dense_tp_size == parallel.attn_tp_size
        and dense_tp_size not in (1, parallel.tp_size)
    )
    if sparse:
        group = SumGroup.MOE_OUTPUT
    else:
        group = SumGroup.ATTN_TP if on_attention_rows else SumGroup.TP
    # An FFN that writes the next stream itself hands on a complete output.
    complete = on_rank_rows or update.writes_stream or output_complete
    if variant is BatchVariant.SEQUENCE_PARALLEL:
        return (
            StageContract(
                InputContract(
                    full,
                    gathered_by_compute=frozenset({TokenAxis.ATTN_TP}),
                    read=read,
                ),
                OutputContract(local, update=update, transform=output_transform),
            ),
            local,
            local,
        )
    if variant is BatchVariant.INPUT_SCATTERED:
        scattered_residual = update.applied_at_exit
        residual = local if scattered_residual else attention
        returned = local if scattered_residual and not terminal else attention
        return (
            StageContract(
                InputContract(full, read=read),
                OutputContract(
                    attention,
                    group=None if complete else group,
                    may_reduce_scatter=not complete
                    and use_reduce_scatter
                    and (scattered_residual or not terminal),
                    update=update,
                    transform=output_transform,
                ),
            ),
            residual,
            returned,
        )
    if cp_shards and _cp_moves().reduce_scatter is not None:
        may_leave = variant is not BatchVariant.CONTEXT_PARALLEL
        may_scatter = True
    elif cp_shards or parallel.attn_dp_size > 1:
        may_leave = may_scatter = parallel.attn_cp_size == 1 or on_cp_shards
    else:
        may_leave = may_scatter = True
    if on_rank_rows:
        rows = local
    elif on_attention_rows:
        rows = attention
        # The output is already on the residual's rows: nothing to scatter.
        use_reduce_scatter = use_reduce_scatterv = False
    else:
        rows = Layout.sharded_over(
            *((TokenAxis.ATTN_CP,) if on_cp_shards else ()), axis_sizes=axes
        )
    produced = (
        OutputContract(rows, update=update, transform=output_transform)
        if complete
        else OutputContract(
            rows,
            group=group,
            may_defer_to_next=may_leave
            and not terminal
            and not update.applied_at_exit
            and update.outlives_layer
            and output_transform is None,
            may_reduce_scatter=use_reduce_scatter and may_scatter,
            may_reduce_scatterv=use_reduce_scatterv and may_leave,
            update=update,
            transform=output_transform,
        )
    )
    returned = local if on_rank_rows and not terminal else attention
    return (
        StageContract(InputContract(rows, read=read), produced),
        local if on_rank_rows else attention,
        returned,
    )


@dataclass(frozen=True)
class StageDeclaration:
    """Describe one computation's boundaries without binding kernels or norms.

    Fields:
        kind: Selects attention/mixer or FFN boundary adapters; does not run
            the corresponding computation.
        read: Consumer operation that derives compute input from the residual.
        update: Producer operation that writes compute output into the residual.
        sparse: Whether this FFN is a MoE; used to resolve input/output rows.
            Expert routing and all-to-all remain inside the MoE computation.
        terminal: Whether this stage ends the model's layer stack, as the
            stack records it. Prevents leaving work that requires a following
            layer; a finalize handoff may still reach the terminal norm when
            the fusion provider allows it.
        output_transform: Optional operation on the contribution before the
            residual update. An FFN's exit runs it under an explicit
            reduction-order contract; an attention's is run by the input of
            the stage that follows, once the attention's sum is complete.
        reduction: Whether the next stage's input always completes the sum
            (an attention's), or the exit decides (an FFN's or a mixer's).
        gathers_attn_tp_input: Whether attention gathers TP-sharded input itself.
        tp_group: Group partitioning attention heads. Full TP consumers require
            full token rows, including when TP shares the prefill CP group.
        dense_tp_size: Dense FFN compute width: None uses the configured width,
            1 means local compute, the attention TP size means compute on the
            attention's rows over attention TP, and the full TP size means TP
            compute.
        exit_rows: Required FFN output rows at the layer or branch exit.
        writes_at_handoff: Whether this FFN writes its output into the residual
            at its exit because the next stage is on another pipeline rank.
        output_complete: Whether the FFN's compute completes its own output
            sum, so the exit owes none. For an FFN whose output sum is fused
            with a reduction its computation needs anyway.
        attn_tp_gather: Optional implementation of the gather over attention
            TP that brings rows into this stage: at its own entry, or at its
            producer's exit, which runs its consumer's gather. Given this rank's
            contiguous slice of the rows, it returns them all, in rank order,
            or None to leave the gather to the boundary. Called on every batch,
            so it must be CUDA-graph safe. Across a pipeline boundary the
            producer's rank takes it from the model's shared declaration
            function, without the declaring layer, so it may use only the
            communication state of the rank it runs on, not that layer's
            weights or buffers.
        previous: The stage whose output this stage consumes, as the stack
            records it: only the facts binding reads of it (see facts_of),
            on this rank or another.
        prepared_from: Declaration whose already-read input a branch reuses.
            Mutually exclusive with previous; avoids a second update/read.

    Sources contain declarations, never executable boundary objects.
    """

    kind: StageKind
    read: ResidualReadout
    update: ResidualUpdate
    sparse: bool = False
    terminal: bool = False
    output_transform: Optional[OutputTransform] = None
    reduction: ProducerReduction = ProducerReduction.EXIT_SCOPED
    gathers_attn_tp_input: bool = False
    tp_group: SumGroup = SumGroup.ATTN_TP
    dense_tp_size: Optional[int] = None
    exit_rows: Optional[ExitRows] = None
    writes_at_handoff: bool = False
    output_complete: bool = False
    attn_tp_gather: Optional[Callable] = None
    # Only declarations participate in construction, never executable stages.
    previous: Optional[StageDeclaration] = None
    prepared_from: Optional[StageDeclaration] = None

    def __post_init__(self):
        if self.previous is not None and self.prepared_from is not None:
            raise ValueError("choose a previous output or a prepared branch input")
        for source in (self.previous, self.prepared_from):
            if source is not None and not isinstance(source, StageDeclaration):
                raise TypeError("stage sources must be declarations")
        _check_update(self.update)
        _check_read(self.read)


# The members every ResidualUpdate and ResidualReadout must set: the binding
# reads each of them, and none has a default that is correct for every
# implementation.
_UPDATE_FACTS = (
    "is_plain_add",
    "applied_at_exit",
    "outlives_layer",
    "writes_stream",
    "quantized_sum",
)
_READ_FACTS = ("is_plain_norm", "reads_before_dp_gather", "reads_after_attn_tp_gather")
# The kernels a read supplies, which the binding also reads; empty when it
# supplies none.
_READ_KERNELS = ("completing_fusions", "gathering_reads")


def _require(protocol, implementation, facts):
    missing = [name for name in facts if not hasattr(implementation, name)]
    if missing:
        raise TypeError(
            f"{type(implementation).__name__} does not set {', '.join(missing)}, "
            f"which every {protocol} must"
        )


def _check_update(update):
    _require("ResidualUpdate", update, _UPDATE_FACTS)
    name = type(update).__name__
    if update.writes_stream and (update.is_plain_add or not update.applied_at_exit):
        raise ValueError(
            f"{name} writes the next stream itself, which only an update "
            "applied at its exit and other than a plain add does"
        )
    if update.quantized_sum and not update.is_plain_add:
        raise ValueError(
            f"{name} lets its sum run quantized, which only a plain add does"
        )


def _check_read(read):
    _require("ResidualReadout", read, _READ_FACTS + _READ_KERNELS)


@dataclass(frozen=True)
class StageConnection:
    """Immutable, norm-free contracts for the two sides of a connection.

    Fields:
        producer: Source declaration, or None at stack entry.
        consumer: Destination declaration, or None at a layer or stack exit.
        exits: Producer-side EdgeContract for each supported BatchVariant.
        entries: Consumer-side EdgeContract for each supported BatchVariant.

    Equivalent declarations can be reconstructed across pipeline partitions;
    adjacent layers need not share the same connection object.
    """

    producer: Optional[StageDeclaration]
    consumer: Optional[StageDeclaration]
    exits: Mapping[BatchVariant, EdgeContract]
    entries: Mapping[BatchVariant, EdgeContract]

    def __post_init__(self):
        object.__setattr__(self, "exits", MappingProxyType(dict(self.exits)))
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))


def declare_attn(
    *,
    read=NORM_QUANT_READOUT,
    update=PLAIN_ADD,
    reduction=ProducerReduction.ALWAYS_PARTIAL,
    gathers_attn_tp_input=True,
    output_transform=None,
    tp_group=SumGroup.ATTN_TP,
    attn_tp_gather=None,
):
    """Declare attention or a mixer; construct its executable boundary later.

    Args:
        read: Input operation; defaults to normalization with quantization support.
        update: Operation that adds this stage's output to the residual.
        reduction: ALWAYS_PARTIAL for an attention whose sum the next stage's
            input always completes; EXIT_SCOPED for a mixer whose exit decides.
        gathers_attn_tp_input: Whether compute gathers attention-TP input slices itself.
        output_transform: Optional operation on the output once its sum is
            complete, before the residual update (a sandwich norm). The next
            stage's input runs it, so no fused add + norm takes that input.
            Requires ALWAYS_PARTIAL.
        tp_group: Head partition; ATTN_TP by default, TP for full-TP consumers.
        attn_tp_gather: Implementation of the gather over attention TP into
            this stage, tried before the boundary's own (see StageDeclaration).

    Returns:
        A StageDeclaration with no norm, tensors or execution plan.
    """
    if (
        output_transform is not None
        and reduction is not ProducerReduction.ALWAYS_PARTIAL
    ):
        raise ValueError("an attention output transform requires ALWAYS_PARTIAL")
    return StageDeclaration(
        StageKind.ATTENTION,
        read,
        update,
        output_transform=output_transform,
        reduction=reduction,
        gathers_attn_tp_input=gathers_attn_tp_input,
        tp_group=tp_group,
        attn_tp_gather=attn_tp_gather,
    )


def declare_ffn(
    *,
    sparse=False,
    read=NORM_READOUT,
    update=PLAIN_ADD,
    output_transform=None,
    next_layer_sparse=False,
    dense_tp_size=None,
    exit_rows=None,
    output_complete=False,
):
    """Declare a dense or MoE FFN independently of its compute module.

    Args:
        sparse: Whether the FFN is a MoE; EP dispatch stays inside compute.
        read: Operation deriving FFN input from the residual.
        update: Operation writing FFN output into the residual.
        output_transform: Optional contribution transform before residual update.
        next_layer_sparse: Whether the next decoder layer's FFN is sparse; used only
            to derive the TBO exit rows when exit_rows is not supplied.
        dense_tp_size: Dense compute width: None for configuration, 1 for local
            compute, the attention TP size, or the full TP size.
        exit_rows: Explicit output-row requirement; otherwise derived from
            the adjacent FFN kinds and TBO configuration.
        output_complete: Whether the compute completes its own output sum,
            fused with a reduction it needs anyway; the exit then owes none.

    Returns:
        A StageDeclaration with no norm, tensors or execution plan.
    """
    return StageDeclaration(
        StageKind.FFN,
        read,
        update,
        sparse=sparse,
        output_transform=output_transform,
        dense_tp_size=dense_tp_size,
        exit_rows=exit_rows or tbo_exit_rows(sparse, next_layer_sparse),
        output_complete=output_complete,
    )


def _reject_unsupported(stage):
    """Reject a stage whose declared capabilities this parallel configuration
    cannot bind, naming the combination. The rejections that depend on the
    rows of one edge are made where that edge binds (``_bind_stage``)."""
    parallel = get_parallel()
    if stage.update.applied_at_exit:
        if stage.sparse and moe_gathers_over_moe_cp():
            raise NotImplementedError(
                "an update applied at the stage's exit with a MoE gathered over "
                "the MoE-CP group"
            )
        if parallel.attn_cp_size > 1 and input_scattered_configured():
            raise NotImplementedError(
                "an update applied at the stage's exit with input-scattered "
                "attention under attention CP"
            )
    if stage.kind is StageKind.FFN and stage.sparse and parallel.attn_cp_size > 1:
        _reject_unsupported_cp_moe(
            _ffn_on_rank_rows(stage.sparse, stage.dense_tp_size),
            _prefill_cp_shards_tokens(),
        )


class _Arrival(NamedTuple):
    """What reaches a stage's entry from its producer, for one batch variant.
    ``plain_add`` is None after an attention that always leaves its sum: the
    entry receives that update with the output."""

    produced: OutputContract
    residual: Layout
    plain_add: Optional[bool]
    written: bool


class _Flow(NamedTuple):
    """One stage's residual data flow for one batch variant."""

    contract: StageContract
    arrival: _Arrival
    during: Layout
    returned: Layout


def _stack_arrival(variant):
    """What reaches the first stage of the model's layer stack: its input, on
    the attention's rows (this rank's slice of them under sequence
    parallelism), owing the TP sum an input-scattered batch leaves."""
    _, attention, local, _ = _row_layouts(variant)
    rows = local if variant is BatchVariant.SEQUENCE_PARALLEL else attention
    owes = variant is BatchVariant.INPUT_SCATTERED
    return _Arrival(
        OutputContract(
            rows,
            group=SumGroup.TP if owes else None,
            always_partial=owes,
            update=None,
        ),
        rows,
        True,
        False,
    )


def _resolve_stage(stage, variant, arrival, following=None):
    """``stage``'s data flow for ``variant``, given what arrives at its entry
    and the stage after it (None when it ends the stack or a branch)."""
    axes, attention, local, full = _row_layouts(variant)
    _reject_unsupported(stage)
    if stage.kind is StageKind.FFN:
        declaration, residual, returned = _resolve_ffn(
            variant,
            sparse=stage.sparse,
            terminal=stage.terminal,
            output_transform=stage.output_transform,
            read=stage.read,
            update=stage.update,
            dense_tp_size=stage.dense_tp_size,
            output_complete=stage.output_complete,
        )
        exit_rows = resolve_exit_rows(stage.exit_rows)
        if exit_rows is ExitRows.ATTENTION or (
            following is not None and following.read.reads_after_attn_tp_gather
        ):
            returned = attention
        elif exit_rows is ExitRows.SLICE:
            # The residual's rows during the FFN: this rank's slice for an FFN
            # on its own rows, else the attention's.
            returned = residual
        return _Flow(declaration, arrival, residual, returned)
    sp = variant is BatchVariant.SEQUENCE_PARALLEL
    scattered = variant is BatchVariant.INPUT_SCATTERED
    gathers = (
        frozenset({TokenAxis.ATTN_TP})
        if stage.gathers_attn_tp_input and (sp or scattered or _use_ag_after_qlora)
        else frozenset()
    )
    owes = not sp and axes[TokenAxis.ATTN_TP] > 1
    group = SumGroup.ATTN_TP
    compute_rows = attention
    if stage.tp_group is SumGroup.TP:
        parallel = get_parallel()
        if parallel.attn_dp_size != 1 or sp or scattered:
            raise NotImplementedError(
                "full-TP attention requires unscattered input without attention DP"
            )
        compute_rows = full
        owes = parallel.tp_size > 1
        # Canonicalize equal groups to the CP transport's reduction contract.
        # initialize_model_parallel aliases ATTN_CP to TP at this topology.
        group = (
            SumGroup.ATTN_CP
            if parallel.attn_cp_size == parallel.tp_size and parallel.attn_cp_size > 1
            else SumGroup.TP
        )
    elif stage.tp_group is not SumGroup.ATTN_TP:
        raise ValueError("attention heads must be partitioned over ATTN_TP or TP")
    declaration = StageContract(
        InputContract(compute_rows, gathered_by_compute=gathers, read=stage.read),
        OutputContract(
            local if sp else compute_rows,
            group=group if owes else None,
            always_partial=owes
            and (
                stage.reduction is ProducerReduction.ALWAYS_PARTIAL
                or (following is not None and following.kind is StageKind.FFN)
            ),
            may_defer_to_next=owes
            and stage.reduction is ProducerReduction.EXIT_SCOPED
            and following is not None
            and following.kind is StageKind.ATTENTION,
            update=stage.update,
            transform=stage.output_transform,
        ),
    )
    # An attention moves no residual: it keeps the rows it arrives on, except
    # that one that always leaves its sum completes an input-scattered sum
    # it receives onto this rank's slice of them.
    during = arrival.residual
    if (
        variant is BatchVariant.INPUT_SCATTERED
        and arrival.produced.always_partial
        and arrival.produced.update is None
        and stage.reduction is ProducerReduction.ALWAYS_PARTIAL
    ):
        during = Layout(during.sharded | {TokenAxis.ATTN_TP})
    return _Flow(declaration, arrival, during, attention)


def _arrival(producer, flow, variant):
    """What reaches the stage after ``producer`` from it, given its flow."""
    output = flow.contract.output
    if (
        producer.kind is StageKind.ATTENTION
        and producer.reduction is ProducerReduction.ALWAYS_PARTIAL
    ):
        # The output owes its sum, and the residual is where it ran.
        return _Arrival(output, flow.during, None, False)
    if (
        producer.kind is StageKind.ATTENTION
        and producer.reduction is ProducerReduction.EXIT_SCOPED
        and (output.always_partial or output.may_defer_to_next)
    ):
        # A mixer's exit leaves the next entry the sum it declares; a sum it
        # may carry on arrives with the value, if at all.
        owes, group = output.always_partial, output.group
    else:
        # On an input-scattered batch, the next entry's reduce-scatter
        # completes the TP sum of an output not written at its exit.
        owes = (
            variant is BatchVariant.INPUT_SCATTERED
            and not producer.update.applied_at_exit
        )
        group = SumGroup.TP
    return _Arrival(
        OutputContract(
            flow.returned,
            group=group if owes else None,
            always_partial=owes,
            update=None,
        ),
        flow.returned,
        producer.update.is_plain_add,
        producer.update.applied_at_exit,
    )


def _entry_edge(stage, flow, variant):
    """The edge into ``stage`` as its entry binds it."""
    arrival = flow.arrival
    joins = (
        variant is BatchVariant.INPUT_SCATTERED
        and arrival.produced.update is not None
        and arrival.produced.update.is_plain_add
        and stage.kind is StageKind.FFN
    )
    return EdgeContract(
        arrival.produced,
        flow.contract.input,
        arrival.residual,
        flow.during,
        residual_joins_sum=joins,
        arriving_plain_add=arrival.plain_add,
        arrives_written=arrival.written,
    )


def _exit_edge(stage, flow, entry):
    """The edge out of ``stage`` as its exit binds it. An attention that always
    leaves its sum has no exit: its edge is its consumer's entry edge, when
    there is a consumer."""
    if entry is not None and (
        stage.kind is StageKind.ATTENTION
        and stage.reduction is ProducerReduction.ALWAYS_PARTIAL
    ):
        return entry
    return EdgeContract(
        flow.contract.output, InputContract(flow.returned), flow.during, flow.returned
    )


def _connect_line(line, origins, *, before=None, after=None, arrivals=None):
    """The connections along a chain of stages, each resolved once per batch
    variant: into ``line[0]``, between each two, and out of ``line[-1]`` to
    ``after`` (a stage on the next rank, or None at the end of the stack or a
    branch). What arrives at ``line[0]`` is ``arrivals``, else what ``before``
    (a stage on the previous rank, or None for the stack input) hands on; that
    rank's own entry is taken as the stack input's. An error resolving a stage
    names the append at ``origins[i]`` that declared ``line[i]``, or the one
    next to the neighbouring rank's stage."""
    variants = tuple(_active_variants())

    def resolve(position, stage, arrivals, following=None):
        try:
            return {
                v: _resolve_stage(stage, v, arrivals[v], following) for v in variants
            }
        except Exception as error:
            _note_origin(error, origins[position])
            raise

    if arrivals is None:
        arrivals = {v: _stack_arrival(v) for v in variants}
        if before is not None:
            if (
                before.output_transform is not None
                and before.kind is StageKind.ATTENTION
                and before.reduction is ProducerReduction.ALWAYS_PARTIAL
            ):
                # Its transform runs at the next stage's input, which would be
                # on this rank, without the module that declares it.
                error = NotImplementedError(
                    "a pipeline rank that ends on an attention transforming the "
                    "output whose sum it leaves"
                )
                _note_origin(error, origins[0])
                raise error
            flow = resolve(0, before, arrivals, line[0])
            arrivals = {v: _arrival(before, flow[v], v) for v in variants}
    flows = []
    for position, stage in enumerate(line):
        following = line[position + 1] if position + 1 < len(line) else after
        flow = resolve(position, stage, arrivals, following)
        flows.append(flow)
        arrivals = {v: _arrival(stage, flow[v], v) for v in variants}
    if after is not None:
        flows.append(resolve(len(line) - 1, after, arrivals))
    entries = [
        {v: _entry_edge(stage, flow[v], v) for v in variants}
        for stage, flow in zip([*line, after], flows)
    ]
    connections = [StageConnection(before, line[0], {}, entries[0])]
    for position, stage in enumerate(line):
        consumer = line[position + 1] if position + 1 < len(line) else after
        following = entries[position + 1] if consumer is not None else {}
        connections.append(
            StageConnection(
                stage,
                consumer,
                {
                    v: _exit_edge(stage, flows[position][v], following.get(v))
                    for v in variants
                },
                following,
            )
        )
    return connections


def _fork_input(prepared):
    """What reaches a branch's first stage: an already-read input, placed onto
    the branch's computation rows by the branch adapter, which forks the
    stream without running the branch's input norm again."""
    arrivals = {}
    for variant in _active_variants():
        if variant is not BatchVariant.ORDINARY:
            raise NotImplementedError(
                "prepared branch transport requires ordinary token rows"
            )
        source = prepared.entries[variant]
        arrivals[variant] = _Arrival(
            OutputContract(source.need.layout, update=None),
            source.residual_to,
            True,
            False,
        )
    return arrivals


class _Append:
    """One append_stages call: its validated stages, the boundaries it
    returned, which the stack fills in when it closes, and where it was made."""

    __slots__ = ("declarations", "bindings", "prepared_from", "boundaries", "origin")

    def __init__(self, declarations, bindings, prepared_from, boundaries, origin):
        self.declarations = declarations
        self.bindings = bindings
        self.prepared_from = prepared_from
        self.boundaries = boundaries
        self.origin = origin


class _LayerStack:
    """A layer stack under construction: what was appended, in order, and how
    to reach the layers other pipeline ranks hold on either side of it."""

    __slots__ = ("appends", "previous_layers", "next_layers", "final_read")

    def __init__(self, previous_layers=(), next_layers=(), final_read=None):
        self.appends = []
        self.previous_layers = previous_layers
        self.next_layers = next_layers
        self.final_read = final_read


# The stack being built; layer_stack saves and restores an outer one.
_stack: Optional[_LayerStack] = None


@contextlib.contextmanager
def layer_stack(*, previous_layers=(), next_layers=(), final_read=None):
    """Open a layer stack that append_stages extends in order.

    Every stage binds when the stack closes, once its producer and its
    consumer are both known: the stage appended before and after it. The last
    stage ends the model's layer stack unless a later layer declares a stage.

    Args:
        previous_layers: For the layers before this stack that another
            pipeline rank holds, nearest first, callables that return the
            stages the layer declares, in order, without building it (the
            model's shared declaration function at that layer's index; an
            empty sequence for a layer that declares none). Called only if
            this stack appended stages, after its own layers are built, until
            one of them declares a stage: its last stage is the producer of
            this stack's first. Only what binding reads of that stage is kept
            (see facts_of).
        next_layers: Likewise for the layers after this stack, whose first
            declared stage is the consumer of this stack's last.
        final_read: The model's final read of the stack's output (a
            ``FinalRead``), when it is not a plain final norm: the consumer of
            the last stage when no later layer declares one. Its
            ``attn_tp_gather`` gathers the rows that stage leaves on this
            rank's attention-TP slice, and with ``reads_attn_tp_slices`` it
            reads that slice instead.
    """
    global _stack
    outer = _stack
    stack = _LayerStack(previous_layers, next_layers, final_read)
    _stack = stack
    try:
        yield stack
        if stack.appends:
            _bind_stack(
                stack.appends,
                previous=_neighbour_stage(stack.previous_layers, last=True),
                following=_neighbour_stage(stack.next_layers, last=False),
                final_read=stack.final_read,
            )
    finally:
        _stack = outer


def check_declared_stages(stack, appended: int, expected, where: str) -> None:
    """Check that what a layer appended to ``stack`` since it held
    ``appended`` appends is what its shared declaration function says it
    declares, as far as binding reads it: another pipeline rank binds this
    layer's stages from that function alone."""
    declared = [
        facts_of(declaration)
        for append in stack.appends[appended:]
        if append.prepared_from is None
        for declaration in append.declarations
    ]
    expected = [facts_of(declaration) for declaration in expected]
    if declared != expected:
        raise ValueError(
            f"{where} declares stages its shared declaration function does "
            f"not: {_first_difference(declared, expected)}"
        )


def _first_difference(declared, expected) -> str:
    """Where the stages a layer declares first differ from those its shared
    declaration function gives, field by field."""
    if len(declared) != len(expected):
        return (
            f"it declares {[stage.kind.name for stage in declared]}, the "
            f"function {[stage.kind.name for stage in expected]}"
        )
    index, (stage, given) = next(
        (index, pair)
        for index, pair in enumerate(zip(declared, expected))
        if pair[0] != pair[1]
    )
    return f"stage {index} ({stage.kind.name}): " + "; ".join(
        f"{name} is {value!r}, the function says {given_value!r}"
        for name, value, given_value in _differing_fields(stage, given)
    )


def _differing_fields(value, given, prefix=""):
    """(dotted name, value, given value) for each leaf field that differs,
    descending into declarations and the facts they hold."""
    if is_dataclass(value):
        names = [field.name for field in fields(value)]
    else:
        names = getattr(type(value), "__struct_fields__", None)
    if names is None or type(value) is not type(given):
        return [(prefix.rstrip("."), value, given)]
    return [
        difference
        for name in names
        if getattr(value, name) != getattr(given, name)
        for difference in _differing_fields(
            getattr(value, name), getattr(given, name), f"{prefix}{name}."
        )
    ]


def _neighbour_stage(layers, *, last):
    """The stage a neighbouring layer declares next to this stack: the last
    one of the nearest layer before it, or the first of the nearest after."""
    for layer in layers:
        declared = tuple(layer())
        if declared:
            return facts_of(declared[-1] if last else declared[0])
    return None


def _require_stack() -> _LayerStack:
    if _stack is None:
        raise RuntimeError(
            "append_stages needs an open layer stack; build the layers inside "
            "make_layers or layer_stack"
        )
    return _stack


def _note_origin(error: Exception, origin: str) -> None:
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(f"while binding the stages appended at {origin}")


def _handed_off(declaration):
    """A stage whose output crosses to another pipeline rank: an FFN leaves it
    on the attention's rows, which is what the handoff carries and the next
    rank's first stage reads, even where it would otherwise stay on this
    rank's attention-TP slice of them. An FFN on its own rows also writes its
    output into the residual there, on every batch, so the handoff carries the
    written stream and the receiver knows it from this declaration."""
    if declaration is None or declaration.kind is not StageKind.FFN:
        return declaration
    return replace(
        declaration,
        exit_rows=ExitRows.ATTENTION,
        writes_at_handoff=_ffn_on_rank_rows(
            declaration.sparse, declaration.dense_tp_size
        ),
    )


def _return_before_trailing_attention(line):
    """When the stack ends on attentions that always leave their sum, the next
    rank or the final read takes the residual on the rows they ran on, which
    is where the FFN before them returned it: no exit moves it after them.
    That FFN returns it on the attention's rows, as one that ends the stack or
    hands off does, instead of on its own rows."""
    trailing = False
    for append in reversed(line):
        declarations = append.declarations
        for index in reversed(range(len(declarations))):
            declaration = declarations[index]
            if (
                declaration.kind is StageKind.ATTENTION
                and declaration.reduction is ProducerReduction.ALWAYS_PARTIAL
            ):
                trailing = True
                continue
            if (
                trailing
                and declaration.kind is StageKind.FFN
                and _ffn_on_rank_rows(declaration.sparse, declaration.dense_tp_size)
            ):
                declarations[index] = replace(declaration, exit_rows=ExitRows.ATTENTION)
            return


def _bind_stack(appends, *, previous, following, final_read=None):
    """Bind every appended stage and fill in the boundaries each append
    returned.

    The stages of the appends that extend the stack form one line, chained in
    append order, and every connection between two of them is built once:
    each stage's outgoing connection is the next one's incoming. A branch
    then starts from the incoming connection of the stage it branches from.
    """
    line = [a for a in appends if a.prepared_from is None]
    if following is not None:
        # The last stage hands off to the next rank.
        line[-1].declarations[-1] = _handed_off(line[-1].declarations[-1])
    _return_before_trailing_attention(line)
    if following is None:
        # Without a stage on a later rank, the last one ends the model's stack,
        # read by the final read.
        line[-1].declarations[-1] = _ended(
            replace(line[-1].declarations[-1], terminal=True), final_read
        )
    stages = [
        (append, index) for append in line for index in range(len(append.bindings))
    ]
    bound = {id(append): [None] * len(append.bindings) for append in appends}
    # A returned declaration's stage as bound, for the branches that start from it.
    sources = {}
    producer = facts_of(_handed_off(previous))
    chained = []
    for append, index in stages:
        chained.append(
            replace(append.declarations[index], previous=producer, prepared_from=None)
        )
        producer = facts_of(chained[-1])
    connections = _connect_line(
        chained,
        [append.origin for append, _ in stages],
        before=chained[0].previous,
        after=following,
    )
    _bind_line(
        stages,
        chained,
        connections,
        bound,
        sources,
        final_read=final_read if following is None else None,
    )
    for append in appends:
        if append.prepared_from is None:
            continue
        source = sources.get(id(append.prepared_from))
        if source is None:
            raise ValueError(
                "prepared_from must be the declaration of a stage appended "
                f"earlier to the same layer stack (at {append.origin})"
            )
        source_declaration, source_incoming = source
        branch = [(append, index) for index in range(len(append.bindings))]
        chained = []
        for index, declaration in enumerate(append.declarations):
            chained.append(
                replace(
                    declaration,
                    previous=facts_of(chained[-1]) if chained else None,
                    prepared_from=None if chained else facts_of(source_declaration),
                )
            )
        try:
            arrivals = _fork_input(source_incoming)
        except Exception as error:
            _note_origin(error, append.origin)
            raise
        connections = _connect_line(
            chained,
            [append.origin] * len(chained),
            before=source_declaration,
            arrivals=arrivals,
        )
        _bind_line(branch, chained, connections, bound, sources)
    bound = [tuple(bound[id(append)]) for append in appends]
    _check_declared_gathers(appends, bound, remote_producer=previous is not None)
    for append, boundaries in zip(appends, bound):
        for returned, boundary in zip(append.boundaries, boundaries):
            returned.plan = boundary.plan
            returned.declaration = boundary.declaration


def _bind_line(stages, chained, connections, bound, sources, *, final_read=None):
    """Bind a chain of stages from its connections, ``connections[i]`` into
    ``chained[i]`` and ``connections[i + 1]`` out of it. An attention that
    always leaves its sum binds after the FFN that follows it in the same
    append, so the attention's entry keeps the residual that FFN's capture
    needs."""
    last = len(chained) - 1
    boundaries = [None] * len(chained)

    def bind(position):
        append, index = stages[position]
        norm, options = append.bindings[index]
        consumer = None
        if position < last and stages[position + 1][0] is append:
            consumer = chained[position + 1]
            if _carries_capture(chained[position], consumer):
                bind(position + 1)
        try:
            boundary = _bind_stage(
                chained[position],
                norm,
                connections[position],
                connections[position + 1],
                final_read=final_read if position == last else None,
                capture_preserves_residual=(
                    _preserves_residual(boundaries[position + 1])
                    if consumer is not None
                    and _carries_capture(chained[position], consumer)
                    else None
                ),
                **options,
            )
        except Exception as error:
            _note_origin(error, append.origin)
            raise
        boundaries[position] = boundary
        bound[id(append)][index] = boundary
        sources[id(append.boundaries[index].declaration)] = (
            boundary.declaration,
            connections[position],
        )

    for position in range(len(chained)):
        if boundaries[position] is None:
            bind(position)


def _carries_capture(producer, consumer):
    """Whether an attention that always leaves its sum keeps the residual its
    FFN's entry needs for capture. Only between stages of one append."""
    return (
        producer.kind is StageKind.ATTENTION
        and producer.reduction is ProducerReduction.ALWAYS_PARTIAL
        and consumer.kind is StageKind.FFN
    )


def _preserves_residual(consumer):
    """Each variant's check that the FFN's entry leaves the residual in place."""
    return {
        variant: path.entry.preserves_residual
        for variant, path in consumer.plan.paths.items()
        if path.entry.preserves_residual is not None
    }


def _check_declared_gathers(appends, bound, *, remote_producer):
    """A stage that declares its gather over attention TP must have one: at
    its own entry, or at its producer's exit. A producer on another pipeline
    rank runs it there."""
    line = [
        (append.origin, boundary)
        for append, boundaries in zip(appends, bound)
        if append.prepared_from is None
        for boundary in boundaries
    ]
    for (origin, consumer), producer in zip(line, [None, *(b for _, b in line)]):
        if consumer.declaration.attn_tp_gather is None or any(
            path.entry.input_gather_declared for path in consumer.plan.paths.values()
        ):
            continue
        if producer is None and remote_producer:
            continue
        if producer is not None and any(
            path.output_gathers_attn_tp for path in producer.plan.paths.values()
        ):
            continue
        error = ValueError(
            "an attention-TP gather is declared, but neither this stage's entry "
            "nor its producer's exit gathers over attention TP"
        )
        _note_origin(error, origin)
        raise error


def _ended(declaration, final_read):
    """The stack's last stage as the final read takes it: an FFN on this
    rank's attention-TP slice leaves its output there for a read that reads
    the slice and gathers what it read."""
    if declaration.kind is StageKind.FFN and getattr(
        final_read, "reads_attn_tp_slices", False
    ):
        return replace(declaration, exit_rows=ExitRows.SLICE)
    return declaration


_STAGE_OPTIONS = {
    StageKind.ATTENTION: frozenset({"qkv_latent_func", "fusions"}),
    StageKind.FFN: frozenset({"fusions"}),
}


def _stage_items(stages):
    declarations, bindings = [], []
    for item in stages:
        if len(item) not in (2, 3):
            raise ValueError("a stage needs (declaration, norm[, options])")
        declaration, norm = item[:2]
        if not isinstance(declaration, StageDeclaration):
            raise TypeError("append_stages requires stage declarations")
        if declaration.previous is not None or declaration.prepared_from is not None:
            raise ValueError("a declaration takes its source from the layer stack")
        if declaration.terminal:
            raise ValueError("the layer stack marks the terminal stage")
        options = dict(item[2] if len(item) == 3 else {})
        unexpected = options.keys() - _STAGE_OPTIONS[declaration.kind]
        if unexpected:
            raise TypeError(
                f"unsupported {declaration.kind.name} stage options: "
                f"{sorted(unexpected)}"
            )
        declarations.append(declaration)
        bindings.append((norm, options))
    return declarations, bindings


def _origin(stack: _LayerStack) -> str:
    """Where the current append_stages call was made, and its place in the stack."""
    frame = sys._getframe(2)
    return (
        f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno} "
        f"(append {len(stack.appends)} of its layer stack)"
    )


def append_stages(*stages, prepared_from=None):
    """Append a local linear sequence of stages to the open layer stack.

    Args:
        *stages: Items of (declaration, norm) or (declaration, norm, options).
            Declarations carry no source or terminal flag. Options are
            constructor keywords: fusions, and qkv_latent_func for attention.
        prepared_from: The ``declaration`` of a boundary an earlier call
            returned on this stack: the already-read stage this sequence
            branches from. A branch is a side path the model merges back
            explicitly: it neither extends the stack nor waits for a consumer,
            so each of its stages binds with only the stages of the branch.

    Returns:
        A tuple of StageBoundary objects in declaration order. Their
        declarations are usable at once and gain their place in the stack when
        it closes, which is also when their plans are bound.
    """
    if not stages:
        raise ValueError("append_stages needs at least one stage")
    stack = _require_stack()
    declarations, bindings = _stage_items(stages)
    declarations = [replace(d) for d in declarations]
    boundaries = tuple(StageBoundary(None, declaration=d) for d in declarations)
    stack.appends.append(
        _Append(declarations, bindings, prepared_from, boundaries, _origin(stack))
    )
    return boundaries
