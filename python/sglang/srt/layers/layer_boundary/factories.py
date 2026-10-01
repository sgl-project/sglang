"""Construct one computing stage and its transport from a predecessor contract."""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Optional

import msgspec

from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.layer_boundary.adapters.overlap import (
    resolve_exit_rows,
    tbo_exit_rows,
)
from sglang.srt.layers.layer_boundary.boundary import _cp_moves
from sglang.srt.layers.layer_boundary.construction import (
    BatchVariant,
    _bind_stage,
    _input_scattered_possible,
    _reject_unsupported_cp_moe,
    _use_ag_after_qlora,
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
from sglang.srt.layers.layer_boundary.layout import (
    Layout,
    SumGroup,
    TokenAxis,
    _cp_gathers_over_attn_cp,
    _prefill_cp_shards_tokens,
    is_dense_ffn_fully_dp,
    moe_gathers_over_moe_cp,
    token_axis_sizes,
)
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.residual import ResidualReadout, ResidualUpdate
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NORM_QUANT_READOUT,
    NORM_READOUT,
    PLAIN_ADD,
    REPLACE_AT_EXIT,
)
from sglang.srt.layers.layer_boundary.stage import StageBoundary
from sglang.srt.layers.moe import is_moe_input_scattered_across_dp_ranks
from sglang.srt.runtime_context import get_exec, get_parallel


def _active_variants():
    yield BatchVariant.ORDINARY
    if _prefill_cp_shards_tokens():
        yield BatchVariant.CONTEXT_PARALLEL
    if _input_scattered_possible():
        yield BatchVariant.INPUT_SCATTERED
    if layernorm_sp.layernorm_sp_enabled():
        yield BatchVariant.SEQUENCE_PARALLEL


def _row_layouts(variant):
    axes = token_axis_sizes(cp_active=variant is BatchVariant.CONTEXT_PARALLEL)
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axes
    )
    local = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, TokenAxis.ATTN_TP, axis_sizes=axes
    )
    return axes, attention, local, Layout.sharded_over(axis_sizes=axes)


def _resolve_ffn(
    variant,
    *,
    sparse,
    terminal=False,
    output_transform=None,
    read=NORM_READOUT,
    update=PLAIN_ADD,
    dense_tp_size=None,
):
    parallel = get_parallel()
    if dense_tp_size not in (None, 1, parallel.tp_size):
        raise ValueError("dense FFN rows support local compute or the full TP group")
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
    on_rank_rows = (
        is_moe_input_scattered_across_dp_ranks()
        if sparse
        else (is_dense_ffn_fully_dp() if dense_tp_size is None else dense_tp_size == 1)
    )
    if parallel.attn_cp_size > 1 and sparse:
        _reject_unsupported_cp_moe(on_rank_rows, cp_shards)
    on_cp_shards = (
        sparse
        and parallel.attn_cp_size > 1
        and parallel.moe_dp_size == parallel.attn_cp_size
        and not _cp_gathers_over_attn_cp()
    )
    group = SumGroup.MOE_OUTPUT if sparse else SumGroup.TP
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
                    group=group,
                    may_reduce_scatter=use_reduce_scatter
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
    rows = (
        local
        if on_rank_rows
        else Layout.sharded_over(
            *((TokenAxis.ATTN_CP,) if on_cp_shards else ()), axis_sizes=axes
        )
    )
    # An FFN that writes the next stream itself hands on a complete output.
    complete = on_rank_rows or update is REPLACE_AT_EXIT
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
        tensor_parallel_over_cp: The mixer partitions its heads over the CP
            group instead of partitioning tokens. Its input gathers CP rows;
            its output owes a sum over that group, including on decode batches.
        dense_tp_size: Dense FFN compute width: None uses the configured width,
            1 means local compute, and the full TP size means TP compute.
        exit_rows: Required FFN output rows at the layer or branch exit.
        previous: Declaration whose output this stage consumes, as the stack
            records it; across pipeline ranks it is built locally.
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
    tensor_parallel_over_cp: bool = False
    dense_tp_size: Optional[int] = None
    exit_rows: Optional[ExitRows] = None
    # Only declarations participate in construction, never executable stages.
    previous: Optional[StageDeclaration] = None
    prepared_from: Optional[StageDeclaration] = None

    def __post_init__(self):
        if self.previous is not None and self.prepared_from is not None:
            raise ValueError("choose a previous output or a prepared branch input")
        for source in (self.previous, self.prepared_from):
            if source is not None and not isinstance(source, StageDeclaration):
                raise TypeError("stage sources must be declarations")


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
    tensor_parallel_over_cp=False,
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
        tensor_parallel_over_cp: Gather context shards for a head-parallel mixer.

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
        tensor_parallel_over_cp=tensor_parallel_over_cp,
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
            compute, or the full TP size.
        exit_rows: Explicit output-row requirement; otherwise derived from
            the adjacent FFN kinds and TBO configuration.

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
    )


def _resolve_stage(stage, variant, following=None):
    axes, attention, local, full = _row_layouts(variant)
    if stage.update.applied_at_exit:
        if stage.sparse and moe_gathers_over_moe_cp():
            raise NotImplementedError(
                "MHC does not support a MoE gathered over the MoE-CP group"
            )
        if get_parallel().attn_cp_size > 1 and _input_scattered_possible():
            raise NotImplementedError(
                "MHC with input-scattered attention under attention CP"
            )
    if stage.kind is StageKind.FFN:
        declaration, residual, returned = _resolve_ffn(
            variant,
            sparse=stage.sparse,
            terminal=stage.terminal,
            output_transform=stage.output_transform,
            read=stage.read,
            update=stage.update,
            dense_tp_size=stage.dense_tp_size,
        )
        if resolve_exit_rows(stage.exit_rows) is ExitRows.ATTENTION:
            returned = attention
        return declaration, residual, returned
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
    if stage.tensor_parallel_over_cp:
        if get_parallel().attn_tp_size != 1 or sp or scattered:
            raise NotImplementedError(
                "head parallelism over CP requires attention TP 1"
            )
        compute_rows = Layout(attention.sharded - {TokenAxis.ATTN_CP})
        owes = get_parallel().attn_cp_size > 1
        group = SumGroup.ATTN_CP
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
    return declaration, None, attention


def _connect(producer, consumer, *, residual_from=None):
    """Resolve producer and consumer declarations; None denotes an external endpoint.

    A missing producer is the stack input; a missing consumer is a layer or stack exit
    whose next read is bound separately (or the terminal output).
    ``residual_from`` names the boundary that placed the producer's residual.
    It is needed for attention, whose residual can stay on finer rows than its
    compute input. No executable stage participates in this construction.
    """
    if producer is None and consumer is None:
        raise ValueError("a boundary needs at least one declared side")
    before, after = producer, consumer
    exits, entries = {}, {}
    for variant in _active_variants():
        _, attention, local, _ = _row_layouts(variant)
        if before is None:
            rows = local if variant is BatchVariant.SEQUENCE_PARALLEL else attention
            owes = variant is BatchVariant.INPUT_SCATTERED
            arrived = OutputContract(
                rows,
                group=SumGroup.TP if owes else None,
                always_partial=owes,
                update=None,
            )
            residual = rows
            capabilities = (True,)
        else:
            decl, during, returned = _resolve_stage(before, variant, following=after)
            if during is None:
                if residual_from is not None:
                    if residual_from.consumer != producer:
                        raise ValueError("residual source must enter the producer")
                    during = residual_from.entries[variant].residual_to
                elif (
                    before.kind is StageKind.ATTENTION
                    and before.reduction is ProducerReduction.EXIT_SCOPED
                ):
                    during = attention
                else:
                    raise ValueError(
                        "attention output needs its incoming residual placement"
                    )
            exits[variant] = EdgeContract(
                decl.output, InputContract(returned), during, returned
            )
            if (
                before.kind is StageKind.ATTENTION
                and before.reduction is ProducerReduction.ALWAYS_PARTIAL
            ):
                arrived, residual, capabilities = decl.output, during, ()
            else:
                owes = (
                    variant is BatchVariant.INPUT_SCATTERED
                    and not before.update.applied_at_exit
                )
                carries = (
                    (
                        before.kind is StageKind.ATTENTION
                        and before.reduction is ProducerReduction.EXIT_SCOPED
                    )
                    or resolve_exit_rows(before.exit_rows) is ExitRows.ATTENTION
                ) and (decl.output.always_partial or decl.output.may_defer_to_next)
                arrived = OutputContract(
                    returned,
                    group=decl.output.group
                    if carries
                    else (SumGroup.TP if owes else None),
                    always_partial=decl.output.always_partial if carries else owes,
                    may_defer_to_next=decl.output.may_defer_to_next
                    if carries
                    else False,
                    update=None,
                )
                residual, capabilities = returned, (before.update.is_plain_add,)
        if after is None:
            continue
        decl, during, _ = _resolve_stage(after, variant)
        if after.kind is StageKind.ATTENTION:
            during = (
                Layout(residual.sharded | {TokenAxis.ATTN_TP})
                if variant is BatchVariant.INPUT_SCATTERED
                and arrived.always_partial
                and arrived.update is None
                and after.reduction is ProducerReduction.ALWAYS_PARTIAL
                else residual
            )
        elif resolve_exit_rows(after.exit_rows) is ExitRows.ATTENTION:
            during = (
                decl.input.layout
                if residual.sharded <= decl.input.layout.sharded
                else residual
            )
        joins = (
            variant is BatchVariant.INPUT_SCATTERED
            and arrived.update is not None
            and arrived.update.is_plain_add
            and after.kind is StageKind.FFN
        )
        edge = EdgeContract(
            arrived,
            decl.input,
            residual,
            during,
            residual_joins_sum=joins,
            arriving_plain_add=capabilities,
        )
        entries[variant] = edge
        if (
            before is not None
            and before.kind is StageKind.ATTENTION
            and before.reduction is ProducerReduction.ALWAYS_PARTIAL
        ):
            exits[variant] = edge
    return StageConnection(producer, consumer, exits, entries)


def _fork_input(prepared, consumer):
    """Place an already-read input onto a branch's computation rows.

    The branch adapter performs the move and forks the stream. It does not
    execute the branch's input norm again.
    """
    entries = {}
    for variant in _active_variants():
        if variant is not BatchVariant.ORDINARY:
            raise NotImplementedError(
                "prepared branch transport requires ordinary token rows"
            )
        source = prepared.entries[variant]
        declaration, during, _ = _resolve_stage(consumer, variant)
        entries[variant] = EdgeContract(
            OutputContract(source.need.layout, update=None),
            declaration.input,
            source.residual_to,
            during,
            arriving_plain_add=(True,),
        )
    return StageConnection(prepared.consumer, consumer, {}, entries)


def _incoming(stage):
    if stage.prepared_from is not None:
        return _fork_input(_incoming(stage.prepared_from), stage)
    previous = stage.previous
    # A normal attention may retain a finer residual than its output rows.
    # FFN and mixer exits declare their returned residual placement directly.
    source = (
        _incoming(previous)
        if previous is not None
        and previous.kind is StageKind.ATTENTION
        and previous.reduction is ProducerReduction.ALWAYS_PARTIAL
        else None
    )
    return _connect(
        previous,
        stage,
        residual_from=source,
    )


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

    __slots__ = ("appends", "previous_layers", "next_layers")

    def __init__(self, previous_layers=(), next_layers=()):
        self.appends = []
        self.previous_layers = previous_layers
        self.next_layers = next_layers


# The stack being built; layer_stack saves and restores an outer one.
_stack: Optional[_LayerStack] = None


@contextlib.contextmanager
def layer_stack(*, previous_layers=(), next_layers=()):
    """Open a layer stack that append_stages extends in order.

    Every stage binds when the stack closes, once its producer and its
    consumer are both known: the stage appended before and after it. The last
    stage ends the model's layer stack unless a later layer declares a stage.

    Args:
        previous_layers: Callables that build, nearest first, the layers before
            this stack that another pipeline rank holds. Called only if this
            stack appended stages, after its own layers are built, until one of
            them declares a stage: its last stage is the producer of this
            stack's first. What they build is discarded.
        next_layers: Likewise for the layers after this stack, whose first
            declared stage is the consumer of this stack's last.
    """
    global _stack
    outer = _stack
    stack = _LayerStack(previous_layers, next_layers)
    _stack = stack
    try:
        yield stack
        if stack.appends:
            _bind_stack(
                stack.appends,
                previous=_neighbour_stage(stack.previous_layers, last=True),
                following=_neighbour_stage(stack.next_layers, last=False),
            )
    finally:
        _stack = outer


def _neighbour_stage(build_layers, *, last):
    """The stage a neighbouring layer declares next to this stack: the last
    one of the nearest layer before it, or the first of the nearest after.
    Branches are side paths, so they never stand next to the stack."""
    global _stack
    for build_layer in build_layers:
        outer = _stack
        _stack = _LayerStack()
        try:
            build_layer()
            appends = [a for a in _stack.appends if a.prepared_from is None]
        finally:
            _stack = outer
        if appends:
            return appends[-1].declarations[-1] if last else appends[0].declarations[0]
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


class _PendingStage:
    """The last stage of a linear append. Its consumer is the next append's
    first stage, or the stack exit, so it binds only once that is known."""

    __slots__ = (
        "boundary",
        "declaration",
        "norm",
        "options",
        "predecessor",
        "predecessor_incoming",
        "predecessor_boundary",
        "origin",
    )

    def __init__(
        self,
        boundary,
        declaration,
        norm,
        options,
        predecessor,
        predecessor_incoming,
        predecessor_boundary,
        origin,
    ):
        self.boundary = boundary
        self.declaration = declaration
        self.norm = norm
        self.options = options
        # The stage before it within the same append, or None when it is the
        # append's only stage and takes its input from the stack.
        self.predecessor = predecessor
        self.predecessor_incoming = predecessor_incoming
        self.predecessor_boundary = predecessor_boundary
        self.origin = origin


class _Chain:
    """The stage a following append extends, and the one stage still waiting
    for its consumer, while the stack binds its appends in order."""

    __slots__ = ("previous", "pending")

    def __init__(self, previous):
        self.previous = _detached(previous)
        self.pending = None


def _detached(declaration):
    """The declaration a following append extends, without its own history.

    A stage's incoming edge reads its producer's own incoming edge only when
    the producer is an attention that always leaves its sum; the copy has no
    history, so that lookback stops at the producer.
    """
    if declaration is None:
        return None
    return replace(declaration, previous=None, prepared_from=None)


def _bind_stack(appends, *, previous, following):
    """Bind every appended stage, in order, and fill in the boundaries each
    append returned."""
    chain = _Chain(previous)
    # A returned declaration's boundary as bound, for the branches that read it.
    sources = {}
    bound = []
    for append in appends:
        prepared_from = append.prepared_from
        if prepared_from is not None:
            source = sources.get(id(prepared_from))
            if source is None:
                raise ValueError(
                    "prepared_from must be the declaration of a stage appended "
                    f"earlier to the same layer stack (at {append.origin})"
                )
            prepared_from = source.declaration
        boundaries = _extend(chain, append, prepared_from)
        for returned, boundary in zip(append.boundaries, boundaries):
            sources[id(returned.declaration)] = boundary
        bound.append(boundaries)
    # The last stage's consumer is the next rank's first stage, if any;
    # without one it ends the model's layer stack.
    _bind_pending(chain, consumer=following, terminal=following is None)
    for append, boundaries in zip(appends, bound):
        for returned, boundary in zip(append.boundaries, boundaries):
            returned.plan = boundary.plan
            returned.declaration = boundary.declaration


def _bind_pending(chain: _Chain, *, consumer, terminal):
    pending = chain.pending
    if pending is None:
        return
    chain.pending = None
    try:
        declaration = replace(pending.declaration, terminal=terminal)
        if pending.predecessor is None:
            incoming = _incoming(declaration)
        else:
            incoming = _connect(
                pending.predecessor,
                declaration,
                residual_from=pending.predecessor_incoming,
            )
        outgoing = _connect(declaration, consumer, residual_from=incoming)
        bound = _bind_stage(
            declaration, pending.norm, incoming, outgoing, **pending.options
        )
    except Exception as error:
        _note_origin(error, pending.origin)
        raise
    pending.boundary.plan = bound.plan
    pending.boundary.declaration = declaration
    if pending.predecessor_boundary is not None:
        _carry_capture(pending.predecessor_boundary, pending.boundary)


def _carry_capture(producer, consumer):
    """Let an attention that always leaves its sum preserve the residual its
    FFN's entry needs for capture. Only between stages of one append."""
    if not (
        producer.kind is StageKind.ATTENTION
        and producer.declaration.reduction is ProducerReduction.ALWAYS_PARTIAL
        and consumer.kind is StageKind.FFN
    ):
        return
    for variant, steps in producer.plan.paths.items():
        next_steps = consumer.plan.paths[variant]
        predicate = next_steps.entry.preserves_residual
        if predicate is not None:
            producer.plan.paths[variant] = msgspec.structs.replace(
                steps,
                entry=msgspec.structs.replace(
                    steps.entry, capture_preserves_residual=predicate
                ),
            )


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


def _extend(chain, append, prepared_from):
    """Chain one append onto the stack and bind what can be bound."""
    declarations, bindings = append.declarations, append.bindings
    branch = prepared_from is not None
    if not branch:
        # This append's first stage is the consumer the pending stage waited for.
        _bind_pending(chain, consumer=declarations[0], terminal=False)
    try:
        chained = []
        for index, declaration in enumerate(declarations):
            chained.append(
                replace(
                    declaration,
                    previous=(
                        chained[-1] if index else (None if branch else chain.previous)
                    ),
                    prepared_from=prepared_from if index == 0 else None,
                )
            )
        boundaries = []
        last = len(chained) - 1
        # incoming feeds the stage being bound; previous_incoming fed the one
        # before.
        previous_incoming, incoming = None, _incoming(chained[0])
        for index, (declaration, (norm, options)) in enumerate(zip(chained, bindings)):
            if index == last and not branch:
                boundaries.append(StageBoundary(None, declaration=declaration))
                chain.pending = _PendingStage(
                    boundaries[-1],
                    declaration,
                    norm,
                    options,
                    predecessor=chained[index - 1] if index else None,
                    predecessor_incoming=previous_incoming,
                    predecessor_boundary=boundaries[-2] if index else None,
                    origin=append.origin,
                )
                break
            following = chained[index + 1] if index < last else None
            outgoing = _connect(declaration, following, residual_from=incoming)
            boundaries.append(
                _bind_stage(declaration, norm, incoming, outgoing, **options)
            )
            previous_incoming, incoming = incoming, outgoing
    except Exception as error:
        _note_origin(error, append.origin)
        raise
    if branch:
        for producer, consumer in zip(boundaries, boundaries[1:]):
            _carry_capture(producer, consumer)
    else:
        # The pair that ends on the pending stage is carried when it binds.
        for producer, consumer in zip(boundaries[:-2], boundaries[1:-1]):
            _carry_capture(producer, consumer)
        chain.previous = _detached(chained[-1])
    return tuple(boundaries)
