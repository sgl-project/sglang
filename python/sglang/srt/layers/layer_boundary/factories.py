"""Construct one computing stage and its transport from a predecessor contract."""

from __future__ import annotations

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
)
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
    reduction=ProducerReduction.EXIT_SCOPED,
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
    produced = (
        OutputContract(rows, update=update, transform=output_transform)
        if on_rank_rows
        else OutputContract(
            rows,
            group=group,
            may_defer_to_next=may_leave
            and not terminal
            and not update.applied_at_exit
            and update.outlives_layer
            and output_transform is None
            and reduction is ProducerReduction.EXIT_SCOPED,
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
        terminal: Whether this stage ends the model's layer stack. Prevents
            leaving work that requires a following layer; a finalize handoff
            may still reach the terminal norm when the fusion provider allows it.
        output_transform: Optional operation on the contribution before the
            residual update. An FFN's exit runs it under an explicit
            reduction-order contract; an attention's is run by the input of
            the stage that follows, once the attention's sum is complete.
        reduction: Whether compute always leaves a partial sum, obeys the
            exit scope, or adds a replicated component after its own sum.
        gathers_attn_tp_input: Whether attention gathers TP-sharded input itself.
        dense_tp_size: Dense FFN compute width: None uses the configured width,
            1 means local compute, and the full TP size means TP compute.
        exit_rows: Required FFN output rows at the layer or branch exit.
        previous: Declaration whose output this stage consumes. It may be
            reconstructed locally, including across pipeline ranks.
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
    previous: Optional[StageDeclaration] = None,
    prepared_from: Optional[StageDeclaration] = None,
    read=NORM_QUANT_READOUT,
    update=PLAIN_ADD,
    terminal=False,
    reduction=ProducerReduction.ALWAYS_PARTIAL,
    gathers_attn_tp_input=True,
    output_transform=None,
):
    """Declare attention or a mixer; construct its executable boundary later.

    Args:
        previous: Producer declaration whose output this stage consumes.
        prepared_from: Source of an already-read branch input, instead of previous.
        read: Input operation; defaults to normalization with quantization support.
        update: Operation that adds this stage's output to the residual.
        terminal: Whether this stage ends the model's layer stack.
        reduction: ALWAYS_PARTIAL for an output projection that always skips reduction;
            EXIT_SCOPED for a mixer that follows its exit scope's reduction decision.
            TAIL_AFTER_SUM is rejected for attention stages.
        gathers_attn_tp_input: Whether compute gathers attention-TP input slices itself.
        output_transform: Optional operation on the output once its sum is
            complete, before the residual update (a sandwich norm). The next
            stage's input runs it, so no fused add + norm takes that input.
            Requires ALWAYS_PARTIAL.

    Returns:
        A StageDeclaration with no norm, tensors or execution plan.
    """
    if reduction is ProducerReduction.TAIL_AFTER_SUM:
        raise ValueError("TAIL_AFTER_SUM is not supported for attention stages")
    if (
        output_transform is not None
        and reduction is not ProducerReduction.ALWAYS_PARTIAL
    ):
        raise ValueError("an attention output transform requires ALWAYS_PARTIAL")
    return StageDeclaration(
        StageKind.ATTENTION,
        read,
        update,
        previous=previous,
        prepared_from=prepared_from,
        terminal=terminal,
        output_transform=output_transform,
        reduction=reduction,
        gathers_attn_tp_input=gathers_attn_tp_input,
    )


def declare_ffn(
    *,
    previous: Optional[StageDeclaration] = None,
    prepared_from: Optional[StageDeclaration] = None,
    sparse=False,
    read=NORM_READOUT,
    update=PLAIN_ADD,
    terminal=False,
    output_transform=None,
    next_layer_sparse=False,
    dense_tp_size=None,
    reduction=ProducerReduction.EXIT_SCOPED,
    exit_rows=None,
):
    """Declare a dense or MoE FFN independently of its compute module.

    Args:
        previous: Producer declaration whose output this stage consumes.
        prepared_from: Source of an already-read branch input, instead of previous.
        sparse: Whether the FFN is a MoE; EP dispatch stays inside compute.
        read: Operation deriving FFN input from the residual.
        update: Operation writing FFN output into the residual.
        terminal: Whether this stage ends the model's layer stack.
        output_transform: Optional contribution transform before residual update.
        next_layer_sparse: Whether the next decoder layer's FFN is sparse; used only
            to derive the TBO exit rows when exit_rows is not supplied.
        dense_tp_size: Dense compute width: None for configuration, 1 for local
            compute, or the full TP size.
        reduction: How compute cooperates with the exit's reduction decision:
            EXIT_SCOPED (default) follows the exit scope; TAIL_AFTER_SUM marks a
            replicated tail after the sum. ALWAYS_PARTIAL is rejected for FFN stages.
        exit_rows: Explicit output-row requirement; otherwise derived from
            the adjacent FFN kinds and TBO configuration.

    Returns:
        A StageDeclaration with no norm, tensors or execution plan.
    """
    if reduction is ProducerReduction.ALWAYS_PARTIAL:
        raise ValueError("ALWAYS_PARTIAL is not supported for ffn stages")
    return StageDeclaration(
        StageKind.FFN,
        read,
        update,
        sparse=sparse,
        previous=previous,
        prepared_from=prepared_from,
        terminal=terminal,
        output_transform=output_transform,
        dense_tp_size=dense_tp_size,
        reduction=reduction,
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
            reduction=stage.reduction,
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
    declaration = StageContract(
        InputContract(attention, gathered_by_compute=gathers, read=stage.read),
        OutputContract(
            local if sp else attention,
            group=SumGroup.ATTN_TP if owes else None,
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


def _connections(stage, following=None):
    if following is not None and following.previous != stage:
        raise ValueError("the following declaration must consume this stage's output")
    incoming = _incoming(stage)
    outgoing = _connect(
        stage,
        following,
        residual_from=incoming,
    )
    return incoming, outgoing


def make_attn_stage(
    *,
    declaration,
    norm,
    following: Optional[StageDeclaration] = None,
    qkv_latent_func=None,
    fusions=None,
):
    """Resolve one attention/mixer's boundaries and bind its input norm.

    Args:
        declaration: Attention StageDeclaration, including its input source.
        norm: This consumer's normalization module, never its neighbour's norm.
        following: Local consumer declaration whose previous is declaration.
            None leaves a layer or stack exit for an independently bound reader.
        qkv_latent_func: Optional attention input hook, invoked after preparation
            and movement onto the compute input rows.
        fusions: Optional backend provider. Consumer side: ordered
            attention_input(plan) and ffn_input(plan) candidates. Producer side
            (FFN exit): can_defer_finalize(plan, batch), called on every exit,
            and can_defer_all_reduce(plan, batch), called when LoRA or TP1 shared
            experts are enabled.

    Returns:
        A StageBoundary with precomputed paths for supported batch variants.
    """
    if declaration.kind is not StageKind.ATTENTION:
        raise TypeError("make_attn_stage requires an attention declaration")
    incoming, outgoing = _connections(declaration, following)
    return _bind_stage(
        declaration,
        norm,
        incoming,
        outgoing,
        qkv_latent_func=qkv_latent_func,
        fusions=fusions,
    )


def make_ffn_stage(
    *,
    declaration,
    norm,
    following: Optional[StageDeclaration] = None,
    fusions=None,
):
    """Resolve one FFN's boundaries and bind its input norm.

    Args:
        declaration: FFN StageDeclaration, including its input source.
        norm: This FFN's input normalization module.
        following: Local consumer declaration whose previous is declaration;
            None leaves a layer or stack exit for an independently bound reader.
        fusions: Optional backend fusion provider, as in make_attn_stage.

    Returns:
        A StageBoundary. Expert routing and all-to-all stay inside compute.
    """
    if declaration.kind is not StageKind.FFN:
        raise TypeError("make_ffn_stage requires an FFN declaration")
    incoming, outgoing = _connections(declaration, following)
    return _bind_stage(
        declaration,
        norm,
        incoming,
        outgoing,
        fusions=fusions,
    )


def make_stages(
    *stages, previous=None, prepared_from=None, following=None, terminal=False
):
    """Bind a local linear sequence of any positive number of stages.

    Args:
        *stages: Items of (declaration, norm) or (declaration, norm, options).
            Declarations must have no source or terminal flag. Options are
            constructor keywords: fusions, and qkv_latent_func for attention.
        previous: External producer declaration consumed by the first stage.
        prepared_from: Already-read input declaration reused by the first stage
            of a branch; mutually exclusive with previous.
        following: External consumer declaration after the last local stage.
            None denotes a layer or stack exit with an independently bound read.
        terminal: Marks only the final stage as the end of the model's stack.

    Returns:
        A tuple of independent StageBoundary objects in declaration order.
        Sources are connected on copied declarations; caller inputs are unchanged.
        No sequence object or runtime routing is retained.
    """
    if not stages:
        raise ValueError("make_stages needs at least one stage")
    if previous is not None and prepared_from is not None:
        raise ValueError("choose a previous output or a prepared branch input")
    declarations = []
    bindings = []
    for index, item in enumerate(stages):
        if len(item) not in (2, 3):
            raise ValueError("a stage needs (declaration, norm[, options])")
        declaration, norm = item[:2]
        if not isinstance(declaration, StageDeclaration):
            raise TypeError("make_stages requires stage declarations")
        if declaration.previous is not None or declaration.prepared_from is not None:
            raise ValueError("pass input sources to make_stages, not its declarations")
        if declaration.terminal:
            raise ValueError("pass terminal to make_stages, not its declarations")
        declaration = replace(
            declaration,
            previous=previous if index == 0 else declarations[-1],
            prepared_from=prepared_from if index == 0 else None,
            terminal=terminal and index == len(stages) - 1,
        )
        declarations.append(declaration)
        bindings.append((norm, item[2] if len(item) == 3 else {}))
    external_consumer = following
    boundaries = []
    incoming = _incoming(declarations[0])
    for index, (declaration, (norm, options)) in enumerate(zip(declarations, bindings)):
        following = (
            declarations[index + 1]
            if index + 1 < len(declarations)
            else external_consumer
        )
        outgoing = _connect(declaration, following, residual_from=incoming)
        options = dict(options)
        if declaration.kind is StageKind.FFN:
            unexpected = options.keys() - {
                "fusions",
            }
            if unexpected:
                raise TypeError(f"unsupported FFN options: {sorted(unexpected)}")
        boundaries.append(_bind_stage(declaration, norm, incoming, outgoing, **options))
        incoming = outgoing
    for producer, consumer in zip(boundaries, boundaries[1:]):
        if not (
            producer.kind is StageKind.ATTENTION
            and producer.declaration.reduction is ProducerReduction.ALWAYS_PARTIAL
            and consumer.kind is StageKind.FFN
        ):
            continue
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
    return tuple(boundaries)
