"""Construct one computing stage and its transport from a predecessor contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Optional

import msgspec

from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.communicator.boundary import (
    EdgeDecl,
    StageDecl,
    StageInput,
    StageKind,
    StageOutput,
    _cp_moves,
)
from sglang.srt.layers.communicator.construction import (
    BatchVariant,
    StageEdges,
    StagePlan,
    _input_can_be_scattered,
    _refuse_uncovered_cp_moe,
    _use_ag_after_qlora,
)
from sglang.srt.layers.communicator.layout import (
    Layout,
    SumGroup,
    TokenAxis,
    _gathers_over_attention_cp,
    _generic_prefill_cp_shards_tokens,
    enable_moe_dense_fully_dp,
    sparse_moe_gathers_over_moe_cp,
    token_axis_sizes,
)
from sglang.srt.layers.communicator.ops import (
    _hand_qkv_hook_its_input,
    _hand_scattered_input_to_attention,
)
from sglang.srt.layers.communicator.output import OutputTransform
from sglang.srt.layers.communicator.residual import StageRead, StageUpdate
from sglang.srt.layers.communicator.residual.add_norm import (
    ADD,
    NORM_QUANT_READ,
    NORM_READ,
)
from sglang.srt.layers.communicator.stage import StageCommunicator
from sglang.srt.layers.moe import is_moe_input_scattered_across_dp_ranks
from sglang.srt.runtime_context import get_exec, get_parallel


def _variants(*, ordinary_only=False):
    yield BatchVariant.ORDINARY
    if ordinary_only:
        return
    if _generic_prefill_cp_shards_tokens():
        yield BatchVariant.CONTEXT_PARALLEL
    if _input_can_be_scattered():
        yield BatchVariant.INPUT_SCATTERED
    if layernorm_sp.layernorm_sp_enabled():
        yield BatchVariant.SEQUENCE_PARALLEL


def _rows(variant):
    axes = token_axis_sizes(cp_active=variant is BatchVariant.CONTEXT_PARALLEL)
    attention = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, axis_sizes=axes
    )
    local = Layout.sharded_over(
        TokenAxis.ATTN_DP, TokenAxis.ATTN_CP, TokenAxis.ATTN_TP_SCATTER, axis_sizes=axes
    )
    return axes, attention, local, Layout.sharded_over(axis_sizes=axes)


def _ffn_decl(
    variant,
    *,
    sparse,
    terminal=False,
    next_sparse=False,
    output=None,
    read=NORM_READ,
    update=ADD,
    local_dense=True,
):
    parallel = get_parallel()
    reduction = get_exec().comm.boundary_reduction
    if reduction not in ("ar", "rs", "rsv", "rs+rsv"):
        raise ValueError(
            "boundary_reduction must be resolved before model construction"
        )
    can_move_output = output is None or output.before_reduce_scatter
    use_reduce_scatter = reduction in ("rs", "rs+rsv") and can_move_output
    use_reduce_scatterv = reduction in ("rsv", "rs+rsv") and can_move_output
    cp_shards = _generic_prefill_cp_shards_tokens()
    axes, attention, local, full = _rows(variant)
    on_local = (
        is_moe_input_scattered_across_dp_ranks()
        if sparse
        else (local_dense and enable_moe_dense_fully_dp())
    )
    if parallel.attn_cp_size > 1 and sparse:
        _refuse_uncovered_cp_moe(on_local, cp_shards)
    on_cp = (
        sparse
        and parallel.attn_cp_size > 1
        and parallel.moe_dp_size == parallel.attn_cp_size
        and not _gathers_over_attention_cp()
    )
    group = SumGroup.MOE_OUTPUT if sparse else SumGroup.TP
    if variant is BatchVariant.SEQUENCE_PARALLEL:
        return (
            StageDecl(
                StageInput(
                    full,
                    gathers_itself=frozenset({TokenAxis.ATTN_TP_SCATTER}),
                    read=read,
                ),
                StageOutput(local, update=update, transform=output),
            ),
            local,
            local,
        )
    if variant is BatchVariant.INPUT_SCATTERED:
        scattered_residual = update.at_producer
        residual = local if scattered_residual else attention
        returned = local if scattered_residual and not terminal else attention
        return (
            StageDecl(
                StageInput(full, read=read),
                StageOutput(
                    attention,
                    group=group,
                    leaves_for_reduce_scatter=use_reduce_scatter
                    and (scattered_residual or not terminal),
                    update=update,
                    transform=output,
                ),
            ),
            residual,
            returned,
        )
    if cp_shards and _cp_moves().reduce_scatter is not None:
        may_leave = variant is not BatchVariant.CONTEXT_PARALLEL
        may_scatter = True
    elif cp_shards or parallel.attn_dp_size > 1:
        may_leave = may_scatter = parallel.attn_cp_size == 1 or on_cp
    else:
        may_leave = may_scatter = True
    rows = (
        local
        if on_local
        else Layout.sharded_over(
            *((TokenAxis.ATTN_CP,) if on_cp else ()), axis_sizes=axes
        )
    )
    output = (
        StageOutput(rows, update=update, transform=output)
        if on_local
        else StageOutput(
            rows,
            group=group,
            leaves_for_next_layer=may_leave
            and not terminal
            and not update.at_producer
            and update.can_defer_across_layers
            and output is None,
            leaves_for_reduce_scatter=use_reduce_scatter and may_scatter,
            leaves_for_reduce_scatterv=use_reduce_scatterv and may_leave,
            update=update,
            transform=output,
        )
    )
    gathers_for_tbo = (
        enable_moe_dense_fully_dp()
        and get_exec().overlap.enable_two_batch_overlap
        and not sparse
        and bool(next_sparse)
    )
    returned = local if on_local and not terminal and not gathers_for_tbo else attention
    return (
        StageDecl(StageInput(rows, read=read), output),
        local if on_local else attention,
        returned,
    )


@dataclass(frozen=True)
class StageDeclaration:
    """A computation and its input source, without an execution plan.

    ``previous`` consumes a producer's output. ``prepared_from`` reuses an
    already-read input for a branch; the two sources are mutually exclusive.
    Sources may be reconstructed locally, including across pipeline ranks.
    """

    kind: StageKind
    read: StageRead
    update: StageUpdate
    sparse: bool = False
    terminal: bool = False
    next_sparse: bool = False
    output: Optional[OutputTransform] = None
    ordinary_only: bool = False
    # A mixer with its own exit publishes reduction decisions before compute.
    mixer_exit: bool = False
    next_kind: Optional[StageKind] = None
    return_to_attention: bool = False
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
    """The producer's exit and consumer's entry for each batch variant.

    Immutable, norm-free declarations. Independently constructed equivalent
    connections work across pipeline partitions; object identity is irrelevant.
    """

    producer: Optional[StageDeclaration]
    consumer: Optional[StageDeclaration]
    exits: Mapping[BatchVariant, EdgeDecl]
    entries: Mapping[BatchVariant, EdgeDecl]

    def __post_init__(self):
        object.__setattr__(self, "exits", MappingProxyType(dict(self.exits)))
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))


def declare_attn(
    *,
    previous: Optional[StageDeclaration] = None,
    prepared_from: Optional[StageDeclaration] = None,
    read=NORM_QUANT_READ,
    update=ADD,
    terminal=False,
    ordinary_only=False,
    mixer_exit=False,
    next_kind=None,
):
    return StageDeclaration(
        StageKind.ATTENTION,
        read,
        update,
        previous=previous,
        prepared_from=prepared_from,
        terminal=terminal,
        ordinary_only=ordinary_only,
        mixer_exit=mixer_exit,
        next_kind=next_kind,
    )


def declare_ffn(
    *,
    previous: Optional[StageDeclaration] = None,
    prepared_from: Optional[StageDeclaration] = None,
    sparse=False,
    read=NORM_READ,
    update=ADD,
    terminal=False,
    next_sparse=False,
    output=None,
    ordinary_only=False,
    return_to_attention=False,
):
    return StageDeclaration(
        StageKind.FFN,
        read,
        update,
        sparse=sparse,
        previous=previous,
        prepared_from=prepared_from,
        terminal=terminal,
        next_sparse=next_sparse,
        output=output,
        ordinary_only=ordinary_only,
        return_to_attention=return_to_attention,
    )


def _resolve(stage, variant):
    axes, attention, local, full = _rows(variant)
    if stage.kind is StageKind.FFN:
        if stage.update.at_producer:
            if stage.sparse and sparse_moe_gathers_over_moe_cp():
                raise NotImplementedError(
                    "MHC does not support a MoE gathered over the MoE-CP group"
                )
            if get_parallel().attn_cp_size > 1 and _input_can_be_scattered():
                raise NotImplementedError(
                    "MHC with input-scattered attention under attention CP"
                )
        declaration, residual, returned = _ffn_decl(
            variant,
            sparse=stage.sparse,
            terminal=stage.terminal,
            next_sparse=stage.next_sparse,
            output=stage.output,
            read=stage.read,
            update=stage.update,
            local_dense=not stage.return_to_attention,
        )
        if stage.return_to_attention:
            returned = attention
        return declaration, residual, returned
    sp = variant is BatchVariant.SEQUENCE_PARALLEL
    scattered = variant is BatchVariant.INPUT_SCATTERED
    gathers = (
        frozenset({TokenAxis.ATTN_TP_SCATTER})
        if not stage.mixer_exit and (sp or scattered or _use_ag_after_qlora)
        else frozenset()
    )
    owes = not sp and axes[TokenAxis.ATTN_TP_SCATTER] > 1
    declaration = StageDecl(
        StageInput(attention, gathers_itself=gathers, read=stage.read),
        StageOutput(
            local if sp else attention,
            group=SumGroup.ATTN_TP if owes else None,
            always_leaves=owes
            and (not stage.mixer_exit or stage.next_kind is StageKind.FFN),
            leaves_for_next_layer=owes
            and stage.mixer_exit
            and stage.next_kind is StageKind.ATTENTION,
            update=stage.update,
        ),
    )
    return declaration, None, attention


def _connect(producer, consumer, *, residual_from=None):
    """Resolve producer and consumer declarations; None denotes an external endpoint.

    A missing producer is the stack input; a missing consumer is a handoff
    whose next read is bound separately (or the terminal output).
    ``residual_from`` names the boundary that placed the producer's residual.
    It is needed for attention, whose residual can stay on finer rows than its
    compute input. No executable stage participates in this construction.
    """
    if producer is None and consumer is None:
        raise ValueError("a boundary needs at least one declared side")
    before, after = producer, consumer
    ordinary_only = any(s.ordinary_only for s in (before, after) if s is not None)
    exits, entries = {}, {}
    for variant in _variants(ordinary_only=ordinary_only):
        _, attention, local, _ = _rows(variant)
        if before is None:
            rows = local if variant is BatchVariant.SEQUENCE_PARALLEL else attention
            owes = variant is BatchVariant.INPUT_SCATTERED
            arrived = StageOutput(
                rows,
                group=SumGroup.TP if owes else None,
                always_leaves=owes,
                update=None,
            )
            residual = rows
            capabilities = (True,)
        else:
            decl, during, returned = _resolve(before, variant)
            if during is None:
                if residual_from is not None:
                    if residual_from.consumer != producer:
                        raise ValueError("residual source must enter the producer")
                    during = residual_from.entries[variant].residual_to
                elif before.mixer_exit:
                    during = attention
                else:
                    raise ValueError(
                        "attention output needs its incoming residual placement"
                    )
            exits[variant] = EdgeDecl(
                decl.output, StageInput(returned), during, returned
            )
            if before.kind is StageKind.ATTENTION and not before.mixer_exit:
                arrived, residual, capabilities = decl.output, during, ()
            else:
                owes = (
                    variant is BatchVariant.INPUT_SCATTERED
                    and not before.update.at_producer
                )
                carries = (before.mixer_exit or before.return_to_attention) and (
                    decl.output.always_leaves or decl.output.leaves_for_next_layer
                )
                arrived = StageOutput(
                    returned,
                    group=decl.output.group
                    if carries
                    else (SumGroup.TP if owes else None),
                    always_leaves=decl.output.always_leaves if carries else owes,
                    leaves_for_next_layer=decl.output.leaves_for_next_layer
                    if carries
                    else False,
                    update=None,
                )
                residual, capabilities = returned, (before.update.adds_plainly,)
        if after is None:
            continue
        decl, during, _ = _resolve(after, variant)
        if after.kind is StageKind.ATTENTION:
            during = (
                Layout(residual.sharded | {TokenAxis.ATTN_TP_SCATTER})
                if variant is BatchVariant.INPUT_SCATTERED
                and arrived.always_leaves
                and arrived.update is None
                and not after.mixer_exit
                else residual
            )
        elif after.return_to_attention:
            during = (
                decl.input.layout
                if residual.sharded <= decl.input.layout.sharded
                else residual
            )
        joins = (
            variant is BatchVariant.INPUT_SCATTERED
            and arrived.update is not None
            and arrived.update.adds_plainly
            and after.kind is StageKind.FFN
        )
        edge = EdgeDecl(
            arrived,
            decl.input,
            residual,
            during,
            residual_joins_sum=joins,
            update_capabilities=capabilities,
        )
        entries[variant] = edge
        if (
            before is not None
            and before.kind is StageKind.ATTENTION
            and not before.mixer_exit
        ):
            exits[variant] = edge
    return StageConnection(producer, consumer, exits, entries)


def _fork_input(prepared, consumer):
    """Place an already-read input onto a branch's computation rows.

    The branch adapter performs the move and forks the stream. It does not
    execute the branch's input norm again.
    """
    entries = {}
    for variant in _variants(ordinary_only=consumer.ordinary_only):
        source = prepared.entries[variant]
        declaration, during, _ = _resolve(consumer, variant)
        entries[variant] = EdgeDecl(
            StageOutput(source.need.layout, update=None),
            declaration.input,
            source.residual_to,
            during,
            update_capabilities=(True,),
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
        and not previous.mixer_exit
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


def _bind_stage(declaration, norm, incoming, outgoing, **options):
    if incoming.consumer != declaration or outgoing.producer != declaration:
        raise ValueError("connections do not match the stage declaration")
    if incoming.entries.keys() != outgoing.exits.keys():
        raise ValueError("incoming and outgoing batch variants disagree")
    if declaration.update.at_producer:
        if declaration.sparse and sparse_moe_gathers_over_moe_cp():
            raise NotImplementedError(
                "MHC does not support a MoE gathered over the MoE-CP group"
            )
        if get_parallel().attn_cp_size > 1 and _input_can_be_scattered():
            raise NotImplementedError(
                "MHC with input-scattered attention under attention CP"
            )
    variants = {}
    for variant, edge in incoming.entries.items():
        if (
            declaration.update.at_producer
            and TokenAxis.ATTN_CP
            in edge.produced.layout.sharded - edge.need.layout.sharded
        ):
            raise NotImplementedError("MHC with a gather over attention CP")
        handoff = None
        if declaration.kind is StageKind.ATTENTION:
            handoff = (
                _hand_scattered_input_to_attention
                if variant is BatchVariant.INPUT_SCATTERED
                and not declaration.update.adds_plainly
                else _hand_qkv_hook_its_input
            )
        moves = _cp_moves() if variant is BatchVariant.CONTEXT_PARALLEL else None
        variants[variant] = StageEdges(edge, outgoing.exits[variant], handoff, moves)
    plan = StagePlan(
        declaration.kind,
        norm,
        variants,
        enters_stack=incoming.producer is None,
        prepared_input=declaration.prepared_from is not None,
        terminal=declaration.terminal,
        fixed_output=declaration.kind is StageKind.ATTENTION
        and not declaration.mixer_exit,
        is_sparse=declaration.sparse,
        **options,
    )
    if declaration.kind is StageKind.ATTENTION and outgoing.consumer is not None:
        # Layout eligibility comes from the connected consumer, not a mutable
        # link to its execution plan. Kernel binding remains consumer-owned.
        from sglang.srt.layers.communicator.boundary import make_boundary

        plan._fusion_rows = {
            v: make_boundary(
                edge,
                cp_moves=_cp_moves() if v is BatchVariant.CONTEXT_PARALLEL else None,
                force_layernorm_before_gather=options.get(
                    "force_layernorm_before_dp_gather", False
                ),
            ).input_rows
            for v, edge in outgoing.entries.items()
        }
    return StageCommunicator(plan, declaration.kind, norm, declaration=declaration)


def make_attn_stage(
    *,
    declaration,
    norm,
    following: Optional[StageDeclaration] = None,
    qkv_latent_func=None,
    force_layernorm_before_dp_gather=False,
    enable_fused_ar_quant=False,
    fused_ar_quant_keep_bf16=False,
    residual_in_hidden=False,
    fusions=None,
):
    """Resolve this attention's boundaries, then bind its norm and hooks.

    ``following`` supplies the consumer declaration for a stage inside a
    layer. With no following stage, the output is handed across a layer or
    stack boundary; the receiver binds its own read independently.
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
        force_layernorm_before_dp_gather=force_layernorm_before_dp_gather,
        enable_fused_ar_quant=enable_fused_ar_quant,
        fused_ar_quant_keep_bf16=fused_ar_quant_keep_bf16,
        residual_in_hidden=residual_in_hidden,
        fusions=fusions,
    )


def make_ffn_stage(
    *,
    declaration,
    norm,
    following: Optional[StageDeclaration] = None,
    force_layernorm_before_dp_gather=False,
    fusions=None,
):
    """Resolve this FFN's boundaries and bind its norm; EP stays inside compute."""
    if declaration.kind is not StageKind.FFN:
        raise TypeError("make_ffn_stage requires an FFN declaration")
    incoming, outgoing = _connections(declaration, following)
    return _bind_stage(
        declaration,
        norm,
        incoming,
        outgoing,
        force_layernorm_before_dp_gather=force_layernorm_before_dp_gather,
        residual_in_hidden=declaration.update.at_producer,
        fusions=fusions,
    )


def make_stages(*stages, previous=None, prepared_from=None, terminal=False):
    """Bind a local linear sequence and return its independent boundaries.

    Each item is ``(declaration, norm)`` or ``(declaration, norm, options)``.
    Options are the stage-specific keyword arguments of ``make_attn_stage``
    or ``make_ffn_stage``; they are consumed only during construction.
    Declarations in the sequence have no input source: this function connects
    them in order. Only the final boundary is terminal when requested.

    ``previous`` describes an external producer's output. ``prepared_from``
    describes an already-read branch input, obtainable from the source
    boundary's immutable ``declaration``. Neither takes an executable boundary.
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
            next_kind=stages[index + 1][0].kind
            if declaration.mixer_exit and index + 1 < len(stages)
            else declaration.next_kind,
        )
        declarations.append(declaration)
        bindings.append((norm, item[2] if len(item) == 3 else {}))
    boundaries = []
    for index, (declaration, (norm, options)) in enumerate(zip(declarations, bindings)):
        build = (
            make_attn_stage
            if declaration.kind is StageKind.ATTENTION
            else make_ffn_stage
        )
        boundaries.append(
            build(
                declaration=declaration,
                norm=norm,
                following=declarations[index + 1]
                if index + 1 < len(declarations)
                else None,
                **options,
            )
        )
    for producer, consumer in zip(boundaries, boundaries[1:]):
        if (
            producer.kind is not StageKind.ATTENTION
            or producer.declaration.mixer_exit
            or consumer.kind is not StageKind.FFN
        ):
            continue
        for variant, steps in producer.plan._paths.items():
            predicate = consumer.plan._paths[variant].ffn.preserves_residual
            if predicate is not None:
                producer.plan._paths[variant] = msgspec.structs.replace(
                    steps,
                    attention=msgspec.structs.replace(
                        steps.attention, capture_preserves_residual=predicate
                    ),
                )
        producer.plan._steps = producer.plan._paths[BatchVariant.ORDINARY]
        producer.plan._cp_steps = producer.plan._paths.get(
            BatchVariant.CONTEXT_PARALLEL
        )
        producer.plan._input_scattered_steps = producer.plan._paths.get(
            BatchVariant.INPUT_SCATTERED
        )
        producer.plan._sp_steps = producer.plan._paths.get(
            BatchVariant.SEQUENCE_PARALLEL
        )
    return tuple(boundaries)
