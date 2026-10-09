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
"""Bind one stage to predeclared boundaries and reusable batch paths."""

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional

from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.adapters.attention import get_attn_tp_context
from sglang.srt.layers.layer_boundary.boundary import (
    ExitMove,
    _cp_moves,
    bind_entry,
    bind_exit,
)
from sglang.srt.layers.layer_boundary.contracts import (
    BatchVariant,
    CpMoves,
    EdgeContract,
    EntryPath,
    ProducerReduction,
    StageKind,
    StagePath,
)
from sglang.srt.layers.layer_boundary.exit import ExitPolicy
from sglang.srt.layers.layer_boundary.fusions.allreduce import (
    attn_input_fusions,
    ffn_input_fusions,
)
from sglang.srt.layers.layer_boundary.layout import (
    TokenAxis,
    _batch_shards_over_cp,
    _cp_gathers_over_attn_cp,
    _prefill_cp_shards_tokens,
    batches_are_unpadded,
    is_dense_ffn_fully_dp,
)
from sglang.srt.layers.layer_boundary.ops import (
    attn_tp_gather_input,
    update_attn_tp_gather_output,
)
from sglang.srt.layers.layer_boundary.prepare import (
    _attn_input_default,
    _attn_input_scattered,
)
from sglang.srt.layers.layer_boundary.stage import StageBoundary
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    is_moe_input_scattered_across_dp_ranks,
)
from sglang.srt.runtime_context import (
    get_forward,
    get_lora,
    get_parallel,
    get_spec,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

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


def _rows_indivisible_over_attn_tp(forward_batch, attn_tp_size: int) -> bool:
    """Whether this batch arrived with rows that do not divide over attention TP."""
    return forward_batch.input_ids.shape[0] % attn_tp_size != 0


def _input_scattered_possible() -> bool:
    """Whether a batch may run this layer with input-scattered attention:
    configured, on TP without attention DP, a prefill CP, an a2a backend or
    a dense MLP on every rank. The rest of what ``AttnTpContext.init_context`` requires is only
    known once the model is built."""
    parallel = get_parallel()
    return (
        parallel.enable_attn_tp_input_scattered
        and parallel.tp_size > 1
        and parallel.attn_dp_size == 1
        and not _prefill_cp_shards_tokens()
        and get_moe_a2a_backend().is_none()
        and not is_dense_ffn_fully_dp()
    )


@dataclass(frozen=True)
class VariantEdges:
    """One stage's incoming and outgoing contracts for one batch variant.

    Fields:
        incoming: Consumer-side contract used to bind prepare.
        outgoing: Producer-side contract used to bind exit transport.
        attn_input_adapter: Optional attention adapter run after input preparation/movement.
        cp_moves: Strategy-specific context-parallel gather and return operations.

    No executable neighbouring stage or neighbouring norm is required.
    """

    incoming: EdgeContract
    outgoing: EdgeContract
    attn_input_adapter: Optional[Callable] = None
    cp_moves: Optional[CpMoves] = None


def _requires_branch_input(*args, **kwargs):
    raise RuntimeError("a prepared branch must enter through branch_input, not prepare")


class StagePlan:
    """Bind reusable entry/exit paths from a stage's resolved declarations.

    Args:
        kind: Boundary role selecting attention or FFN fusion adapters.
        norm: This stage's consumer normalization module.
        variants: Mapping from BatchVariant to VariantEdges. Construction binds
            each entry once; a forward selects an existing path by batch facts.
        enters_stack: Whether this stage initializes the residual from embeddings.
        is_branch: Branch reusing an already-read input; requires branch_input
            instead of a normal prepare to avoid repeating the read/update.
        terminal: Whether the stage ends the model's layer stack.
        writes_at_handoff: Whether the stage, an FFN handing off to another
            pipeline rank, writes its output into the residual at its exit.
        finishes_directly: Attention can publish its output with finish instead of
            an exit scope and output transport.
        qkv_latent_func: Optional hook for prepared attention input.
        fusions: Optional backend provider. Consumer side: ordered
            attention_input(plan) and ffn_input(plan) candidates. Producer side
            (FFN exit): can_defer_finalize(plan, batch), called on every exit,
            and can_defer_all_reduce(plan, batch), called when LoRA or TP1
            shared experts are enabled.

    The plan owns static paths, not per-forward tensors or a neighbour's norm.
    Runtime residual state belongs to the ForwardBatch's ResidualStream.
    """

    def __init__(
        self,
        kind,
        norm,
        variants,
        *,
        enters_stack=False,
        is_branch=False,
        terminal=False,
        writes_at_handoff=False,
        finishes_directly=False,
        qkv_latent_func=None,
        fusions=None,
        attn_tp_gather=None,
    ):
        self.norm = norm
        self.edges = dict(variants)
        # The attention TP size an unpadded batch's rows are checked against,
        # when such a batch may arrive (see BatchVariant.UNPADDED).
        self._unpadded_attn_tp_size = (
            get_parallel().attn_tp_size if BatchVariant.UNPADDED in self.edges else None
        )
        self.enters_stack = enters_stack
        self.terminal = terminal
        self.finishes_directly = finishes_directly
        self.qkv_latent_func = qkv_latent_func
        self.fusions = fusions
        self._speculative_algo = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        self._publish_lora_layout = get_parallel().attn_dp_enabled and bool(
            get_lora().enable_lora
        )
        self._next_input_rows = None
        carried = (
            attn_input_fusions(self, next(iter(self.edges.values())).incoming.need.read)
            if kind is StageKind.ATTENTION
            else ()
        )
        fused = ffn_input_fusions(self) if kind is StageKind.FFN else ()
        self.paths = {}
        for variant, edges in self.edges.items():
            if is_branch:
                # The branch adapter moves an already-read input and forks its
                # stream. Binding a normal prepare would add or norm it twice.
                entry = EntryPath(
                    prepare=_requires_branch_input,
                    input_rows=edges.incoming.need.layout,
                )
            else:
                entry = bind_entry(
                    edges.incoming,
                    fusions=fused,
                    carried_fusions=carried,
                    cp_moves=edges.cp_moves,
                    enters_stack=enters_stack,
                    attn_input_adapter=edges.attn_input_adapter,
                    attn_tp_gather=attn_tp_gather,
                )
            out = (
                ExitMove()
                if finishes_directly
                else bind_exit(
                    edges.outgoing,
                    cp_moves=edges.cp_moves,
                    attn_tp_gather=attn_tp_gather,
                )
            )
            self.paths[variant] = StagePath(
                entry=entry,
                output=edges.outgoing.produced,
                output_move=out.output_move,
                output_move_completes_sum=out.output_move_completes_sum,
                returns_over_dp=out.returns_over_dp,
                writes_at_handoff=writes_at_handoff,
            )
        if attn_tp_gather is not None and not any(
            getattr(path.output_move, "func", None) is update_attn_tp_gather_output
            or getattr(path.entry.input_move, "func", None) is attn_tp_gather_input
            for path in self.paths.values()
        ):
            raise ValueError(
                "an attention-TP gather is declared, but no batch variant "
                "gathers over attention TP at this stage's boundaries"
            )

    @property
    def incoming_residual_rows(self):
        return self.edges[BatchVariant.ORDINARY].incoming.residual

    @property
    def input_on_attn_tp_slices(self):
        entry = self.paths[BatchVariant.ORDINARY].entry
        return TokenAxis.ATTN_TP in entry.input_rows.sharded

    def variant_for(self, forward_batch):
        # The batch determines its rows; missing paths must not change them.
        if get_forward().sp_active:
            return BatchVariant.SEQUENCE_PARALLEL
        if get_attn_tp_context().input_scattered:
            return BatchVariant.INPUT_SCATTERED
        if _batch_shards_over_cp(forward_batch):
            return BatchVariant.CONTEXT_PARALLEL
        if self._unpadded_attn_tp_size is not None and _rows_indivisible_over_attn_tp(
            forward_batch, self._unpadded_attn_tp_size
        ):
            return BatchVariant.UNPADDED
        return BatchVariant.ORDINARY

    def path_for(self, forward_batch):
        variant = self.variant_for(forward_batch)
        try:
            return self.paths[variant]
        except KeyError:
            raise NotImplementedError(
                f"no stage boundary path for the active {variant.name} batch"
            ) from None

    def fused_input_rows(self, forward_batch):
        if self._next_input_rows is not None:
            return self._next_input_rows[self.variant_for(forward_batch)]
        entry = self.path_for(forward_batch)
        return entry.entry.input_rows

    def produced(self, forward_batch):
        return self.path_for(forward_batch).output

    @cached_property
    def output(self):
        return ExitPolicy(self)

    def branch_rows(self, forward_batch):
        if self.variant_for(forward_batch) is not BatchVariant.ORDINARY:
            raise NotImplementedError("branch transport requires ordinary batch rows")
        edges = self.edges[BatchVariant.ORDINARY]
        return (
            edges.incoming.need.layout,
            edges.incoming.residual_to,
            edges.outgoing.need.layout,
        )


def _bind_stage(declaration, norm, incoming, outgoing, **options):
    if incoming.consumer != declaration or outgoing.producer != declaration:
        raise ValueError("connections do not match the stage declaration")
    if incoming.entries.keys() != outgoing.exits.keys():
        raise ValueError("incoming and outgoing batch variants disagree")
    variants = {}
    for variant, edge in incoming.entries.items():
        if (
            declaration.update.applied_at_exit
            and TokenAxis.ATTN_CP
            in edge.produced.layout.sharded - edge.need.layout.sharded
            and not _cp_gathers_over_attn_cp()
        ):
            raise NotImplementedError("MHC with a gather over attention CP")
        attn_input_adapter = None
        if declaration.kind is StageKind.ATTENTION:
            attn_input_adapter = (
                _attn_input_scattered
                if variant is BatchVariant.INPUT_SCATTERED
                and not declaration.update.is_plain_add
                else _attn_input_default
            )
        moves = _cp_moves() if variant is BatchVariant.CONTEXT_PARALLEL else None
        variants[variant] = VariantEdges(
            edge, outgoing.exits[variant], attn_input_adapter, moves
        )
    plan = StagePlan(
        declaration.kind,
        norm,
        variants,
        enters_stack=incoming.producer is None,
        is_branch=declaration.prepared_from is not None,
        terminal=declaration.terminal,
        writes_at_handoff=declaration.writes_at_handoff,
        attn_tp_gather=declaration.attn_tp_gather,
        finishes_directly=declaration.kind is StageKind.ATTENTION
        and declaration.reduction is ProducerReduction.ALWAYS_PARTIAL,
        **options,
    )
    if declaration.kind is StageKind.ATTENTION and outgoing.consumer is not None:
        # Layout eligibility comes from the connected consumer, not a mutable
        # link to its execution plan. Kernel binding remains consumer-owned.
        from sglang.srt.layers.layer_boundary.boundary import input_rows

        plan._next_input_rows = {
            v: input_rows(edge) for v, edge in outgoing.entries.items()
        }

    return StageBoundary(plan, declaration=declaration)
