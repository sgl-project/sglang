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
    _cp_moves,
    make_boundary,
    make_output_boundary,
)
from sglang.srt.layers.layer_boundary.contracts import (
    BatchVariant,
    CpMoves,
    EdgeDecl,
    ProducerReduction,
    StageEntry,
    StageKind,
    StageSteps,
)
from sglang.srt.layers.layer_boundary.exit import OutputBoundary
from sglang.srt.layers.layer_boundary.fusions.allreduce import (
    attention_fusions,
    ffn_fusions,
)
from sglang.srt.layers.layer_boundary.layout import (
    TokenAxis,
    _batch_shards_over_cp,
    _gathers_over_attention_cp,
    _generic_prefill_cp_shards_tokens,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.layer_boundary.prepare import (
    _hand_qkv_hook_its_input,
    _hand_scattered_input_to_attention,
)
from sglang.srt.layers.layer_boundary.stage import StageBoundary
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
)
from sglang.srt.runtime_context import (
    get_forward,
    get_lora,
    get_parallel,
    get_spec,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()


def _refuse_uncovered_cp_moe(moe_on_local_rows: bool, cp_shards: bool) -> None:
    """A MoE layer under attention CP whose tokens the steps cannot bring
    to it: under a prefill CP, one dispatched per DP shard under attention DP
    and GQA CP, and one on the TP group whose data-parallel groups are the CP
    ranks under DSA or MLA CP; and under attention DP, one on the TP group
    whose data-parallel groups are the CP ranks."""
    parallel = get_parallel()
    gqa = not _gathers_over_attention_cp()
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


def _input_can_be_scattered() -> bool:
    """Whether a batch may run this layer with input-scattered attention:
    configured, on TP without attention DP, a prefill CP, an a2a backend or
    a dense MLP on every rank. The rest of what ``AttnTpContext.init_context`` requires is only
    known once the model is built."""
    parallel = get_parallel()
    return (
        parallel.enable_attn_tp_input_scattered
        and parallel.tp_size > 1
        and parallel.attn_dp_size == 1
        and not _generic_prefill_cp_shards_tokens()
        and get_moe_a2a_backend().is_none()
        and not enable_moe_dense_fully_dp()
    )


@dataclass(frozen=True)
class StageEdges:
    """One stage's incoming and outgoing contracts for one batch variant.

    Fields:
        incoming: Consumer-side contract used to bind prepare.
        outgoing: Producer-side contract used to bind exit transport.
        handoff: Optional attention adapter run after input preparation/movement.
        cp_moves: Strategy-specific context-parallel gather and return operations.

    No executable neighbouring stage or neighbouring norm is required.
    """

    incoming: EdgeDecl
    outgoing: EdgeDecl
    handoff: Optional[Callable] = None
    cp_moves: Optional[CpMoves] = None


def _requires_branch_input(*args, **kwargs):
    raise RuntimeError("a prepared branch must enter through branch_input, not prepare")


class StagePlan:
    """Bind reusable entry/exit paths from a stage's resolved declarations.

    Args:
        kind: Boundary role selecting attention or FFN fusion adapters.
        norm: This stage's consumer normalization module.
        variants: Mapping from BatchVariant to StageEdges. Construction binds
            each entry once; a forward selects an existing path by batch facts.
        enters_stack: Whether this stage initializes the residual from embeddings.
        prepared_input: Branch reusing an already-read input; requires branch_input
            instead of a normal prepare to avoid repeating the read/update.
        terminal: Whether the stage ends the model's layer stack.
        direct_handoff: Attention can publish its output with finish instead of
            an exit scope and output transport.
        qkv_latent_func: Optional hook for prepared attention input.
        fusions: Optional backend provider of ordered consumer fusion candidates
            and the producer deferral policies (see make_attn_stage).

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
        prepared_input=False,
        terminal=False,
        direct_handoff=False,
        qkv_latent_func=None,
        fusions=None,
    ):
        self.norm = norm
        self.variants = dict(variants)
        self.enters_stack = enters_stack
        self.terminal = terminal
        self.direct_handoff = direct_handoff
        self.qkv_latent_func = qkv_latent_func
        self.fusions = fusions
        self._speculative_algo = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        self._publish_lora_layout = get_parallel().enable_dp_attention and bool(
            get_lora().enable_lora
        )
        self._fusion_rows = None
        carried = (
            attention_fusions(
                self, next(iter(self.variants.values())).incoming.need.read
            )
            if kind is StageKind.ATTENTION
            else ()
        )
        fused = ffn_fusions(self) if kind is StageKind.FFN else ()
        self._paths = {}
        for variant, edges in self.variants.items():
            if prepared_input:
                # The branch adapter moves an already-read input and forks its
                # stream. Binding a normal prepare would add or norm it twice.
                entry = StageEntry(
                    prepare=_requires_branch_input,
                    input_rows=edges.incoming.need.layout,
                )
            else:
                into = make_boundary(
                    edges.incoming,
                    fusions=fused,
                    carried_fusions=carried,
                    cp_moves=edges.cp_moves,
                    enters_stack=enters_stack,
                )
                entry = StageEntry(
                    prepare=into.prepare,
                    input_rows=into.input_rows,
                    input_move=into.input_move,
                    handoff=edges.handoff,
                    capture_move=into.capture_move,
                    capture_move_allocates=into.capture_move_allocates,
                    preserves_residual=into.preserves_residual,
                    input_sum=edges.incoming.produced.group
                    if edges.incoming.produced.always_leaves
                    else None,
                )
            out = (
                None
                if direct_handoff
                else make_output_boundary(edges.outgoing, cp_moves=edges.cp_moves)
            )
            self._paths[variant] = StageSteps(
                entry=entry,
                output=edges.outgoing.produced,
                output_move=None if out is None else out.output_move,
                returns_over_dp=out is not None and out.returns_over_dp,
                output_move_completes_sum=False
                if out is None
                else out.output_move_completes_sum,
            )

    @property
    def incoming_residual_rows(self):
        return self.variants[BatchVariant.ORDINARY].incoming.residual

    @property
    def input_on_attention_tp_slices(self):
        return TokenAxis.ATTN_TP_SCATTER in self.incoming_residual_rows.sharded

    def _variant(self, forward_batch):
        # The batch determines its rows; missing paths must not change them.
        if get_forward().sp_active:
            return BatchVariant.SEQUENCE_PARALLEL
        if get_attn_tp_context().input_scattered:
            return BatchVariant.INPUT_SCATTERED
        if _batch_shards_over_cp(forward_batch):
            return BatchVariant.CONTEXT_PARALLEL
        return BatchVariant.ORDINARY

    def _batch_steps(self, forward_batch):
        variant = self._variant(forward_batch)
        try:
            return self._paths[variant]
        except KeyError:
            raise NotImplementedError(
                f"no stage boundary path for the active {variant.name} batch"
            ) from None

    def fusion_rows(self, forward_batch):
        if self._fusion_rows is not None:
            return self._fusion_rows[self._variant(forward_batch)]
        entry = self._batch_steps(forward_batch)
        return entry.entry.input_rows

    def produced(self, forward_batch):
        return self._batch_steps(forward_batch).output

    @cached_property
    def output(self):
        return OutputBoundary(self)

    def _branch_rows(self, forward_batch):
        if self._variant(forward_batch) is not BatchVariant.ORDINARY:
            raise NotImplementedError("branch transport requires ordinary batch rows")
        edges = self.variants[BatchVariant.ORDINARY]
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
        direct_handoff=declaration.kind is StageKind.ATTENTION
        and declaration.reduction is ProducerReduction.PARTIAL,
        **options,
    )
    if declaration.kind is StageKind.ATTENTION and outgoing.consumer is not None:
        # Layout eligibility comes from the connected consumer, not a mutable
        # link to its execution plan. Kernel binding remains consumer-owned.
        from sglang.srt.layers.layer_boundary.boundary import input_rows

        plan._fusion_rows = {
            v: input_rows(edge) for v, edge in outgoing.entries.items()
        }

    return StageBoundary(plan, declaration=declaration)
