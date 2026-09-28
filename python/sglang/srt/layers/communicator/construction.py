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
from enum import Enum, auto
from functools import cached_property
from typing import Callable, Optional, Protocol, Tuple

from sglang.srt.environ import envs
from sglang.srt.layers.communicator.adapters.attention import get_attn_tp_context
from sglang.srt.layers.communicator.boundary import (
    BoundarySteps,
    EdgeDecl,
    FusedMlpInput,
    StageDecl,
    StageEntry,
    StageKind,
    make_boundary,
    make_output_boundary,
)
from sglang.srt.layers.communicator.exit import OutputBoundary
from sglang.srt.layers.communicator.fusions.allreduce import (
    attention_fusions,
    ffn_fusions,
)
from sglang.srt.layers.communicator.layout import (
    TokenAxis,
    _batch_shards_over_cp,
    _gathers_over_attention_cp,
    _generic_prefill_cp_shards_tokens,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.communicator.stage import StageCommunicator
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import (
    get_forward,
    get_lora,
    get_parallel,
    get_spec,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()


class BoundaryFusions(Protocol):
    """The fused kernels a fusion backend gives a layer, bound to it and tried
    before the layer's own: at its attention input (each takes what the previous
    layer left, the residual, the batch and a post-residual addition), at its
    FFN input, and at its FFN exit (each says what the exit does when its kernel
    takes the batch)."""

    def attention_input(self, layer: "StagePlan") -> Tuple[Callable, ...]: ...

    def ffn_input(self, layer: "StagePlan") -> Tuple["FusedMlpInput", ...]: ...

    requires_local_reduction: bool

    def can_defer_finalize(
        self, layer: "StagePlan", forward_batch: ForwardBatch
    ) -> bool: ...


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


class BatchVariant(Enum):
    ORDINARY = auto()
    CONTEXT_PARALLEL = auto()
    INPUT_SCATTERED = auto()
    SEQUENCE_PARALLEL = auto()


@dataclass(frozen=True)
class StageEdges:
    """One stage's incoming and outgoing declarations for one batch variant.

    No adjacent stage object is required. Edges can come from a local model
    assembler, pipeline contract or branch adapter.
    """

    incoming: EdgeDecl
    outgoing: EdgeDecl
    handoff: Optional[Callable] = None
    cp_moves: object = None


def _requires_branch_input(*args, **kwargs):
    raise RuntimeError("a prepared branch must enter through branch_input, not prepare")


class StagePlan:
    """Precomputed paths for one stage; never owns the neighbouring norm."""

    def __init__(
        self,
        kind,
        norm,
        variants,
        *,
        enters_stack=False,
        prepared_input=False,
        terminal=False,
        fixed_output=False,
        is_sparse=False,
        qkv_latent_func=None,
        force_layernorm_before_dp_gather=False,
        enable_fused_ar_quant=False,
        fused_ar_quant_keep_bf16=False,
        residual_in_hidden=False,
        fusions=None,
    ):
        self.kind = kind
        self.norm = norm
        self.variants = dict(variants)
        self.is_first_layer = enters_stack
        self.is_last_layer = terminal
        self.fixed_output = fixed_output
        self.is_sparse = is_sparse
        self.qkv_latent_func = qkv_latent_func
        self.enable_fused_ar_quant = enable_fused_ar_quant
        self.fused_ar_quant_keep_bf16 = fused_ar_quant_keep_bf16
        self.residual_in_hidden = residual_in_hidden
        self.fusions = fusions
        self._speculative_algo = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        self._publish_lora_layout = get_parallel().enable_dp_attention and bool(
            get_lora().enable_lora
        )
        self._fusion_rows = None
        carried = attention_fusions(self) if kind is StageKind.ATTENTION else ()
        self._carried_fusions = carried
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
                    force_layernorm_before_gather=force_layernorm_before_dp_gather,
                    cp_moves=edges.cp_moves,
                    enters_stack=enters_stack,
                )
                entry = StageEntry(
                    prepare=into.prepare,
                    input_rows=into.input_rows,
                    input_move=into.input_move,
                    handoff=edges.handoff,
                    fused=into.fused,
                    capture_move=into.capture_move,
                    capture_move_allocates=into.capture_move_allocates,
                    preserves_residual=into.preserves_residual,
                    input_sum=edges.incoming.produced.group
                    if edges.incoming.produced.always_leaves
                    else None,
                )
            out = (
                None
                if fixed_output
                else make_output_boundary(edges.outgoing, cp_moves=edges.cp_moves)
            )
            self._paths[variant] = BoundarySteps(
                attention=entry if kind is StageKind.ATTENTION else None,
                ffn=entry if kind is StageKind.FFN else None,
                ffn_output=edges.outgoing.produced,
                ffn_output_move=None if out is None else out.output_move,
                ffn_output_move_completes_sum=False
                if out is None
                else out.output_move_completes_sum,
                ffn_sum_is_movable=edges.outgoing.produced.group is not None,
            )
        self._steps = self._paths[BatchVariant.ORDINARY]
        self._cp_steps = self._paths.get(BatchVariant.CONTEXT_PARALLEL)
        self._input_scattered_steps = self._paths.get(BatchVariant.INPUT_SCATTERED)
        self._sp_steps = self._paths.get(BatchVariant.SEQUENCE_PARALLEL)

    @property
    def input_rows(self):
        return self.variants[BatchVariant.ORDINARY].incoming.residual

    @property
    def input_on_attention_tp_slices(self):
        return TokenAxis.ATTN_TP_SCATTER in self.input_rows.sharded

    def _variant(self, forward_batch):
        if self._sp_steps is not None and get_forward().sp_active:
            return BatchVariant.SEQUENCE_PARALLEL
        if (
            self._input_scattered_steps is not None
            and get_attn_tp_context().input_scattered
        ):
            return BatchVariant.INPUT_SCATTERED
        if self._cp_steps is not None and _batch_shards_over_cp(forward_batch):
            return BatchVariant.CONTEXT_PARALLEL
        return BatchVariant.ORDINARY

    def _batch_steps(self, forward_batch):
        return self._paths[self._variant(forward_batch)]

    def fusion_rows(self, forward_batch):
        if self._fusion_rows is not None:
            return self._fusion_rows[self._variant(forward_batch)]
        entry = self._batch_steps(forward_batch)
        return (entry.attention or entry.ffn).input_rows

    def produced(self, kind, forward_batch):
        return self._batch_steps(forward_batch).ffn_output

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


def make_stage(
    declaration: StageDecl,
    *,
    kind: StageKind,
    norm,
    incoming: EdgeDecl,
    outgoing: EdgeDecl,
    variants=None,
    handoff=None,
    **options,
) -> StageCommunicator:
    """Build one real stage from its own declaration and adjacent edges.

    The optional variants supply the same stage's CP/SP/input-scattered
    declarations. Neither a paired stage nor its layer facts/norm is needed.
    """
    if incoming.need != declaration.input or outgoing.produced != declaration.output:
        raise ValueError("stage declaration disagrees with its boundaries")
    paths = dict(variants or {})
    paths[BatchVariant.ORDINARY] = StageEdges(incoming, outgoing, handoff)
    return StageCommunicator(StagePlan(kind, norm, paths, **options), kind, norm)
