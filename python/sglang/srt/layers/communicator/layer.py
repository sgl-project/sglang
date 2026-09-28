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
"""The communicator a model layer uses."""

from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property, partial
from typing import Callable, Optional, Protocol, Tuple, Union

import msgspec
import torch

from sglang.srt.distributed import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.layers import layernorm_sp
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateAccumulator
from sglang.srt.layers.communicator.adapters.attention import get_attn_tp_context
from sglang.srt.layers.communicator.boundary import (
    BoundarySteps,
    DecoderLayerSides,
    FusedMlpInput,
    LayerStage,
    StageEntry,
    StageKind,
    _cp_moves,
    _select_boundary_steps,
    decoder_layer_sides,
    input_scattered_layer_sides,
    make_boundary,
    make_output_boundary,
    scattered_residual_layer_sides,
    sequence_parallel_layer_sides,
    with_residual,
)
from sglang.srt.layers.communicator.layout import (
    CommunicateContext,
    Layout,
    SumGroup,
    TokenAxis,
    _batch_shards_over_cp,
    _batch_size,
    _ffn_has_tokens,
    _gathers_over_attention_cp,
    _generic_prefill_cp_shards_tokens,
    _sum_group,
    enable_moe_dense_fully_dp,
    sparse_moe_gathers_over_moe_cp,
    token_axis_sizes,
)
from sglang.srt.layers.communicator.ops import (
    _all_reduce_then_to_local_tokens,
    _hand_qkv_hook_its_input,
    _hand_scattered_input_to_attention,
    _redistribute_output,
    _reduce_and_redistribute_output_step,
    _to_local_tokens,
    move_rows,
)
from sglang.srt.layers.communicator.output import (
    HandoffOutput,
    UnreducedOutput,
    reduce_output,
)
from sglang.srt.layers.communicator.residual import LayerResidual
from sglang.srt.layers.communicator.residual.add_norm import (
    PLAIN_RESIDUAL,
    aiter_all_reduce_fusion_enabled_for,
    apply_aiter_all_reduce_fusion,
    apply_flashinfer_allreduce_fusion,
)
from sglang.srt.layers.communicator.residual.mhc import MHCState
from sglang.srt.layers.dp_attention import (
    is_dp_attention_enabled,
    is_enable_moe_cp_allgather,
)
from sglang.srt.layers.moe import (
    can_merge_post_experts_all_reduce,
    get_moe_a2a_backend,
    is_moe_input_scattered_across_dp_ranks,
    post_experts_reduction_group,
    post_experts_sum_is_one_all_reduce,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import (
    LoRABatchLayout,
    get_exec,
    get_forward,
    get_lora,
    get_parallel,
    get_spec,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import get_bool_env_var, is_hip

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()


@dataclass
class LayerFacts:
    """Where a layer sits in the model and whether it and its neighbours have a
    sparse MLP: the facts its boundaries are declared from."""

    is_layer_sparse: bool = False
    # The model's first layer: its input is the embedding, not a layer output.
    is_first_layer: bool = False
    # The model's last layer: its output goes to the final norm, not a next layer.
    is_last_layer: bool = False
    # Whether the layer before this one has a sparse MLP; None when the facts
    # were given directly, not planned from the layer sequence.
    is_previous_layer_sparse: Optional[bool] = None
    # Whether the layer after this one has a sparse MLP; None likewise.
    is_next_layer_sparse: Optional[bool] = None

    @classmethod
    def init_new(
        cls,
        *,
        layer_id: int,
        num_layers: int,
        is_layer_sparse: bool,
        is_previous_layer_sparse: Optional[bool],
        is_next_layer_sparse: Optional[bool],
    ) -> "LayerFacts":
        return cls(
            is_layer_sparse=is_layer_sparse,
            is_first_layer=layer_id == 0,
            is_last_layer=layer_id == num_layers - 1,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )


def _unfused_completion_matches_the_ffn(forward_batch: ForwardBatch) -> bool:
    """Whether the next layer's all-reduce is the one the FFN would have run
    itself, as the MoE declares it (post_experts_sum_is_one_all_reduce), on a
    batch the FFN runs."""
    return _ffn_has_tokens(forward_batch) and post_experts_sum_is_one_all_reduce()


class StageCommunicator:
    """One stage of a layer, its attention or its FFN: the boundary into it,
    run with the stage's norm on the steps the layer chose for the batch."""

    def __init__(self, layer: "LayerCommunicator", name: str, norm_name: str):
        self._layer = layer
        self._name = name
        self._norm_name = norm_name

    @property
    def norm(self) -> torch.nn.Module:
        """The norm the stage reads its input with, one of the layer's."""
        return getattr(self._layer, self._norm_name)

    def entry(
        self, forward_batch: ForwardBatch, steps: Optional[BoundarySteps] = None
    ) -> StageEntry:
        if steps is None:
            steps = self._layer._batch_steps(forward_batch)
        entry = getattr(steps, self._name)
        if entry is None:
            raise NotImplementedError(
                f"a layer that is one stage has no {self._name} stage"
            )
        return entry

    def prepare(
        self,
        hidden_states,
        residual,
        forward_batch: ForwardBatch,
        steps: Optional[BoundarySteps] = None,
        **call,
    ):
        """The stage's input and the residual, from the previous stage's output
        (``call``: what the stage's read takes, e.g. the attention's
        ``quant_format``)."""
        entry = self.entry(forward_batch, steps)
        context = self._layer._context
        hidden_states, residual = entry.prepare(
            hidden_states, residual, forward_batch, self.norm, context, **call
        )
        if entry.input_move is not None:
            hidden_states = entry.input_move(
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                context=context,
            )
        if entry.handoff is not None:
            hidden_states = entry.handoff(
                hidden_states, forward_batch, self._layer.qkv_latent_func
            )
        return hidden_states, residual


class LayerFusions(Protocol):
    """The fused kernels a fusion backend gives a layer, bound to it and tried
    before the layer's own: at its attention input (each takes what the previous
    layer left, the residual, the batch and a post-residual addition), at its
    FFN input, and at its FFN exit (each says what the exit does when its kernel
    takes the batch)."""

    def attention_input(self, layer: "LayerCommunicator") -> Tuple[Callable, ...]: ...

    def ffn_input(self, layer: "LayerCommunicator") -> Tuple["FusedMlpInput", ...]: ...

    def ffn_exit(
        self, layer: "LayerCommunicator"
    ) -> Tuple[Callable[[ForwardBatch], Optional["FfnExitFusion"]], ...]: ...


class LayerCommunicator:
    # Communicators built without __init__ (e.g. test doubles) publish no LoRA
    # layout and try no fused kernel at the FFN exit.
    _publish_lora_layout: bool = False
    _ffn_exit_fusions: Tuple[Callable, ...] = ()
    # A plain residual unless the layer is built with its own.
    _residual: LayerResidual = PLAIN_RESIDUAL
    # The fused kernels a backend gives the layer, if any.
    fusions: Optional[LayerFusions] = None

    def __init__(
        self,
        layer_facts: LayerFacts,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        # Reduce scatter requires skipping all-reduce in model code after MoE/MLP, so only enable for models which have that implemented. Remove flag once done for all models that use LayerCommunicator.
        allow_reduce_scatter: bool = False,
        qkv_latent_func: Optional[Callable] = None,
        force_layernorm_before_dp_gather: bool = False,
        enable_fused_ar_quant: bool = False,
        fused_ar_quant_keep_bf16: bool = False,
        # False for a layer whose FFN always completes its own reduction.
        allow_deferred_ffn_reduction: bool = True,
        # How the layer's attention and FFN read their inputs from the residual
        # and write their outputs into it.
        residual: LayerResidual = PLAIN_RESIDUAL,
        # A layer that is one stage of a sequence of stages, instead of an
        # attention followed by an FFN.
        stage: Optional["LayerStage"] = None,
        # The fused kernels a backend gives the layer, tried before its own.
        fusions: Optional[LayerFusions] = None,
    ):
        self.layer_facts = layer_facts
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.allow_reduce_scatter = allow_reduce_scatter
        self.is_last_layer = layer_facts.is_last_layer
        self.qkv_latent_func = qkv_latent_func
        self.force_layernorm_before_dp_gather = force_layernorm_before_dp_gather
        self.enable_fused_ar_quant = enable_fused_ar_quant
        self.fused_ar_quant_keep_bf16 = fused_ar_quant_keep_bf16
        self.allow_deferred_ffn_reduction = allow_deferred_ffn_reduction
        self._residual = residual
        self.fusions = fusions

        self._context = CommunicateContext.init_new()
        self._context.force_layernorm_before_dp_gather = (
            force_layernorm_before_dp_gather
        )
        # The fused kernels every batch's attention input tries first.
        self._attn_input_fusions = self._select_attn_input_fusions()
        # The fused kernels of the next layer's input the FFN exit tries first.
        self._ffn_exit_fusions = self._select_ffn_exit_fusions()
        self._speculative_algo = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        # LoRA kernels need the per-layer token layout only under DP attention.
        self._publish_lora_layout = get_parallel().enable_dp_attention and bool(
            get_lora().enable_lora
        )
        if stage is not None:
            if residual != PLAIN_RESIDUAL:
                raise ValueError(
                    "a layer that is one stage takes its reads and updates from "
                    "the stage's declarations"
                )
            self._init_stage(stage)
            return
        # Its two boundaries, for a layer that is one stage.
        self.stage_edges = None
        # The steps the layer's ordinary batches run.
        sides = self._declared_sides()
        self._declared = sides
        self._steps = self._steps_from_declarations(
            sides,
            fusions=self._select_mlp_input_fusions(),
            force_layernorm_before_gather=force_layernorm_before_dp_gather,
        )
        # The steps a batch that shards its tokens over attention CP runs; None
        # when no batch does.
        self._cp_steps = (
            self._steps_from_declarations(
                self._declared_sides(cp_active=True),
                fusions=self._select_mlp_input_fusions(),
                force_layernorm_before_gather=force_layernorm_before_dp_gather,
                cp_moves=_cp_moves(),
            )
            if _generic_prefill_cp_shards_tokens()
            else None
        )
        # The steps a batch with input-scattered attention runs; None where it
        # cannot run.
        self._input_scattered_steps = (
            self._steps_for_input_scattered(sides)
            if self._input_can_be_scattered()
            else None
        )

        # Under LayerNorm SP, the steps the layer runs while the region is
        # active; None without SP. The two are exclusive: SP runs a model without
        # q_lora, which input-scattered attention needs.
        self._sp_steps = (
            _select_boundary_steps(
                with_residual(
                    sequence_parallel_layer_sides(axis_sizes=token_axis_sizes()),
                    residual,
                ),
                attention_fusions=self._attn_input_fusions,
                enters_stack=self.layer_facts.is_first_layer,
            )
            if layernorm_sp.layernorm_sp_enabled()
            else None
        )

    def _init_stage(self, stage: "LayerStage") -> None:
        """A layer that is one stage: every batch runs the two boundaries its
        declarations give."""
        if get_parallel().attn_cp_size > 1 or layernorm_sp.layernorm_sp_enabled():
            raise NotImplementedError(
                "a layer that is one stage with attention CP or LayerNorm SP"
            )
        self._declared = None
        self._cp_steps = self._input_scattered_steps = self._sp_steps = None
        into_edge, out_edge = stage.edges
        is_ffn = stage.kind is StageKind.FFN
        self.stage_edges = stage.edges
        into = make_boundary(
            into_edge,
            fusions=self._select_mlp_input_fusions() if is_ffn else (),
            carried_fusions=() if is_ffn else self._attn_input_fusions,
            force_layernorm_before_gather=self.force_layernorm_before_dp_gather,
            enters_stack=stage.enters_stack,
        )
        out = make_output_boundary(out_edge)
        entry = StageEntry(
            prepare=into.prepare,
            input_rows=into.input_rows,
            input_move=into.input_move,
            handoff=None if is_ffn else _hand_qkv_hook_its_input,
            fused=into.fused,
        )
        self._steps = BoundarySteps(
            attention=None if is_ffn else entry,
            ffn=entry if is_ffn else None,
            ffn_output=out_edge.produced,
            ffn_output_move=out.output_move,
            ffn_output_move_completes_sum=out.output_move_completes_sum,
            ffn_sum_is_movable=out_edge.produced.group is not None,
        )

    @cached_property
    def attn(self) -> StageCommunicator:
        """The layer's attention stage."""
        return StageCommunicator(self, "attention", "input_layernorm")

    @cached_property
    def ffn(self) -> StageCommunicator:
        """The layer's FFN stage."""
        return StageCommunicator(self, "ffn", "post_attention_layernorm")

    @property
    def input_rows(self) -> Layout:
        """The rows the layer's input and residual arrive on in a batch that
        runs its ordinary steps."""
        if self._declared is not None:
            return self._declared.input_rows
        return self.stage_edges[0].residual

    @property
    def input_on_attention_tp_slices(self) -> bool:
        """Whether the layer's input arrives on each attention-TP rank's slice
        of its rows (after a layer on local rows), in a batch that runs its
        ordinary steps."""
        return TokenAxis.ATTN_TP_SCATTER in self.input_rows.sharded

    def _input_can_be_scattered(self) -> bool:
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

    def _declared_sides(self, *, cp_active: bool = False) -> DecoderLayerSides:
        """The declarations this layer's boundaries are chosen from: an
        attention and an FFN, in a layer whose previous layer is known.
        ``cp_active`` gives a batch that shards its tokens over CP; the others
        hold every token on each CP rank. An active LayerNorm SP region and
        input-scattered attention have their own. Raises NotImplementedError
        for the combinations the steps do not cover."""
        modes = self.layer_facts
        parallel = get_parallel()
        if not (modes.is_first_layer or modes.is_previous_layer_sparse is not None):
            raise NotImplementedError(
                "a layer built without the facts of the layer before it"
            )
        # A MoE dispatched per DP shard computes on this rank's local rows and
        # hands its layer's output on there; so does a dense MLP on every rank.
        moe_on_local_rows = is_moe_input_scattered_across_dp_ranks()
        dense_on_local_rows = enable_moe_dense_fully_dp()

        # Some batch shards its tokens over attention CP.
        cp_shards = _generic_prefill_cp_shards_tokens()
        if parallel.attn_cp_size > 1 and modes.is_layer_sparse:
            self._refuse_uncovered_cp_moe(moe_on_local_rows, cp_shards)
        # A MoE whose data-parallel groups are the CP ranks computes each CP
        # shard of a GQA prefill CP on its own ranks.
        ffn_on_cp_shards = (
            modes.is_layer_sparse
            and parallel.attn_cp_size > 1
            and parallel.moe_dp_size == parallel.attn_cp_size
            and not _gathers_over_attention_cp()
        )

        def on_local_rows(sparse: bool) -> bool:
            return moe_on_local_rows if sparse else dense_on_local_rows

        def gathers_for_tbo(sparse: bool, next_sparse: Optional[bool]) -> bool:
            # Under two-batch overlap a dense layer on local rows hands the
            # sparse layer after it, where the split happens, the attention's
            # rows.
            return (
                dense_on_local_rows
                and get_exec().overlap.enable_two_batch_overlap
                and not sparse
                and bool(next_sparse)
            )

        if cp_shards and _cp_moves().reduce_scatter is not None:
            # A CP extend's FFN may leave its sum to the reduce-scatter that
            # takes each rank's shard back (DSA and MLA CP).
            may_leave = not cp_active
            may_leave_to_reduce_scatter = True
        elif cp_shards or parallel.attn_dp_size > 1:
            # Otherwise under CP an FFN over every CP shard completes its own sum;
            # one on its own CP shard hands on its rows as without CP.
            may_leave = may_leave_to_reduce_scatter = (
                parallel.attn_cp_size == 1 or ffn_on_cp_shards
            )
        else:
            # No batch shards its tokens and there is no attention DP: as
            # without CP.
            may_leave = may_leave_to_reduce_scatter = True
        sides = decoder_layer_sides(
            axis_sizes=token_axis_sizes(cp_active=cp_active),
            ffn_on_local_rows=on_local_rows(modes.is_layer_sparse),
            ffn_shards_over_cp=ffn_on_cp_shards,
            previous_on_local_rows=(
                not modes.is_first_layer
                and on_local_rows(modes.is_previous_layer_sparse)
                and not gathers_for_tbo(
                    modes.is_previous_layer_sparse, modes.is_layer_sparse
                )
            ),
            is_last_layer=modes.is_last_layer,
            hands_on_attention_rows=gathers_for_tbo(
                modes.is_layer_sparse, modes.is_next_layer_sparse
            ),
            attention_gathers_local_rows=_use_ag_after_qlora,
            ffn_group=SumGroup.MOE_OUTPUT if modes.is_layer_sparse else SumGroup.TP,
            leaves_for_next_layer=self.allow_deferred_ffn_reduction and may_leave,
            leaves_for_reduce_scatter=self.allow_reduce_scatter
            and may_leave_to_reduce_scatter,
            # A MoE block leaves its sum to reduce_scatterv whenever that combine
            # applies (should_skip_post_experts_all_reduce).
            leaves_for_reduce_scatterv=(
                self.allow_reduce_scatter or modes.is_layer_sparse
            )
            and may_leave,
        )
        return with_residual(sides, self._residual)

    @staticmethod
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
                    "a MoE dispatched per DP shard under attention DP and GQA "
                    "prefill CP"
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

    def _steps_for_input_scattered(self, sides: DecoderLayerSides) -> "BoundarySteps":
        """The steps a batch with input-scattered attention runs at this layer,
        for the layer's ordinary declarations ``sides``. A plain residual comes
        back to the full rows inside the attention output's sum; one whose
        write-back is not a plain add stays on each rank's slice."""
        if self._residual.attention_update.adds_plainly:
            scattered = input_scattered_layer_sides(
                axis_sizes=token_axis_sizes(),
                ffn_group=sides.ffn_output.group,
                hands_on_partial=self.allow_reduce_scatter and not self.is_last_layer,
            )
            handoff = _hand_qkv_hook_its_input
        else:
            scattered = scattered_residual_layer_sides(
                axis_sizes=token_axis_sizes(),
                ffn_group=sides.ffn_output.group,
                is_first_layer=self.layer_facts.is_first_layer,
                is_last_layer=self.is_last_layer,
                leaves_for_reduce_scatter=self.allow_reduce_scatter,
            )
            # DSA and hook-less attention take the slice gathered.
            handoff = _hand_scattered_input_to_attention
        return _select_boundary_steps(
            with_residual(scattered, self._residual),
            attention_handoff=handoff,
            attention_fusions=self._attn_input_fusions,
            enters_stack=self.layer_facts.is_first_layer,
        )

    def _steps_from_declarations(
        self, sides: DecoderLayerSides, **kwargs
    ) -> "BoundarySteps":
        """The steps this layer runs for a set of declarations."""
        return _select_boundary_steps(
            sides,
            attention_fusions=self._attn_input_fusions,
            enters_stack=self.layer_facts.is_first_layer,
            **kwargs,
        )

    def prepare_attn_and_capture_last_layer_outputs(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        captured_last_layer_outputs: Optional[AuxHiddenStateAccumulator] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
        quant_format: str = "",
    ):
        hidden_states, residual = self.prepare_attn(
            hidden_states,
            residual,
            forward_batch,
            quant_format=quant_format,
            post_residual_addition=post_residual_addition,
        )
        if captured_last_layer_outputs is not None:
            move = self.attn.entry(forward_batch).input_move
            gathered_last_layer_output = (
                residual
                if move is None
                else move(
                    hidden_states=residual,
                    forward_batch=forward_batch,
                    context=self._context,
                )
            )
            if (
                gathered_last_layer_output is residual
                # An accumulator that copies on append already holds a snapshot.
                and not getattr(captured_last_layer_outputs, "copies_on_append", False)
                and not self._post_attn_residual_is_read_only(residual, forward_batch)
            ):
                gathered_last_layer_output = residual.clone()
            captured_last_layer_outputs.append(gathered_last_layer_output)
        return hidden_states, residual

    def _post_attn_residual_is_read_only(
        self, residual: torch.Tensor, forward_batch: ForwardBatch
    ) -> bool:
        """True if ``prepare_mlp``'s post-attention RMSNorm leaves ``residual``
        untouched, so Eagle3 aux capture can keep its reference and skip the clone.

        Of the base fused entry's kernels only flashinfer's writes a fresh
        ``residual_out`` (see ``flashinfer_allreduce_residual_rmsnorm``); the aiter
        kernel and every plain norm fold into ``residual`` in place. It runs when
        ``prepare_mlp`` selected a fused entry that may return a new residual and
        the batch is not input-scattered.
        """
        return (
            any(f.may_return_new_residual for f in self.ffn.entry(forward_batch).fused)
            and not get_attn_tp_context().input_scattered
            and apply_flashinfer_allreduce_fusion(residual.shape[0])
        )

    def publish_attn_lora_layout(self) -> None:
        """Attention consumes the DP-local token batch."""
        if self._publish_lora_layout:
            get_forward().set("lora_batch_layout", LoRABatchLayout.DP_LOCAL)

    def publish_mlp_lora_layout(self, steps: "BoundarySteps") -> None:
        """The MLP consumes the TP-global batch only when the batch's FFN input
        is gathered over attention DP."""
        if self._publish_lora_layout:
            get_forward().set(
                "lora_batch_layout",
                (
                    LoRABatchLayout.DP_LOCAL
                    if TokenAxis.ATTN_DP in steps.ffn.input_rows.sharded
                    else LoRABatchLayout.TP_GLOBAL
                ),
            )

    def prepare_attn(
        self,
        hidden_states: Union[torch.Tensor, UnreducedOutput],
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        quant_format: str = "",
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        self.publish_attn_lora_layout()
        # The SP region opens at the first layer, re-evaluated per forward so a
        # crash mid-loop cannot leak into the next one. It sets what the batch's
        # steps are chosen from, and the first layer's input owes nothing.
        if self._sp_steps is not None and self.layer_facts.is_first_layer:
            get_forward().set(
                "sp_active", layernorm_sp.runs_sp(forward_batch.forward_mode)
            )
            if get_forward().sp_active:
                hidden_states = layernorm_sp.sp_entry_scatter(hidden_states)
        return self.attn.prepare(
            hidden_states,
            residual,
            forward_batch,
            quant_format=quant_format,
            post_residual_addition=post_residual_addition,
        )

    def _in_sp_region(self) -> bool:
        return self._sp_steps is not None and get_forward().sp_active

    def _batch_steps(self, forward_batch: ForwardBatch) -> "BoundarySteps":
        """The steps this batch runs: an active LayerNorm SP region's,
        input-scattered attention's, a CP extend's, or the layer's ordinary
        ones."""
        if self._in_sp_region():
            return self._sp_steps
        if (
            self._input_scattered_steps is not None
            and get_attn_tp_context().input_scattered
        ):
            return self._input_scattered_steps
        if self._cp_steps is not None and _batch_shards_over_cp(forward_batch):
            return self._cp_steps
        return self._steps

    def _select_attn_input_fusions(self) -> Tuple[Callable, ...]:
        """The fused kernels that complete what the previous layer left together
        with the residual update and the input norm, in the order they are tried.
        Each takes (owed, residual, forward_batch, post_residual_addition) and
        returns None when it does not take the batch. They add the residual
        plainly; the boundary tries them only when the update it writes in is a
        plain add. A backend's come first."""
        given = self.fusions.attention_input(self) if self.fusions else ()
        if not hasattr(self.input_layernorm, "forward_with_allreduce_fusion"):
            return given
        self._attn_input_fuses_quant = (
            self.enable_fused_ar_quant
            and _use_aiter
            and hasattr(
                self.input_layernorm, "forward_with_allreduce_fusion_quant_per_group"
            )
        )
        return (*given, self._reduce_output_and_update_and_read_residual)

    def _reduce_output_and_update_and_read_residual(
        self,
        owed: UnreducedOutput,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        post_residual_addition: Optional[torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Complete the sum the previous layer left, add it to the residual and
        apply the input norm in one aiter or flashinfer kernel; None when the
        kernel does not take this batch. The result is not quantized for
        ``quant_format``."""
        if (
            not isinstance(owed, UnreducedOutput)
            # The kernel does not add it.
            or post_residual_addition is not None
            # The kernel reduces over the MoE output's group.
            or owed.group is not post_experts_reduction_group()
        ):
            return None
        hidden_states = owed.partial
        if not (
            apply_aiter_all_reduce_fusion(hidden_states, forward_batch)
            or apply_flashinfer_allreduce_fusion(hidden_states.shape[0])
        ):
            return None
        if self._attn_input_fuses_quant:
            # Falls back to AR+RMSNorm + separate quant internally when the
            # fully-fused kernel cannot service the shape.
            quant_result = (
                self.input_layernorm.forward_with_allreduce_fusion_quant_per_group(
                    hidden_states,
                    residual,
                    use_attn_tp_group=False,
                    keep_bf16=self.fused_ar_quant_keep_bf16,
                )
            )
            if quant_result is not None:
                return quant_result
        return self.input_layernorm.forward_with_allreduce_fusion(
            hidden_states, residual, use_attn_tp_group=False
        )

    def _select_mlp_input_fusions(self) -> Tuple["FusedMlpInput", ...]:
        """The fused kernels that can take the attention -> FFN steps, in the
        order they are tried. They add the residual plainly; the boundary tries
        them only when the update it writes in is a plain add. A backend's come
        first."""
        given = self.fusions.ffn_input(self) if self.fusions else ()
        if not hasattr(self.post_attention_layernorm, "forward_with_allreduce_fusion"):
            return given
        return (
            *given,
            FusedMlpInput(
                completes=SumGroup.ATTN_TP,
                run=self._mlp_input_reduce_output_and_update_and_read_residual,
                may_return_new_residual=True,
            ),
        )

    def _mlp_input_reduce_output_and_update_and_read_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """The attention-TP all-reduce, residual add and norm in one aiter or
        flashinfer kernel; None when neither takes the batch."""
        if not (
            apply_aiter_all_reduce_fusion(hidden_states, forward_batch)
            or apply_flashinfer_allreduce_fusion(hidden_states.shape[0])
        ):
            return None
        return self.post_attention_layernorm.forward_with_allreduce_fusion(
            hidden_states, residual, use_attn_tp_group=True
        )

    def prepare_mlp(
        self,
        hidden_states: Union[torch.Tensor, UnreducedOutput],
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        cache=None,
    ):
        steps = self._batch_steps(forward_batch)
        self.publish_mlp_lora_layout(steps)
        if cache is not None:
            self._context.cache = cache

        return self.ffn.prepare(hidden_states, residual, forward_batch, steps)

    def maybe_prefetch_next_full_attention_kv(
        self,
        forward_batch: ForwardBatch,
        next_full_attention_layer_id: Optional[int],
    ) -> None:
        return

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return self._complete_ffn_output_now(
            hidden_states,
            residual,
            forward_batch=forward_batch,
            dp_step=self._postprocess_dp_step(forward_batch),
        )

    def _local_token_move_can_go_to_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        """Whether the next layer's input can run this layer's move of its FFN
        output back to this rank's tokens: the base postprocess scatter, when
        the next layer's input also writes the output into the residual."""
        steps = self._batch_steps(forward_batch)
        return steps.returns_over_dp and not steps.ffn_output.update.at_producer

    def _postprocess_dp_step(
        self, forward_batch: ForwardBatch
    ) -> Optional[Callable[[torch.Tensor, torch.Tensor, ForwardBatch], None]]:
        """The reduce-scatter that brings this layer's FFN output back to this
        rank's tokens under attention DP; None when the base postprocess would
        only scatter, or does not move tokens."""
        steps = self._batch_steps(forward_batch)
        if not steps.returns_over_dp:
            return None
        return _reduce_and_redistribute_output_step(
            forward_batch,
            leaves_for_reduce_scatter=steps.ffn_output.leaves_for_reduce_scatter,
            leaves_for_reduce_scatterv=steps.ffn_output.leaves_for_reduce_scatterv,
        )

    def _ffn_leaves_sum_to_reduce_scatter(
        self, forward_batch: ForwardBatch, dp_step: Optional[Callable]
    ) -> bool:
        """Whether the FFN leaves its sum out because a reduce-scatter completes
        it: the attention-DP one ``dp_step`` names, or the CP / input-scattered
        one."""
        steps = self._batch_steps(forward_batch)
        if not steps.ffn_output.leaves_for_reduce_scatter:
            return False
        if dp_step is not None or steps.ffn_output_move_completes_sum:
            return True
        return get_attn_tp_context().input_scattered and not self.is_last_layer

    def ffn_reduction_group(self, forward_batch: ForwardBatch) -> GroupCoordinator:
        """The group this layer's FFN output owes its sum over: the MoE output's
        group on a sparse layer, the TP group a dense MLP reduces over."""
        return _sum_group(self._batch_steps(forward_batch).ffn_output.group)

    def _select_ffn_completion(self, forward_batch: ForwardBatch) -> "FfnCompletion":
        """Decide once, before the FFN runs, what it skips and what completes its
        output: the next layer's input, or this layer's postprocess step."""
        dp_step = self._postprocess_dp_step(forward_batch)
        mlp_reduce_scatter = self._ffn_leaves_sum_to_reduce_scatter(
            forward_batch, dp_step
        )
        complete_now = partial(
            self._complete_ffn_output_now,
            forward_batch=forward_batch,
            dp_step=dp_step,
        )
        steps = self._batch_steps(forward_batch)
        if not steps.ffn_output.leaves_for_next_layer:
            return FfnCompletion(
                defer_moe_finalize=False,
                fuse_mlp_allreduce=False,
                mlp_reduce_scatter=mlp_reduce_scatter,
                complete=complete_now,
            )
        fusion = next(
            filter(None, (fused(forward_batch) for fused in self._ffn_exit_fusions)),
            None,
        )
        defer_moe_finalize = fusion is FfnExitFusion.DEFER_MOE_FINALIZE
        # A fused kernel that takes the sum, a handoff included, skips the
        # post-experts all-reduce.
        fuse_mlp_allreduce = fusion is not None or self._ffn_sum_moves_to_next_layer(
            forward_batch, mlp_reduce_scatter=mlp_reduce_scatter, dp_step=dp_step
        )
        if fuse_mlp_allreduce:
            group = self.ffn_reduction_group(forward_batch)
            if steps.returns_over_dp:
                # Under attention DP the next layer also brings the sum back to
                # this rank's tokens.
                wrap = partial(
                    UnreducedOutput,
                    reduce_and_redistribute=partial(
                        _all_reduce_then_to_local_tokens, group, forward_batch
                    ),
                )
            else:
                wrap = partial(UnreducedOutput, group=group)
            complete = partial(_leave_to_next_layer, wrap)
        elif (
            dp_step is not None
            and not self.is_last_layer
            and self._local_token_move_can_go_to_next_layer(forward_batch)
        ):
            complete = partial(
                _leave_to_next_layer,
                partial(
                    UnreducedOutput,
                    reduce_and_redistribute=partial(
                        _to_local_tokens, dp_step, forward_batch
                    ),
                ),
            )
        else:
            complete = complete_now
        return FfnCompletion(
            defer_moe_finalize=defer_moe_finalize,
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
            complete=complete,
        )

    def _complete_ffn_output_now(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        *,
        forward_batch: ForwardBatch,
        dp_step: Optional[Callable],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This layer's postprocess, run with the attention-DP step already
        chosen: the move back to where the next layer reads the FFN output, then
        the write-back into the residual for a layer that does it itself."""
        steps = self._batch_steps(forward_batch)
        if steps.returns_over_dp:
            hidden_states = _to_local_tokens(
                dp_step or _redistribute_output, forward_batch, hidden_states
            )
        else:
            hidden_states, residual = steps.ffn_output_move(
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
                context=self._context,
                allow_reduce_scatter=self.allow_reduce_scatter,
                is_layer_sparse=self.layer_facts.is_layer_sparse,
            )
        update = steps.ffn_output.update
        if residual is not None and update.at_producer:
            hidden_states = update.update(hidden_states, residual)
            residual = None
        return hidden_states, residual

    def mixer_exit(self, forward_batch: ForwardBatch) -> "MixerExit":
        """Decide once whether this stage's mixer (an attention-like stage)
        skips its output all-reduce. Use the result as a context manager around
        the mixer, then call ``finish``."""
        return MixerExit(self, forward_batch)

    def ffn_exit(self, forward_batch: ForwardBatch) -> "FfnExit":
        """Decide once how this layer's FFN output reduction completes. Use the
        result as a context manager around the FFN call, then call ``finish``."""
        return FfnExit(self, forward_batch)

    def _branch_rows(
        self, forward_batch: ForwardBatch
    ) -> Tuple[Layout, Layout, Layout]:
        """The rows of this layer's FFN input, of its residual while the FFN
        runs, and of what the layer hands on, for a batch that runs its ordinary
        steps."""
        if self._batch_steps(forward_batch) is not self._steps:
            raise NotImplementedError(
                "a branch on a batch with attention CP, LayerNorm SP or "
                "input-scattered attention"
            )
        if self._declared is None:
            raise NotImplementedError("a branch on a layer that is one stage")
        sides = self._declared
        return sides.ffn.layout, sides.ffn_residual_rows, sides.output_rows

    def branch_input(
        self,
        source: "LayerCommunicator",
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The FFN input and residual that ``source``'s boundary read for its
        own FFN, for this layer's FFN, which branches from the same input: moved
        to the rows this FFN needs and its residual's rows."""
        rows, residual_rows, _ = source._branch_rows(forward_batch)
        to, residual_to, _ = self._branch_rows(forward_batch)
        return (
            move_rows(hidden_states, rows, to, forward_batch),
            move_rows(residual, residual_rows, residual_to, forward_batch),
        )

    def branch_output(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        """This layer's complete FFN output as a branch's contribution, which
        adds to the layer's output without writing the residual: moved to the
        rows the layer hands on."""
        rows, _, to = self._branch_rows(forward_batch)
        return move_rows(hidden_states, rows, to, forward_batch)

    def merge_branch(
        self,
        contribution: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        source: "LayerCommunicator",
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A contribution from ``branch_output`` summed with what ``source``'s
        layer hands on, ``hidden_states`` and ``residual``, moved to the rows
        this layer hands on."""
        _, _, rows = source._branch_rows(forward_batch)
        _, _, to = self._branch_rows(forward_batch)
        hidden_states = move_rows(hidden_states, rows, to, forward_batch)
        residual = move_rows(residual, rows, to, forward_batch)
        return contribution + hidden_states, residual

    def finish_layer_stack(
        self,
        hidden_states: Union[torch.Tensor, UnreducedOutput, HandoffOutput],
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        *,
        final_norm_takes_handoff: bool = False,
    ) -> Tuple[Union[torch.Tensor, HandoffOutput], Optional[torch.Tensor]]:
        """Complete what this layer left for a next layer. Call it on the last
        layer of this rank before its output reaches the final norm, the next
        pipeline rank, or any other consumer outside the layers. A final norm
        that does a producer's handoff together with its own work
        (``final_norm_takes_handoff``) receives it as it is."""
        if final_norm_takes_handoff and isinstance(hidden_states, HandoffOutput):
            return hidden_states, residual
        return reduce_output(hidden_states), residual

    def _select_ffn_exit_fusions(
        self,
    ) -> Tuple[Callable[[ForwardBatch], Optional["FfnExitFusion"]], ...]:
        """The fused kernels of the next layer's input that may take this layer's
        FFN sum, in the order they are tried. Each returns what the exit does
        when its kernel takes this batch, or None. A backend's come first."""
        given = self.fusions.ffn_exit(self) if self.fusions else ()
        return (*given, self._next_input_norm_takes_ffn_sum)

    def _next_input_norm_takes_ffn_sum(
        self, forward_batch: ForwardBatch
    ) -> Optional["FfnExitFusion"]:
        """The aiter / flashinfer AR + add + norm of the next layer's input."""
        if self.should_fuse_mlp_allreduce_with_next_layer(forward_batch):
            return FfnExitFusion.NEXT_INPUT
        return None

    def _ffn_sum_can_move_to_next_layer(self, forward_batch: ForwardBatch) -> bool:
        # Under the MoE-CP all-gather the fusion path would skip postprocess_layer
        # and its MoE-CP scatter, leaving hidden_states longer than the residual.
        if (
            is_enable_moe_cp_allgather()
            or not self._batch_steps(forward_batch).ffn_sum_is_movable
        ):
            return False

        # The fused residual+LN reduces over a single group. Hybrid EP+TP spans
        # two disjoint groups; post_experts_all_reduce() merges them into one
        # _TP reduction when moe_dp_size == 1, which the fused kernel can absorb.
        # When merging is blocked, no single group covers both, so fusion stays off.
        parallel = get_parallel()
        if (
            parallel.moe_ep_size > 1
            and parallel.moe_tp_size > 1
            and not can_merge_post_experts_all_reduce()
        ):
            return False

        if (
            is_dp_attention_enabled()
            and self._speculative_algo is not None
            and self._speculative_algo.is_eagle()
        ):
            return False

        return not get_attn_tp_context().input_scattered

    # NOTE: This function will cause torch recompilation
    def should_fuse_mlp_allreduce_with_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        if not self._ffn_sum_can_move_to_next_layer(forward_batch):
            return False
        batch_size = _batch_size(forward_batch)
        return (
            (
                apply_flashinfer_allreduce_fusion(batch_size)
                or (
                    _use_aiter
                    and batch_size > 0
                    and get_parallel().tp_size != 6
                    and not is_dp_attention_enabled()
                    and get_moe_a2a_backend().is_none()
                    and aiter_all_reduce_fusion_enabled_for(forward_batch.forward_mode)
                )
            )
            and (not self.is_last_layer)
            and (self._context.tp_size > 1)
        )

    def _ffn_sum_moves_to_next_layer(
        self,
        forward_batch: ForwardBatch,
        *,
        mlp_reduce_scatter: bool,
        dp_step: Optional[Callable],
    ) -> bool:
        """Whether the FFN leaves its output's all-reduce to the next layer's
        input when no fused kernel takes it: when the next layer would run the
        same all-reduce the FFN itself would have."""
        return (
            self._context.tp_size > 1
            and not self.is_last_layer
            and self._ffn_sum_can_move_to_next_layer(forward_batch)
            and _unfused_completion_matches_the_ffn(forward_batch)
            and not mlp_reduce_scatter
            # Under attention DP the next layer must also run postprocess's
            # scatter back to this rank's tokens, and nothing more.
            and (
                not is_dp_attention_enabled()
                or (
                    self._local_token_move_can_go_to_next_layer(forward_batch)
                    and dp_step is None
                )
            )
        )


class FfnExitFusion(Enum):
    """What an FFN exit does when a fused kernel of the next layer's input takes
    its sum."""

    # The MoE hands its unfinalized output on; the next input finalizes it.
    DEFER_MOE_FINALIZE = auto()
    # The FFN leaves its all-reduce to the next input norm.
    NEXT_INPUT = auto()


class FfnCompletion(msgspec.Struct, frozen=True):
    """One FFN's reduction decision. The flags are published while the FFN runs;
    ``complete(hidden_states, residual)`` then either wraps the output for the
    next layer's input to complete, or runs this layer's postprocess step."""

    defer_moe_finalize: bool
    fuse_mlp_allreduce: bool
    mlp_reduce_scatter: bool
    complete: Callable[[torch.Tensor, torch.Tensor], Tuple]


def _leave_to_next_layer(
    wrap: Callable[[torch.Tensor], UnreducedOutput],
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
) -> Tuple[UnreducedOutput, torch.Tensor]:
    return wrap(hidden_states), residual


class MixerExit:
    """The scope that publishes a mixer's decision while it runs: inside the
    ``with`` block ``fuse_mlp_allreduce`` on ``get_forward()`` tells its
    row-parallel output projection to skip the all-reduce. It skips when the
    stage's output always leaves its sum (to an FFN stage, which completes it in
    its input), and when it may leave it and the fused kernel takes it into the
    next input norm."""

    __slots__ = ("skips_reduction", "_hands_on", "_scope")

    def __init__(self, communicator: LayerCommunicator, forward_batch: ForwardBatch):
        produced = communicator._batch_steps(forward_batch).ffn_output
        self._hands_on = (
            produced.leaves_for_next_layer
            and communicator.should_fuse_mlp_allreduce_with_next_layer(forward_batch)
        )
        self.skips_reduction = produced.always_leaves or self._hands_on
        self._scope = get_forward().scoped(fuse_mlp_allreduce=self.skips_reduction)

    def __enter__(self) -> "MixerExit":
        self._scope.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._scope.__exit__(*exc_info)

    def finish(
        self, hidden_states: torch.Tensor
    ) -> Union[torch.Tensor, UnreducedOutput]:
        """The mixer's output: its partial sum to the FFN stage after it, as a
        value that owes the sum to a mixer after it, or complete."""
        if self._hands_on:
            return UnreducedOutput(hidden_states, group=get_parallel().tp_group)
        return hidden_states


class FfnExit:
    """The scope that publishes an FfnCompletion while the FFN runs: inside the
    ``with`` block it is ``fuse_mlp_allreduce`` / ``mlp_reduce_scatter`` /
    ``defer_moe_finalize`` on ``get_forward()``."""

    __slots__ = (
        "communicator",
        "forward_batch",
        "defer_moe_finalize",
        "fuse_mlp_allreduce",
        "mlp_reduce_scatter",
        "_complete",
        "_scope",
    )

    def __init__(self, communicator: LayerCommunicator, forward_batch: ForwardBatch):
        self.communicator = communicator
        self.forward_batch = forward_batch
        completion = communicator._select_ffn_completion(forward_batch)
        self.defer_moe_finalize = completion.defer_moe_finalize
        self.fuse_mlp_allreduce = completion.fuse_mlp_allreduce
        self.mlp_reduce_scatter = completion.mlp_reduce_scatter
        self._complete = completion.complete
        self._scope = get_forward().scoped(
            fuse_mlp_allreduce=self.fuse_mlp_allreduce,
            mlp_reduce_scatter=self.mlp_reduce_scatter,
            defer_moe_finalize=self.defer_moe_finalize,
        )

    def __enter__(self) -> "FfnExit":
        self._scope.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._scope.__exit__(*exc_info)

    def finish(
        self, hidden_states: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, UnreducedOutput], torch.Tensor]:
        """Leave the reduction to the next layer's input, or postprocess."""
        if not isinstance(hidden_states, torch.Tensor):
            # A deferred MoE finalize handoff, consumed by the next prepare_attn.
            assert self.defer_moe_finalize, "unrequested deferred MoE handoff"
            return hidden_states, residual
        return self._complete(hidden_states, residual)


class MHCLayerCommunicator(LayerCommunicator):
    """A layer whose residual is hyper-connection streams: the shared boundary
    steps, run with MHCState's residual operations."""

    def __init__(
        self,
        layer_facts: LayerFacts,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        allow_reduce_scatter: bool = False,
        qkv_latent_func: Optional[Callable] = None,
        *,
        hc_mult: int,
        hc_attn_pre: Callable,
        hc_ffn_pre: Callable,
        hc_post: Callable,
        hc_ffn_post_pre: Optional[Callable] = None,
    ):
        self.mhc = MHCState(
            hc_mult=hc_mult,
            hc_attn_pre=hc_attn_pre,
            hc_ffn_pre=hc_ffn_pre,
            hc_post=hc_post,
            hc_ffn_post_pre=hc_ffn_post_pre,
            is_last_layer=layer_facts.is_last_layer,
        )
        if layer_facts.is_layer_sparse and sparse_moe_gathers_over_moe_cp():
            raise NotImplementedError(
                "MHCLayerCommunicator does not support a MoE gathered over the "
                "MoE-CP group (moe_dp_size < attention_context_parallel_size). "
                "Increase moe_dp_size to match attention_context_parallel_size."
            )
        # The postprocess writes the FFN output into the streams, so the FFN's
        # sum never waits for the next layer.
        super().__init__(
            layer_facts,
            input_layernorm,
            post_attention_layernorm,
            allow_reduce_scatter,
            qkv_latent_func,
            allow_deferred_ffn_reduction=False,
            residual=self.mhc.layer_residual(),
        )
        # MHC has not been run with input-scattered attention under attention CP.
        if self._context.attn_cp_size > 1 and self._input_can_be_scattered():
            raise NotImplementedError(
                "MHCLayerCommunicator with input-scattered attention under attention CP"
            )

    def _steps_from_declarations(
        self, sides: DecoderLayerSides, **kwargs
    ) -> BoundarySteps:
        # MHC has not been run with an FFN input gathered over attention CP.
        if (
            TokenAxis.ATTN_CP
            in sides.attention_output.layout.sharded - sides.ffn.layout.sharded
        ):
            raise NotImplementedError(
                f"MHCLayerCommunicator with a gather over attention CP: {sides=}"
            )
        return super()._steps_from_declarations(sides, **kwargs)
