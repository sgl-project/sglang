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
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import torch

from sglang.kernels.ops.layernorm.mhc import hc_contract, hc_expand
from sglang.srt.distributed.communication_op import (
    attention_tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.boundary_layout import (
    DecoderLayerSides,
    SumGroup,
    TokenAxis,
    scattered_residual_layer_sides,
)
from sglang.srt.layers.communicator import (
    AttentionInputs,
    BoundarySteps,
    CommunicateContext,
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    FfnCompletion,
    LayerCommunicator,
    LayerScatterModes,
    MlpInputKind,
    ScatterMode,
    _select_attention_input_move,
    _token_axis_sizes,
    get_attn_tp_context,
    mlp_input_kind,
    tp_reduce_scatter,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    attn_tp_reduce_scatter_tensor,
    dp_gather_replicate,
    dp_reduce_scatter_tensor,
    dp_scatter,
    get_dp_global_num_tokens,
    get_global_dp_buffer,
    get_local_dp_buffer_mhc,
    is_allocation_symmetric,
)
from sglang.srt.layers.moe import should_use_dp_reduce_scatterv
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel


def tp_all_gather_hidden_states(hidden_states, forward_batch):
    assert get_attn_tp_context().input_scattered, (
        "Input scattered guarantees same num tokens in TP group."
    )
    total_tokens = forward_batch.input_ids.shape[0]
    output = hidden_states.new_empty((total_tokens, hidden_states.shape[-1]))
    get_parallel().tp_group.all_gather_into_tensor(output, hidden_states)

    return output


def _attention_input(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    context: CommunicateContext,
    *,
    move: Callable,
    qkv_latent_func: Optional[Callable],
):
    """Move the attention input to the attention's rows and hand the QKV hook
    its input."""
    hidden_states = move(
        hidden_states=hidden_states, forward_batch=forward_batch, context=context
    )
    if qkv_latent_func is not None:
        get_attn_tp_context().set_attn_inputs(
            AttentionInputs(hidden_states, forward_batch, qkv_latent_func)
        )
    return hidden_states


def _input_scattered_attention_input(
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    context: CommunicateContext,
    *,
    qkv_latent_func: Optional[Callable],
):
    """Input-scattered attention: the QKV hook gathers the rows after its
    projection, except that DSA and attention without a hook consume full
    hidden states, so those are gathered here."""
    ctx = get_attn_tp_context()
    if ctx.is_dsa or qkv_latent_func is None:
        hidden_states = tp_all_gather_hidden_states(hidden_states, forward_batch)
    if qkv_latent_func is not None:
        ctx.set_attn_inputs(
            AttentionInputs(
                hidden_states,
                forward_batch,
                qkv_latent_func,
                is_pre_gathered=ctx.is_dsa,
            )
        )
    return hidden_states


def _select_mhc_ffn_input(sides: DecoderLayerSides, mhc: "MHCState") -> Callable:
    """MHC's steps from the attention output to the FFN input, for the rows
    the declarations call for. The residual is written back with hc_post,
    which is not a plain add, so a DP gather always runs after it (the
    replicate order)."""
    produced, need = sides.attention_output, sides.ffn
    residual, residual_to = sides.input_rows, sides.ffn_residual_rows
    fns = MHCCommunicateWithAllReduceAndLayerNormFn
    gathered = produced.layout.sharded - need.layout.sharded - need.gathers_itself
    sliced = need.layout.sharded - produced.layout.sharded
    if TokenAxis.ATTN_TP_SCATTER in residual_to.sharded - need.layout.sharded:
        # The residual stays on each rank's slice while the FFN takes all rows
        # (input-scattered): complete the sum onto the slice, update and read
        # the residual there, and gather the FFN input.
        if gathered or sliced or produced.group is not SumGroup.ATTN_TP:
            raise NotImplementedError(f"{produced=} {need=}")
        return partial(fns._reduce_scatter_update_and_gather, mhc=mhc)
    if sliced:
        # Each attention-TP rank takes its own slice.
        if (
            sliced != {TokenAxis.ATTN_TP_SCATTER}
            or gathered
            or produced.group is not SumGroup.ATTN_TP
        ):
            raise NotImplementedError(f"{produced=} {need=}")
        return partial(
            fns._scatter_hidden_states_and_residual,
            scatters_residual=residual != residual_to,
            mhc=mhc,
        )
    if gathered - {TokenAxis.ATTN_DP}:
        raise NotImplementedError(f"{produced=} {need=}")
    if gathered or produced.group is SumGroup.ATTN_TP:
        return partial(
            fns._gather_hidden_states_and_residual,
            residual_on_slice=TokenAxis.ATTN_TP_SCATTER in residual.sharded,
            mhc=mhc,
        )
    return partial(fns._simple, mhc=mhc)


def _select_mhc_ffn_output_move(
    sides: DecoderLayerSides,
) -> Tuple[Optional[Callable], bool]:
    """MHC's move of the FFN output to the rows the layer hands on, None when
    it goes back over attention DP (MHC's postprocess runs that itself), and
    whether the move completes the sum the FFN leaves."""
    produced, residual, to = (
        sides.ffn_output,
        sides.ffn_residual_rows,
        sides.output_rows,
    )
    fns = MHCCommunicateSummableTensorPairFn
    returned = residual.sharded - produced.layout.sharded
    if produced.layout == residual:
        if to == residual:
            return fns._trivial, False
        if to.sharded == residual.sharded - {TokenAxis.ATTN_TP_SCATTER}:
            return fns._gather, False
    elif to == residual and returned == {TokenAxis.ATTN_DP}:
        return None, False
    elif (
        returned == {TokenAxis.ATTN_TP_SCATTER}
        and produced.leaves_for_reduce_scatter
        and to in (residual, produced.layout)
    ):
        # The reduce-scatter onto each rank's slice completes the FFN's sum.
        return fns._reduce_scatter_and_combine, True
    raise NotImplementedError(f"{produced=} {residual=} {to=}")


@dataclass
class MHCState:
    """Parameters belong to the owning layer; this state only holds scratch
    shared across communication stages."""

    hc_mult: int
    hc_attn_pre: Callable
    hc_ffn_pre: Callable
    hc_post: Callable
    hc_ffn_post_pre: Optional[Callable] = None
    h_res: Optional[torch.Tensor] = None
    h_post: Optional[torch.Tensor] = None

    @staticmethod
    def _resolve_out_norm(out_norm):
        if out_norm is None:
            return None, None
        return out_norm.weight.data, out_norm.variance_epsilon

    def attn_split(self, hidden_states, out_norm: Optional[torch.nn.Module] = None):
        residual = hidden_states
        out_norm_weight, out_norm_eps = self._resolve_out_norm(out_norm)
        hidden_states, self.h_res, self.h_post, norm_fused = self.hc_attn_pre(
            hidden_states, out_norm_weight, out_norm_eps
        )
        if out_norm is not None and not norm_fused and hidden_states.shape[0] != 0:
            hidden_states = out_norm(hidden_states)
        return hidden_states, residual

    def attn_to_mlp(
        self, hidden_states, residual, out_norm: Optional[torch.nn.Module] = None
    ):
        out_norm_weight, out_norm_eps = self._resolve_out_norm(out_norm)
        if self.hc_ffn_post_pre is not None and hidden_states.shape[0] != 0:
            # Returns None when it declines -- no fused kernel for this platform
            # or shape, or a shape the fusion is slower at -- and the chain runs.
            fused = self.hc_ffn_post_pre(
                hidden_states=hidden_states,
                residual=residual,
                h_res=self.h_res,
                h_post=self.h_post,
                out_norm_weight=out_norm_weight,
                out_norm_eps=out_norm_eps,
            )
            if fused is not None:
                hidden_states, residual, self.h_res, self.h_post, norm_fused = fused
                if out_norm is not None and not norm_fused:
                    hidden_states = out_norm(hidden_states)
                return hidden_states, residual

        hidden_states = self.hc_post(hidden_states, residual, self.h_res, self.h_post)
        residual = hidden_states
        hidden_states, self.h_res, self.h_post, norm_fused = self.hc_ffn_pre(
            hidden_states, out_norm_weight, out_norm_eps
        )
        if out_norm is not None and not norm_fused and hidden_states.shape[0] != 0:
            hidden_states = out_norm(hidden_states)
        return hidden_states, residual

    def mlp_combine(self, hidden_states, residual):
        return self.hc_post(hidden_states, residual, self.h_res, self.h_post)

    def reset_aux(self):
        self.h_res = None
        self.h_post = None


class MHCCommunicateWithAllReduceAndLayerNormFn:
    @staticmethod
    def _scatter_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        scatters_residual: bool,
        mhc: MHCState,
    ):
        input_hidden_states = hidden_states
        hidden_states = hidden_states.tensor_split(context.attn_tp_size)[
            context.attn_tp_rank
        ]
        attn_tp_reduce_scatter_tensor(hidden_states, input_hidden_states)
        if scatters_residual:
            residual = residual.tensor_split(context.attn_tp_size)[context.attn_tp_rank]
            mhc.h_res = mhc.h_res.tensor_split(context.attn_tp_size)[
                context.attn_tp_rank
            ]
            mhc.h_post = mhc.h_post.tensor_split(context.attn_tp_size)[
                context.attn_tp_rank
            ]

        hidden_states, residual = mhc.attn_to_mlp(
            hidden_states, residual, out_norm=layernorm
        )
        return hidden_states, residual

    @staticmethod
    def _simple(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        mhc: MHCState,
    ):
        hidden_states, residual = mhc.attn_to_mlp(
            hidden_states, residual, out_norm=layernorm
        )
        return hidden_states, residual

    @staticmethod
    def _reduce_scatter_update_and_gather(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        mhc: MHCState,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states

        scatter_states = hidden_states.tensor_split(context.tp_size)[context.tp_rank]
        get_parallel().tp_group.reduce_scatter_tensor(scatter_states, hidden_states)

        scatter_states, residual = mhc.attn_to_mlp(
            scatter_states, residual, out_norm=layernorm
        )

        attn_tp_all_gather_into_tensor(hidden_states, scatter_states)

        return hidden_states, residual

    @staticmethod
    def _gather_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_on_slice: bool,
        mhc: MHCState,
    ):
        if residual_on_slice:
            raise NotImplementedError(
                "Unsupported: h_res/h_post allgather not implemented."
            )

        hidden_states = attention_tensor_model_parallel_all_reduce(hidden_states)
        if context.attn_dp_size != 1:
            if hidden_states.shape[0] != 0:
                with use_symmetric_memory(
                    get_parallel().tp_group,
                    disabled=not is_allocation_symmetric(),
                ):
                    hidden_states, residual = mhc.attn_to_mlp(
                        hidden_states, residual, out_norm=layernorm
                    )
            else:
                hidden_states, residual = mhc.attn_to_mlp(hidden_states, residual)

            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_parallel().tp_group),
                hidden_states,
            )
            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
        else:
            hidden_states, residual = mhc.attn_to_mlp(
                hidden_states, residual, out_norm=layernorm
            )
        return hidden_states, residual


class MHCCommunicateSummableTensorPairFn(CommunicateSummableTensorPairFn):
    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        fn = CommunicateSummableTensorPairFn.get_fn(
            hidden_states_input_mode,
            residual_input_mode,
            output_mode,
            context,
        )
        replacements = {
            CommunicateSummableTensorPairFn._trivial: MHCCommunicateSummableTensorPairFn._trivial,
            CommunicateSummableTensorPairFn._scatter_hidden_states: MHCCommunicateSummableTensorPairFn._scatter_hidden_states,
            CommunicateSummableTensorPairFn._gather: MHCCommunicateSummableTensorPairFn._gather,
            CommunicateSummableTensorPairFn._scatter: MHCCommunicateSummableTensorPairFn._scatter,
        }
        return replacements.get(fn, fn)

    @staticmethod
    def _trivial(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        *,
        mhc: MHCState,
        is_last_layer: bool,
        **kwargs,
    ):
        hidden_states = mhc.mlp_combine(hidden_states, residual)
        if not is_last_layer:
            return hidden_states, None
        return hc_contract(hidden_states, mhc.hc_mult), None

    @staticmethod
    def _reduce_scatter_and_combine(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        *,
        mhc: MHCState,
        is_last_layer: bool,
        **kwargs,
    ):
        """Input-scattered: reduce-scatter the FFN output onto this rank's
        slice, which completes its sum, and combine the streams there. The last
        layer contracts them and gathers the full rows back."""
        hidden_states, _ = tp_reduce_scatter(hidden_states, None, context)
        hidden_states = mhc.mlp_combine(hidden_states, residual)
        if not is_last_layer:
            return hidden_states, None

        local_states = hc_contract(hidden_states, mhc.hc_mult)
        hidden_states = local_states.new_empty(
            local_states.shape[0] * context.tp_size, *local_states.shape[1:]
        )
        get_parallel().tp_group.all_gather_into_tensor(hidden_states, local_states)
        return hidden_states, None

    @staticmethod
    def _scatter_hidden_states(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        allow_reduce_scatter: bool = False,
        *,
        mhc: MHCState,
        is_last_layer: bool,
        **kwargs,
    ):
        hidden_states, global_hidden_states = (
            get_local_dp_buffer_mhc(get_parallel().tp_group, 1),
            hidden_states,
        )
        # MoE skips its post-expert all-reduce with reduce_scatterv, so this
        # scatter must reduce while combining local-expert partial sums.
        if should_use_dp_reduce_scatterv():
            get_parallel().tp_group.reduce_scatterv(
                global_hidden_states,
                output=hidden_states,
                sizes=get_dp_global_num_tokens(),
            )
        elif allow_reduce_scatter and forward_batch.dp_padding_mode.is_max_len():
            dp_reduce_scatter_tensor(hidden_states, global_hidden_states)
        else:
            dp_scatter(hidden_states, global_hidden_states, forward_batch)

        hidden_states = mhc.mlp_combine(hidden_states, residual)
        if not is_last_layer:
            return hidden_states, None

        hidden_states = hc_contract(hidden_states, mhc.hc_mult)
        return hidden_states, None

    @staticmethod
    def _gather(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        *,
        mhc: MHCState,
        is_last_layer: bool,
        **kwargs,
    ):
        hidden_states = mhc.mlp_combine(hidden_states, residual)
        if is_last_layer:
            hidden_states = hc_contract(hidden_states, mhc.hc_mult)

        hidden_states, local_hidden_states = (
            get_local_dp_buffer_mhc(
                get_parallel().tp_group, 1 if is_last_layer else mhc.hc_mult
            ),
            hidden_states,
        )

        attn_tp_all_gather_into_tensor(hidden_states, local_hidden_states)
        return hidden_states, None

    @staticmethod
    def _scatter(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        *,
        mhc: MHCState,
        is_last_layer: bool,
        **kwargs,
    ):
        hidden_states = hidden_states.tensor_split(context.attn_tp_size)[
            context.attn_tp_rank
        ]
        residual = residual.tensor_split(context.attn_tp_size)[context.attn_tp_rank]

        hidden_states = mhc.mlp_combine(hidden_states, residual)

        return hidden_states, None


class MHCLayerCommunicator(LayerCommunicator):
    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        allow_reduce_scatter: bool = False,
        qkv_latent_func: Optional[Callable] = None,
        *,
        is_first_layer: bool,
        hc_mult: int,
        hc_attn_pre: Callable,
        hc_ffn_pre: Callable,
        hc_post: Callable,
        hc_ffn_post_pre: Optional[Callable] = None,
    ):
        self.is_first_layer = is_first_layer
        self.mhc = MHCState(
            hc_mult=hc_mult,
            hc_attn_pre=hc_attn_pre,
            hc_ffn_pre=hc_ffn_pre,
            hc_post=hc_post,
            hc_ffn_post_pre=hc_ffn_post_pre,
        )

        # The postprocess combines the hyper-connection streams, so the FFN's
        # sum never waits for the next layer.
        super().__init__(
            layer_scatter_modes,
            input_layernorm,
            post_attention_layernorm,
            allow_reduce_scatter,
            qkv_latent_func,
            allow_deferred_ffn_reduction=False,
        )

    def _steps_from_declarations(
        self, sides: DecoderLayerSides, **kwargs
    ) -> BoundarySteps:
        """MHC's own implementation of each step the declarations call for."""
        move, completes_sum = _select_mhc_ffn_output_move(sides)
        return BoundarySteps(
            attention_input=partial(
                _attention_input,
                move=_select_attention_input_move(sides.input_rows, sides.attention),
                qkv_latent_func=self.qkv_latent_func,
            ),
            ffn_input=_select_mhc_ffn_input(sides, self.mhc),
            ffn_output=sides.ffn_output,
            ffn_output_move=move,
            ffn_sum_is_movable=sides.ffn_output.group is not None,
            ffn_output_move_completes_sum=completes_sum,
        )

    def _steps_for_input_scattered(self, sides: DecoderLayerSides) -> BoundarySteps:
        """An input-scattered batch keeps the residual on each rank's slice."""
        scattered = scattered_residual_layer_sides(
            axis_sizes=_token_axis_sizes(),
            ffn_group=sides.ffn_output.group,
            is_first_layer=self.is_first_layer,
            is_last_layer=self.is_last_layer,
        )
        move, completes_sum = _select_mhc_ffn_output_move(scattered)
        return BoundarySteps(
            attention_input=partial(
                _input_scattered_attention_input,
                qkv_latent_func=self.qkv_latent_func,
            ),
            ffn_input=_select_mhc_ffn_input(scattered, self.mhc),
            ffn_output=scattered.ffn_output,
            ffn_output_move=move,
            ffn_sum_is_movable=False,
            ffn_output_move_completes_sum=completes_sum,
            layer_input=(
                tp_reduce_scatter if scattered.input_owes is SumGroup.TP else None
            ),
        )

    def _local_token_move_can_go_to_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        # MHC's move back to this rank's tokens also combines the streams.
        return False

    def _post_init_communicate(self):
        # Base MOE_FULL callables do not accept ``mhc``, so reject this
        # combination at construction.
        if self.layer_scatter_modes.mlp_mode == ScatterMode.MOE_FULL:
            raise NotImplementedError(
                "MHCLayerCommunicator does not support MOE_FULL "
                "(moe_dp_size < attention_context_parallel_size). Increase "
                "moe_dp_size to match attention_context_parallel_size."
            )
        attention_input = CommunicateSimpleFn.get_fn(
            input_mode=self.layer_scatter_modes.layer_input_mode,
            output_mode=self.layer_scatter_modes.attn_mode,
            context=self._context,
        )
        postprocess = MHCCommunicateSummableTensorPairFn.get_fn(
            hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,
            residual_input_mode=self.layer_scatter_modes.middle_residual_mode,
            output_mode=self.layer_scatter_modes.layer_output_mode,
            context=self._context,
        )
        # MHC's own prepare_attn and postprocess run these; its move back over
        # attention DP is the empty move, as on the declarations.
        if postprocess is MHCCommunicateSummableTensorPairFn._scatter_hidden_states:
            postprocess = None
        return (
            partial(
                _attention_input,
                move=attention_input,
                qkv_latent_func=self.qkv_latent_func,
            ),
            postprocess,
        )

    def prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        self.publish_attn_lora_layout()
        steps = self._batch_steps(forward_batch)
        if self.is_first_layer:
            if steps.layer_input is not None:
                hidden_states, _ = steps.layer_input(hidden_states, None, self._context)
            hidden_states = hc_expand(hidden_states, self.mhc.hc_mult)

        hidden_states, residual = self.mhc.attn_split(
            hidden_states, out_norm=self.input_layernorm
        )
        hidden_states = steps.attention_input(
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            context=self._context,
        )
        return hidden_states, residual

    def _select_mlp_input(self):
        """MHC's own implementation of each boundary kind, applied to this
        layer's MHC state."""
        kind = mlp_input_kind(self.layer_scatter_modes, self._context)
        residual_input_mode = self.layer_scatter_modes.layer_input_mode
        fns = MHCCommunicateWithAllReduceAndLayerNormFn
        if kind is MlpInputKind.NORM:
            return partial(fns._simple, mhc=self.mhc), ()
        if kind is MlpInputKind.GATHER:
            return (
                partial(
                    fns._gather_hidden_states_and_residual,
                    residual_on_slice=residual_input_mode == ScatterMode.SCATTERED
                    and self._context.attn_tp_size > 1,
                    mhc=self.mhc,
                ),
                (),
            )
        if kind is MlpInputKind.SCATTER:
            return (
                partial(
                    fns._scatter_hidden_states_and_residual,
                    scatters_residual=residual_input_mode == ScatterMode.TP_ATTN_FULL,
                    mhc=self.mhc,
                ),
                (),
            )
        raise NotImplementedError(f"MHCLayerCommunicator does not support {kind}")

    def postprocess_layer(self, hidden_states, residual, forward_batch):
        move = self._batch_steps(forward_batch).ffn_output_move
        if move is None:
            # Back over attention DP, combining the streams on this rank's rows.
            move = MHCCommunicateSummableTensorPairFn._scatter_hidden_states
        hidden_states, residual = move(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            context=self._context,
            allow_reduce_scatter=self.allow_reduce_scatter,
            mhc=self.mhc,
            is_last_layer=self.is_last_layer,
        )
        self.mhc.reset_aux()

        return hidden_states, residual

    def _select_ffn_completion(self, forward_batch: ForwardBatch) -> FfnCompletion:
        """An MHC layer's own postprocess completes its FFN output, combining the
        hyper-connection streams; nothing is left to the next layer."""
        return FfnCompletion(
            defer_moe_finalize=False,
            fuse_mlp_allreduce=False,
            mlp_reduce_scatter=self.should_use_reduce_scatter(forward_batch),
            complete=partial(
                self._complete_ffn_output_now,
                forward_batch=forward_batch,
                dp_step=None,
            ),
        )

    def should_use_reduce_scatter(self, forward_batch: ForwardBatch):
        if not self.allow_reduce_scatter:
            return False
        steps = self._batch_steps(forward_batch)
        if steps.returns_over_dp:
            # reduce_scatterv already combines expert outputs; returning False
            # would make RowParallelLinear perform an extra all-reduce.
            if should_use_dp_reduce_scatterv():
                return True
            if forward_batch.dp_padding_mode.is_max_len():
                return True
        return steps.ffn_output_move_completes_sum
