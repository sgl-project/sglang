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
from typing import Callable, Optional

import torch

from sglang.kernels.ops.layernorm.mhc import hc_contract, hc_expand
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.communication_op import (
    attention_tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.communicator import (
    AttentionInputs,
    CommunicateContext,
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
    get_attn_tp_context,
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


def tp_all_gather_hidden_states(hidden_states, forward_batch):
    assert (
        get_attn_tp_context().input_scattered
    ), "Input scattered guarantees same num tokens in TP group."
    total_tokens = forward_batch.input_ids.shape[0]
    output = hidden_states.new_empty((total_tokens, hidden_states.shape[-1]))
    get_tp_group().all_gather_into_tensor(output, hidden_states)

    return output


@dataclass
class MHCState:
    """Parameters belong to the owning layer; this state only holds scratch
    shared across communication stages."""

    hc_mult: int
    hc_attn_pre: Callable
    hc_ffn_pre: Callable
    hc_post: Callable
    hc_attn_to_mlp: Optional[Callable] = None
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
        if self.hc_attn_to_mlp is not None:
            fused = self.hc_attn_to_mlp(
                hidden_states,
                residual,
                self.h_res,
                self.h_post,
                out_norm_weight,
                out_norm_eps,
            )
            if fused is not None:
                (
                    hidden_states,
                    residual,
                    self.h_res,
                    self.h_post,
                    norm_fused,
                ) = fused
                if (
                    out_norm is not None
                    and not norm_fused
                    and hidden_states.shape[0] != 0
                ):
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


class MHCCommunicateWithAllReduceAndLayerNormFn(CommunicateWithAllReduceAndLayerNormFn):
    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        hidden_states_output_mode: ScatterMode,
        residual_output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        fn = CommunicateWithAllReduceAndLayerNormFn.get_fn(
            hidden_states_input_mode,
            residual_input_mode,
            hidden_states_output_mode,
            residual_output_mode,
            context,
        )
        replacements = {
            CommunicateWithAllReduceAndLayerNormFn._simple: MHCCommunicateWithAllReduceAndLayerNormFn._simple,
            CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual: MHCCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
            CommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual: MHCCommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual,
        }
        if isinstance(fn, partial):
            return partial(
                replacements.get(fn.func, fn.func),
                *fn.args,
                **(fn.keywords or {}),
            )
        return replacements.get(fn, fn)

    @staticmethod
    def _scatter_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
        mhc: MHCState,
    ):
        input_hidden_states = hidden_states
        hidden_states = hidden_states.tensor_split(context.attn_tp_size)[
            context.attn_tp_rank
        ]
        attn_tp_reduce_scatter_tensor(hidden_states, input_hidden_states)
        if residual_input_mode == ScatterMode.TP_ATTN_FULL:
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
    def _tp_all_reduce_with_scattered_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        mhc: MHCState,
    ):
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states

        scatter_states = hidden_states.tensor_split(context.tp_size)[context.tp_rank]
        get_tp_group().reduce_scatter_tensor(scatter_states, hidden_states)

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
        residual_input_mode,
        mhc: MHCState,
    ):
        if get_attn_tp_context().input_scattered:
            return MHCCommunicateWithAllReduceAndLayerNormFn._tp_all_reduce_with_scattered_residual(
                hidden_states,
                residual,
                layernorm,
                context,
                mhc=mhc,
            )

        if residual_input_mode == ScatterMode.SCATTERED and context.attn_tp_size > 1:
            raise NotImplementedError(
                "Unsupported: h_res/h_post allgather not implemented."
            )

        hidden_states = attention_tensor_model_parallel_all_reduce(hidden_states)
        if context.attn_dp_size != 1:
            if hidden_states.shape[0] != 0:
                with use_symmetric_memory(
                    get_tp_group(),
                    disabled=not is_allocation_symmetric(),
                ):
                    hidden_states, residual = mhc.attn_to_mlp(
                        hidden_states, residual, out_norm=layernorm
                    )
            else:
                hidden_states, residual = mhc.attn_to_mlp(hidden_states, residual)

            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_tp_group()),
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
        if get_attn_tp_context().input_scattered:
            hidden_states, _ = tp_reduce_scatter(hidden_states, None, context)

        hidden_states = mhc.mlp_combine(hidden_states, residual)
        if not is_last_layer:
            return hidden_states, None

        hidden_states = hc_contract(hidden_states, mhc.hc_mult)
        if get_attn_tp_context().input_scattered:
            local_states = hidden_states
            hidden_states = local_states.new_empty(
                local_states.shape[0] * context.tp_size, *local_states.shape[1:]
            )
            get_tp_group().all_gather_into_tensor(hidden_states, local_states)

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
            get_local_dp_buffer_mhc(get_tp_group(), 1),
            hidden_states,
        )
        # MoE skips its post-expert all-reduce with reduce_scatterv, so this
        # scatter must reduce while combining local-expert partial sums.
        if should_use_dp_reduce_scatterv():
            get_tp_group().reduce_scatterv(
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
                get_tp_group(), 1 if is_last_layer else mhc.hc_mult
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
        is_last_layer: bool = False,
        qkv_latent_func: Optional[Callable] = None,
        *,
        is_first_layer: bool,
        hc_mult: int,
        hc_attn_pre: Callable,
        hc_ffn_pre: Callable,
        hc_post: Callable,
        hc_attn_to_mlp: Optional[Callable] = None,
    ):
        self.is_first_layer = is_first_layer
        self.mhc = MHCState(
            hc_mult=hc_mult,
            hc_attn_pre=hc_attn_pre,
            hc_ffn_pre=hc_ffn_pre,
            hc_post=hc_post,
            hc_attn_to_mlp=hc_attn_to_mlp,
        )

        super().__init__(
            layer_scatter_modes,
            input_layernorm,
            post_attention_layernorm,
            allow_reduce_scatter,
            is_last_layer,
            qkv_latent_func,
        )

    def _post_init_communicate(self):
        # Base MOE_FULL callables do not accept ``mhc``, so reject this
        # combination at construction.
        if self.layer_scatter_modes.mlp_mode == ScatterMode.MOE_FULL:
            raise NotImplementedError(
                "MHCLayerCommunicator does not support MOE_FULL "
                "(moe_dp_size < attention_context_parallel_size without "
                "--enable-prefill-cp). Set --enable-prefill-cp or raise "
                "moe_dp_size to match attention_context_parallel_size."
            )
        self._communicate_simple_fn = CommunicateSimpleFn.get_fn(
            input_mode=self.layer_scatter_modes.layer_input_mode,
            output_mode=self.layer_scatter_modes.attn_mode,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = (
            MHCCommunicateWithAllReduceAndLayerNormFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.attn_mode,
                residual_input_mode=self.layer_scatter_modes.layer_input_mode,
                hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,
                residual_output_mode=self.layer_scatter_modes.middle_residual_mode,
                context=self._context,
            )
        )
        self._communicate_summable_tensor_pair_fn = (
            MHCCommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,
                residual_input_mode=self.layer_scatter_modes.middle_residual_mode,
                output_mode=self.layer_scatter_modes.layer_output_mode,
                context=self._context,
            )
        )

    def prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if self.is_first_layer:
            if get_attn_tp_context().input_scattered:
                hidden_states, _ = tp_reduce_scatter(
                    hidden_states,
                    None,
                    self._context,
                )
            hidden_states = hc_expand(hidden_states, self.mhc.hc_mult)

        hidden_states, residual = self.mhc.attn_split(
            hidden_states, out_norm=self.input_layernorm
        )

        hidden_states = self._communicate_simple_fn(
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            context=self._context,
        )

        # DSA and attention without a QKV hook consume full hidden states, so
        # gather them before attention.
        ctx = get_attn_tp_context()
        dsa_pre_gather = ctx.input_scattered and ctx.is_dsa
        no_qkv_latent_pre_gather = ctx.input_scattered and self.qkv_latent_func is None
        if dsa_pre_gather or no_qkv_latent_pre_gather:
            hidden_states = tp_all_gather_hidden_states(hidden_states, forward_batch)

        if self.qkv_latent_func is not None:
            attn_inputs = AttentionInputs(
                hidden_states,
                forward_batch,
                self.qkv_latent_func,
                is_pre_gathered=dsa_pre_gather,
            )
            ctx.set_attn_inputs(attn_inputs)

        return hidden_states, residual

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        cache=None,
    ):
        if cache is not None:
            self._context.cache = cache

        hidden_states, residual = self._communicate_with_all_reduce_and_layer_norm_fn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            layernorm=self.post_attention_layernorm,
            context=self._context,
            mhc=self.mhc,
        )

        return hidden_states, residual

    def postprocess_layer(self, hidden_states, residual, forward_batch):
        hidden_states, residual = self._communicate_summable_tensor_pair_fn(
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

    def should_fuse_mlp_allreduce_with_next_layer(self, forward_batch):
        return False

    def should_use_reduce_scatter(self, forward_batch: ForwardBatch):
        if not self.allow_reduce_scatter:
            return False
        if (
            self._communicate_summable_tensor_pair_fn
            is MHCCommunicateSummableTensorPairFn._scatter_hidden_states
        ):
            # reduce_scatterv already combines expert outputs; returning False
            # would make RowParallelLinear perform an extra all-reduce.
            if should_use_dp_reduce_scatterv():
                return True
            if forward_batch.dp_padding_mode.is_max_len():
                return True

        if dsa_use_prefill_cp(forward_batch):
            return True

        if get_attn_tp_context().input_scattered:
            return True
        return False
