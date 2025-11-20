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


from functools import partial
from typing import Callable, Optional

import torch

from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def nsa_enable_prefill_cp():
    # After using cp, the communication mode of this part changes.
    # The three parts of prepare_attn, prepare_mlp, and postprocess_layer
    # no longer require additional communication for reduce, scatter, etc.
    return is_nsa_enable_prefill_cp()


class NSACPLayerCommunicator(LayerCommunicator):
    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        # Reduce scatter requires skipping all-reduce in model code after MoE/MLP, so only enable for models which have that implemented. Remove flag once done for all models that use LayerCommunicator.
        allow_reduce_scatter: bool = False,
        is_last_layer: bool = False,
        qkv_latent_func: Optional[Callable] = None,
    ):
        super().__init__(
            layer_scatter_modes,
            input_layernorm,
            post_attention_layernorm,
            allow_reduce_scatter,
            is_last_layer,
            qkv_latent_func,
        )
        self._communicate_simple_fn = NSACPCommunicateSimpleFn.get_fn(
            input_mode=self.layer_scatter_modes.layer_input_mode,
            output_mode=self.layer_scatter_modes.attn_mode,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = (
            NSACPCommunicateWithAllReduceAndLayerNormFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.attn_mode,
                residual_input_mode=self.layer_scatter_modes.layer_input_mode,
                hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,
                residual_output_mode=self.layer_scatter_modes.middle_residual_mode,
                context=self._context,
            )
        )
        self._communicate_summable_tensor_pair_fn = (
            NSACPCommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,
                residual_input_mode=self.layer_scatter_modes.middle_residual_mode,
                output_mode=self.layer_scatter_modes.layer_output_mode,
                context=self._context,
            )
        )


class NSACPCommunicateSimpleFn(CommunicateSimpleFn):
    @staticmethod
    def get_fn(
        input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(input_mode, output_mode):
            return NSACPCommunicateSimpleFn._trivial

        if (input_mode == ScatterMode.SCATTERED) and (
            output_mode == ScatterMode.TP_ATTN_FULL
        ):
            return NSACPCommunicateSimpleFn._scattered_to_tp_attn_full

        raise NotImplementedError(f"{input_mode=} {output_mode=}")

    @staticmethod
    def _scattered_to_tp_attn_full(
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ) -> torch.Tensor:

        if nsa_enable_prefill_cp():
            return hidden_states
        else:
            assert False, "Not implemented"


class NSACPCommunicateWithAllReduceAndLayerNormFn(
    CommunicateWithAllReduceAndLayerNormFn
):
    """Besides communication, needs to
    1. All reduce in tp_attn_group on hidden_states
    2. Apply layer norm
    """

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        hidden_states_output_mode: ScatterMode,
        residual_output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if (
            context.is_same_group_size(
                hidden_states_input_mode, hidden_states_output_mode
            )
            and context.is_same_group_size(residual_input_mode, residual_output_mode)
            and context.attn_tp_size == 1
        ):
            return NSACPCommunicateWithAllReduceAndLayerNormFn._simple

        if (
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (
                residual_input_mode in [ScatterMode.SCATTERED, ScatterMode.TP_ATTN_FULL]
            )
            and (hidden_states_output_mode == ScatterMode.FULL)
            and (residual_output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return partial(
                NSACPCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,
            )

        if (
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (
                residual_input_mode in [ScatterMode.SCATTERED, ScatterMode.TP_ATTN_FULL]
            )
            and (hidden_states_output_mode == ScatterMode.SCATTERED)
            and (residual_output_mode == ScatterMode.SCATTERED)
        ):
            return partial(
                NSACPCommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,
            )

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {hidden_states_output_mode=} {residual_output_mode=}"
        )

    @staticmethod
    def _gather_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
    ):
        if context.attn_dp_size != 1:
            if nsa_enable_prefill_cp():
                hidden_states += residual
                if hidden_states.shape[0] != 0:
                    hidden_states = layernorm(hidden_states)
                return hidden_states, residual
            else:
                assert False, "not yet handled"

    @staticmethod
    def _scatter_hidden_states_and_residual(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
        *,
        residual_input_mode,
    ):
        if nsa_enable_prefill_cp():
            if hidden_states.shape[0] != 0:
                hidden_states, residual = layernorm(hidden_states, residual)
            return hidden_states, residual
        else:
            assert False, "not yet handled"


class NSACPCommunicateSummableTensorPairFn(CommunicateSummableTensorPairFn):
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""

    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(
            hidden_states_input_mode, output_mode
        ) and context.is_same_group_size(residual_input_mode, output_mode):
            return NSACPCommunicateSummableTensorPairFn._trivial

        if (
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.TP_ATTN_FULL)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return NSACPCommunicateSummableTensorPairFn._scatter_hidden_states

        if (
            (hidden_states_input_mode == ScatterMode.SCATTERED)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return NSACPCommunicateSummableTensorPairFn._gather

        if (
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (residual_input_mode == ScatterMode.TP_ATTN_FULL)
            and (output_mode == ScatterMode.SCATTERED)
        ):
            return NSACPCommunicateSummableTensorPairFn._scatter

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

    @staticmethod
    def _scatter_hidden_states(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        allow_reduce_scatter: bool = False,
    ):
        if nsa_enable_prefill_cp():
            return hidden_states, residual
        else:
            assert False, "not yet handled"

    @staticmethod
    def _gather(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        hidden_states += residual
        residual = None
        if nsa_enable_prefill_cp():
            return hidden_states, residual
        else:
            assert False, "not yet handled"

    @staticmethod
    def _scatter(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ):
        if nsa_enable_prefill_cp():
            return hidden_states, residual
        else:
            assert False, "not yet handled"
