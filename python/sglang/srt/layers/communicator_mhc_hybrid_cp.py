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
from typing import Optional

import torch

from sglang.kernels.ops.layernorm.mhc import hc_contract
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.communicator import (
    CommunicateContext,
    ScatterMode,
)
from sglang.srt.layers.communicator_dsa_cp import (
    DSACPCommunicateSimpleFn,
    DSACPCommunicateSummableTensorPairFn,
    DSACPCommunicateWithAllReduceAndLayerNormFn,
    maybe_prefetch_next_full_attention_kv,
)
from sglang.srt.layers.communicator_mhc import MHCLayerCommunicator, MHCState
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    attn_cp_reduce_scatter_tensor,
    get_local_dp_buffer_mhc,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class MHCHybridDSACPLayerCommunicator(MHCLayerCommunicator):
    def _post_init_communicate(self):
        # SCATTERED in attn tp is different from SCATTERED in global tp when dp_size > 1
        if self.layer_scatter_modes.mlp_mode != ScatterMode.SCATTERED:
            assert (
                self._context.attn_dp_size == 1
            ), "dp_size should be 1 when moe_runner_backend is none"
        self._communicate_simple_fn = DSACPCommunicateSimpleFn.get_fn(
            input_mode=ScatterMode.SCATTERED,
            output_mode=ScatterMode.SCATTERED,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = (
            MHCHybridDSACPCommunicateWithAllReduceAndLayerNormFn.get_fn(
                hidden_states_input_mode=ScatterMode.SCATTERED,
                residual_input_mode=ScatterMode.SCATTERED,
                hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,
                residual_output_mode=ScatterMode.SCATTERED,
                context=self._context,
            )
        )
        self._communicate_summable_tensor_pair_fn = (
            MHCHybridDSACPCommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,
                residual_input_mode=ScatterMode.SCATTERED,
                output_mode=ScatterMode.SCATTERED,
                context=self._context,
            )
        )

    def maybe_prefetch_next_full_attention_kv(
        self,
        forward_batch: ForwardBatch,
        next_full_attention_layer_id: Optional[int],
    ) -> None:
        maybe_prefetch_next_full_attention_kv(
            forward_batch, next_full_attention_layer_id
        )


class MHCHybridDSACPCommunicateWithAllReduceAndLayerNormFn(
    DSACPCommunicateWithAllReduceAndLayerNormFn
):
    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        hidden_states_output_mode: ScatterMode,
        residual_output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        assert hidden_states_input_mode == ScatterMode.SCATTERED
        assert residual_input_mode == ScatterMode.SCATTERED
        assert residual_output_mode == ScatterMode.SCATTERED
        if hidden_states_output_mode == ScatterMode.SCATTERED:
            return MHCHybridDSACPCommunicateWithAllReduceAndLayerNormFn._simple

        if hidden_states_output_mode == ScatterMode.FULL:
            return partial(
                MHCHybridDSACPCommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,
            )

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} "
            f"{hidden_states_output_mode=} {residual_output_mode=}"
        )

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
        hidden_states, residual = mhc.attn_to_mlp(
            hidden_states, residual, out_norm=layernorm
        )

        if dsa_use_prefill_cp(forward_batch):
            assert context.attn_dp_size == 1
            hidden_states, local_hidden_states = (
                get_local_dp_buffer_mhc(get_tp_group(), 1),
                hidden_states,
            )
            attn_cp_all_gather_into_tensor(
                hidden_states,
                local_hidden_states,
            )
        return hidden_states, residual


class MHCHybridDSACPCommunicateSummableTensorPairFn(
    DSACPCommunicateSummableTensorPairFn
):
    @staticmethod
    def get_fn(
        hidden_states_input_mode: ScatterMode,
        residual_input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if (
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.SCATTERED)
        ):
            return MHCHybridDSACPCommunicateSummableTensorPairFn._scatter_hidden_states

        if context.is_same_group_size(
            hidden_states_input_mode, output_mode
        ) and context.is_same_group_size(residual_input_mode, output_mode):
            return MHCHybridDSACPCommunicateSummableTensorPairFn._trivial

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

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

        hidden_states = hc_contract(hidden_states, mhc.hc_mult)
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
        if dsa_use_prefill_cp(forward_batch):
            assert context.attn_dp_size == 1
            input_hidden_states = hidden_states
            hidden_states = hidden_states.tensor_split(context.attn_cp_size)[
                context.attn_cp_rank
            ]
            attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)

        hidden_states = mhc.mlp_combine(hidden_states, residual)
        if not is_last_layer:
            return hidden_states, None

        hidden_states = hc_contract(hidden_states, mhc.hc_mult)
        return hidden_states, None
