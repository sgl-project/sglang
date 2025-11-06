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
from enum import Enum, auto
from functools import partial
from typing import Dict, List, Optional

import torch

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather_into_tensor,
    attn_tp_reduce_scatter_tensor,
    dp_gather_partial,
    dp_reduce_scatter_tensor,
    dp_scatter,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_allocation_symmetric,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    get_bool_env_var,
    is_cuda,
    is_flashinfer_available,
    is_gfx95_supported,
    is_hip,
    is_sm90_supported,
    is_sm100_supported,
    prepare_weight_cache,
)

_is_flashinfer_available = is_flashinfer_available()
_is_sm90_supported = is_cuda() and is_sm90_supported()
_is_sm100_supported = is_cuda() and is_sm100_supported()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()
_is_gfx95_supported = is_gfx95_supported()

if _use_aiter and _is_gfx95_supported:
    from sglang.srt.layers.quantization.rocm_mxfp4_utils import fused_rms_mxfp4_quant

FUSE_ALLREDUCE_MAX_BATCH_SIZE = 2048


class ScatterMode(Enum):
    """
    Suppose we have TP=4, DP=2, enable-dp-attention, and the system handles seq a,b,c,d
    Model input/output: [ab, ab, cd, cd] for four ranks respectively
    SCATTERED: [a, b, c, d]
    TP_ATTN_FULL: [ab, ab, cd, cd], i.e. all ranks inside a TP attn group have full data of the group
    FULL: [abcd, abcd, abcd, abcd]
    """

    SCATTERED = auto()
    TP_ATTN_FULL = auto()
    FULL = auto()

    @staticmethod
    def model_input_output():
        """The scatter mode for model forward pass input and output data"""
        return ScatterMode.TP_ATTN_FULL


@dataclass
class _LayerModeComputationContext:
    num_layers: int
    layer_id: int
    is_layer_sparse: bool
    is_previous_layer_sparse: Optional[bool]

    def previous_layer(self):
        assert self.is_previous_layer_sparse is not None
        return _LayerModeComputationContext(
            layer_id=self.layer_id - 1,
            is_layer_sparse=self.is_previous_layer_sparse,
            is_previous_layer_sparse=None,
            num_layers=self.num_layers,
        )


@dataclass
class LayerScatterModes:
    layer_input_mode: ScatterMode
    attn_mode: ScatterMode
    # Can be further split into e.g. mlp_input_mode and mlp_output_mode if needed
    mlp_mode: ScatterMode
    middle_residual_mode: ScatterMode
    layer_output_mode: ScatterMode

    @classmethod
    def init_new(cls, **kwargs):
        context = _LayerModeComputationContext(**kwargs)
        return cls(
            layer_input_mode=cls._compute_layer_input_mode(context),
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=cls._compute_mlp_mode(context),
            middle_residual_mode=cls._compute_middle_residual_mode(context),
            layer_output_mode=cls._compute_layer_output_mode(context),
        )

    @classmethod
    def _compute_layer_input_mode(cls, context: _LayerModeComputationContext):
        if context.layer_id == 0:
            return ScatterMode.model_input_output()
        return cls._compute_layer_output_mode(context.previous_layer())

    @classmethod
    def _compute_mlp_mode(cls, context: _LayerModeComputationContext):
        if context.is_layer_sparse:
            return (
                ScatterMode.SCATTERED
                if (
                    # Token dispatch/combine will be handled outside of LayerCommunicator for these modes.
                    not get_moe_a2a_backend().is_none()
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                )
                else ScatterMode.FULL
            )
        else:
            return (
                ScatterMode.SCATTERED
                if enable_moe_dense_fully_dp()
                else ScatterMode.FULL
            )

    @classmethod
    def _compute_middle_residual_mode(cls, context: _LayerModeComputationContext):
        mlp_mode = cls._compute_mlp_mode(context)
        if mlp_mode == ScatterMode.SCATTERED:
            return ScatterMode.SCATTERED
        if mlp_mode == ScatterMode.FULL:
            return ScatterMode.TP_ATTN_FULL
        raise NotImplementedError

    @classmethod
    def _compute_layer_output_mode(cls, context: _LayerModeComputationContext):
        mlp_mode = cls._compute_mlp_mode(context)
        if context.layer_id == context.num_layers - 1:
            return ScatterMode.model_input_output()
        if mlp_mode == ScatterMode.SCATTERED:
            return ScatterMode.SCATTERED
        if mlp_mode == ScatterMode.FULL:
            return ScatterMode.TP_ATTN_FULL
        raise NotImplementedError


def enable_moe_dense_fully_dp():
    return get_global_server_args().moe_dense_tp_size == 1


class LayerCommunicator:
    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        # Reduce scatter requires skipping all-reduce in model code after MoE/MLP, so only enable for models which have that implemented. Remove flag once done for all models that use LayerCommunicator.
        allow_reduce_scatter: bool = False,
        is_last_layer: bool = False,
    ):
        self.layer_scatter_modes = layer_scatter_modes
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.allow_reduce_scatter = allow_reduce_scatter
        self.is_last_layer = is_last_layer

        self._context = CommunicateContext.init_new()
        self._communicate_simple_fn = CommunicateSimpleFn.get_fn(
            input_mode=self.layer_scatter_modes.layer_input_mode,
            output_mode=self.layer_scatter_modes.attn_mode,
            context=self._context,
        )
        self._communicate_with_all_reduce_and_layer_norm_fn = (
            CommunicateWithAllReduceAndLayerNormFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.attn_mode,
                residual_input_mode=self.layer_scatter_modes.layer_input_mode,
                hidden_states_output_mode=self.layer_scatter_modes.mlp_mode,
                residual_output_mode=self.layer_scatter_modes.middle_residual_mode,
                context=self._context,
            )
        )
        self._communicate_summable_tensor_pair_fn = (
            CommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=self.layer_scatter_modes.mlp_mode,
                residual_input_mode=self.layer_scatter_modes.middle_residual_mode,
                output_mode=self.layer_scatter_modes.layer_output_mode,
                context=self._context,
            )
        )

        self._speculative_algo = SpeculativeAlgorithm.from_string(
            get_global_server_args().speculative_algorithm
        )

    def prepare_attn_and_capture_last_layer_outputs(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
    ):
        hidden_states, residual = self.prepare_attn(
            hidden_states, residual, forward_batch
        )
        if captured_last_layer_outputs is not None:
            gathered_last_layer_output = self._communicate_simple_fn(
                hidden_states=residual,
                forward_batch=forward_batch,
                context=self._context,
            )
            if gathered_last_layer_output is residual:
                # Clone to avoid modifying the original residual by Custom RMSNorm inplace operation
                gathered_last_layer_output = residual.clone()
            captured_last_layer_outputs.append(gathered_last_layer_output)
        return hidden_states, residual

    def prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        qaunt_format: str = "",
    ):
        if hidden_states.shape[0] == 0:
            residual = hidden_states
        else:
            if (
                residual is not None
                and hasattr(hidden_states, "_sglang_needs_allreduce_fusion")
                and hidden_states._sglang_needs_allreduce_fusion
            ):
                hidden_states, residual = (
                    self.input_layernorm.forward_with_allreduce_fusion(
                        hidden_states, residual
                    )
                )
            else:
                if residual is None:
                    residual = hidden_states

                    if _use_aiter and _is_gfx95_supported and ("mxfp4" in qaunt_format):
                        hidden_states = fused_rms_mxfp4_quant(
                            hidden_states,
                            self.input_layernorm.weight,
                            self.input_layernorm.variance_epsilon,
                            None,
                            None,
                            None,
                            None,
                        )
                    else:
                        hidden_states = self.input_layernorm(hidden_states)
                else:
                    if _use_aiter and _is_gfx95_supported and ("mxfp4" in qaunt_format):
                        hidden_states, residual = fused_rms_mxfp4_quant(
                            hidden_states,
                            self.input_layernorm.weight,
                            self.input_layernorm.variance_epsilon,
                            None,
                            None,
                            None,
                            residual,
                        )
                    else:
                        hidden_states, residual = self.input_layernorm(
                            hidden_states, residual
                        )

        hidden_states = self._communicate_simple_fn(
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            context=self._context,
        )

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

        return self._communicate_with_all_reduce_and_layer_norm_fn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            layernorm=self.post_attention_layernorm,
            context=self._context,
        )

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        return self._communicate_summable_tensor_pair_fn(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
            context=self._context,
            allow_reduce_scatter=self.allow_reduce_scatter,
        )

    def should_use_reduce_scatter(self, forward_batch: ForwardBatch):
        return (
            self.allow_reduce_scatter
            and self._communicate_summable_tensor_pair_fn
            is CommunicateSummableTensorPairFn._scatter_hidden_states
            and forward_batch.dp_padding_mode.is_max_len()
        )

    def should_fuse_mlp_allreduce_with_next_layer(
        self, forward_batch: ForwardBatch
    ) -> bool:
        if (
            is_dp_attention_enabled()
            and self._speculative_algo is not None
            and self._speculative_algo.is_eagle()
        ):
            return False

        batch_size = (
            forward_batch.input_ids.shape[0]
            if hasattr(forward_batch, "input_ids")
            else 0
        )
        if batch_size > FUSE_ALLREDUCE_MAX_BATCH_SIZE:
            return False

        static_conditions_met = (
            (not self.is_last_layer)
            and (self._context.tp_size > 1)
            and not is_dp_attention_enabled()
            and get_global_server_args().enable_flashinfer_allreduce_fusion
            and _is_flashinfer_available
        )

        if not static_conditions_met:
            return False

        return (
            batch_size > 0
            and batch_size <= FUSE_ALLREDUCE_MAX_BATCH_SIZE
            and (not self.is_last_layer)
        )


@dataclass
class CommunicateContext:
    process_group_sizes: Dict[ScatterMode, int]
    attn_tp_rank: int
    attn_tp_size: int
    attn_dp_size: int
    tp_size: int
    cache = None

    def is_same_group_size(self, a: ScatterMode, b: ScatterMode):
        return self.process_group_sizes[a] == self.process_group_sizes[b]

    @classmethod
    def init_new(cls):
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        attn_dp_size = get_attention_dp_size()
        tp_size = get_tensor_model_parallel_world_size()
        process_group_sizes = {
            ScatterMode.SCATTERED: 1,
            ScatterMode.TP_ATTN_FULL: attn_tp_size,
            # TODO: support --moe-dense-tp-size > 1
            ScatterMode.FULL: tp_size,
        }
        return cls(
            process_group_sizes=process_group_sizes,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=attn_tp_size,
            attn_dp_size=attn_dp_size,
            tp_size=tp_size,
        )


class CommunicateSimpleFn:
    @staticmethod
    def get_fn(
        input_mode: ScatterMode,
        output_mode: ScatterMode,
        context: CommunicateContext,
    ):
        if context.is_same_group_size(input_mode, output_mode):
            return CommunicateSimpleFn._trivial

        if (input_mode == ScatterMode.SCATTERED) and (
            output_mode == ScatterMode.TP_ATTN_FULL
        ):
            return CommunicateSimpleFn._scattered_to_tp_attn_full

        raise NotImplementedError(f"{input_mode=} {output_mode=}")

    @staticmethod
    def _trivial(
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ) -> torch.Tensor:
        return hidden_states

    @staticmethod
    def _scattered_to_tp_attn_full(
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ) -> torch.Tensor:
        hidden_states, local_hidden_states = (
            get_local_dp_buffer(),
            hidden_states,
        )
        attn_tp_all_gather_into_tensor(
            hidden_states,
            local_hidden_states,
        )
        return hidden_states


class CommunicateWithAllReduceAndLayerNormFn:
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
            return CommunicateWithAllReduceAndLayerNormFn._simple

        if (
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (
                residual_input_mode in [ScatterMode.SCATTERED, ScatterMode.TP_ATTN_FULL]
            )
            and (hidden_states_output_mode == ScatterMode.FULL)
            and (residual_output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return partial(
                CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual,
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
                CommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual,
                residual_input_mode=residual_input_mode,
            )

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {hidden_states_output_mode=} {residual_output_mode=}"
        )

    @staticmethod
    def _simple(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        layernorm: torch.nn.Module,
        context: CommunicateContext,
    ):
        # TODO move these `if shape != 0` into LayerNorm itself
        if hidden_states.shape[0] != 0:
            hidden_states, residual = layernorm(hidden_states, residual)
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
    ):
        if residual_input_mode == ScatterMode.SCATTERED and context.attn_tp_size > 1:
            residual, local_residual = (
                get_local_dp_buffer(),
                residual,
            )
            attn_tp_all_gather_into_tensor(residual, local_residual)
        if context.attn_dp_size != 1:
            if context.attn_tp_rank == 0:
                hidden_states += residual

            # Perform layernorm on smaller data before comm. Only valid when attn_tp_size is 1 (tp_size == dp_size)
            use_layer_norm_before_gather = context.attn_tp_size == 1
            if use_layer_norm_before_gather and hidden_states.shape[0] != 0:
                residual = hidden_states
                with use_symmetric_memory(
                    get_tp_group(),
                    disabled=not is_allocation_symmetric(),
                ):
                    hidden_states = layernorm(hidden_states)

            hidden_states, local_hidden_states = (
                get_global_dp_buffer(),
                hidden_states,
            )
            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)

            if not use_layer_norm_before_gather:
                dp_scatter(residual, hidden_states, forward_batch)
                if hidden_states.shape[0] != 0:
                    hidden_states = layernorm(hidden_states)
        else:
            # According to the discussion in https://github.com/flashinfer-ai/flashinfer/issues/1223#issuecomment-3047256465
            # We set the max token num to 128 for allreduce fusion with min-latency case(use_oneshot=True).
            if (
                (_is_sm100_supported or _is_sm90_supported)
                and _is_flashinfer_available
                and hasattr(layernorm, "forward_with_allreduce_fusion")
                and get_global_server_args().enable_flashinfer_allreduce_fusion
                and hidden_states.shape[0] <= 4096
            ):
                hidden_states, residual = layernorm.forward_with_allreduce_fusion(
                    hidden_states, residual
                )
            else:
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
                if context.cache is not None:
                    _ = prepare_weight_cache(hidden_states, context.cache)
                hidden_states, residual = layernorm(hidden_states, residual)
        return hidden_states, residual

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
        input_hidden_states = hidden_states
        hidden_states = hidden_states.tensor_split(context.attn_tp_size)[
            context.attn_tp_rank
        ]
        attn_tp_reduce_scatter_tensor(hidden_states, input_hidden_states)
        if residual_input_mode == ScatterMode.TP_ATTN_FULL:
            residual = residual.tensor_split(context.attn_tp_size)[context.attn_tp_rank]
        if hidden_states.shape[0] != 0:
            hidden_states, residual = layernorm(hidden_states, residual)
        return hidden_states, residual


class CommunicateSummableTensorPairFn:
    """It is allowed to make (hidden_states, residual) := (hidden_states + residual, None) if needed."""

    @classmethod
    def execute(
        cls,
        hidden_states_input_mode,
        residual_input_mode,
        output_mode,
        context,
        **kwargs,
    ):
        return cls.get_fn(
            hidden_states_input_mode=hidden_states_input_mode,
            residual_input_mode=residual_input_mode,
            output_mode=output_mode,
            context=context,
        )(context=context, **kwargs)

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
            return CommunicateSummableTensorPairFn._trivial

        if (
            (hidden_states_input_mode == ScatterMode.FULL)
            and (residual_input_mode == ScatterMode.TP_ATTN_FULL)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return CommunicateSummableTensorPairFn._scatter_hidden_states

        if (
            (hidden_states_input_mode == ScatterMode.SCATTERED)
            and (residual_input_mode == ScatterMode.SCATTERED)
            and (output_mode == ScatterMode.TP_ATTN_FULL)
        ):
            return CommunicateSummableTensorPairFn._gather

        if (
            (hidden_states_input_mode == ScatterMode.TP_ATTN_FULL)
            and (residual_input_mode == ScatterMode.TP_ATTN_FULL)
            and (output_mode == ScatterMode.SCATTERED)
        ):
            return CommunicateSummableTensorPairFn._scatter

        raise NotImplementedError(
            f"{hidden_states_input_mode=} {residual_input_mode=} {output_mode=}"
        )

    @staticmethod
    def _trivial(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        **kwargs,
    ):
        return hidden_states, residual

    @staticmethod
    def _scatter_hidden_states(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
        allow_reduce_scatter: bool = False,
    ):
        hidden_states, global_hidden_states = (
            get_local_dp_buffer(),
            hidden_states,
        )
        if allow_reduce_scatter and forward_batch.dp_padding_mode.is_max_len():
            # When using padding, all_reduce is skipped after MLP and MOE and reduce scatter is used here instead.
            dp_reduce_scatter_tensor(hidden_states, global_hidden_states)
        else:
            dp_scatter(hidden_states, global_hidden_states, forward_batch)
        return hidden_states, residual

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
        hidden_states, local_hidden_states = (
            get_local_dp_buffer(),
            hidden_states,
        )
        attn_tp_all_gather_into_tensor(
            hidden_states,
            local_hidden_states,
        )
        return hidden_states, residual

    @staticmethod
    def _scatter(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch: ForwardBatch,
        context: CommunicateContext,
    ):
        assert residual is None, "not yet handled residual!=None"
        tensor_list = list(hidden_states.tensor_split(context.attn_tp_size))
        hidden_states = tensor_list[context.attn_tp_rank]
        return hidden_states, residual
