# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/mxfp4.py

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import get_dp_global_num_tokens, get_local_dp_buffer
from sglang.srt.layers.moe.utils import (
    get_moe_runner_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.utils import (
    direct_register_custom_op,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_sm100_supported,
    is_triton_kernels_available,
    log_info_on_rank0,
    mxfp_supported,
    next_power_of_2,
    round_up,
    set_weight_attrs,
)

_is_sm100_supported = is_cuda() and is_sm100_supported()
has_triton_kernels = is_triton_kernels_available()


if is_flashinfer_available():
    from flashinfer import (
        block_scale_interleave,
        mxfp8_quantize,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
        trtllm_fp4_block_scale_moe,
    )

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.topk import TopKOutput

_is_hip = is_hip()

if _is_hip:
    # import aiter
    from aiter import ActivationType, QuantType, dtypes
    from aiter.fused_moe import fused_moe
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
except ImportError:
    flashinfer_cutlass_fused_moe = None


def _swizzle_mxfp4(quant_tensor, scale, num_warps):
    """weight swizzle for mxfp4 moe, used for OAI mxfp4 kernel"""
    import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
    from triton_kernels.numerics import InFlexData
    from triton_kernels.tensor import FP4, convert_layout, wrap_torch_tensor
    from triton_kernels.tensor_details import layout

    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(
        mx_axis=1
    )
    scale_layout, scale_layout_opts = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=1, num_warps=num_warps
    )
    if _is_sm100_supported:
        constraints = {
            "is_persistent": True,
            "epilogue_subtile": 1,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    # transpose the tensor so that the quantization axis is on dim1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout, **value_layout_opts
    )
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout, **scale_layout_opts)
    return quant_tensor, InFlexData(), scale


def _dequant_mxfp4(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype
) -> torch.Tensor:
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError(
            "The package `amd-quark` is required to use "
            "MX-FP4 models. Please install it with `pip install "
            "amd-quark`."
        ) from err

    return mx.dq_mxfp4(x, scale, float_dtype)


def _dequant_mxfp4_fake(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype
) -> torch.Tensor:
    return torch.empty(
        (*x.shape[:-1], x.shape[-1] * 2), dtype=float_dtype, device=x.device
    )


def _quant_dequant_mxfp4(
    x: torch.Tensor, scale_calculation_mode: str = "even"
) -> torch.Tensor:
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError(
            "The package `amd-quark` is required to use "
            "MX-FP4 models. Please install it with `pip install "
            "amd-quark`."
        ) from err

    return mx.qdq_mxfp4(x, scale_calculation_mode)


def _quant_dequant_mxfp4_fake(
    x: torch.Tensor, scale_calculation_mode: str = "even"
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="dequant_mxfp4",
    op_func=_dequant_mxfp4,
    mutates_args=[],
    fake_impl=_dequant_mxfp4_fake,
)
dequant_mxfp4 = torch.ops.sglang.dequant_mxfp4

direct_register_custom_op(
    op_name="quant_dequant_mxfp4",
    op_func=_quant_dequant_mxfp4,
    mutates_args=[],
    fake_impl=_quant_dequant_mxfp4_fake,
)
quant_dequant_mxfp4 = torch.ops.sglang.quant_dequant_mxfp4


class Mxfp4Config(QuantizationConfig):

    def __init__(
        self,
        ignored_layers: Optional[list[str]] = None,
        is_checkpoint_mxfp4_serialized: bool = False,
    ):
        super().__init__()
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):

        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_mxfp4_serialized = "mxfp4" in quant_method

        if _is_hip:
            if mxfp_supported():
                return cls(
                    is_checkpoint_mxfp4_serialized=is_checkpoint_mxfp4_serialized
                )
            else:

                platform = torch.cuda.get_device_properties(0).gcnArchName
                raise ValueError(
                    f"Current platform {platform} not support mxfp4 computation"
                )

        return cls(is_checkpoint_mxfp4_serialized=is_checkpoint_mxfp4_serialized)

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def is_static_cfg(self):
        return self.is_checkpoint_mxfp4_serialized

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:

        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        if isinstance(layer, LinearBase):
            if self.ignored_layers and is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            elif _is_hip:
                return UnquantizedLinearMethod()
        elif isinstance(layer, FusedMoE):
            if self.is_checkpoint_mxfp4_serialized:
                return Mxfp4MoEMethod(prefix=prefix)
            else:
                return Mxfp4DynamicQuantMoEMethod()
        else:
            if self.is_checkpoint_mxfp4_serialized:
                raise NotImplementedError("Mxfp4 attention layer is not implemented")
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Mxfp4MoEMethod(FusedMoEMethodBase):

    def __init__(
        self,
        prefix: str,
    ):
        super().__init__()

        self.prefix = prefix
        self.topk_indices_dtype = None
        self.use_triton_kernels = get_moe_runner_backend().is_triton_kernel()
        self.with_bias = False
        self.enable_flashinfer_cutlass_moe = (
            get_moe_runner_backend().is_flashinfer_cutlass()
        )
        self.use_flashinfer = (
            get_moe_runner_backend().is_flashinfer_trtllm()
            or self.enable_flashinfer_cutlass_moe
        )
        self.flashinfer_mxfp4_moe_precision = global_server_args_dict[
            "flashinfer_mxfp4_moe_precision"
        ]

        self.triton_kernel_moe_forward = None
        self.triton_kernel_moe_with_bias_forward = None
        if torch.cuda.is_available() and has_triton_kernels:
            from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
                triton_kernel_moe_forward as _tk_forward,
            )
            from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import (
                triton_kernel_moe_with_bias_forward as _tk_with_bias_forward,
            )

            self.triton_kernel_moe_forward = _tk_forward
            self.triton_kernel_moe_with_bias_forward = _tk_with_bias_forward

        self.weight_dtype = (
            torch.int64 if self.enable_flashinfer_cutlass_moe else torch.uint8
        )
        self.initial_weight_dtype = torch.uint8
        self.initial_scale_dtype = torch.uint8
        self.scale_dtype = (
            torch.int32 if self.enable_flashinfer_cutlass_moe else torch.uint8
        )
        self.weight_alignment = 128 if self.enable_flashinfer_cutlass_moe else 256
        self.weight_vec_size = torch.iinfo(self.weight_dtype).bits // 4
        self.scaling_vector_size = 32

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        self.with_bias = with_bias

        self.initial_intermediate_size = intermediate_size
        self.initial_hidden_size = hidden_size
        # pad the intermediate size to be a multiple of 2 * mxfp4_block
        # for to hold non-uniform sharded tensor as well as swizzling
        intermediate_size_per_partition_after_pad = intermediate_size
        if _is_sm100_supported:
            if self.use_flashinfer:
                intermediate_size_per_partition_after_pad = round_up(
                    intermediate_size, self.weight_alignment
                )
                hidden_size = round_up(hidden_size, self.weight_alignment)
            else:
                intermediate_size_per_partition_after_pad = round_up(
                    intermediate_size, 64
                )
        elif has_triton_kernels:
            # TODO: this is a hack to make
            # intermediate_size_per_partition_after_pad the same as the
            # per_rank_intermediate_size during weight loading
            intermediate_size_per_partition_after_pad = round_up(
                intermediate_size, self.scaling_vector_size
            )

        self.intermediate_size = intermediate_size_per_partition_after_pad

        self.hidden_size = hidden_size
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                layer.num_local_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // self.weight_vec_size,
                dtype=self.weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                layer.num_local_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // self.scaling_vector_size,
                dtype=self.initial_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w13_weight_bias = torch.nn.Parameter(
            torch.zeros(
                layer.num_local_experts,
                2 * intermediate_size_per_partition_after_pad,
                dtype=torch.bfloat16,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_bias", w13_weight_bias)
        set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.zeros(
                layer.num_local_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // self.weight_vec_size,
                dtype=self.weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                layer.num_local_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // self.scaling_vector_size,
                dtype=self.initial_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_weight_bias = torch.nn.Parameter(
            torch.zeros(layer.num_local_experts, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_bias", w2_weight_bias)
        set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        if self.enable_flashinfer_cutlass_moe:
            fake_input_scale = torch.nn.Parameter(
                torch.ones(layer.num_local_experts, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("fake_input_scale", fake_input_scale)

        if self.use_flashinfer:
            # TODO: these values are hardcoded for now, we need to get them from the model
            layer.gemm1_alpha = Parameter(
                torch.tensor([1.702] * self.num_experts, dtype=torch.float32).cuda(),
                requires_grad=False,
            )
            layer.gemm1_beta = Parameter(
                torch.tensor([1.0] * self.num_experts, dtype=torch.float32).cuda(),
                requires_grad=False,
            )
            layer.gemm1_clamp_limit = Parameter(
                torch.tensor([7.0] * self.num_experts, dtype=torch.float32).cuda(),
                requires_grad=False,
            )

            assert (
                layer.w13_weight.dim() == 3
                and layer.w13_weight.shape[0] == self.num_experts
                and layer.w13_weight.shape[1] == self.intermediate_size * 2
                and layer.w13_weight.shape[2]
                == self.hidden_size // self.weight_vec_size
            )
            assert (
                layer.w13_weight_scale.dim() == 3
                and layer.w13_weight_scale.shape[0] == self.num_experts
                and layer.w13_weight_scale.shape[1] == self.intermediate_size * 2
                and layer.w13_weight_scale.shape[2]
                == self.hidden_size // self.scaling_vector_size
            )
            assert (
                layer.w2_weight.dim() == 3
                and layer.w2_weight.shape[0] == self.num_experts
                and layer.w2_weight.shape[1] == self.hidden_size
                and layer.w2_weight.shape[2]
                == self.intermediate_size // self.weight_vec_size
            )
            assert (
                layer.w2_weight_scale.dim() == 3
                and layer.w2_weight_scale.shape[1] == self.hidden_size
                and layer.w2_weight_scale.shape[2]
                == self.intermediate_size // self.scaling_vector_size
            )
            assert (
                layer.w13_weight_bias.dim() == 2
                and layer.w13_weight_bias.shape[0] == self.num_experts
                and layer.w13_weight_bias.shape[1] == self.intermediate_size * 2
            )
            assert (
                layer.w2_weight_bias.dim() == 2
                and layer.w2_weight_bias.shape[0] == self.num_experts
                and layer.w2_weight_bias.shape[1] == self.hidden_size
            )

            w13_weight_scale = layer.w13_weight_scale.data
            w2_weight_scale = layer.w2_weight_scale.data
            w13_weight = layer.w13_weight.data
            w2_weight = layer.w2_weight.data
            w13_bias = layer.w13_weight_bias.data
            w2_bias = layer.w2_weight_bias.data

            if self.enable_flashinfer_cutlass_moe:
                # Deinterleave gate and up
                gate_weight, up_weight = w13_weight[:, ::2, :], w13_weight[:, 1::2, :]
                w13_weight = torch.cat([up_weight, gate_weight], dim=-2)
                gate_weight_scale, up_weight_scale = (
                    w13_weight_scale[:, ::2, :],
                    w13_weight_scale[:, 1::2, :],
                )
                w13_weight_scale = torch.cat(
                    [up_weight_scale, gate_weight_scale], dim=-2
                )
                gate_bias, up_bias = w13_bias[:, ::2], w13_bias[:, 1::2]
                w13_bias = torch.cat([up_bias, gate_bias], dim=-1)

                layer.w13_weight = Parameter(w13_weight, requires_grad=False)
                layer.w13_weight_bias = Parameter(w13_bias, requires_grad=False)

                w13_original_shape = w13_weight_scale.shape
                w2_original_shape = w2_weight_scale.shape
                interleaved_w13_weight_scale = []
                interleaved_w2_weight_scale = []
                for i in range(self.num_experts):
                    interleaved_w13_weight_scale.append(
                        block_scale_interleave(w13_weight_scale[i])
                    )
                    interleaved_w2_weight_scale.append(
                        block_scale_interleave(w2_weight_scale[i])
                    )

                w13_weight_scale = (
                    torch.stack(interleaved_w13_weight_scale)
                    .reshape(w13_original_shape)
                    .view(self.scale_dtype)
                )
                w2_weight_scale = (
                    torch.stack(interleaved_w2_weight_scale)
                    .reshape(w2_original_shape)
                    .view(self.scale_dtype)
                )
                layer.w13_weight_scale = Parameter(
                    w13_weight_scale, requires_grad=False
                )
                layer.w2_weight_scale = Parameter(w2_weight_scale, requires_grad=False)
            else:
                log_info_on_rank0(
                    logger,
                    f"Shuffling MoE weights for FlashInfer MXFP4 moe kernel (layer: {self.prefix}), it might take a while...",
                )

                # Swap w1 and w3 as the definition of
                # swiglu is different in the trtllm-gen
                def swap_every_two_rows(x, axis=-1):
                    shape = x.shape
                    if axis < 0:
                        axis = len(shape) + axis

                    # Create a new shape with pairs swapped along specified axis
                    new_shape = list(shape)
                    new_shape[axis] = shape[axis] // 2
                    new_shape.insert(axis + 1, 2)

                    # Reshape to expose pairs, swap them, and reshape back
                    x = x.reshape(*new_shape)
                    x = x.flip(axis + 1)
                    new_shape = list(shape)
                    return x.reshape(*new_shape)

                w13_weight_scale = swap_every_two_rows(w13_weight_scale, -2)
                w13_weight = swap_every_two_rows(w13_weight, -2)
                w13_bias = swap_every_two_rows(w13_bias, -1)

                # Shuffle weights and scaling factors for transposed mma output
                gemm1_weights_mxfp4_shuffled = []
                gemm1_scales_mxfp4_shuffled = []
                gemm2_weights_mxfp4_shuffled = []
                gemm2_scales_mxfp4_shuffled = []
                gemm1_bias_shuffled = []
                gemm2_bias_shuffled = []
                epilogue_tile_m = 128  # FIXME: this depends on the kernel internals
                w13_bias = w13_bias.to(torch.float32)
                w2_bias = w2_bias.to(torch.float32)
                for i in range(self.num_experts):
                    gemm1_weights_mxfp4_shuffled.append(
                        shuffle_matrix_a(
                            w13_weight[i].view(torch.uint8), epilogue_tile_m
                        )
                    )
                    gemm1_scales_mxfp4_shuffled.append(
                        shuffle_matrix_sf_a(
                            w13_weight_scale[i].view(torch.uint8), epilogue_tile_m
                        )
                    )
                    gemm1_bias_shuffled.append(
                        shuffle_matrix_a(
                            w13_bias[i].clone().reshape(-1, 1), epilogue_tile_m
                        )
                    )

                    gemm2_weights_mxfp4_shuffled.append(
                        shuffle_matrix_a(
                            w2_weight[i].view(torch.uint8), epilogue_tile_m
                        )
                    )
                    gemm2_scales_mxfp4_shuffled.append(
                        shuffle_matrix_sf_a(
                            w2_weight_scale[i].view(torch.uint8), epilogue_tile_m
                        )
                    )
                    gemm2_bias_shuffled.append(
                        shuffle_matrix_a(
                            w2_bias[i].clone().reshape(-1, 1), epilogue_tile_m
                        )
                    )

                w13_weight = torch.stack(gemm1_weights_mxfp4_shuffled)
                w13_weight_scale = (
                    torch.stack(gemm1_scales_mxfp4_shuffled)
                    .reshape(
                        self.num_experts,
                        2 * self.intermediate_size,
                        self.hidden_size // self.scaling_vector_size,
                    )
                    .view(torch.float8_e4m3fn)
                )

                w2_weight = torch.stack(gemm2_weights_mxfp4_shuffled)
                w2_weight_scale = (
                    torch.stack(gemm2_scales_mxfp4_shuffled)
                    .reshape(
                        self.num_experts,
                        self.hidden_size,
                        self.intermediate_size // self.scaling_vector_size,
                    )
                    .view(torch.float8_e4m3fn)
                )

                layer.w13_weight = Parameter(w13_weight, requires_grad=False)
                layer.w13_weight_scale = Parameter(
                    w13_weight_scale, requires_grad=False
                )
                layer.w2_weight = Parameter(w2_weight, requires_grad=False)
                layer.w2_weight_scale = Parameter(w2_weight_scale, requires_grad=False)
                layer.w13_weight_bias = Parameter(
                    torch.stack(gemm1_bias_shuffled).reshape(self.num_experts, -1),
                    requires_grad=False,
                )
                layer.w2_weight_bias = Parameter(
                    torch.stack(gemm2_bias_shuffled).reshape(self.num_experts, -1),
                    requires_grad=False,
                )
            return

        if self.use_triton_kernels:

            from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

            w13_weight_bias = layer.w13_weight_bias.to(torch.float32)
            w2_weight_bias = layer.w2_weight_bias.to(torch.float32)

            layer.w13_weight_bias = Parameter(w13_weight_bias, requires_grad=False)
            layer.w2_weight_bias = Parameter(w2_weight_bias, requires_grad=False)

            num_warps = 8

            w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
                layer.w13_weight, layer.w13_weight_scale, num_warps
            )
            w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
                layer.w2_weight, layer.w2_weight_scale, num_warps
            )

            self.w13_precision_config = PrecisionConfig(
                weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex)
            )
            self.w2_precision_config = PrecisionConfig(
                weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex)
            )

            self.w13_weight_triton_tensor = w13_weight
            self.w2_weight_triton_tensor = w2_weight
            del layer.w13_weight
            del layer.w2_weight
        else:
            from triton_kernels.numerics_details.mxfp import upcast_from_mxfp

            w13_weight = upcast_from_mxfp(
                layer.w13_weight, layer.w13_weight_scale, dtype=torch.bfloat16, axis=-1
            )
            w2_weight = upcast_from_mxfp(
                layer.w2_weight, layer.w2_weight_scale, dtype=torch.bfloat16, axis=-1
            )
            del layer.w13_weight
            del layer.w2_weight
            del layer.w13_weight_scale
            del layer.w2_weight_scale
            layer.w13_weight = Parameter(w13_weight.data, requires_grad=False)
            layer.w2_weight = Parameter(w2_weight.data, requires_grad=False)
        torch.cuda.empty_cache()

    def _get_tile_tokens_dim(self, x: torch.Tensor, top_k: int):
        # Number of tokens in the input tensor.
        num_tokens = x.shape[0]
        # Factor to account for the imbalance of the experts.
        # factor equals to the
        # max_real_num_tokens_per_expert / perfect_num_tokens_per_expert
        # - 1.0 means perfect expert distribution.
        # - > 1.0 means some experts have more
        #     tokens than the perfect distribution.
        # - < 1.0 does not make sense.
        imbalance_factor = 1.3
        # Calculate the number of tokens per expert
        # assuming perfect distribution.
        num_tokens_per_expert = (num_tokens * top_k) // self.num_experts
        # Apply the imbalance factor.
        num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
        # And pad the number to the next power of 2.
        tile_tokens_dim = next_power_of_2(num_tokens_per_expert)
        # Cap to 8-64 tokens per CTA tile
        # as it's the range supported by the kernel.
        tile_tokens_dim = min(max(tile_tokens_dim, 8), 64)

        return tile_tokens_dim

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_output: TopKOutput,
        moe_runner_config: MoeRunnerConfig,
    ) -> torch.Tensor:

        from sglang.srt.layers.moe.topk import TopKOutputChecker

        if self.use_flashinfer:
            # When bf16 mode is enabled, we don't need to quantize the input,
            # TRT-LLM automatically handles quantization in the kernel implementation and pipelines it with GEMM operations,
            # which can theoretically improve performance
            if self.flashinfer_mxfp4_moe_precision == "bf16":
                assert x.dtype == torch.bfloat16
                x_quant = x
                x_scale = None

                # May be fused later if this code branch is frequently needed
                origin_hidden_states_dim = x_quant.shape[-1]
                if self.hidden_size != origin_hidden_states_dim:
                    x_quant = torch.nn.functional.pad(
                        x_quant,
                        (0, self.hidden_size - origin_hidden_states_dim),
                        mode="constant",
                        value=0.0,
                    )
            elif self.flashinfer_mxfp4_moe_precision == "default":
                swizzle = (
                    self.enable_flashinfer_cutlass_moe
                    and not should_use_flashinfer_cutlass_moe_fp4_allgather()
                )
                if x.shape[0] == 0:
                    x_quant = torch.zeros(
                        0, self.hidden_size, dtype=torch.uint8, device=x.device
                    )
                    x_scale = torch.zeros(
                        0,
                        self.hidden_size // self.scaling_vector_size,
                        dtype=torch.uint8,
                        device=x.device,
                    )
                else:
                    x_quant, x_scale = mxfp8_quantize(
                        x,
                        swizzle,
                        alignment=self.weight_alignment,
                    )
                x_scale = x_scale.view(torch.float8_e4m3fn).reshape(-1)
            else:
                raise NotImplementedError

            assert x_quant.shape[-1] == self.hidden_size

        if self.enable_flashinfer_cutlass_moe:
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
            output_dtype = x.dtype

            if should_use_flashinfer_cutlass_moe_fp4_allgather():
                # Quantize before comm, swizzle after.
                x_quant = x_quant.view(torch.uint8)
                x_scale = x_scale.view(torch.uint8).reshape(
                    x.shape[0], self.hidden_size // self.scaling_vector_size
                )
                topk_weights, topk_ids, x_quant, x_scale = get_tp_group().all_gatherv(
                    [topk_weights, topk_ids, x_quant, x_scale],
                    sizes=get_dp_global_num_tokens(),
                )
                x_quant = x_quant.view(torch.float8_e4m3fn)
                x_scale = block_scale_interleave(x_scale)

            output = flashinfer_cutlass_fused_moe(
                input=x_quant,
                token_selected_experts=topk_ids.to(torch.int),
                token_final_scales=topk_weights,
                fc1_expert_weights=layer.w13_weight,
                fc1_expert_biases=layer.w13_weight_bias,
                fc2_expert_weights=layer.w2_weight,
                fc2_expert_biases=layer.w2_weight_bias,
                output_dtype=output_dtype,
                input_sf=x_scale,
                swiglu_alpha=layer.gemm1_alpha,
                swiglu_beta=layer.gemm1_beta,
                swiglu_limit=layer.gemm1_clamp_limit,
                quant_scales=[
                    layer.w13_weight_scale,
                    layer.fake_input_scale,
                    layer.w2_weight_scale,
                    layer.fake_input_scale,
                ],
                ep_size=layer.moe_ep_size,
                ep_rank=layer.moe_ep_rank,
                tp_size=layer.moe_tp_size,
                tp_rank=layer.moe_tp_rank,
                tune_max_num_tokens=next_power_of_2(x.shape[0]),
                use_mxfp8_act_scaling=True,
            )[0]
            if should_use_flashinfer_cutlass_moe_fp4_allgather():
                global_output = output[:, : self.initial_hidden_size].contiguous()
                local_output = get_local_dp_buffer()
                get_tp_group().reduce_scatterv(
                    global_output, output=local_output, sizes=get_dp_global_num_tokens()
                )
            return local_output

        if self.use_flashinfer:
            assert TopKOutputChecker.format_is_bypassed(topk_output)

            top_k = topk_output.topk_config.top_k
            router_logits = topk_output.router_logits

            trtllm_gen_output = trtllm_fp4_block_scale_moe(
                router_logits.to(torch.bfloat16),
                None,  # routing_bias
                x_quant,
                x_scale,
                layer.w13_weight,  # uint8 (e2m1 x 2)
                layer.w13_weight_scale,  # uint8 (e4m3 x 2)
                layer.w13_weight_bias,  # fp32 per expert per channel
                layer.gemm1_alpha,  # fp32 per expert
                layer.gemm1_beta,  # fp32 per expert
                layer.gemm1_clamp_limit,  # fp32 per expert
                layer.w2_weight,  # uint8 (e2m1 x 2)
                layer.w2_weight_scale,  # ue8m0
                layer.w2_weight_bias,  # fp32 per expert per channel
                None,  # output1_scale_scalar
                None,  # output1_scale_gate_scalar
                None,  # output2_scale_scalar
                layer.num_experts,
                top_k,
                None,  # n_group      # TODO: support n_group
                None,  # topk_group   # TODO: support topk_group
                self.intermediate_size,  # padded to multiple of 256
                layer.moe_ep_rank * layer.num_local_experts,  # local_expert_offset
                layer.num_local_experts,  # local num experts
                None,
                self._get_tile_tokens_dim(x, top_k),
                1,  # routing_method_type, renormalize
                True,  # do finalize
            )[0]
            return trtllm_gen_output

        if self.use_triton_kernels:
            assert (
                layer.moe_ep_size == 1
            ), "Expert parallel is not supported when using triton kernels"
            if self.with_bias:
                return self.triton_kernel_moe_with_bias_forward(
                    hidden_states=x,
                    w1=self.w13_weight_triton_tensor,
                    w1_pcg=self.w13_precision_config,
                    w2=self.w2_weight_triton_tensor,
                    w2_pcg=self.w2_precision_config,
                    b1=layer.w13_weight_bias,
                    b2=layer.w2_weight_bias,
                    topk_output=topk_output,
                    moe_runner_config=moe_runner_config,
                )
            else:
                return self.triton_kernel_moe_forward(
                    hidden_states=x,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    topk_output=topk_output,
                    moe_runner_config=moe_runner_config,
                )
        else:
            from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts

            return fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_output=topk_output,
                moe_runner_config=moe_runner_config,
                b1=layer.w13_weight_bias,
                b2=layer.w2_weight_bias,
            )


class Mxfp4DynamicQuantMoEMethod(FusedMoEMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Allocate 2 scales for w1 and w3 respectively.
        # They will be combined to a single scale after weight loading.
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )

        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def mxfp4_quantize(self, w):
        w_shape = w.shape
        w_need_reshape = True if w.dim() != 2 else False

        if w_need_reshape:
            w_last_dim_size = w_shape[-1]
            w = w.view(-1, w_last_dim_size)

        w, mx_scales = dynamic_mxfp4_quant(w)

        if w_need_reshape:
            w_new_shape = w_shape[:-1] + (w.shape[-1],)
            w = w.view(w_new_shape)

        mx_scales = e8m0_shuffle(mx_scales)

        return w, mx_scales

    def process_weights_after_loading(self, layer: Module) -> None:
        w13, w13_mx_scales = self.mxfp4_quantize(layer.w13_weight.data)
        w2, w2_mx_scales = self.mxfp4_quantize(layer.w2_weight.data)

        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_mx_scales, requires_grad=False)

        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_mx_scales, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_output: TopKOutput,
        moe_runner_config: MoeRunnerConfig,
    ) -> torch.Tensor:
        topk_weights, topk_ids, _ = topk_output

        return fused_moe(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            quant_type=QuantType.per_1x32,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            activation=(
                ActivationType.Silu
                if moe_runner_config.activation == "silu"
                else ActivationType.Gelu
            ),
            doweight_stage1=False,
        )
