# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.utils import (
    direct_register_custom_op,
    get_bool_env_var,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    next_power_of_2,
    round_up,
    set_weight_attrs,
)

has_triton_kernels = importlib.util.find_spec("triton_kernels") is not None


if is_flashinfer_available():
    # from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer import (
        mxfp8_quantize,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
        trtllm_fp4_block_scale_moe,
    )

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput

OCP_MX_BLOCK_SIZE = 32


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
    if is_cuda() and torch.cuda.get_device_capability()[0] == 10:
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


try:
    direct_register_custom_op(
        op_name="dequant_mxfp4",
        op_func=_dequant_mxfp4,
        mutates_args=[],
        fake_impl=_dequant_mxfp4_fake,
    )
    dequant_mxfp4 = torch.ops.sglang.dequant_mxfp4
except AttributeError as error:
    raise error

try:
    direct_register_custom_op(
        op_name="quant_dequant_mxfp4",
        op_func=_quant_dequant_mxfp4,
        mutates_args=[],
        fake_impl=_quant_dequant_mxfp4_fake,
    )
    quant_dequant_mxfp4 = torch.ops.sglang.quant_dequant_mxfp4
except AttributeError as error:
    raise error


class Mxfp4Config(QuantizationConfig):

    def __init__(self, ignored_layers: Optional[list[str]] = None):
        super().__init__()
        self.ignored_layers = ignored_layers

    @classmethod
    def from_config(cls, config):
        return cls()

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
        elif isinstance(layer, FusedMoE):
            use_flashinfer = global_server_args_dict.get("enable_flashinfer_mxfp4_moe", False)
            return Mxfp4MoEMethod(
                use_triton_kernels=True, with_bias=True, use_flashinfer=use_flashinfer
            )
        else:
            raise NotImplementedError("Mxfp4 attention layer is not implemented")
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Mxfp4MoEMethod(FusedMoEMethodBase):

    def __init__(
        self,
        use_triton_kernels: bool = True,
        with_bias: bool = True,
        use_flashinfer: bool = False,
    ):
        super().__init__()
        self.topk_indices_dtype = None
        self.use_triton_kernels = use_triton_kernels
        self.with_bias = with_bias
        self.use_flashinfer = use_flashinfer

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

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # print(f"hi {self=} create_weights {layer=}")
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8
        mxfp4_block = 32

        # pad the intermediate size to be a multiple of 2 * mxfp4_block
        # for to hold non-uniform sharded tensor as well as swizzling
        if self.use_flashinfer:
            intermediate_size_per_partition_after_pad = round_up(intermediate_size, 256)
            hidden_size = round_up(hidden_size, 256)
        elif is_hip():
            intermediate_size_per_partition_after_pad = round_up(intermediate_size, 128)
        else:
            intermediate_size_per_partition_after_pad = round_up(intermediate_size, 64)

        self.intermediate_size = intermediate_size_per_partition_after_pad

        self.hidden_size = hidden_size
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w13_weight_bias = torch.nn.Parameter(
            torch.zeros(
                num_experts,
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
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w2_weight_bias = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_bias", w2_weight_bias)
        set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer):
        if self.use_flashinfer:
            logger.info(
                "Shuffling MoE weights for FlashInfer, it might take a while..."
            )
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
            sf_block_size = 32  # mxfp4 block size

            assert (
                layer.w13_weight.dim() == 3
                and layer.w13_weight.shape[0] == self.num_experts
                and layer.w13_weight.shape[1] == self.intermediate_size * 2
                and layer.w13_weight.shape[2] == self.hidden_size // 2
            )
            assert (
                layer.w13_weight_scale.dim() == 3
                and layer.w13_weight_scale.shape[0] == self.num_experts
                and layer.w13_weight_scale.shape[1] == self.intermediate_size * 2
                and layer.w13_weight_scale.shape[2] == self.hidden_size // sf_block_size
            )
            assert (
                layer.w2_weight.dim() == 3
                and layer.w2_weight.shape[0] == self.num_experts
                and layer.w2_weight.shape[1] == self.hidden_size
                and layer.w2_weight.shape[2] == self.intermediate_size // 2
            )
            assert (
                layer.w2_weight_scale.dim() == 3
                and layer.w2_weight_scale.shape[1] == self.hidden_size
                and layer.w2_weight_scale.shape[2]
                == self.intermediate_size // sf_block_size
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
            w13_bias = layer.w13_weight_bias.data.to(torch.float32)
            w2_bias = layer.w2_weight_bias.data.to(torch.float32)

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
            for i in range(self.num_experts):
                gemm1_weights_mxfp4_shuffled.append(
                    shuffle_matrix_a(w13_weight[i].view(torch.uint8), epilogue_tile_m)
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
                    shuffle_matrix_a(w2_weight[i].view(torch.uint8), epilogue_tile_m)
                )
                gemm2_scales_mxfp4_shuffled.append(
                    shuffle_matrix_sf_a(
                        w2_weight_scale[i].view(torch.uint8), epilogue_tile_m
                    )
                )
                gemm2_bias_shuffled.append(
                    shuffle_matrix_a(w2_bias[i].clone().reshape(-1, 1), epilogue_tile_m)
                )

            w13_weight = torch.stack(gemm1_weights_mxfp4_shuffled)
            w13_weight_scale = (
                torch.stack(gemm1_scales_mxfp4_shuffled)
                .reshape(
                    self.num_experts,
                    2 * self.intermediate_size,
                    self.hidden_size // sf_block_size,
                )
                .view(torch.float8_e4m3fn)
            )

            w2_weight = torch.stack(gemm2_weights_mxfp4_shuffled)
            w2_weight_scale = (
                torch.stack(gemm2_scales_mxfp4_shuffled)
                .reshape(
                    self.num_experts,
                    self.hidden_size,
                    self.intermediate_size // sf_block_size,
                )
                .view(torch.float8_e4m3fn)
            )

            layer.w13_weight = Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = Parameter(w13_weight_scale, requires_grad=False)
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

        # need to delete the original weights to save memory on single GPU
        del layer.w13_weight
        del layer.w2_weight
        layer.w13_weight = None
        layer.w2_weight = None
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
        *,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
        activation_alpha: Optional[float] = None,
        swiglu_limit: Optional[float] = None,
    ) -> torch.Tensor:
        if self.use_flashinfer:
            # Based on profiling results, we need to quantize x to mxfp8 here to achieve better performance
            x_quant, x_scale = mxfp8_quantize(x, False)  # to mxfp8
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(-1)

            top_k, router_logits = topk_output

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
                self.num_experts,
                top_k,
                None,  # n_group
                None,  # topk_group
                self.intermediate_size,  # padded to multiple of 256
                0,  # local_expert_offset
                self.num_experts,  # local num experts
                None,
                self._get_tile_tokens_dim(x, top_k),
                1,  # routing_method_type, renormalize
                True,  # do finalize
            )[0]
            return trtllm_gen_output

        if self.use_triton_kernels:
            if self.with_bias:
                # TODO why we do not put weights on layer?
                assert layer.w13_weight is None
                assert layer.w2_weight is None
                return self.triton_kernel_moe_with_bias_forward(
                    hidden_states=x,
                    w1=self.w13_weight_triton_tensor,
                    w1_pcg=self.w13_precision_config,
                    w2=self.w2_weight_triton_tensor,
                    w2_pcg=self.w2_precision_config,
                    b1=layer.w13_weight_bias,
                    b2=layer.w2_weight_bias,
                    topk_output=topk_output,
                    activation=activation,
                    activation_alpha=activation_alpha,
                    swiglu_limit=swiglu_limit,
                )
            else:
                return self.triton_kernel_moe_forward(
                    hidden_states=x,
                    w1=layer.w13_weight,
                    w2=layer.w2_weight,
                    topk_output=topk_output,
                )
        else:
            raise NotImplementedError()
