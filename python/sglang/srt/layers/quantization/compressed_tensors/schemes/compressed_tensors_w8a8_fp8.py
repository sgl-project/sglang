# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter

from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.utils import requantize_with_max_scale

__all__ = ["CompressedTensorsW8A8Fp8"]


class CompressedTensorsW8A8Fp8(CompressedTensorsScheme):

    def __init__(self, strategy: str, is_static_input_scheme: bool):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer) -> None:
        # If per tensor, when we have a fused module (e.g. QKV) with per
        # tensor scales (thus N scales being passed to the kernel),
        # requantize so we can always run per tensor

        def _process_input_scale_after_loading(layer):
            if (
                self.strategy == QuantizationStrategy.TENSOR
                or self.strategy == QuantizationStrategy.CHANNEL
            ):
                if is_fp8_fnuz():
                    input_scale = getattr(layer, "input_scale", None)

                    if input_scale is not None:
                        input_scale = input_scale * 2.0
                        layer.input_scale = Parameter(input_scale, requires_grad=False)

            # INPUT SCALE
            if self.is_static_input_scheme and hasattr(layer, "input_scale"):
                layer.input_scale = Parameter(
                    layer.input_scale.max(), requires_grad=False
                )
            else:
                layer.input_scale = None

            return

        def _process_weights_after_loading(weight, weight_scale, logical_widths):
            rectified_weight = None
            rectified_weight_scale = None

            if self.strategy == QuantizationStrategy.TENSOR:
                max_w_scale, rectified_weight = requantize_with_max_scale(
                    weight=weight,
                    weight_scale=weight_scale,
                    logical_widths=logical_widths,
                )

                if is_fp8_fnuz():
                    # input_scale = getattr(layer, "input_scale", None)

                    # NOTE (yiakwy) : for fused kv parameter, this function is exclusive to ROCm to load NV fp8 data correctly
                    # but it does not make sense to double (2*input_scale) a shared input scale twice for different weight shards
                    rectified_weight, max_w_scale, input_scale = (
                        normalize_e4m3fn_to_e4m3fnuz(
                            weight=rectified_weight,
                            weight_scale=max_w_scale,
                            input_scale=None,
                        )
                    )

                rectified_weight = Parameter(rectified_weight.t(), requires_grad=False)
                rectified_weight_scale = Parameter(max_w_scale, requires_grad=False)

            # If channelwise, scales are already lined up, so just transpose.
            elif self.strategy == QuantizationStrategy.CHANNEL:
                if is_fp8_fnuz():
                    # input_scale = getattr(layer, "input_scale", None)

                    # NOTE (yiakwy) : the same issue as above
                    rectified_weight, rectified_weight_scale, input_scale = (
                        normalize_e4m3fn_to_e4m3fnuz(
                            weight=weight,
                            weight_scale=weight_scale,
                            input_scale=None,
                        )
                    )
                else:
                    rectified_weight_scale = weight_scale.data

                rectified_weight = Parameter(rectified_weight.t(), requires_grad=False)
                # required by torch.compile to be torch.nn.Parameter
                rectified_weight_scale = Parameter(
                    rectified_weight_scale, requires_grad=False
                )

            else:
                raise ValueError(f"Unknown quantization strategy {self.strategy}")

            return rectified_weight, rectified_weight_scale

        if "fused_qkv_a_proj_with_mqa" in layer.prefix:

            # See DeepSeek V2
            assert layer.fused_parameters == 2

            q_a_proj_w, kv_a_proj_with_mqa_w = layer.weight.split(
                layer.fused_shapes, dim=0
            )
            q_a_proj_w_scale, kv_a_proj_with_mqa_w_scale = layer.weight_scale.split(
                [1, 1], dim=0
            )

            rectified_q_a_proj_w_t, rectified_q_a_proj_w_t_scale = (
                _process_weights_after_loading(
                    q_a_proj_w, q_a_proj_w_scale, [layer.fused_shapes[0]]
                )
            )
            rectified_kv_a_proj_with_mqa_w_t, rectified_kv_a_proj_with_mqa_w_t_scale = (
                _process_weights_after_loading(
                    kv_a_proj_with_mqa_w,
                    kv_a_proj_with_mqa_w_scale,
                    [layer.fused_shapes[1]],
                )
            )

            # assigned merged weights back
            merged_weight = torch.cat(
                [rectified_q_a_proj_w_t.data, rectified_kv_a_proj_with_mqa_w_t.data],
                dim=1,
            )

            merged_weight_scale = torch.cat(
                [
                    rectified_q_a_proj_w_t_scale.data.unsqueeze(0),
                    rectified_kv_a_proj_with_mqa_w_t_scale.data.unsqueeze(0),
                ],
            )

            layer.weight = Parameter(merged_weight, requires_grad=False)
            layer.weight_scale = Parameter(merged_weight_scale, requires_grad=False)
        else:
            rectified_weight, rectified_weight_scale = _process_weights_after_loading(
                layer.weight, layer.weight_scale, layer.logical_widths
            )
            layer.weight = rectified_weight
            layer.weight_scale = rectified_weight_scale

        _process_input_scale_after_loading(layer)

        return

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: List[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        # TODO: update create_xxx_parameter functions to return
        # the newly added parameters

        if hasattr(layer, "fused_parameters") and layer.fused_parameters > 0:
            fused_parameters = layer.fused_parameters
        else:
            fused_parameters = 1

        if self.strategy == QuantizationStrategy.CHANNEL:
            if fused_parameters > 1:
                raise NotImplemented(
                    "Support for channel-wise weight scaling for fused parameters is not supported yet. Stay tuned!"
                )

            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty(
                    (sum(output_partition_sizes), 1),
                    dtype=torch.float32,
                ),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(
                    # TODO (yiakwy) : impl logics inside PerTensorScaleParameter, i.e. each of fused prameteres has (fused_parameters,) tensor-wise weight scale parameters
                    len(output_partition_sizes) * fused_parameters,
                    dtype=torch.float32,
                ),
                weight_loader=weight_loader,
            )

        # min requirement for fp8 kernels
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(
                    len(output_partition_sizes) * fused_parameters, dtype=torch.float32
                ),
                weight_loader=weight_loader,
            )
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # NOTE (fallback to non-fused solution) : concat([input * w[:shard1] * w_shard1_scale, input * w[shart1:shart1+shart2] * w_shard2_scale], dim=1)
        # Will write kernel to improve it later

        if hasattr(layer, "fused_parameters") and layer.fused_parameters > 0:
            if "fused_qkv_a_proj_with_mqa" in layer.prefix:
                assert layer.fused_parameters == 2

                q_a_proj_w, kv_a_proj_with_mqa_w = layer.weight.split(
                    layer.fused_shapes, dim=1
                )
                q_a_proj_w_scale, kv_a_proj_with_mqa_w_scale = layer.weight_scale.split(
                    [1, 1], dim=0
                )

                partial_out_q = apply_fp8_linear(
                    input=x,
                    weight=Parameter(q_a_proj_w, requires_grad=False),
                    weight_scale=Parameter(q_a_proj_w_scale, requires_grad=False),
                    input_scale=layer.input_scale,
                    bias=bias,
                    use_per_token_if_dynamic=True,
                    compressed_tensor_quant=True,
                )
                partial_out_kv_latent_cache = apply_fp8_linear(
                    input=x,
                    weight=Parameter(kv_a_proj_with_mqa_w, requires_grad=False),
                    weight_scale=Parameter(
                        kv_a_proj_with_mqa_w_scale, requires_grad=False
                    ),
                    input_scale=layer.input_scale,
                    bias=bias,
                    use_per_token_if_dynamic=True,
                    compressed_tensor_quant=True,
                )

                return torch.concat([partial_out_q, partial_out_kv_latent_cache], dim=1)
            else:
                raise Exception(
                    f"fused_parameters for this type of layer#{layer.prefix}: {type(layer)} is not supported yet."
                )
        else:
            return apply_fp8_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
                use_per_token_if_dynamic=True,
                compressed_tensor_quant=True,
            )
