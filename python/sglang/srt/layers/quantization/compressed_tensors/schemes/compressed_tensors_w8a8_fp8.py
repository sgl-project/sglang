# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch.nn import Parameter

from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    apply_fp8_ptpc_linear,
    dispatch_w8a8_block_fp8_linear,
    normalize_e4m3fn_to_e4m3fnuz,
    validate_fp8_block_shape,
)
from sglang.srt.layers.quantization.utils import requantize_with_max_scale
from sglang.srt.utils import get_bool_env_var, is_hip

__all__ = ["CompressedTensorsW8A8Fp8"]

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight


FP8_ALIGNMENT = 16

strategy_to_parameter_type = {
    QuantizationStrategy.BLOCK: BlockQuantScaleParameter,
    QuantizationStrategy.CHANNEL: ChannelQuantScaleParameter,
    QuantizationStrategy.TENSOR: PerTensorScaleParameter,
}


def _pad_weight_for_fp8(
    weight: torch.Tensor,
    weight_scale: Optional[torch.Tensor] = None,
):
    """
    Pad weight (out_dim, in_dim) so both dims are divisible by FP8_ALIGNMENT.

    Returns:
        weight_t: padded and transposed weight
        weight_scale: padded scale if needed
        orig_output_dim: original N dimension (for output slice)
    """
    orig_out, orig_in = weight.shape
    pad_in = (-orig_in) % FP8_ALIGNMENT
    pad_out = (-orig_out) % FP8_ALIGNMENT

    if pad_in > 0 or pad_out > 0:
        weight = F.pad(weight, (0, pad_in, 0, pad_out))
        if weight_scale is not None and pad_out > 0:
            if weight_scale.dim() == 2 and weight_scale.shape[0] == orig_out:
                weight_scale = F.pad(
                    weight_scale, (0, 0, 0, pad_out), value=1.0  # 0*scale=0 for padded rows
                )
    return weight.t(), weight_scale, orig_out


class CompressedTensorsW8A8Fp8(CompressedTensorsLinearScheme):
    def __init__(self, weight_quant: QuantizationArgs, is_static_input_scheme: bool):
        self.weight_quant = weight_quant
        self.strategy = self.weight_quant.strategy
        self.is_static_input_scheme = is_static_input_scheme
        self.weight_block_size = self.weight_quant.block_structure
        if self.weight_block_size is not None:
            self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.weight_block_size = None
        layer.orig_dtype = params_dtype

        if self.strategy == QuantizationStrategy.BLOCK:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            # Validate block quantization shapes
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

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
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
            weight_scale[:] = torch.finfo(torch.float32).min
        elif self.strategy == QuantizationStrategy.TENSOR:
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            weight_scale[:] = torch.finfo(torch.float32).min
        elif self.strategy == QuantizationStrategy.BLOCK:
            assert layer.weight_block_size is not None
            block_n, block_k = layer.weight_block_size[0], layer.weight_block_size[1]
            output_size_per_partition = sum(output_partition_sizes)
            weight_scale = BlockQuantScaleParameter(
                data=torch.empty(
                    (output_size_per_partition + block_n - 1) // block_n,
                    (input_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            weight_scale.format_ue8m0 = False
            weight_scale[:] = torch.finfo(torch.float32).min

        layer.register_parameter("weight_scale", weight_scale)
        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    def process_weights_after_loading(self, layer) -> None:
        if self.strategy == QuantizationStrategy.TENSOR:
            max_w_scale, weight = requantize_with_max_scale(
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                logical_widths=layer.logical_widths,
            )

            if is_fp8_fnuz():
                input_scale = getattr(layer, "input_scale", None)

                weight, max_w_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight, weight_scale=max_w_scale, input_scale=input_scale
                )
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale, requires_grad=False)

            weight_t, max_w_scale, orig_out = _pad_weight_for_fp8(weight, max_w_scale)
            layer.weight = Parameter(weight_t, requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
            layer.register_buffer("_orig_output_dim", torch.tensor(orig_out, dtype=torch.int))

        elif self.strategy == QuantizationStrategy.CHANNEL:
            weight = layer.weight

            if is_fp8_fnuz():
                input_scale = getattr(layer, "input_scale", None)

                weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight,
                    weight_scale=layer.weight_scale,
                    input_scale=input_scale,
                )
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale, requires_grad=False)
            else:
                weight_scale = layer.weight_scale.data

            if _use_aiter:
                # aiter kernel handles non-16-aligned shapes internally; weight/scale not padded.
                # Output dimension unchanged; register for uniform output slicing logic.
                orig_out = weight.shape[0]
                layer.weight = Parameter(
                    shuffle_weight(weight, (FP8_ALIGNMENT, FP8_ALIGNMENT)),
                    requires_grad=False,
                )
                layer.register_buffer("_orig_output_dim", torch.tensor(orig_out, dtype=torch.int))
            else:
                weight_t, weight_scale, orig_out = _pad_weight_for_fp8(
                    weight, weight_scale
                )
                layer.weight = Parameter(weight_t, requires_grad=False)
                layer.register_buffer("_orig_output_dim", torch.tensor(orig_out, dtype=torch.int))

            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        elif self.strategy == QuantizationStrategy.BLOCK:
            # Block kernel uses its own layout; no padding applied (assumes block-aligned dims)
            assert self.is_static_input_scheme is False
            weight = layer.weight
            weight_scale = layer.weight_scale

            if is_fp8_fnuz():
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight, weight_scale=weight_scale
                )
            layer.weight = Parameter(weight.detach(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale.detach(), requires_grad=False)
            layer.register_buffer("_orig_output_dim", torch.tensor(-1, dtype=torch.int))

        else:
            raise ValueError(f"Unknown quantization strategy {self.strategy}")

        # INPUT SCALE
        if self.is_static_input_scheme and hasattr(layer, "input_scale"):
            layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)
        else:
            layer.input_scale = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.weight_block_size is not None:
            return self.w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.weight_block_size,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
            )

        if _use_aiter and self.strategy == QuantizationStrategy.CHANNEL:
            output = apply_fp8_ptpc_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
                use_per_token_if_dynamic=True,
                compressed_tensor_quant=True,
            )
        else:
            expected_in = layer.weight.shape[0]
            pad = expected_in - x.shape[-1]
            if pad < 0:
                raise RuntimeError(
                    f"Input dim {x.shape[-1]} larger than padded weight dim {expected_in}"
                )
            if pad > 0:
                x = F.pad(x, (0, pad))
            output = apply_fp8_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
                use_per_token_if_dynamic=True,
                compressed_tensor_quant=True,
            )

        n = layer._orig_output_dim.item() if hasattr(layer, "_orig_output_dim") else -1
        if n >= 0:
            output = output[..., :n]
        return output
