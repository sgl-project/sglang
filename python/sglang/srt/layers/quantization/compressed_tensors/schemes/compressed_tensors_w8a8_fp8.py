# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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


# torch._scaled_mm requires both matrix dimensions to be divisible by 16.
# Models with intermediate_size not divisible by 16 after tensor-parallel
# sharding (e.g. 10944 / tp=8 = 1368) fail CUDA Graph capture without this.
FP8_ALIGNMENT = 16


def _pad_to_alignment(t: torch.Tensor, dim: int, alignment: int) -> torch.Tensor:
    """Pad tensor along *dim* so its size is a multiple of *alignment*.

    Returns the original tensor unchanged when already aligned.
    """
    size = t.shape[dim]
    remainder = size % alignment
    if remainder == 0:
        return t
    pad_size = alignment - remainder
    # F.pad takes a flat tuple starting from the last dimension:
    # (last_right, last_left, second_last_right, second_last_left, ...)
    ndim = t.dim()
    pad: list[int] = [0] * (2 * ndim)
    pad_idx = 2 * (ndim - 1 - dim) + 1  # right-pad index for `dim`
    pad[pad_idx] = pad_size
    return F.pad(t, pad)


strategy_to_parameter_type = {
    QuantizationStrategy.BLOCK: BlockQuantScaleParameter,
    QuantizationStrategy.CHANNEL: ChannelQuantScaleParameter,
    QuantizationStrategy.TENSOR: PerTensorScaleParameter,
}


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

            # Pad to FP8_ALIGNMENT so torch._scaled_mm succeeds in CUDA Graph.
            # Per-tensor scale is a scalar — no scale padding needed.
            orig_out = weight.shape[0]
            weight = _pad_to_alignment(weight, dim=0, alignment=FP8_ALIGNMENT)
            weight = _pad_to_alignment(weight, dim=1, alignment=FP8_ALIGNMENT)
            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
            layer.register_buffer(
                "_orig_output_dim", torch.tensor(orig_out), persistent=False
            )

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

            orig_out = weight.shape[0]

            if _use_aiter:
                # aiter kernel handles alignment internally; keep weight as (N, K).
                layer.weight = Parameter(
                    shuffle_weight(weight, (FP8_ALIGNMENT, FP8_ALIGNMENT)),
                    requires_grad=False,
                )
                # required by torch.compile to be torch.nn.Parameter
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            else:
                # Pad both dims so torch._scaled_mm succeeds in CUDA Graph.
                weight = _pad_to_alignment(weight, dim=0, alignment=FP8_ALIGNMENT)
                weight = _pad_to_alignment(weight, dim=1, alignment=FP8_ALIGNMENT)
                # Pad per-channel scale to match new output dim.
                weight_scale = _pad_to_alignment(
                    weight_scale, dim=0, alignment=FP8_ALIGNMENT
                )
                layer.weight = Parameter(weight.t(), requires_grad=False)
                # required by torch.compile to be torch.nn.Parameter
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                # Slice output back to original N after GEMM (padded rows are noise).
                layer.register_buffer(
                    "_orig_output_dim", torch.tensor(orig_out), persistent=False
                )

        elif self.strategy == QuantizationStrategy.BLOCK:
            assert self.is_static_input_scheme is False
            weight = layer.weight
            weight_scale = layer.weight_scale

            if is_fp8_fnuz():
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight, weight_scale=weight_scale
                )
            # Block kernels do not use torch._scaled_mm, so no alignment padding.
            layer.weight = Parameter(weight.detach(), requires_grad=False)
            layer.weight_scale = Parameter(weight_scale.detach(), requires_grad=False)

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
            # Block kernels handle alignment internally; no padding needed.
            return self.w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.weight_block_size,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
            )

        if _use_aiter and self.strategy == QuantizationStrategy.CHANNEL:
            # aiter keeps weight as (N, K) and handles alignment internally;
            # no input padding or output slicing needed.
            return apply_fp8_ptpc_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
                use_per_token_if_dynamic=True,
                compressed_tensor_quant=True,
            )

        # weight is stored transposed as (K_padded, N_padded).
        # Pad the input K-dim to match so torch._scaled_mm aligns correctly.
        weight_k_dim = layer.weight.shape[0]
        if x.shape[-1] < weight_k_dim:
            x = F.pad(x, (0, weight_k_dim - x.shape[-1]))
        elif x.shape[-1] > weight_k_dim:
            raise RuntimeError(
                f"Input last dim {x.shape[-1]} > padded weight K dim {weight_k_dim}."
                " This should never happen."
            )

        output = apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            use_per_token_if_dynamic=True,
            compressed_tensor_quant=True,
        )

        # Trim any output columns added by N-dim padding.
        orig_n = layer._orig_output_dim.item()
        return output[..., :orig_n]
