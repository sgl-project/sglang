# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch.nn import Parameter

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    apply_fp8_ptpc_linear,
    deepgemm_w8a8_block_fp8_linear_with_fallback,
    dispatch_w8a8_block_fp8_linear,
    normalize_e4m3fn_to_e4m3fnuz,
    requant_block_scale_ue8m0_for_deepgemm,
    validate_fp8_block_shape,
)
from sglang.srt.layers.quantization.utils import requantize_with_max_scale
from sglang.srt.utils import ceil_align, get_bool_env_var, is_hip

__all__ = ["CompressedTensorsW8A8Fp8", "FP8_ALIGNMENT", "pad_to_alignment"]

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
    from aiter.ops.shuffle import shuffle_weight

# torch._scaled_mm requires both matrix dimensions divisible by 16.
# Models where intermediate_size / tp_size is not 16-aligned
# (e.g. 10944 / tp=8 = 1368) fail CUDA Graph capture without padding.
FP8_ALIGNMENT = 16


def pad_to_alignment(
    tensor: torch.Tensor, dim: int, alignment: int = FP8_ALIGNMENT
) -> torch.Tensor:
    """Zero-pad tensor along dim to the next multiple of alignment.

    Returns the original tensor unchanged when already aligned.
    """
    size = tensor.shape[dim]
    pad = ceil_align(size, alignment) - size
    if pad == 0:
        return tensor
    # F.pad takes padding in reverse-dim order, two values per dim (left, right).
    # Index 2*(ndim-1-dim)+1 selects the right-side padding for `dim`.
    pad_arg = [0] * (2 * tensor.ndim)
    pad_arg[2 * (tensor.ndim - 1 - dim) + 1] = pad
    return F.pad(tensor, pad_arg)


def _pad_weight_to_alignment(
    weight: torch.Tensor, weight_scale: Optional[torch.Tensor]
) -> tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
    """Pad weight (N, K) so both dims are multiples of FP8_ALIGNMENT.

    Returns (weight_t, weight_scale, orig_N, orig_K) where weight_t is transposed.
    weight_scale is padded along dim-0 when it is per-channel (shape [N, 1]
    or [N,]).
    """
    orig_n = weight.shape[0]
    orig_k = weight.shape[1]
    weight = pad_to_alignment(weight, dim=0)
    weight = pad_to_alignment(weight, dim=1)
    if weight_scale is not None and weight.shape[0] > orig_n:
        if weight_scale.shape == (orig_n,):
            weight_scale = weight_scale.unsqueeze(1)
        if weight_scale.shape != (orig_n, 1):
            raise ValueError(
                f"Expected per-channel scale shape ({orig_n}, 1) or ({orig_n},), "
                f"got {tuple(weight_scale.shape)}"
            )
        weight_scale = pad_to_alignment(weight_scale, dim=0)
    return weight.t(), weight_scale, orig_n, orig_k


def _cache_padding_metadata(
    layer, weight_t: torch.Tensor, orig_n: int, orig_k: int
) -> None:
    """Cache padded shapes on the layer and pre-pad a copy of bias if present.

    Keeps apply_weights branch-light and CUDA-Graph-friendly: no per-forward
    shape math, no F.pad on bias, no F.pad on input when K is already aligned.

    Bias is stored under `_padded_bias` rather than overwriting `layer.bias`,
    because callers using `skip_bias_add=True` read `layer.bias` directly and
    would see the wrong shape if we mutated it in place.
    """
    # weight_t is (K_padded, N_padded).
    k_padded, n_padded = weight_t.shape
    layer._orig_output_dim = orig_n
    layer._pad_input_k = k_padded - orig_k
    layer._needs_output_slice = n_padded > orig_n

    bias = getattr(layer, "bias", None)
    if bias is not None and layer._needs_output_slice:
        layer._padded_bias = pad_to_alignment(bias.data, dim=0)
    else:
        layer._padded_bias = None


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

            weight_t, _, orig_n, orig_k = _pad_weight_to_alignment(weight, None)
            layer.weight = Parameter(weight_t, requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
            _cache_padding_metadata(layer, weight_t, orig_n, orig_k)

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
                # keep the weight as (N, K); (16, 16) is the aiter shuffle tile
                # size, unrelated to the torch._scaled_mm alignment requirement
                layer.weight = Parameter(
                    shuffle_weight(weight, (16, 16)), requires_grad=False
                )
                # required by torch.compile to be torch.nn.Parameter
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
            else:
                weight_t, weight_scale, orig_n, orig_k = _pad_weight_to_alignment(
                    weight, weight_scale
                )
                layer.weight = Parameter(weight_t, requires_grad=False)
                # required by torch.compile to be torch.nn.Parameter
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                _cache_padding_metadata(layer, weight_t, orig_n, orig_k)

        elif self.strategy == QuantizationStrategy.BLOCK:
            assert self.is_static_input_scheme is False
            if is_fp8_fnuz():
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.weight, weight_scale=layer.weight_scale
                )
                layer.weight = Parameter(weight.data, requires_grad=False)
                layer.weight_scale = Parameter(weight_scale.data, requires_grad=False)
                layer.weight_scale.format_ue8m0 = False
            else:
                layer.weight.requires_grad_(False)
                layer.weight_scale.requires_grad_(False)

            # On Blackwell, block-FP8 dispatches to DeepGEMM, which needs the
            # weight scales UE8M0-packed to match its UE8M0 activation scales.
            use_deepgemm_runner = (
                self.w8a8_block_fp8_linear
                is deepgemm_w8a8_block_fp8_linear_with_fallback
            )
            requant_block_scale_ue8m0_for_deepgemm(
                layer.weight,
                layer.weight_scale,
                self.weight_block_size,
                use_deepgemm_runner=use_deepgemm_runner,
                output_dtype=getattr(layer, "orig_dtype", None),
                weight_shape=layer.weight.shape,
            )

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
            return apply_fp8_ptpc_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
                use_per_token_if_dynamic=True,
                compressed_tensor_quant=True,
            )

        # weight is stored transposed as (K_padded, N_padded). All shape math
        # is precomputed in process_weights_after_loading and cached on `layer`
        # so this path is branch-light and CUDA-Graph-friendly.
        #
        # _pad_input_k is typically 0 for column-parallel layers (K == hidden_size
        # is naturally aligned) and non-zero for row-parallel layers where
        # K == intermediate_size/tp_size can be misaligned.
        if layer._pad_input_k:
            x = F.pad(x, (0, layer._pad_input_k))

        # Swap in the pre-padded bias when N was padded at load time.
        # `_padded_bias` is None when no padding was needed OR when the layer had
        # no bias; if the caller passed bias=None (skip_bias_add=True), keep None.
        if bias is not None and layer._padded_bias is not None:
            bias = layer._padded_bias

        output = apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            use_per_token_if_dynamic=True,
            compressed_tensor_quant=True,
        )

        if layer._needs_output_slice:
            output = output[..., : layer._orig_output_dim]
        return output
