# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 dense-dequant fallback for compressed-tensors weight-only int4
checkpoints whose ``format`` is not ``pack_quantized``. Unblocks loading by
unpacking the 4-bit integer weights, dequantizing with per-group/channel
scales (and optional zero points) into a dense float tensor and running
``torch.nn.functional.linear`` (no fused 2:4-sparse kernel; lossless dequant).

Only integer (``QuantizationType.INT``) 4-bit weights are handled; float4 /
NVFP4 stay a loud ``NotImplementedError`` at dispatch (the uint4b8 bias is
only valid for int quantization, not microscaling FP4).
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from compressed_tensors.quantization import QuantizationType

from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.utils import unpack_cols

__all__ = ["CompressedTensorsW4A16Sparse24"]


class CompressedTensorsW4A16Sparse24(CompressedTensorsLinearScheme):
    """Dense-dequant fallback for weight-only int4 (W4A16) compressed-tensors
    checkpoints outside the Marlin ``pack_quantized`` path."""

    def __init__(
        self,
        num_bits: int,
        strategy: str,
        quant_type: QuantizationType,
        group_size: Optional[int] = None,
        symmetric: bool = True,
    ):
        assert num_bits == 4, "CompressedTensorsW4A16Sparse24 only supports num_bits=4"
        assert strategy in (
            "channel",
            "group",
        ), f"Unsupported strategy: {strategy!r} (expected 'channel' or 'group')"
        # uint4b8 dequant is only valid for int weights; reject float4/NVFP4.
        assert quant_type == QuantizationType.INT, (
            f"CompressedTensorsW4A16Sparse24 only supports INT weight "
            f"quantization, got type={quant_type!r}"
        )
        self.num_bits = num_bits
        self.strategy = strategy
        self.quant_type = quant_type
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.pack_factor = 32 // num_bits
        # Symmetric int4: unsigned [0,15] + implicit bias (uint4b8).
        self._bias = 2 ** (num_bits - 1)

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        self._params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)
        # group_size == -1 = channelwise; scales partition-aware under row-parallel.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = input_size != input_size_per_partition
        scales_and_zp_size = input_size // group_size
        if row_parallel and self.group_size != -1:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self.pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
        )
        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition, scales_and_zp_size, dtype=params_dtype
            ),
        }
        if self.group_size == -1:
            weight_scale = ChannelQuantScaleParameter(output_dim=0, **weight_scale_args)
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args
            )
        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )
        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)
        if not self.symmetric:
            # Asymmetric: packed int4 zero point [out // pack_factor, n_groups], packed_dim=0 (WNA16 axis).
            weight_zero_point = PackedvLLMParameter(
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
                packed_factor=self.pack_factor,
                packed_dim=0,
                data=torch.zeros(
                    output_size_per_partition // self.pack_factor,
                    scales_and_zp_size,
                    dtype=torch.int32,
                ),
            )
            layer.register_parameter("weight_zero_point", weight_zero_point)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_packed = layer.weight_packed
        weight_scale = layer.weight_scale
        # weight_packed is [out, in // pack_factor]; unpack to [out, in].
        out_features = weight_packed.shape[0]
        in_features = weight_packed.shape[1] * self.pack_factor
        q = unpack_cols(
            weight_packed,
            num_bits=self.num_bits,
            size_k=out_features,
            size_n=in_features,
        ).to(torch.float32)
        group_size = self.group_size if self.group_size != -1 else in_features
        scales_exp = weight_scale.to(torch.float32).repeat_interleave(group_size, dim=1)
        if self.symmetric:
            w = (q - self._bias) * scales_exp
        else:
            weight_zero_point = layer.weight_zero_point
            # zero point packed [out // pack_factor, n_groups] (packed_dim=0);
            # unpack to [n_groups, out], transpose to [out, n_groups], expand.
            n_groups = scales_exp.shape[1] // group_size
            zp = unpack_cols(
                weight_zero_point.t().contiguous(),
                num_bits=self.num_bits,
                size_k=n_groups,
                size_n=out_features,
            )
            zp = zp.t().to(torch.float32).repeat_interleave(group_size, dim=1)
            w = (q - zp) * scales_exp
        target_dtype = getattr(self, "_params_dtype", torch.float16)
        layer.weight = torch.nn.Parameter(w.to(target_dtype), requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)
