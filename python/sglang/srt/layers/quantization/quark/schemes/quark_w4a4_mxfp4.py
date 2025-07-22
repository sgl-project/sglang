# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional

import aiter
import torch
import torch.nn.functional as F
from aiter.ops.gemm_op_a4w4 import gemm_a4w4
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility import dtypes
from aiter.utility.fp4_utils import e8m0_shuffle

from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.quark.schemes import QuarkScheme
from sglang.srt.utils import get_bool_env_var

__all__ = ["QuarkW4A4MXFP4"]

OCP_MX_BLOCK_SIZE = 32


class QuarkW4A4MXFP4(QuarkScheme):

    def __init__(
        self, weight_quant_spec: dict[str, Any], input_quant_spec: dict[str, Any]
    ):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return

        # for aiter implement
        # wshuffle = shuffle_weight(layer.weight.data, layout=(16, 16))
        # w_scales_shuffle = e8m0_shuffle(layer.weight_scale.data).view(dtypes.fp8_e8m0)

        # layer.weight = torch.nn.Parameter(wshuffle,
        #                                  requires_grad=False)
        # layer.weight_scale = torch.nn.Parameter(w_scales_shuffle,
        #                                        requires_grad=False)

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        out_dtype = x.dtype
        # M = x.shape[0]
        # N = layer.weight.shape[0]

        # quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
        # x, x_scales_shuffle = quant_func(x, shuffle=True)

        # y = torch.zeros((M + 255) // 256 * 256, N, device=x.device, dtype=self.out_dtype)

        # out = gemm_a4w4(x, layer.weight.data, x_scales_shuffle, layer.weight_scale.data, y, bias=bias)

        # return out[:M]

        # triton implement
        x_q, x_s = dynamic_mxfp4_quant(x)
        y = torch.empty(
            x_q.shape[0], layer.weight.shape[0], device=x_q.device, dtype=out_dtype
        )

        out = gemm_afp4wfp4(x_q, layer.weight, x_s, layer.weight_scale, out_dtype, y)

        return out
