# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional

import torch
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.gemm_afp4wfp4_pre_quant_atomic import gemm_afp4wfp4_pre_quant
from aiter.ops.triton.quant import dynamic_mxfp4_quant

from sglang.srt.layers.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.quark.schemes import QuarkScheme

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
        # This path does not have support for bias currently
        assert bias is None, "bias is not supported"

        three_d = False
        x_s = None
        y = None
        if isinstance(x, tuple):
            assert len(x) in [
                2,
                3,
            ], "For tuple input, only (x, x_s) or (x, x_s, y) formats are accepted"
            if len(x) == 2:
                x, x_s = x
            elif len(x) == 3:
                x, x_s, y = x

        use_fused_quant_gemm = (
            x_s is None and y is not None and layer.weight.shape[0] == y.shape[1]
        )

        if x.dim() == 3:
            three_d = True
            x = x.view(-1, x.shape[-1])
            output_shape = [*x.shape[:-1], layer.weight.shape[0]]

        # use_fused_quant_gemm = true, x_q is a bf16/fp16 num
        # x_s is not None = true, x_q is uint8 num
        if use_fused_quant_gemm or x_s is not None:
            x_q = x
        else:
            x_q, x_s = dynamic_mxfp4_quant(x)

        if y is None:
            y = torch.empty(
                x_q.shape[0],
                layer.weight.shape[0],
                device=x_q.device,
                dtype=self.out_dtype,
            )

        if use_fused_quant_gemm:
            gemm_afp4wfp4_pre_quant(x_q, layer.weight, layer.weight_scale, y.dtype, y)
            y = y.to(x.dtype)
        else:
            gemm_afp4wfp4(x_q, layer.weight, x_s, layer.weight_scale, self.out_dtype, y)

        if three_d:
            return y.view(*output_shape)

        return y
