# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional

import aiter
import torch
import torch.nn.functional as F
from aiter.ops.gemm_op_a4w4 import gemm_a4w4
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.gemm_afp4wfp4_pre_quant_atomic import gemm_afp4wfp4_pre_quant
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
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any],
        online: bool,
        layer_name: Optional[str] = None,
    ):
        """
        Initializes the quantization scheme for the layer.

        Args:
            weight_quant_spec (dict[str, Any]): Specification for weight quantization.
            input_quant_spec (dict[str, Any]): Specification for input quantization.
            online (bool): Whether to use online quantization, which loads full/half precision weight and quantize post loading.
            layer_name (Optional[str], optional): Name of the layer for debugging purpose. Defaults to None.
        """
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec
        self.online = online
        self.layer_name = layer_name

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def mxfp4_quantize(w):
        w_shape = w.shape
        w_need_reshape = w.dim() != 2

        if w_need_reshape:
            w_last_dim_size = w_shape[-1]
            w = w.view(-1, w_last_dim_size)

        w, mx_scales = dynamic_mxfp4_quant(w)

        if w_need_reshape:
            w_new_shape = w_shape[:-1] + (w.shape[-1],)
            w = w.view(w_new_shape)

        return w, mx_scales

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.online:
            # NOTE: This step happens after `post_load_weights` for deepseek models.
            # So it is safe for kv_b_proj quantization of w_kc and w_vc in `quark_post_load_weights`
            # since kv_b_proj.weight at that point is still bfloat16.
            assert layer.weight.dtype in {torch.bfloat16, torch.float16}
            w, mx_scales = self.mxfp4_quantize(layer.weight.data)
            weight = PackedvLLMParameter(
                data=w,
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=2,
                weight_loader=layer.weight.weight_loader,
            )
            layer.register_parameter("weight", weight)
            layer.weight_scale.data = mx_scales

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
        if self.online:
            # Create and load bfloat16 weight
            # quantize to mxfp4 during `process_weights_after_loading`
            pack_factor = 1
            dtype = torch.bfloat16
        else:
            pack_factor = 2
            dtype = torch.uint8

        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // pack_factor,
                dtype=dtype,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=pack_factor,
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
