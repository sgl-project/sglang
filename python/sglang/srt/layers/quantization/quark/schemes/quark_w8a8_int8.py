# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional, Set

import torch

from vllm import _custom_ops as ops

from vllm.logger import init_logger
"""
from vllm.model_executor.layers.quantization.kernels.scaled_mm import (
    ScaledMMLinearLayerConfig, choose_scaled_mm_linear_kernel)
"""
from sglang.srt.layers.quantization.quark.schemes import QuarkScheme
from sglang.srt.layers.parameter  import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)

logger = init_logger(__name__)


class QuarkW8A8Int8(QuarkScheme):
    _kernel_backends_being_used: Set[str] = set()

    def __init__(self, qscheme: str, is_static_input_scheme: Optional[bool],
            input_symmetric: Optional[bool], online_rotation_method: Optional[Callable]):
        self.qscheme = qscheme
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric
        self.online_rotation_method = online_rotation_method
        #Lazy Import
        from sglang.srt.layers.quantization.quark.schemes import process_weights_after_loading as pwal
        self.pwal=pwal

    @classmethod
    def get_min_capability(cls) -> int:
        # turing and up
        return 75

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        #print("output_partition_sizes: ",sum(output_partition_sizes),len(output_partition_sizes))

        layer.logical_widths = output_partition_sizes
        self.logical_widths = output_partition_sizes

        """
        scaled_mm_linear_kernel_config = ScaledMMLinearLayerConfig(
            is_channelwise=(self.qscheme == "per_channel"),
            is_static_input_scheme=(self.is_static_input_scheme is True),
            input_symmetric=(self.input_symmetric is True))

        kernel_type = choose_scaled_mm_linear_kernel(
            scaled_mm_linear_kernel_config)

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for QuarkW8A8Int8", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)
        """
            
        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=torch.int8),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)

        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.qscheme == "per_channel":
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1),
                                 dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader)
        else:
            assert self.qscheme == "per_tensor"
            #print("per_tensor?",len(output_partition_sizes))
            weight_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                   weight_loader=weight_loader)
        layer.register_parameter("weight_scale", weight_scale)

        #print("before processing: ",layer.weight_scale.shape)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = BasevLLMParameter(data=torch.empty(
                1, dtype=torch.float32),
                                            weight_loader=weight_loader)
            layer.register_parameter("input_scale", input_scale)

            if not self.input_symmetric:
                # Note: quark stores the zp using the same dtype
                # as the weights
                # AZP loaded as int8 but used as int32
                input_zero_point = BasevLLMParameter(
                    data=torch.empty(1, dtype=torch.int8),
                    weight_loader=weight_loader)
                layer.register_parameter("input_zero_point", input_zero_point)

        """
        self.kernel = kernel_type(c=scaled_mm_linear_kernel_config,
                                  w_q_param_name="weight",
                                  w_s_param_name="weight_scale",
                                  i_s_param_name="input_scale",
                                  i_zp_param_name="input_zero_point",
                                  azp_adj_param_name="azp_adj")
        """
                                  
        if self.online_rotation_method:
            func_name,func_args=self.online_rotation_method
            self.online_rotation_method_callable=func_name(layer,*func_args)

    # Checkpoints are serialized in quark format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        print("before processing: ",layer.weight_scale.shape)
        print("before processing: ",layer.weight_scale)
        self.pwal(layer)
        #self.kernel.process_weights_after_loading(layer)
        #print("after processing: ",layer.weight_scale.shape)
        #print("after processing: ",layer.weight_scale)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
        bias: Optional[torch.Tensor]) -> torch.Tensor:

        #raise ValueError("wrong")

        #print("A, x : ",layer.weight.shape,x.shape)

        if self.online_rotation_method:

            x=self.online_rotation_method_callable(x)
            
        #return self.kernel.apply_weights(layer, x, bias)

        """begin - replace kernel.apply_weights"""

        #x_q=(x/layer.input_scale).to(torch.int8)
        #x_q_uq=x_q.bfloat16()

        #w_uq = (layer.weight.bfloat16().T).contiguous()

        #y=torch.nn.functional.linear(x_q_uq,w_uq)
        #print(y.shape)
        #print(layer.weight_scale.shape)
        #y=y*layer.weight_scale.squeeze()

        i_s=layer.input_scale
        #print(x.dtype)
        #print(i_s.dtype)
        x_q, x_s, x_zp = ops.scaled_int8_quant(x, i_s, None, symmetric=True)
        w_q=layer.weight
        w_s=layer.weight_scale
        return ops.cutlass_scaled_mm(x_q, w_q, scale_a=x_s, scale_b=w_s, out_dtype=x.dtype, bias=None)


        """end - replace kernel.apply_weights"""

        return y.bfloat16()