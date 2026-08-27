"""ModelSlim MXFP8 scheme for pre-quantized weight inference on Ascend NPU.

Loads weights pre-quantized by msmodelslim (float8_e4m3fn weights,
uint8 scales) and runs MXFP8 matmul at inference.
"""

from typing import List, Optional

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform

_is_npu = current_platform.is_npu()

if _is_npu:
    import torch_npu

from sglang.multimodal_gen.runtime.models.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme

MXFP8_BLOCK_SIZE = 32


class ModelSlimMXFP8Scheme(ModelSlimLinearScheme):

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)

        # msmodelslim exports weight as float8_e4m3fn, shape [out, in]
        weight = ModelWeightParameter(
            data=torch.empty(
                (output_size_per_partition, input_size_per_partition),
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # msmodelslim exports weight_scale as uint8, shape [out, in/32].
        # NOTE: This parameter is intentionally named "weight_scale" (not
        # "weight_scale_inv" as used in mxfp8_npu.py) because the weight loader
        # matches parameter names to checkpoint keys, and msmodelslim checkpoints
        # store this tensor under the key "<layer>.weight_scale".
        scale_dim = input_size_per_partition // MXFP8_BLOCK_SIZE
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition, scale_dim),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # weight is already float8_e4m3fn, no cast needed
        weight = layer.weight.data
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)

        # Reshape weight_scale: [out, in/32] -> [out, in/32//2, 2]
        weight_scale = layer.weight_scale.data
        weight_scale = weight_scale.reshape(weight_scale.shape[0], -1, 2)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            # npu_dynamic_mx_quant only accepts fp16/bf16 activations
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        # npu_dynamic_mx_quant requires a 2D input [tokens, hidden_size].
        # Diffusion transformer inputs are typically 3D [batch, seq, hidden] or
        # higher. Flattening to 2D merges all leading dimensions into a single
        # token axis so the NPU kernel can compute per-token MXFP8 scales, then
        # we restore the original shape from the output.
        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        # Dynamic MXFP8 activation quantisation
        qx, input_scale = torch_npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch_npu.float8_e4m3fn
        )

        # MXFP8 matmul
        output = torch_npu.npu_quant_matmul(
            qx,
            layer.weight.transpose(0, 1),
            layer.weight_scale.transpose(0, 1),
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
            group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
        )

        # Restore original shape
        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        output = output.reshape(output_shape)

        return output
