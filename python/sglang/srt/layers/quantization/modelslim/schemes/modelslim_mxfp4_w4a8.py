"""ModelSlim W4A8_MXFP scheme for pre-quantized weight inference on Ascend NPU (SRT).

Loads weights pre-quantized by msmodelslim:
  weight:       float8_e4m3fn, shape [out, in],    group_size=32
  weight_scale: uint8 (+127 biased),  shape [out, in/32]

Inference:
  activation → npu_dynamic_mx_quant(float8_e4m3fn) → qx + per-token scale
  npu_quant_matmul(qx, weight, weight_scale, scale_dtype=FP8_E8M0)
"""

from typing import List, Optional

import torch

from sglang.srt.layers.parameter import GroupQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme
from sglang.srt.platforms import current_platform

_is_npu = current_platform.is_npu()

if _is_npu:
    import torch_npu

MXFP4_W4A8_BLOCK_SIZE = 32

_FLOAT8_E8M0FNU_DTYPE = (
    getattr(torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None))
    if _is_npu
    else getattr(torch, "float8_e8m0fnu", None)
)


class ModelSlimMXFP4W4A8Scheme(ModelSlimLinearScheme):

    def __init__(
        self,
        quant_config: Optional[dict] = None,
        prefix: Optional[str] = None,
    ):
        # quant_config / prefix are accepted to match the linear-scheme dispatch
        # signature used by ModelSlimConfig.get_linear_scheme; W4A8_MXFP needs no
        # per-layer config beyond what create_weights derives.
        del quant_config, prefix

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

        scale_dim = input_size_per_partition // MXFP4_W4A8_BLOCK_SIZE
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
        # weight_scale: [out, in/32] → reshape [out, in/64, 2] → transpose [in/64, out, 2]
        # weight:       [out, in]    → transpose [in, out]
        n_dim, k_dim = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim // 2, 2)
        layer.weight.data = layer.weight.data.transpose(0, 1)
        layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        if original_dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
            original_dtype = torch.bfloat16

        input_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])

        qx, input_scale = torch_npu.npu_dynamic_mx_quant(
            x_2d, dst_type=torch_npu.float8_e4m3fn
        )

        output = torch_npu.npu_quant_matmul(
            qx,
            layer.weight,
            layer.weight_scale,
            scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            pertoken_scale=input_scale,
            pertoken_scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
            bias=bias.to(torch.float32) if bias is not None else None,
            output_dtype=original_dtype,
            group_sizes=[1, 1, MXFP4_W4A8_BLOCK_SIZE],
        )

        output_shape = list(input_shape[:-1]) + [output.shape[-1]]
        return output.reshape(output_shape)
