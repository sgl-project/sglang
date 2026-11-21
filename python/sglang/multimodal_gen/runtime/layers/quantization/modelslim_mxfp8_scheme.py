"""ModelSlim MXFP8 scheme for pre-quantized weight inference on NPU.

Loads weights pre-quantized by msmodelslim (float8_e4m3fn weights,
uint8 scales) and runs MXFP8 matmul at inference.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUMXFP8LinearMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme

MXFP8_BLOCK_SIZE = 32


class ModelSlimMXFP8Scheme(ModelSlimLinearScheme):
    def __init__(self):
        self.kernel = NPUMXFP8LinearMethod()

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

        # msmodelslim exports UE8M0 bytes, including a partial final block.
        # NOTE: This parameter is intentionally named "weight_scale" (not
        # "weight_scale_inv" as used by SRT's NPUMXFP8LinearMethod) because the
        # weight loader
        # matches parameter names to checkpoint keys, and msmodelslim checkpoints
        # store this tensor under the key "<layer>.weight_scale".
        scale_dim = (
            input_size_per_partition + MXFP8_BLOCK_SIZE - 1
        ) // MXFP8_BLOCK_SIZE
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
        # The SRT kernel pairs UE8M0 scales and pre-transposes weight/scale.
        # Pad only an odd scale count, preserving the partial checkpoint block;
        # the extra byte has no corresponding weight element and is not used.
        scale = layer.weight_scale.data
        if scale.shape[-1] % 2:
            layer.weight_scale = torch.nn.Parameter(
                F.pad(scale, (0, 1), value=0), requires_grad=False
            )
        self.kernel.process_weights_after_loading(layer)
        # Preserve the offline adapter's per-call bias conversion, including
        # callers that update layer.bias after loading.
        layer.bias_fp32 = None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        return self.kernel.apply(layer, x, bias)
