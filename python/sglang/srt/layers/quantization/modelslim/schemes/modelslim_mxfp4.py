"""ModelSlim W4A4_MXFP4 scheme for pre-quantized weight inference on Ascend NPU (SRT).

The msmodelslim ``W4A4_MXFP4`` checkpoint stores weights as **packed FP4**:

    weight:       uint8           shape [out, in//2]  (two FP4 values per byte)
    weight_scale: uint8 (UE8M0)  shape [out, in//32] (block scales, group_size=32)

This is a true W4(weight) A4(activation) scheme: both weights and activations are
single-level MXFP4. Weight post-processing and the matmul are delegated to
``NPUSingleLevelMXFP4OfflineLinearMethod`` (``self.kernel``): the packed weight is
transposed and the scale reshaped to 3D, then ``npu_quant_matmul`` runs with
``x1_dtype = x2_dtype = float4_e2m1fn_x2`` and
``group_sizes=[1, 1, 32]`` — sharing the online
``NPUSingleLevelMXFP4LinearMethod`` matmul exactly (only the weight source differs).

This differs from ``W4A8_MXFP`` only in the activation dtype (FP4 vs FP8), and
from ``W8A8_MXFP8`` in both weight packing and activation dtype.
"""

from typing import Dict, List, Optional

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUSingleLevelMXFP4OfflineLinearMethod,
)
from sglang.srt.layers.parameter import GroupQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme

# Fixed by the msmodelslim W4A4_MXFP4 export format (group_size=32).
MXFP4_BLOCK_SIZE = 32
MXFP4_PACK_FACTOR = 2


class ModelSlimMXFP4Scheme(ModelSlimLinearScheme):
    """W4A4_MXFP4 offline scheme — packed-FP4 weights, MXFP4 activations."""

    def __init__(
        self,
        quant_config: Optional[Dict[str, any]] = None,
        prefix: Optional[str] = None,
    ):
        # quant_config / prefix accepted to match ModelSlimConfig.get_linear_scheme's
        # dispatch signature; W4A4_MXFP4 needs no per-layer config beyond create_weights.
        del quant_config, prefix
        self.kernel = NPUSingleLevelMXFP4OfflineLinearMethod()

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

        # msmodelslim packs two FP4 values per byte along the input dimension.
        weight = ModelWeightParameter(
            data=torch.empty(
                (
                    output_size_per_partition,
                    input_size_per_partition // MXFP4_PACK_FACTOR,
                ),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # UE8M0 block scales: uint8, shape [out, in//32].
        scale_dim = input_size_per_partition // MXFP4_BLOCK_SIZE
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
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.kernel.apply(layer, x, bias)
