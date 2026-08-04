"""ModelSlim W4A8_MXFP scheme for pre-quantized weight inference on Ascend NPU (SRT).

The msmodelslim ``W4A8_MXFP`` checkpoint stores weights as **packed FP4**:

    weight:       uint8 (pack_fp4_to_uint8),  shape [out, in//2],  group_size=32
    weight_scale: uint8 (UE8M0, +127 biased), shape [out, in//32]

(verified on ``Qwen3-8B-mxw4a8-pack-full`` and matching the msmodelslim exporter
``ascendv1.py:on_w4a8_mx_dynamic_per_block``). This is a true W4(weight) A8(activation)
scheme: weights are 4-bit FP4, activations are dynamically quantised to MXFP8.

This is NOT the same layout as ``W8A8_MXFP8`` (which stores float8_e4m3fn weights
of shape [out, in]) — so weight creation and the forward pass differ from MXFP8.
Weight post-processing and the matmul are delegated to ``NPUMXFP4W4A8OfflineLinearMethod``
(``self.kernel``), mirroring vllm-ascend's ``AscendW4A8MXFPDynamicLinearMethod``:
``npu_format_cast`` the packed FP4 to FRACTAL_NZ + transpose, then ``x2_dtype=
float4_e2m1fn_x2`` matmul with ``group_sizes=[0, 0, 32]``. Requires a recent
torch_npu for the FP4 matmul on Ascend 950/A5 (older builds reject the NZ weight) —
see ``NPUMXFP4W4A8OfflineLinearMethod`` for the version caveat.
"""

from typing import Dict, List, Optional

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUMXFP4W4A8OfflineLinearMethod,
)
from sglang.srt.layers.parameter import GroupQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme

# Fixed by the msmodelslim W4A8_MXFP export format (ascendv1.py sets group_size=32).
MXFP4_W4A8_BLOCK_SIZE = 32
# FP4 weights are bit-packed two-per-byte along the input (reduction) dim.
MXFP4_W4A8_PACK_FACTOR = 2


class ModelSlimMXFP4W4A8Scheme(ModelSlimLinearScheme):
    """W4A8_MXFP offline scheme — packed-FP4 weights, MXFP8 activations."""

    def __init__(
        self,
        quant_config: Optional[Dict[str, any]] = None,
        prefix: Optional[str] = None,
    ):
        # quant_config / prefix accepted to match ModelSlimConfig.get_linear_scheme's
        # dispatch signature; W4A8_MXFP needs no per-layer config beyond create_weights.
        del quant_config, prefix
        self.kernel = NPUMXFP4W4A8OfflineLinearMethod()

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

        # Packed-FP4 weight: uint8, shape [out, in//2] (two FP4 nibbles per byte
        # along the input dim). input_dim=1 is the packed dim; TP row-parallel
        # sharding narrows by self.data.shape[input_dim] (already halved), so a
        # plain ModelWeightParameter shards correctly without packing metadata
        # (FP4 packs the reduction dim only; the output dim stays unpacked).
        weight = ModelWeightParameter(
            data=torch.empty(
                (
                    output_size_per_partition,
                    input_size_per_partition // MXFP4_W4A8_PACK_FACTOR,
                ),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # UE8M0 block scales: uint8, shape [out, in//32]. Named "weight_scale" to
        # match the checkpoint key; the kernel re-layouts it into weight_scale_inv
        # during process_weights_after_loading.
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
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.kernel.apply(layer, x, bias)
