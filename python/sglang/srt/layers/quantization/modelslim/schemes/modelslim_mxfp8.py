"""ModelSlim MXFP8 scheme for pre-quantized weight inference on Ascend NPU (SRT).

Loads weights pre-quantized by msmodelslim (float8_e4m3fn weights,
uint8 scales) and runs MXFP8 matmul at inference.

This scheme owns weight creation and alignment of TP shards to checkpoint
quantization blocks; weight post-processing and the forward pass are otherwise
delegated to an NPUMXFP8LinearMethod kernel (self.kernel). Its
process_weights_after_loading detects the pre-quantized float8_e4m3fn weight and
takes the offline (transpose-only) branch.
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUMXFP8LinearMethod,
)
from sglang.srt.layers.parameter import GroupQuantScaleParameter, ModelWeightParameter
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme

MXFP8_BLOCK_SIZE = 32


class ModelSlimMXFP8Scheme(ModelSlimLinearScheme):
    def __init__(
        self,
        quant_config: Optional[Dict[str, any]] = None,
        prefix: Optional[str] = None,
    ):
        # quant_config / prefix are accepted to match the linear-scheme
        # dispatch signature used by ModelSlimConfig.get_linear_scheme;
        # MXFP8 needs no per-layer config beyond what create_weights derives.
        del quant_config, prefix
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

        # msmodelslim exports one uint8 scale per 32 input elements. Keep the
        # final partial block: Qwen3-VL has visual projections such as K=4304,
        # whose checkpoint scale shape is [out, ceil(4304/32)] = [out, 135].
        # NOTE: Named "weight_scale" (not "weight_scale_inv") to match the
        # checkpoint key exported by msmodelslim; the kernel re-layouts it into
        # weight_scale_inv during process_weights_after_loading.
        scale_dim = (
            input_size_per_partition + MXFP8_BLOCK_SIZE - 1
        ) // MXFP8_BLOCK_SIZE
        scale_loader = weight_loader
        if (
            input_size_per_partition != input_size
            and input_size_per_partition % MXFP8_BLOCK_SIZE
        ):
            tp_rank, _ = get_group_rank_size(layer.tp_group)
            start = tp_rank * input_size_per_partition
            left = start % MXFP8_BLOCK_SIZE
            right = -(left + input_size_per_partition) % MXFP8_BLOCK_SIZE
            layer.mxfp8_tp_padding = (left, right)
            scale_start = start // MXFP8_BLOCK_SIZE
            scale_dim = (left + input_size_per_partition + right) // MXFP8_BLOCK_SIZE

            # A shard can start inside a checkpoint block, so adjacent ranks
            # may share a scale. Include that block and align both W and X to it.
            def scale_loader(param, loaded_weight):
                if not layer.use_presharded_weights:
                    loaded_weight = loaded_weight.narrow(1, scale_start, scale_dim)
                assert param.shape == loaded_weight.shape, (
                    f"{param.shape=} {loaded_weight.shape=}"
                )
                param.data.copy_(loaded_weight)

        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                (output_size_per_partition, scale_dim),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=scale_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if padding := getattr(layer, "mxfp8_tp_padding", None):
            layer.weight.data = F.pad(layer.weight.data, padding)
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if padding := getattr(layer, "mxfp8_tp_padding", None):
            if isinstance(x, tuple):
                raise ValueError(
                    "ModelSlim MXFP8 with unaligned TP shards requires "
                    "unquantized inputs for block alignment."
                )
            x = F.pad(x, padding)
        return self.kernel.apply(layer, x, bias)
