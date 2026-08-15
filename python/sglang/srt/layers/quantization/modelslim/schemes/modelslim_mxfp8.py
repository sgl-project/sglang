"""ModelSlim MXFP8 scheme for pre-quantized weight inference on Ascend NPU (SRT).

Loads weights pre-quantized by msmodelslim (float8_e4m3fn weights,
uint8 scales) and runs MXFP8 matmul at inference.

Following the modelslim-scheme convention (see ModelSlimW8A8Int8), this scheme
owns only the hardware-agnostic weight creation; weight post-processing and the
forward pass are delegated to an NPUMXFP8LinearMethod kernel (self.kernel). Its
process_weights_after_loading detects the pre-quantized float8_e4m3fn weight and
takes the offline (transpose-only) branch.
"""

from typing import Dict, List, Optional

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    MXFP_E8M0_NOT_LOADED,
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
        # MLAProlog needs the checkpoint-layout source only for QKV-A and Q-B.
        # Other MXFP8 linears must not retain a second full-size weight view.
        del quant_config
        self.kernel = NPUMXFP8LinearMethod()
        self.configure_runtime_prefix(prefix)

    def configure_runtime_prefix(self, prefix: Optional[str]) -> None:
        """Select the two runtime projections consumed by MLAProlog.

        The quant-description prefix for fused QKV-A is usually rewritten to
        ``q_a_proj`` by ``packed_modules_mapping``.  Selection must therefore
        use the runtime module prefix, not the checkpoint lookup prefix.
        """

        module_name = "" if prefix is None else prefix.rsplit(".", 1)[-1]
        self.kernel.preserve_mlaprolog_source = module_name in {
            "fused_qkv_a_proj_with_mqa",
            "q_b_proj",
        }

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
        # NOTE: Named "weight_scale" (not "weight_scale_inv") to match the
        # checkpoint key exported by msmodelslim; the kernel re-layouts it into
        # weight_scale_inv during process_weights_after_loading.
        scale_dim = input_size_per_partition // MXFP8_BLOCK_SIZE
        weight_scale = GroupQuantScaleParameter(
            data=torch.full(
                (output_size_per_partition, scale_dim),
                MXFP_E8M0_NOT_LOADED,
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
