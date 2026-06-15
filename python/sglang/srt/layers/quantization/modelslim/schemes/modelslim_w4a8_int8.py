# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUW4A8DynamicLinearMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimLinearScheme
from sglang.srt.utils import set_weight_attrs


class ModelSlimW4A8Int8(ModelSlimLinearScheme):
    """ModelSlim offline W4A8 Dense Linear scheme.

    Handles ``W4A8_DYNAMIC`` quant_type from ``quant_model_description.json``.

    Weight layout in the checkpoint:
    - ``new_quant_version`` (version == "1.0.0"): INT4×2 pre-packed into INT8,
      so on-disk shape is ``[N/2, K]``.
    - Old version: each INT8 stores one INT4, on-disk shape is ``[N, K]``.

    Delegates weight processing and matmul to ``NPUW4A8DynamicLinearMethod``
    which uses ``torch_npu.npu_weight_quant_batchmatmul`` (weight-dequant path).
    """

    def __init__(
        self,
        quant_config: Dict[str, Any],
        prefix: str,
    ):
        self.quant_config = quant_config
        self.group_size: int = quant_config.get("group_size", 256)
        self.new_quant_version: bool = quant_config.get("version", "0") == "1.0.0"
        self.kernel = NPUW4A8DynamicLinearMethod(
            group_size=self.group_size,
            new_quant_version=self.new_quant_version,
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # ── Weight ──────────────────────────────────────────────────────────
        # new_quant_version: 2 INT4 packed per INT8 → shape [N/2, K]
        # old version      : 1 INT4 per INT8         → shape [N,   K]
        weight_n = (
            output_size_per_partition // 2
            if self.new_quant_version
            else output_size_per_partition
        )
        weight = torch.nn.Parameter(
            torch.empty(weight_n, input_size_per_partition, dtype=torch.int8),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

        # ── Per-channel L1 scale & offset: [N, 1] ───────────────────────────
        weight_scale = torch.nn.Parameter(
            torch.empty(output_size_per_partition, 1, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale, {"output_dim": 0})
        layer.register_parameter("weight_scale", weight_scale)
        set_weight_attrs(weight_scale, extra_weight_attrs)

        weight_offset = torch.nn.Parameter(
            torch.empty(output_size_per_partition, 1, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(weight_offset, {"output_dim": 0})
        layer.register_parameter("weight_offset", weight_offset)
        set_weight_attrs(weight_offset, extra_weight_attrs)

        # ── Per-group L2 scale & offset: [N, K//group_size] ─────────────────
        # Note: for RowParallelLinear (K partitioned), input_dim=1 would be needed;
        # for ColumnParallelLinear (N partitioned), output_dim=0 suffices.
        # Initial implementation covers the column-parallel case.
        group_num = input_size_per_partition // self.group_size
        weight_scale_second = torch.nn.Parameter(
            torch.empty(output_size_per_partition, group_num, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(weight_scale_second, {"output_dim": 0})
        layer.register_parameter("weight_scale_second", weight_scale_second)
        set_weight_attrs(weight_scale_second, extra_weight_attrs)

        weight_offset_second = torch.nn.Parameter(
            torch.empty(output_size_per_partition, group_num, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(weight_offset_second, {"output_dim": 0})
        layer.register_parameter("weight_offset_second", weight_offset_second)
        set_weight_attrs(weight_offset_second, extra_weight_attrs)

        # ── scale_bias (new_quant_version only): [N, 1] ─────────────────────
        # Shape is [N, 16] for RowParallelLinear (down_proj / o_proj),
        # but [N, 1] for ColumnParallelLinear. Using [N, 1] for simplicity;
        # process_weights_after_loading handles both shapes dynamically.
        if self.new_quant_version:
            scale_bias = torch.nn.Parameter(
                torch.empty(output_size_per_partition, 1, dtype=torch.float32),
                requires_grad=False,
            )
            set_weight_attrs(scale_bias, {"output_dim": 0})
            layer.register_parameter("scale_bias", scale_bias)
            set_weight_attrs(scale_bias, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.kernel.apply(layer, x, bias)
