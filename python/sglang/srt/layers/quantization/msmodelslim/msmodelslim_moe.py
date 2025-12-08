# Adapted from https://github.com/vllm-project/vllm/tree/v0.8.2/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import enum
import logging
from enum import Enum
from typing import TYPE_CHECKING

import torch
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

logger = logging.getLogger(__name__)


__all__ = [
    "ModelSlimMoEMethod",
]


class ModelSlimMoEMethod(FusedMoEMethodBase):
    def __new__(cls, *args, **kwargs):
        if cls is ModelSlimMoEMethod:
            return super().__new__(cls)
        return super().__new__(cls)

    @staticmethod
    def get_moe_method(
        quant_config: ModelSlimConfig,
        layer: torch.nn.Module,
        prefix: str,
    ) -> "ModelSlimMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.

        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get("input_activations")
        is_moe_w4_dynamic = quant_config.is_dynamic_token_w4(weight_quant, input_quant)
        is_moe_input_quant = input_quant

        if (
            is_moe_w4_dynamic and is_moe_input_quant is not None
        ) or quant_config._is_moe_w4a8_dynamic(prefix, weight_quant, input_quant):
            return NPUW4A8Int4DynamicMoEMethod(quant_config)
        elif is_moe_w4_dynamic and is_moe_input_quant is None:
            return NPUW4A16Int4DynamicMoEMethod(quant_config)
        else:
            return NPUW8A8Int8DynamicMoEMethod(quant_config)
        # else:
        #     raise RuntimeError(
        #         f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
        #     )
