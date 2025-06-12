# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter

from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from sglang.srt.layers.quantization.int8_kernel import (
    get_w8a8_block_int8_configs,
    w8a8_block_int8_matmul,
)
from sglang.srt.layers.quantization.int8_utils import (
    apply_w8a8_block_int8_linear,
    block_dequant,
    input_to_int8,
)
from sglang.srt.layers.quantization.utils import requantize_with_max_scale
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8LinearMethod

__all__ = ["CompressedTensorsW8A8Int8"]


class CompressedTensorsW8A8Int8(W8A8Int8LinearMethod):

    def __init__(
        self, strategy: str, is_static_input_scheme: bool, input_symmetric: bool
    ):
        super().__init__()
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.apply(layer, x, bias)
