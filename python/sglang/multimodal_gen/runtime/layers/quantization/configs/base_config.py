# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/quantization/base_config.py

import inspect

import torch

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig as SRTQuantizationConfig,
)
from sglang.srt.layers.quantization.base_config import (
    QuantizeMethodBase as SRTQuantizeMethodBase,
)


class QuantizeMethodBase(SRTQuantizeMethodBase):
    def embedding(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


def method_has_implemented_embedding(method_class: type[QuantizeMethodBase]) -> bool:
    """
    Not all quant methods have embedding implemented, so we need to check that
    it exists for our given method. We check this by making sure the function
    has been changed from the base implementation.
    """
    base_embedding = inspect.getattr_static(QuantizeMethodBase, "embedding", None)
    class_embedding = inspect.getattr_static(method_class, "embedding", None)

    return class_embedding is not None and class_embedding is not base_embedding


class QuantizationConfig(SRTQuantizationConfig):
    # for quantization frameworks with a separate quantized model provided, e.g. Nunchaku
    quantized_model_path: str | None = None
    checkpoint_uses_native_qkv_layout: bool = False
    checkpoint_uses_comfy_quantization: bool = False
    supports_srt_linear_layers: bool = False
    supports_quantized_embeddings: bool = False

    def get_scaled_act_names(self) -> list[str]:
        return []

    def supports_input_partition(
        self, prefix: str, input_size_per_partition: int
    ) -> bool:
        """Whether a row-parallel shard preserves this format's input layout."""
        return True

    def quantizes_embedding(self, prefix: str) -> bool:
        """Whether this checkpoint config owns the named embedding table."""
        return False

    def remap_checkpoint_prefixes(self, param_names_mapping: dict) -> None:
        """Translate checkpoint module names to the native model namespace."""
        return
