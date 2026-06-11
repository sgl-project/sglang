# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/ab3e80042eac24dd362408e6d63ad98768046359/vllm/model_executor/layers/quantization/gguf.py

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_xpu

from .schemes import (
    GGUFAscendEmbeddingScheme,
    GGUFAscendLinearScheme,
    GGUFAscendMoEScheme,
    GGUFEmbeddingScheme,
    GGUFLinearScheme,
    GGUFMoEScheme,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
_is_musa = is_musa()
_is_npu = is_npu()

if not (_is_cuda or _is_hip or _is_xpu or _is_musa or _is_npu):
    warnings.warn("Only CUDA, MUSA and NPU support GGUF quantization currently.")


def is_layer_skipped_gguf(prefix: str, modules_to_not_convert: list[str]) -> bool:
    return any(module_name in prefix for module_name in modules_to_not_convert)


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, modules_to_not_convert: list[str] | None = None) -> None:
        super().__init__()
        if _is_hip:
            warnings.warn("Only CUDA and MUSA support GGUF quantization currently.")
        self.modules_to_not_convert = modules_to_not_convert or []

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_name(self) -> str:
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60 if not _is_musa else 21

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GGUFConfig":
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            layer.scheme = self.get_linear_scheme(layer)
            return GGUFLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            layer.scheme = self.get_embedding_scheme(layer)
            return GGUFEmbeddingMethod(self)
        elif isinstance(layer, FusedMoE):
            layer.scheme = self.get_moe_scheme(layer)
            return GGUFMoEMethod(self)
        return None

    def get_linear_scheme(self, layer: torch.nn.Module):
        assert isinstance(layer, LinearBase)
        if _is_npu:
            return GGUFAscendLinearScheme(self)

        return GGUFLinearScheme(self)

    def get_embedding_scheme(self, layer: torch.nn.Module):
        if _is_npu:
            return GGUFAscendEmbeddingScheme(self)

        return GGUFEmbeddingScheme(self)

    def get_moe_scheme(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        assert isinstance(layer, FusedMoE)
        if _is_npu:
            return GGUFAscendMoEScheme(self)

        return GGUFMoEScheme(self)


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF."""

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if not hasattr(layer, "scheme"):
            layer.scheme = self.quant_config.get_linear_scheme(layer)
        layer.scheme.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return layer.scheme.apply_weights(layer, x, bias)


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF."""

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return layer.scheme.embedding(layer, x)


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF."""

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if not hasattr(layer, "scheme"):
            layer.scheme = self.quant_config.get_moe_scheme(layer)
        layer.scheme.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        layer.scheme.create_moe_runner(layer, moe_runner_config)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.scheme.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        return layer.scheme.apply_weights(layer, dispatch_output)
