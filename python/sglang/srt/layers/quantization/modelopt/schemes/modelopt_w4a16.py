# SPDX-License-Identifier: Apache-2.0
"""ModelOpt W4A16 AWQ: uint8 packed int4 weights + float32 group scales (+ optional pre_quant_scale)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.parameter import ModelWeightParameter, PackedvLLMParameter
from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.layers.quantization.modelopt.modelopt import ModelOptQuantConfig
from sglang.srt.layers.quantization.modelopt.utils import (
    modelopt_w4a16_linear_ref,
    pre_quant_scale_sharded_loader,
)
from sglang.srt.utils.common import set_weight_attrs


class ModelOptW4A16AWQConfig(ModelOptQuantConfig):
    """ModelOpt W4A16_AWQ: uint8 packed 4-bit weight + float32 per-group scales, no qzeros."""

    def __init__(
        self,
        group_size: int = 128,
        kv_cache_quant_algo: Optional[str] = None,
        exclude_modules: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__(kv_cache_quant_algo, exclude_modules, packed_modules_mapping)
        self.group_size = group_size

    @classmethod
    def override_quantization_method(cls, hf_quant_config, user_quant):
        if hf_quant_config is None:
            return None
        quant_algo = str(hf_quant_config.get("quant_algo", "")).upper()
        if "W4A16" in quant_algo and "AWQ" in quant_algo:
            if user_quant in ("modelopt", None, "modelopt_w4a16_awq"):
                return "modelopt_w4a16_awq"
        return None

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_w4a16_awq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ModelOptW4A16AWQConfig:
        if isinstance(config, dict) and "quantization" in config:
            flat = config.get("quantization") or config
        else:
            flat = config
        group_size = flat.get("group_size", 128)
        exclude = flat.get("ignore") or flat.get("exclude_modules") or []
        return cls(
            group_size=group_size,
            exclude_modules=exclude if isinstance(exclude, list) else list(exclude),
            packed_modules_mapping=config.get("packed_modules_mapping"),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return self._get_quant_method(
            layer,
            prefix,
            Linear=ModelOptW4A16AWQLinearMethod,
            Moe=None,
        )


class ModelOptW4A16AWQLinearMethod(LinearMethodBase):
    """W4A16 AWQ linear: checkpoint (out_packed, in) -> transpose to TRT-LLM (in, out_packed)."""

    def __init__(self, quant_config: ModelOptW4A16AWQConfig):
        self.quant_config = quant_config

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
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        group_size = self.quant_config.group_size
        if input_size_per_partition % group_size != 0:
            raise ValueError(
                f"input_size_per_partition ({input_size_per_partition}) must be "
                f"divisible by group_size ({group_size})"
            )
        num_groups = input_size_per_partition // group_size
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition // 2,
                input_size_per_partition,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=2,
            packed_dim=0,
        )
        layer.register_parameter("weight", weight)
        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                num_groups,
                dtype=torch.float32,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)
        pre_quant_scale = Parameter(
            torch.ones(input_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("pre_quant_scale", pre_quant_scale)
        set_weight_attrs(
            layer.pre_quant_scale,
            {"weight_loader": pre_quant_scale_sharded_loader},
        )
        if extra_weight_attrs.get("bias", True):
            bias_param = Parameter(
                torch.empty(output_size_per_partition, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("bias", bias_param)
            set_weight_attrs(
                bias_param,
                {"weight_loader": weight_loader, "output_dim": 0},
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = Parameter(
            layer.weight.data.t().contiguous(), requires_grad=False
        )
        layer.weight_scale = Parameter(
            layer.weight_scale.data.t().contiguous(), requires_grad=False
        )
        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias = Parameter(layer.bias.data, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return modelopt_w4a16_linear_ref(
            x,
            layer.weight,
            layer.weight_scale,
            self.quant_config.group_size,
            pre_quant_scale=getattr(layer, "pre_quant_scale", None),
            bias=bias,
            out_dtype=x.dtype,
        )
