# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.petit_utils import (
    apply_petit_mxfp4_linear,
    prepare_mxfp4_layer_for_petit,
    verify_petit_mxfp4_supported,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import (
    is_layer_excluded_by_patterns,
    is_layer_skipped,
)
from sglang.srt.utils import is_hip

_is_hip = is_hip()
logger = logging.getLogger(__name__)


class PetitMxfp4Config(QuantizationConfig):
    """Config class for Petit MXFP4 linear inference on ROCm."""

    def __init__(
        self,
        is_checkpoint_mxfp4_serialized: bool = False,
        group_size: int = 32,
        exclude_modules: Optional[List[str]] = None,
    ) -> None:
        self.is_checkpoint_mxfp4_serialized = is_checkpoint_mxfp4_serialized
        self.group_size = group_size
        self.exclude_modules = exclude_modules or []
        if is_checkpoint_mxfp4_serialized:
            logger.warning(
                "Detected mxfp4 checkpoint for petit kernel path. "
                "This format is experimental and subject to change."
            )

    @classmethod
    def get_name(cls) -> str:
        return "petit_mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        # Petit MXFP4 kernel currently supports BF16 activations on ROCm.
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 90

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "PetitMxfp4Config":
        quant_section = config.get("quantization", config)
        quant_method = (
            quant_section.get("quant_algo")
            or quant_section.get("quant_method")
            or config.get("quant_method")
            or ""
        )
        group_size = quant_section.get("group_size", 32)
        verify_petit_mxfp4_supported(quant_method, group_size)

        exclude_modules = quant_section.get("exclude_modules", [])
        is_checkpoint_mxfp4_serialized = "mxfp4" in quant_method.lower()
        return cls(
            is_checkpoint_mxfp4_serialized=is_checkpoint_mxfp4_serialized,
            group_size=group_size,
            exclude_modules=exclude_modules,
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        # Keep legacy MXFP4 flows unless user explicitly opts into petit_mxfp4.
        if str(user_quant).lower() != cls.get_name():
            return None
        if cls.is_petit_mxfp4_compatible(hf_quant_cfg):
            return cls.get_name()
        return None

    @classmethod
    def is_petit_mxfp4_compatible(cls, quant_config: Dict[str, Any]) -> bool:
        if not _is_hip or not quant_config:
            return False

        quant_method = str(quant_config.get("quant_method", "")).lower()
        quant_algo = str(quant_config.get("quant_algo", "")).lower()
        nested_quant = quant_config.get("quantization")
        if isinstance(nested_quant, dict):
            quant_algo = str(nested_quant.get("quant_algo", quant_algo)).lower()
        return "mxfp4" in quant_method or "mxfp4" in quant_algo

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix, self.exclude_modules
            ) or is_layer_excluded_by_patterns(prefix, self.exclude_modules):
                return UnquantizedLinearMethod()
            return PetitMxfp4LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class PetitMxfp4LinearMethod(LinearMethodBase):
    """Linear method for MXFP4 weights + Petit kernel execution."""

    def __init__(self, quant_config: PetitMxfp4Config):
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
    ) -> None:
        del input_size, output_size, params_dtype

        if not self.quant_config.is_checkpoint_mxfp4_serialized:
            raise ValueError(
                "MXFP4 quantization was selected, but dynamic quantization "
                "is not supported for petit_mxfp4."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "Unsupported model when in features size is not divisible by "
                f"group_size={self.quant_config.group_size}."
            )
        if input_size_per_partition % 2 != 0:
            raise ValueError("MXFP4 packed weights require even K dimension.")

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        prepare_mxfp4_layer_for_petit(layer)
        layer.weight_scale_2 = Parameter(
            torch.ones(1, device=layer.weight.device, dtype=torch.float32),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_petit_mxfp4_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            global_scale=layer.weight_scale_2,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
