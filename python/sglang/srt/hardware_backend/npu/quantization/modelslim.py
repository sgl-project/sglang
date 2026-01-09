from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A8Int4DynamicMoEMethod,
    NPUW4A16Int4DynamicMoEMethod,
    NPUW8A8Int8DynamicMoEMethod,
)
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUW8A8Int8DynamicLinearMethod,
    NPUW8A8Int8LinearMethod,
)
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import apply_module_patch


# func refers to RMSNorm.__init__
def npu_wrapper_rmsnorm_init(func):
    def init(self, hidden_size: int, **extra_args) -> None:
        func(self, hidden_size, **extra_args)
        self.ignore_anti = True
        # The Ascend w8a8_int8 quantization requires adding a bias in rmsnorm
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)

    return init


# func refers to RMSNorm.forward_oot
def npu_wrapper_rmsnorm_forward(func):
    def _rmsnorm_forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        from sgl_kernel_npu.norm.add_rmsnorm_bias import add_rmsnorm_bias

        if not x.is_contiguous():
            x = x.contiguous()
        if residual is not None:
            out, residual_out = add_rmsnorm_bias(
                x,
                residual,
                self.weight.data,
                self.bias,
                self.variance_epsilon,
            )
            return out.to(x.dtype), residual_out

        out = torch.ops.npu.npu_rms_norm(x, self.weight.data, self.variance_epsilon)[0]
        out = out + self.bias
        return out.to(x.dtype)

    return _rmsnorm_forward_oot


class ModelSlimConfig(QuantizationConfig):
    """
    Config class for ModelSlim Quantization, a NPU-specific quantization type.
    """

    def __init__(self, quant_config: Dict[str, Any] = {}):
        super().__init__()
        self.quant_description = quant_config
        self.is_dynamic = quant_config.get("is_dynamic", False)
        self.is_moe_w4_dynamic = False
        ignore = cast(List[str], quant_config.get("ignore", []))
        self.ignore = ignore if ignore is not None else []
        packed_modules_mapping = quant_config.get("packed_modules_mapping", {})
        self.packed_modules_mapping = (
            packed_modules_mapping if packed_modules_mapping is not None else {}
        )
        self.target_scheme_map = (
            CompressedTensorsConfig._quantization_scheme_map_from_config(
                config=quant_config
            )
        )
        target = "MoEGMM" if "MoEGMM" in self.target_scheme_map else "Linear"
        target_scheme = self.target_scheme_map.get(target, None)
        if target_scheme is None:
            self.is_moe_w4_dynamic = False
        else:
            weight_quant = target_scheme.get("weights")
            input_quant = target_scheme.get("input_activations")
            self.is_moe_w4_dynamic = self.is_dynamic_token_w4(weight_quant, input_quant)
            self.is_moe_input_quant = input_quant

        for name in self.quant_description.keys():
            if "norm.bias" in name:
                apply_module_patch(
                    "sglang.srt.layers.layernorm.RMSNorm",
                    "__init__",
                    [npu_wrapper_rmsnorm_init],
                )
                apply_module_patch(
                    "sglang.srt.layers.layernorm.RMSNorm",
                    "forward_npu",
                    [npu_wrapper_rmsnorm_forward],
                )

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_name(self) -> str:
        return "modelslim"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        filenames = ["quant_model_description.json"]
        return filenames

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ModelSlimConfig:
        return cls(config)

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if should_ignore_layer(
                prefix,
                ignore=self.ignore,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            key = "model"
            if "vision_model" in prefix:
                key = "vision_model"
            elif "visual" in prefix:
                key = "visual"
            packed_modules_mapping_subset = self.packed_modules_mapping.get(key, {})
            prefix_in_quant_config = prefix
            proj_name = prefix.split(".")[-1]
            if proj_name in packed_modules_mapping_subset:
                prefix_in_quant_config = prefix.replace(
                    proj_name, packed_modules_mapping_subset[proj_name][0]
                )
            self.is_dynamic = (
                self.quant_description.get(prefix_in_quant_config + ".weight", "")
                == "W8A8_DYNAMIC"
                or self.quant_description.get("quant_method", "")
                == "modelslim"  # TODO: This path is for compress-tensor config，needs refactor @zhengdqin
            )
            if self.is_layer_skipped(prefix, packed_modules_mapping_subset):
                return UnquantizedLinearMethod()
            return (
                NPUW8A8Int8DynamicLinearMethod(self)
                if self.is_dynamic
                else NPUW8A8Int8LinearMethod(self)
            )
        elif isinstance(layer, FusedMoE):
            prefix_in_quant_config = prefix + ".0.down_proj.weight"
            is_moe_w4a8_dynamic = (
                self.quant_description.get(prefix_in_quant_config, "STATIC")
                == "W4A8_DYNAMIC"
            )
            if (
                self.is_moe_w4_dynamic and self.is_moe_input_quant is not None
            ) or is_moe_w4a8_dynamic:
                return NPUW4A8Int4DynamicMoEMethod()
            elif self.is_moe_w4_dynamic and self.is_moe_input_quant is None:
                return NPUW4A16Int4DynamicMoEMethod(self)
            else:
                return NPUW8A8Int8DynamicMoEMethod()
        return None

    def is_layer_skipped(
        self, prefix: str, fused_mapping: Mapping[str, List[str]] = MappingProxyType({})
    ):
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = (
                    self.quant_description.get(shard_prefix + ".weight", "") == "FLOAT"
                )

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision."
                    )
        else:
            is_skipped = self.quant_description.get(prefix + ".weight", "") == "FLOAT"

        assert is_skipped is not None
        return is_skipped

    def get_scaled_act_names(self) -> List[str]:
        return []

    def is_dynamic_token_w4(self, weight_quant, input_quant) -> bool:
        is_w4 = weight_quant.num_bits == 4
        weight_strategy = (
            weight_quant.strategy == QuantizationStrategy.TENSOR.value
            or weight_quant.strategy == QuantizationStrategy.CHANNEL.value
            or weight_quant.strategy == QuantizationStrategy.GROUP.value
        )
        if input_quant is not None:
            is_token = (
                weight_strategy
                and input_quant.strategy == QuantizationStrategy.TOKEN.value
            )
            is_dynamic = not weight_quant.dynamic and input_quant.dynamic
        else:
            is_token = weight_strategy
            is_dynamic = not weight_quant.dynamic

        # Both symmetric and asymmetric input quantization supported.
        # Only symmetric weight quantization supported.
        return is_w4 and weight_quant.symmetric and is_token and is_dynamic
