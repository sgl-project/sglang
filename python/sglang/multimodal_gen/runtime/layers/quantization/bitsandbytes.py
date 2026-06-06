# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from packaging import version

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.utils import set_weight_attrs


def _require_bitsandbytes() -> None:
    try:
        import bitsandbytes

        if version.parse(bitsandbytes.__version__) < version.parse("0.46.1"):
            raise ImportError(
                "bitsandbytes version is wrong. Please install bitsandbytes>=0.46.1."
            )
    except ImportError as err:
        raise ImportError(
            "Please install bitsandbytes>=0.46.1 via "
            "`pip install bitsandbytes>=0.46.1` to use bitsandbytes quantizer."
        ) from err


def _calculate_quant_ratio(dtype: torch.dtype) -> int:
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits // torch.iinfo(torch.uint8).bits
    return torch.iinfo(dtype).bits // torch.iinfo(torch.uint8).bits


def _is_layer_skipped(prefix: str, skipped_modules: list[str]) -> bool:
    components = prefix.split(".")
    if any(module_name in components for module_name in skipped_modules):
        return True

    prefixes = {".".join(components[: i + 1]) for i in range(len(components))}
    return bool(set(skipped_modules) & prefixes)


class BitsAndBytesConfig(QuantizationConfig):
    """Config class for pre-quantized bitsandbytes 4-bit checkpoints."""

    def __init__(
        self,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float32",
        bnb_4bit_quant_storage: str = "uint8",
        bnb_4bit_quant_type: str = "fp4",
        bnb_4bit_use_double_quant: bool = False,
        llm_int8_enable_fp32_cpu_offload: bool = False,
        llm_int8_has_fp16_weight: bool = False,
        llm_int8_skip_modules: list[str] | None = None,
        llm_int8_threshold: float = 6.0,
    ) -> None:
        super().__init__()
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_storage = bnb_4bit_quant_storage
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.llm_int8_skip_modules = llm_int8_skip_modules or []
        self.llm_int8_threshold = llm_int8_threshold

        if self.load_in_8bit or not self.load_in_4bit:
            raise ValueError("SGLang diffusion only supports bitsandbytes 4-bit.")
        if self.bnb_4bit_quant_storage != "uint8":
            raise ValueError(
                f"Unsupported bnb_4bit_quant_storage: {self.bnb_4bit_quant_storage}"
            )

    @classmethod
    def get_name(cls) -> str:
        return "bitsandbytes"

    def get_scaled_act_names(self) -> list[str]:
        return []

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float32, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BitsAndBytesConfig":
        def get_safe_value(keys, default_value=None):
            try:
                value = QuantizationConfig.get_from_keys(config, keys)
                return value if value is not None else default_value
            except ValueError:
                return default_value

        return cls(
            load_in_8bit=get_safe_value(["load_in_8bit"], False),
            load_in_4bit=get_safe_value(["load_in_4bit"], True),
            bnb_4bit_compute_dtype=get_safe_value(
                ["bnb_4bit_compute_dtype"], "float32"
            ),
            bnb_4bit_quant_storage=get_safe_value(["bnb_4bit_quant_storage"], "uint8"),
            bnb_4bit_quant_type=get_safe_value(["bnb_4bit_quant_type"], "fp4"),
            bnb_4bit_use_double_quant=get_safe_value(
                ["bnb_4bit_use_double_quant"], False
            ),
            llm_int8_enable_fp32_cpu_offload=get_safe_value(
                ["llm_int8_enable_fp32_cpu_offload"], False
            ),
            llm_int8_has_fp16_weight=get_safe_value(
                ["llm_int8_has_fp16_weight"], False
            ),
            llm_int8_skip_modules=get_safe_value(["llm_int8_skip_modules"], []),
            llm_int8_threshold=get_safe_value(["llm_int8_threshold"], 6.0),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            if _is_layer_skipped(prefix, self.llm_int8_skip_modules):
                return UnquantizedLinearMethod()
            return BitsAndBytesLinearMethod(self)
        return None


class BitsAndBytesLinearMethod(LinearMethodBase):
    """Linear method for pre-quantized bitsandbytes 4-bit weights."""

    def __init__(self, quant_config: BitsAndBytesConfig):
        _require_bitsandbytes()
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
    ) -> None:
        del input_size, output_size
        quant_ratio = _calculate_quant_ratio(params_dtype)
        total_size = input_size_per_partition * sum(output_partition_sizes)
        if total_size % quant_ratio != 0:
            raise ValueError(
                "The input size is not aligned with the quantized weight shape."
            )

        qweight = nn.Parameter(
            torch.empty(total_size // quant_ratio, 1, dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight,
            {
                "input_dim": 0,
                "output_dim": 0,
                "pack_factor": quant_ratio,
                "use_bitsandbytes_4bit": True,
            },
        )
        layer.register_parameter("weight", qweight)
        set_weight_attrs(qweight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_type = x.dtype
        original_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))

        out_dim = sum(
            quant_state.shape[0]
            for quant_state in layer.weight.bnb_quant_state.values()
        )
        out = torch.empty(x.shape[0], out_dim, dtype=torch.bfloat16, device=x.device)
        apply_bnb_4bit(x.to(torch.bfloat16), layer.weight, out)
        out = out.to(original_type)

        if len(original_shape) > 2:
            out = out.view(*original_shape[:-1], out.size(-1))

        if bias is not None:
            out = out + bias
        return out


def apply_bnb_4bit(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> None:
    from bitsandbytes import matmul_4bit

    offsets = weight.bnb_shard_offsets
    quant_states = weight.bnb_quant_state
    current_index = 0
    for i in range(len(quant_states)):
        output_size = quant_states[i].shape[0]
        out[:, current_index : current_index + output_size] = matmul_4bit(
            x,
            weight[offsets[i] : offsets[i + 1]].t(),
            quant_states[i],
        )
        current_index += output_size


class BitsAndBytes4BitLinear(nn.Module):
    """Storage-only bitsandbytes 4-bit linear for nn.Linear-based encoders."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _require_bitsandbytes()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        quant_ratio = _calculate_quant_ratio(compute_dtype or torch.get_default_dtype())
        total_size = in_features * out_features
        if total_size % quant_ratio != 0:
            raise ValueError(
                "The input size is not aligned with the quantized weight shape."
            )

        self.weight = nn.Parameter(
            torch.empty(total_size // quant_ratio, 1, dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            self.weight,
            {
                "pack_factor": quant_ratio,
                "use_bitsandbytes_4bit": True,
            },
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_features, dtype=compute_dtype or torch.get_default_dtype()
                ),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_type = x.dtype
        original_shape = x.shape
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))

        out = torch.empty(
            x.shape[0], self.out_features, dtype=torch.bfloat16, device=x.device
        )
        apply_bnb_4bit(x.to(torch.bfloat16), self.weight, out)
        out = out.to(original_type)

        if len(original_shape) > 2:
            out = out.view(*original_shape[:-1], out.size(-1))

        if self.bias is not None:
            out = out + self.bias
        return out


def swap_linears_to_bitsandbytes_4bit(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            replacement = BitsAndBytes4BitLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=child.weight.dtype,
            )
            setattr(module, name, replacement)
        else:
            swap_linears_to_bitsandbytes_4bit(child)


_BNB_4BIT_STATE_SUFFIXES = {
    "absmax",
    "quant_map",
    "nested_absmax",
    "nested_quant_map",
    "bitsandbytes",
}


def is_bitsandbytes_4bit_state_name(weight_name: str) -> bool:
    suffix = weight_name.split(".")[-1]
    return any(state_suffix in suffix for state_suffix in _BNB_4BIT_STATE_SUFFIXES)


def split_bitsandbytes_4bit_state(
    weights: Any,
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, torch.Tensor]]:
    normal_weights: list[tuple[str, torch.Tensor]] = []
    quant_state_dict: dict[str, torch.Tensor] = {}
    for name, tensor in weights:
        if is_bitsandbytes_4bit_state_name(name):
            if "quant_state.bitsandbytes" in name:
                tensor = tensor.cpu().data
            quant_state_dict[name] = tensor
            continue
        normal_weights.append((name, tensor))
    return normal_weights, quant_state_dict


def build_bitsandbytes_4bit_quant_states(
    normal_weight_names: list[str],
    quant_state_dict: dict[str, torch.Tensor],
    device: torch.device,
    param_names_mapping=None,
) -> dict[str, Any]:
    from bitsandbytes.functional import QuantState

    quant_states: dict[str, Any] = {}
    device_str = str(device)
    for source_name in normal_weight_names:
        if (
            f"{source_name}.quant_state.bitsandbytes__nf4" not in quant_state_dict
            and f"{source_name}.quant_state.bitsandbytes__fp4" not in quant_state_dict
        ):
            continue
        target_name = source_name
        if param_names_mapping is not None:
            target_name, _, _ = param_names_mapping(source_name)
        state_tensors = {
            name: tensor
            for name, tensor in quant_state_dict.items()
            if name.startswith(f"{source_name}.")
        }
        quant_states[target_name] = QuantState.from_dict(
            state_tensors, device=device_str
        )
    return quant_states


def attach_bitsandbytes_4bit_quant_states(
    params_dict: dict[str, torch.nn.Parameter],
    quant_states: dict[str, Any],
) -> None:
    for param_name, quant_state in quant_states.items():
        param = params_dict.get(param_name)
        if param is None:
            raise ValueError(f"Parameter {param_name} not found in the model.")

        state_by_shard = {0: quant_state}
        set_weight_attrs(param, {"bnb_quant_state": state_by_shard})
        offsets = torch.tensor([0, param.numel()]).cpu()
        set_weight_attrs(param, {"bnb_shard_offsets": offsets})
