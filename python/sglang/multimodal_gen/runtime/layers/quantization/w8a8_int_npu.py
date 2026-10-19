"""Online W8A8 INT8 quantization for diffusion linear layers on Ascend NPU."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, NamedTuple, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import LinearBase, LinearMethodBase
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.parameter import ModelWeightParameter
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.quantization.utils import _module_path_match, is_layer_skipped


class _INT8Activation(NamedTuple):
    data: torch.Tensor
    scale: torch.Tensor
    source_dtype: torch.dtype


@lru_cache(maxsize=None)
def _supports_swiglu_int8(device: torch.device, dtype: torch.dtype, width: int) -> bool:
    try:
        op = torch.ops.npu.npu_dequant_swiglu_quant
        probe = torch.zeros((1, width), device=device, dtype=dtype)
        data, scale = op(probe, quant_mode=1, activate_left=True)
        return (
            data.dtype == torch.int8
            and data.shape == (1, width // 2)
            and scale.dtype == torch.float32
            and scale.shape == (1,)
        )
    except torch.OutOfMemoryError:
        raise
    except (AttributeError, NotImplementedError, RuntimeError):
        return False


@lru_cache(maxsize=None)
def _supports_add_rmsnorm_int8(device: torch.device, dtype: torch.dtype, width: int):
    try:
        x = torch.zeros((1, width), device=device, dtype=dtype)
        weight = torch.ones(width, device=device, dtype=dtype)
        data, _, residual, scale, _ = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
            x, x, weight, epsilon=1e-6
        )
        return (
            data.dtype == torch.int8
            and data.shape == x.shape
            and residual.dtype == dtype
            and residual.shape == x.shape
            and scale.dtype == torch.float32
            and scale.shape == (1,)
        )
    except torch.OutOfMemoryError:
        raise
    except (AttributeError, NotImplementedError, RuntimeError):
        return False


class NPUOnlineW8A8DiffusionConfig(QuantizationConfig):
    """Quantize full-precision diffusion transformer linears to W8A8."""

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        if not current_platform.is_npu():
            raise ValueError("w8a8_int diffusion quantization requires Ascend NPU")
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}

    @classmethod
    def get_name(cls) -> str:
        return "w8a8_int"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NPUOnlineW8A8DiffusionConfig:
        return cls(
            ignored_layers=cls.get_from_keys_or(config, ["ignored_layers"], None)
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if not isinstance(layer, LinearBase):
            return None
        if any(
            _module_path_match(ignored, prefix) for ignored in self.ignored_layers
        ) or is_layer_skipped(
            prefix,
            self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            from sglang.multimodal_gen.runtime.layers.linear import (
                UnquantizedLinearMethod,
            )

            return UnquantizedLinearMethod()
        return NPUOnlineW8A8DiffusionLinearMethod(self)

    def get_scaled_act_names(self) -> list[str]:
        return []


class NPUOnlineW8A8DiffusionLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: NPUOnlineW8A8DiffusionConfig) -> None:
        self.quant_config = quant_config

    def quantize_rmsnorm(
        self,
        x: torch.Tensor,
        norm: torch.nn.RMSNorm,
        shift: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        *,
        keep_float: bool = True,
    ):
        from sglang.multimodal_gen.runtime.layers.minimax_h3_npu import (
            rmsnorm_modulate,
        )

        result = rmsnorm_modulate(
            x, norm, shift, scale, indices, quantize=True, keep_float=keep_float
        )
        if result is None:
            return None
        normalized, data, scales = result
        return normalized, _INT8Activation(data, scales, x.dtype)

    def quantize_add_rmsnorm(
        self, x: torch.Tensor, residual: torch.Tensor, norm: torch.nn.RMSNorm
    ):
        if (
            x.device.type != "npu"
            or x.dtype not in (torch.float16, torch.bfloat16)
            or x.ndim != 2
            or x.numel() == 0
            or residual.shape != x.shape
            or residual.dtype != x.dtype
            or residual.device != x.device
            or norm.weight is None
            or norm.weight.shape != (x.shape[-1],)
            or norm.weight.dtype != x.dtype
            or norm.weight.device != x.device
        ):
            return None
        if not _supports_add_rmsnorm_int8(x.device, x.dtype, x.shape[-1]):
            return None
        eps = norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps
        data, _, residual_out, scales, _ = torch.ops.npu.npu_add_rms_norm_dynamic_quant(
            x, residual, norm.weight, epsilon=eps
        )
        return _INT8Activation(data, scales, x.dtype), residual_out

    def quantize_swiglu(self, x: torch.Tensor) -> _INT8Activation | None:
        if (
            x.device.type != "npu"
            or x.dtype not in (torch.float16, torch.bfloat16)
            or x.ndim != 2
            or x.shape[0] == 0
            or x.shape[1] == 0
            or x.shape[1] % 2
            or not x.is_contiguous()
        ):
            return None
        if not _supports_swiglu_int8(x.device, x.dtype, x.shape[1]):
            return None
        data, scale = torch.ops.npu.npu_dequant_swiglu_quant(
            x, quant_mode=1, activate_left=True
        )
        return _INT8Activation(data, scale, x.dtype)

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
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "w8a8_int diffusion quantization requires FP16 or BF16 "
                f"weights, got {weight.dtype}"
            )
        if not weight.is_npu:
            weight = weight.to(f"npu:{torch.npu.current_device()}")
        quantized_weight, weight_scale = torch.ops.npu.npu_dynamic_quant(weight)
        weight = Parameter(
            npu_format_cast(quantized_weight.transpose(-2, -1).contiguous()),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.weight = weight
        layer.weight_scale = Parameter(
            weight_scale.flatten(),
            requires_grad=False,
        )
        # Convert GEMM bias once instead of on every denoising step.
        if layer.bias is not None and not layer.skip_bias_add:
            layer.bias.data = layer.bias.data.to(torch.float32)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | _INT8Activation,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, _INT8Activation):
            quantized_x, dynamic_scale, output_dtype = x
            input_shape = quantized_x.shape
        else:
            input_shape = x.shape
            output_dtype = x.dtype
        if output_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "w8a8_int diffusion quantization requires FP16 or BF16 "
                f"activations, got {output_dtype}"
            )
        if not isinstance(x, _INT8Activation):
            quantized_x, dynamic_scale = torch.ops.npu.npu_dynamic_quant(
                x.reshape(-1, x.shape[-1]),
                dst_type=torch.int8,
            )
        output = torch.ops.npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=dynamic_scale.flatten(),
            bias=bias,
            output_dtype=output_dtype,
        )
        return output.reshape(*input_shape[:-1], output.shape[-1])
