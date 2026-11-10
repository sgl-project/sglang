"""Online W4A16 storage for MiniMax-H3 on NVIDIA Volta.

The checkpoint is loaded and TP-sharded in FP16, then each rank-local linear
weight is quantized to asymmetric groupwise UINT4 in the conventional AWQ
checkpoint layout.  On SM70, the layout is converted once to TurboMind's
packed format and evaluated by its fused FP16 x UINT4 GEMM.

This online path uses round-to-nearest (RTN) group quantization.  It does not
perform AWQ's activation-aware calibration/search; calibrated AWQ checkpoints
can use the same runtime layout, but producing one is a separate offline step.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from sglang.kernels.sm70_paths import sm70_prebuilt
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.parameter import ModelWeightParameter
from torch import nn

_AWQ_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
_AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


@lru_cache(maxsize=1)
def _load_sm70_awq_ops():
    extension = sm70_prebuilt("_sm70_turbomind_v100.so")
    if not extension.is_file():
        return None
    try:
        torch.ops.load_library(str(extension))
    except (ImportError, OSError, RuntimeError):
        return None
    ops = torch.ops.sglang_sm70_turbomind
    required = ("awq_prepare", "awq_gemm_out")
    return ops if all(hasattr(ops, name) for name in required) else None


def _pack_awq_columns(values: torch.Tensor) -> torch.Tensor:
    """Pack logical UINT4 [rows, cols] values into AWQ int32 columns."""
    if values.ndim != 2 or values.shape[1] % 8:
        raise ValueError("AWQ UINT4 packing requires a 2D tensor and N % 8 == 0")
    order = torch.tensor(_AWQ_PACK_ORDER, dtype=torch.long, device=values.device)
    chunks = values.reshape(values.shape[0], -1, 8).index_select(-1, order)
    packed = torch.zeros(chunks.shape[:-1], dtype=torch.int32, device=values.device)
    for index in range(8):
        packed.bitwise_or_((chunks[..., index].to(torch.int32) & 0xF) << (4 * index))
    return packed.contiguous()


def _unpack_awq_columns(packed: torch.Tensor) -> torch.Tensor:
    """Unpack AWQ int32 columns back to logical UINT4 values."""
    values = torch.stack(
        [((packed >> (4 * index)) & 0xF) for index in range(8)], dim=-1
    )
    reverse = torch.tensor(_AWQ_REVERSE_ORDER, dtype=torch.long, device=packed.device)
    return values.index_select(-1, reverse).reshape(packed.shape[0], -1)


class V100W4A16AWQConfig(QuantizationConfig):
    """Online groupwise W4 storage with FP16 activations and SM70 GEMM."""

    requires_fp16_source = True

    def __init__(
        self,
        group_size: int = 128,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        if group_size not in (32, 64, 128):
            raise ValueError("v100_w4a16_awq group_size must be 32, 64, or 128")
        self.group_size = group_size
        self.ignored_layers = tuple(ignored_layers or ())

    @classmethod
    def get_name(cls) -> str:
        return "v100_w4a16_awq"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> V100W4A16AWQConfig:
        return cls(
            group_size=cls.get_from_keys_or(config, ["group_size"], 128),
            ignored_layers=cls.get_from_keys_or(
                config, ["ignored_layers", "modules_to_not_convert"], None
            ),
        )

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if any(pattern in prefix for pattern in self.ignored_layers):
            return UnquantizedLinearMethod()
        return V100W4A16AWQLinearMethod(self)


class V100W4A16AWQLinearMethod(LinearMethodBase):
    """Asymmetric groupwise UINT4 storage and fused SM70 execution."""

    def __init__(self, quant_config: V100W4A16AWQConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        if params_dtype != torch.float16:
            raise ValueError(
                f"v100_w4a16_awq requires FP16 source parameters, got {params_dtype}."
            )
        output_size_per_partition = sum(output_partition_sizes)
        if input_size_per_partition % self.quant_config.group_size:
            raise ValueError(
                "v100_w4a16_awq requires the rank-local input dimension to be "
                f"divisible by group_size={self.quant_config.group_size}, got "
                f"K={input_size_per_partition}."
            )
        if output_size_per_partition % 8:
            raise ValueError(
                "v100_w4a16_awq requires the rank-local output dimension to be "
                f"divisible by 8, got N={output_size_per_partition}."
            )

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.logical_widths = output_partition_sizes
        layer.orig_dtype = params_dtype
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )
        layer.register_parameter("weight", weight)

    @torch.no_grad()
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if layer.weight.dtype == torch.int32 and hasattr(layer, "weight_scale"):
            return
        if layer.weight.dtype != torch.float16:
            raise ValueError(
                "v100_w4a16_awq expected loaded FP16 weights, "
                f"got {layer.weight.dtype}."
            )

        group_size = self.quant_config.group_size
        output_size, input_size = layer.weight.shape
        grouped = layer.weight.float().reshape(
            output_size, input_size // group_size, group_size
        )
        weight_min = grouped.amin(dim=-1)
        weight_max = grouped.amax(dim=-1)
        scales = ((weight_max - weight_min) / 15.0).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        zeros = torch.round(-weight_min / scales).clamp_(0, 15).to(torch.int32)
        quantized = torch.round(grouped / scales.unsqueeze(-1))
        quantized.add_(zeros.unsqueeze(-1)).clamp_(0, 15)
        quantized = quantized.to(torch.uint8)

        # The runtime/checkpoint convention is [K, N // 8], with the AWQ
        # nibble interleave applied within each group of eight output columns.
        logical_qweight = quantized.permute(1, 2, 0).reshape(input_size, output_size)
        qweight = _pack_awq_columns(logical_qweight)
        qzeros = _pack_awq_columns(zeros.transpose(0, 1).contiguous())
        scales = scales.transpose(0, 1).to(torch.float16).contiguous()

        layer.weight = nn.Parameter(qweight, requires_grad=False)
        layer.weight_scale = nn.Parameter(scales, requires_grad=False)
        layer.weight_zero = nn.Parameter(qzeros, requires_grad=False)
        layer.v100_awq_group_size = group_size
        layer._v100_awq_prepared = False
        layer._v100_awq_prepared_in_place = False

        # The text projection's output exceeds fp16 on real prompts. Keep a
        # dense fp32 copy so that one GEMM is not saturated to the fp16 max.
        if str(getattr(layer, "prefix", "")).endswith("condition_proj"):
            layer.register_buffer(
                "_v100_fp32_weight",
                self._dequantize_checkpoint_weight(layer).float().contiguous(),
                persistent=False,
            )

        # With a resident DiT the source tensor is already CUDA-resident, so
        # convert immediately and discard the checkpoint layout.  Offloaded
        # components retain it and lazily cache the converted tensors as
        # buffers; this keeps snapshot-based residency reusable across calls.
        if layer.weight.is_cuda:
            self._prepare_sm70_weight(layer, replace_source=True)

    @staticmethod
    @torch.no_grad()
    def _prepare_sm70_weight(layer: nn.Module, *, replace_source: bool) -> None:
        if layer._v100_awq_prepared:
            return
        ops = _load_sm70_awq_ops()
        if ops is None:
            raise RuntimeError(
                "v100_w4a16_awq requires the SM70 TurboMind extension. Run "
                "`python scripts/build_sm70_turbomind.py` in the serving "
                "environment."
            )
        tm_weight, tm_scales, meta = ops.awq_prepare(
            layer.weight,
            layer.weight_scale,
            layer.weight_zero,
            layer.v100_awq_group_size,
            False,
        )
        layer._v100_awq_k_ld = int(meta[0].item())
        layer._v100_awq_q_ld = int(meta[1].item())
        layer._v100_awq_prepared = True
        layer._v100_awq_prepared_in_place = replace_source
        if replace_source:
            layer.weight = nn.Parameter(tm_weight, requires_grad=False)
            layer.weight_scale = nn.Parameter(tm_scales, requires_grad=False)
            layer.weight_zero = nn.Parameter(
                torch.empty(0, dtype=torch.int32, device=tm_weight.device),
                requires_grad=False,
            )
        else:
            layer.register_buffer("_v100_awq_tm_weight", tm_weight, persistent=False)
            layer.register_buffer("_v100_awq_tm_scales", tm_scales, persistent=False)

    @staticmethod
    def _dequantize_checkpoint_weight(layer: nn.Module) -> torch.Tensor:
        if layer._v100_awq_prepared_in_place:
            raise RuntimeError(
                "The TurboMind-packed weight cannot be dequantized on CPU; "
                "move the layer and its input back to CUDA together."
            )
        qweight = _unpack_awq_columns(layer.weight)
        qzeros = _unpack_awq_columns(layer.weight_zero)
        group_size = layer.v100_awq_group_size
        row_groups = torch.arange(
            qweight.shape[0], device=qweight.device, dtype=torch.long
        ).div_(group_size, rounding_mode="floor")
        weight = (
            qweight.to(torch.float32) - qzeros.index_select(0, row_groups).float()
        ) * layer.weight_scale.index_select(0, row_groups).float()
        return weight.transpose(0, 1).to(torch.float16).contiguous()

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if layer.weight.dtype != torch.int32 or not hasattr(layer, "weight_scale"):
            raise RuntimeError(
                "v100_w4a16_awq weights were used before post-load quantization"
            )
        fp32_weight = getattr(layer, "_v100_fp32_weight", None)
        if fp32_weight is not None:
            return F.linear(
                x.float(),
                fp32_weight,
                None if bias is None else bias.float(),
            )
        if x.dtype != torch.float16:
            # Values past the fp16 range become inf under a raw cast, and an
            # inf activation turns the packed GEMM into NaNs.
            limit = torch.finfo(torch.float16).max
            x = x.clamp(min=-limit, max=limit).to(dtype=torch.float16)
        if bias is not None and bias.dtype != torch.float16:
            bias = bias.to(torch.float16)

        if not x.is_cuda:
            return F.linear(x, self._dequantize_checkpoint_weight(layer), bias)
        if torch.cuda.get_device_capability(x.device) != (7, 0):
            dequant_weight = self._dequantize_checkpoint_weight(layer)
            return F.linear(x, dequant_weight, bias)

        if not layer._v100_awq_prepared:
            self._prepare_sm70_weight(layer, replace_source=False)
        if layer._v100_awq_prepared_in_place:
            tm_weight = layer.weight
            tm_scales = layer.weight_scale
        else:
            tm_weight = layer._v100_awq_tm_weight
            tm_scales = layer._v100_awq_tm_scales

        input_shape = x.shape
        x_2d = x.reshape(-1, input_shape[-1]).contiguous()
        out = torch.empty(
            (x_2d.shape[0], layer.output_size_per_partition),
            dtype=torch.float16,
            device=x.device,
        )
        ops = _load_sm70_awq_ops()
        assert ops is not None
        ops.awq_gemm_out(
            out,
            x_2d,
            tm_weight,
            tm_scales,
            layer.v100_awq_group_size,
            layer._v100_awq_k_ld,
            layer._v100_awq_q_ld,
            False,
        )
        if bias is not None:
            out.add_(bias)
        # TurboMind accumulates in FP32 but writes FP16. Saturate cast
        # overflows immediately so they cannot become NaNs in packed attention.
        torch.nan_to_num_(out, nan=0.0, posinf=65504.0, neginf=-65504.0)
        return out.view(*input_shape[:-1], layer.output_size_per_partition)


__all__ = ["V100W4A16AWQConfig", "V100W4A16AWQLinearMethod"]
