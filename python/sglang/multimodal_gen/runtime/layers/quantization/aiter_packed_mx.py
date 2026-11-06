# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for AITER's packed-blob MX GEMMs (A6W6, A6W4).

Both kernels take operands that a packer has already flattened into an opaque
uint8 blob plus packed E8M0 scales, laid out for the kernel's 256-row / 128-K
tiling. The blob carries no shape, so N and K travel to the GEMM as ints, and
the GEMM slices its 256-padded output, so each custom op makes the result dense
again before it returns.
Both are ROCm gfx95x (MI350+) only and emit bf16, so a layer whose parameters
are not bf16 keeps the unquantized linear.
"""

import logging
from typing import Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
    apply_unquantized_linear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.utils import is_gfx95_supported, is_hip

logger = logging.getLogger(__name__)


def aiter_import_error(module_name: str) -> Optional[Exception]:
    """Import one AITER GEMM module, returning the failure instead of raising.

    These modules load whenever any quantization method is resolved, so an
    aiter build without a given kernel must not break the other ROCm paths.
    """
    if not is_hip():
        return ImportError("not a ROCm build")
    try:
        __import__(module_name)
    except Exception as error:  # noqa: BLE001 - reported at layer construction
        logger.warning(f"aiter module {module_name} not available: {error}")
        return error
    return None


class PackedMXConfig(QuantizationConfig):
    """Online quantization from an unquantized BF16 checkpoint.

    There is no serialized form for these formats: the packed blob is tied to
    the kernel's tiling, so a checkpoint cannot carry it portably.
    """

    method_name: str = ""

    def __init__(
        self,
        ignored_layers: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
    ):
        super().__init__()
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}

    @classmethod
    def get_name(cls) -> str:
        return cls.method_name

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 95  # gfx95x, Note: is_gfx95_supported() is a better check

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "PackedMXConfig":
        raise ValueError(
            f"{cls.method_name} has no serialized checkpoint format; select it "
            f"on a BF16 checkpoint with --quantization {cls.method_name}"
        )

    def linear_method_cls(self) -> type["PackedMXLinearMethod"]:
        raise NotImplementedError

    def get_quant_method(self, layer, prefix: str):
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        name = self.method_name.upper()
        if not isinstance(layer, LinearBase):
            logger.debug(f"{name}: Skipping layer {prefix} (not a LinearBase)")
            return None
        if is_layer_skipped(
            prefix,
            self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            logger.debug(f"{name}: Keeping layer {prefix} unquantized (ignored_layers)")
            return UnquantizedLinearMethod()
        logger.debug(f"{name}: Replacing layer {prefix} with {name} linear method")
        return self.linear_method_cls()(self)


class PackedMXLinearMethod(LinearMethodBase):
    """Quantizes BF16 weights into the kernel's packed blob after loading."""

    format_name: str = ""

    def _quantize_weight(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack one [N, K] bf16 weight into (blob, scales) for this kernel."""
        raise NotImplementedError

    def _packed_gemm(
        self,
        x_2d: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_features: int,
        in_features: int,
    ) -> torch.Tensor:
        """Quantize activations and run the GEMM, returning dense bf16 [M, N]."""
        raise NotImplementedError

    def _require_kernels(self) -> None:
        raise NotImplementedError

    def __init__(self, quant_config: PackedMXConfig):
        if not is_gfx95_supported():
            platform = "unknown"
            if is_hip():
                try:
                    platform = torch.cuda.get_device_properties(0).gcnArchName
                except Exception:
                    platform = "ROCm (unknown arch)"
            raise RuntimeError(
                f"{self.format_name} requires ROCm and MI350+ (gfx95x). "
                f"Current platform: {platform}."
            )
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
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        # The packed weight loses its logical shape; the gemm needs both dims.
        layer.packed_mx_out_features = output_size_per_partition
        layer.packed_mx_in_features = input_size_per_partition

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            weight_loader=weight_loader,
            input_dim=1,
            output_dim=0,
        )
        layer.register_parameter("weight", weight)

        # Placeholder scale (will be created during quantization)
        weight_scale = PerTensorScaleParameter(
            data=torch.empty(1, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # These kernels emit bf16 only, so anything else keeps the plain linear.
        layer.packed_mx = layer.weight.dtype == torch.bfloat16
        if not layer.packed_mx:
            logger.info(
                f"{self.format_name}: Keeping a {layer.weight.dtype} layer "
                f"unquantized; the gemm requires bfloat16"
            )
            return

        self._require_kernels()

        weight_data = layer.weight.data
        was_on_cpu = weight_data.device.type == "cpu"
        if was_on_cpu:
            weight_data = weight_data.cuda()

        w_quant, mx_scales = self._quantize_weight(weight_data)

        if was_on_cpu:
            w_quant = w_quant.cpu()
            mx_scales = mx_scales.cpu()

        layer.weight = Parameter(w_quant, requires_grad=False)
        layer.weight_scale = Parameter(mx_scales, requires_grad=False)

        logger.debug(
            f"{self.format_name}: Quantized layer weights - "
            f"weight {layer.weight.shape} {layer.weight.dtype}, "
            f"scale {layer.weight_scale.shape}"
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not layer.packed_mx:
            return apply_unquantized_linear(x, layer.weight, bias)

        out_features = layer.packed_mx_out_features
        original_shape = x.shape

        y = self._packed_gemm(
            x.reshape(-1, original_shape[-1]),
            layer.weight,
            layer.weight_scale,
            out_features,
            layer.packed_mx_in_features,
        )

        if bias is not None:
            y = y + bias

        return y.reshape(*original_shape[:-1], out_features)


__all__ = ["PackedMXConfig", "PackedMXLinearMethod", "aiter_import_error"]
