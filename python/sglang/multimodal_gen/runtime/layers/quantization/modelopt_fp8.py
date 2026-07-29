"""ModelOpt FP8 quantization support for diffusion models.

Handles checkpoints produced by NVIDIA Model Optimizer (ModelOpt) with
``quant_algo: "FP8"`` and ``quant_method: "modelopt"``.

Per quantized linear layer the checkpoint contains:
    .weight         float8_e4m3fn  [out, in]   FP8 quantized weight
    .weight_scale   float32        scalar       per-tensor weight scale
    .input_scale    float32        scalar       per-tensor static activation scale
    .bias           bfloat16       [out]        bias (unquantized)
    ._amax          (ignored)                   calibration artifact

Layers listed in the ``ignore`` field of the quantization config remain in
bfloat16 and use the standard unquantized linear method.
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Any, Dict, List, Optional

import torch

from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
)
from sglang.srt.layers.quantization.utils import convert_to_channelwise

logger = logging.getLogger(__name__)


class ModelOptFp8Config(QuantizationConfig):
    """Config for ModelOpt static per-tensor FP8 quantization."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = True,
        ignore: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.ignore = ignore or []

    # -- QuantizationConfig interface ----------------------------------------

    @classmethod
    def get_name(cls) -> str:
        return "modelopt"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelOptFp8Config":
        quant_algo = config.get("quant_algo")
        if quant_algo is None:
            raise ValueError(
                "ModelOptFp8Config requires 'quant_algo' in the quantization config."
            )
        if "FP8" not in quant_algo:
            raise ValueError(
                f"ModelOptFp8Config only supports FP8, got quant_algo={quant_algo!r}."
            )
        ignore = config.get("ignore", [])
        return cls(is_checkpoint_fp8_serialized=True, ignore=ignore)

    def _is_layer_ignored(self, prefix: str) -> bool:
        """Check whether *prefix* matches any pattern in the ignore list.

        ModelOpt ignore patterns are matched against the full prefix as a glob
        (e.g. ``"norm_out*"`` matches ``"norm_out.linear"``) **and** against the
        first path component (e.g. ``"proj_out"`` matches only the top-level
        ``proj_out``, not ``single_transformer_blocks.0.proj_out``).
        """
        first_component = prefix.split(".")[0]
        for pattern in self.ignore:
            if fnmatch.fnmatch(prefix, pattern):
                return True
            if fnmatch.fnmatch(first_component, pattern):
                return True
        return False

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            if self._is_layer_ignored(prefix):
                return UnquantizedLinearMethod()
            return ModelOptFp8LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []


class ModelOptFp8LinearMethod(LinearMethodBase):
    """Linear method for ModelOpt static per-tensor FP8 quantization.

    Uses ``torch._scaled_mm`` (or CUTLASS FP8 GEMM when available) for
    the FP8 matrix multiply — the same kernels used by the LLM runtime.
    """

    def __init__(self, quant_config: ModelOptFp8Config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

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
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        for scale_name in ("weight_scale", "input_scale"):
            scale = PerTensorScaleParameter(
                data=torch.full(
                    (len(output_partition_sizes),),
                    torch.finfo(torch.float32).min,
                    dtype=torch.float32,
                ),
                weight_loader=weight_loader,
            )
            layer.register_parameter(scale_name, scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Diffusion models use single-partition layers (no TP, no fused QKV),
        # so we just take the max scale directly without the
        # dequantize-requantize round-trip that the LLM path does (which
        # requires CUDA kernels that are unavailable during CPU-phase loading).
        max_w_scale = layer.weight_scale.max()

        # Transpose weight to [in, out] column-major layout for
        # apply_fp8_linear / CUTLASS fp8_scaled_mm.  Do NOT call
        # .contiguous() — the kernel requires column-major stride.
        layer.weight = torch.nn.Parameter(layer.weight.data.t(), requires_grad=False)

        if self.cutlass_fp8_supported:
            max_w_scale = convert_to_channelwise(max_w_scale, layer.logical_widths)
        layer.weight_scale = torch.nn.Parameter(max_w_scale, requires_grad=False)
        layer.input_scale = torch.nn.Parameter(
            layer.input_scale.max(), requires_grad=False
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
        )
