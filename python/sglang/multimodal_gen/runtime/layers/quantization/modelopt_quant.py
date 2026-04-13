# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/modelopt_quant.py
from __future__ import annotations

import logging
import re
from functools import lru_cache
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
from sglang.multimodal_gen.runtime.models.utils import set_weight_attrs
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight,
    slice_nvfp4_output,
)
from sglang.srt.layers.quantization.utils import (
    convert_to_channelwise,
    is_layer_skipped,
    requantize_with_max_scale,
)
from sglang.srt.layers.utils.common import copy_or_rebind_param
from sglang.srt.utils.common import round_up

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_fp4_quantize_op():
    return current_platform.get_modelopt_fp4_quantize_op()


@lru_cache(maxsize=1)
def _get_fp4_gemm_op():
    return current_platform.get_modelopt_fp4_gemm_op()


class ModelOptQuantConfig(QuantizationConfig):
    def __init__(
        self,
        exclude_modules: Optional[List[str]],
        packed_modules_mapping: Optional[Dict[str, List[str]]],
    ):
        super().__init__()
        self.packed_modules_mapping = packed_modules_mapping or {}
        self.exclude_modules = exclude_modules or []

    def _get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        *,
        Linear: type[LinearMethodBase],
    ) -> Optional[QuantizeMethodBase]:
        from sglang.multimodal_gen.runtime.layers.linear import LinearBase

        if isinstance(layer, LinearBase):
            if self.is_layer_excluded(prefix) or (
                self.packed_modules_mapping
                and is_layer_skipped(prefix, [], self.packed_modules_mapping)
            ):
                return UnquantizedLinearMethod()
            return Linear(self)
        return None

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def override_quantization_method(cls, hf_quant_config, user_quant) -> Optional[str]:
        if hf_quant_config is None:
            return None

        quant_algo = (
            hf_quant_config.get("quant_algo")
            or hf_quant_config.get("quantization", {}).get("quant_algo")
            or ""
        ).upper()
        if user_quant in {"modelopt", "modelopt_fp8"} and "FP8" in quant_algo:
            return "modelopt_fp8"
        if user_quant in {"modelopt", "modelopt_fp4"} and (
            "NVFP4" in quant_algo or "FP4" in quant_algo
        ):
            return "modelopt_fp4"
        return None

    def is_layer_excluded(self, prefix: str) -> bool:
        for pattern in self.exclude_modules:
            regex_str = re.escape(pattern).replace(r"\*", r".*")
            if re.fullmatch(regex_str, prefix):
                return True
        return False


class ModelOptFp8Config(ModelOptQuantConfig):
    """Config class for ModelOpt FP8 diffusion checkpoints."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        exclude_modules: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__(exclude_modules, packed_modules_mapping)
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning(
                "Detected ModelOpt FP8 checkpoint. The format is experimental and subject to change."
            )

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelOptFp8Config":
        quant_method = config.get("quant_algo")
        exclude_modules = config.get("ignore")
        if quant_method is None:
            try:
                quantization_section = cls.get_from_keys(config, ["quantization"])
                quant_method = quantization_section.get("quant_algo")
                exclude_modules = quantization_section.get("exclude_modules")
            except ValueError as exc:
                raise ValueError(
                    "Cannot find 'quant_algo' in the model's quantization config."
                ) from exc

        if quant_method is None or "FP8" not in quant_method:
            raise ValueError(
                "ModelOptFp8Config only supports static FP8 quantization in SGLang diffusion."
            )

        return cls(
            is_checkpoint_fp8_serialized=True,
            exclude_modules=exclude_modules,
            packed_modules_mapping=config.get("packed_modules_mapping"),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return self._get_quant_method(layer, prefix, Linear=ModelOptFp8LinearMethod)


class ModelOptFp4Config(ModelOptQuantConfig):
    """Config class for NVFP4."""

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool = False,
        group_size: int = None,
        exclude_modules: List[str] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        checkpoint_uses_packed_qkv: bool = False,
    ) -> None:
        super().__init__(exclude_modules, packed_modules_mapping)
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected nvfp4 checkpoint. Please note that the "
                "format is experimental and subject to change."
            )
        self.group_size = group_size
        self.checkpoint_uses_packed_qkv = checkpoint_uses_packed_qkv

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_fp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @staticmethod
    def common_group_size(cfg: dict) -> int:
        """Return the unique group_size across the config; raise if missing/mismatched."""
        sizes = set()

        def _add_group_size_from_dict(config: dict):
            group_size = config.get("group_size")
            if isinstance(group_size, int):
                sizes.add(group_size)

        # Top-level and 'quantization' block
        _add_group_size_from_dict(cfg)
        quantization = cfg.get("quantization")
        if isinstance(quantization, dict):
            _add_group_size_from_dict(quantization)

        # config_groups: accept group-level or nested dicts (e.g., weights/input_activations)
        for config_groups in (cfg.get("config_groups") or {}).values():
            if isinstance(config_groups, dict):
                _add_group_size_from_dict(config_groups)
                for config_group in config_groups.values():
                    if isinstance(config_group, dict):
                        _add_group_size_from_dict(config_group)

        if not sizes:
            raise ValueError("No group_size found in config.")
        if len(sizes) > 1:
            raise ValueError(f"Inconsistent group_size values: {sorted(sizes)}")
        return next(iter(sizes))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ModelOptFp4Config:
        group_size = None
        exclude_modules = []

        # Flat format (config.json quantization_config)
        quant_method = config.get("quant_algo")
        if quant_method is not None:
            group_size = config.get("group_size")
            if group_size is None:
                config_groups = config.get("config_groups", {})
                if config_groups:
                    first_group = next(iter(config_groups.values()), {})
                    group_size = first_group.get("weights", {}).get("group_size")
            exclude_modules = config.get("ignore", [])
        else:
            # Nested format (hf_quant_config.json)
            try:
                quant_config = cls.get_from_keys(config, ["quantization"])
                quant_method = quant_config["quant_algo"]
                group_size = ModelOptFp4Config.common_group_size(config)
                exclude_modules = quant_config.get("exclude_modules", [])
            except (ValueError, KeyError):
                raise ValueError("Cannot find 'quant_algo' in quantization config.")

        if quant_method not in ["NVFP4"]:
            raise ValueError(
                f"Only NVFP4 quantization is supported for diffusion, got '{quant_method}'."
            )

        if group_size is None or exclude_modules is None:
            raise ValueError(
                "NVFP4 quantization requires group_size and exclude_modules "
                "in the quantization config"
            )
        return cls(
            is_checkpoint_nvfp4_serialized=True,
            group_size=group_size,
            exclude_modules=exclude_modules,
            packed_modules_mapping=config.get("packed_modules_mapping"),
            checkpoint_uses_packed_qkv=config.get("checkpoint_uses_packed_qkv", False),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return self._get_quant_method(layer, prefix, Linear=ModelOptFp4LinearMethod)


class ModelOptFp8LinearMethod(LinearMethodBase):
    """Linear method for ModelOpt static FP8 checkpoints."""

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
    ):
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
            else params_dtype
        )
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition,
                    dtype=weight_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )

        if self.quant_config.is_checkpoint_fp8_serialized:
            for scale_name in ["weight_scale", "input_scale"]:
                layer.register_parameter(
                    scale_name,
                    PerTensorScaleParameter(
                        data=torch.full(
                            (len(output_partition_sizes),),
                            torch.finfo(torch.float32).min,
                            dtype=torch.float32,
                        ),
                        weight_loader=weight_loader,
                    ),
                )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        max_w_scale, quantized_weight = requantize_with_max_scale(
            layer.weight, layer.weight_scale, layer.logical_widths
        )
        # Preserve the parameter subclass metadata while rebinding to the
        # transposed FP8 view expected by the runtime.
        layer.weight.data = quantized_weight.t().detach()
        layer.weight.requires_grad_(False)
        if self.cutlass_fp8_supported:
            max_w_scale = convert_to_channelwise(max_w_scale, layer.logical_widths)
        copy_or_rebind_param(layer, "weight_scale", max_w_scale)
        copy_or_rebind_param(layer, "input_scale", layer.input_scale.max())

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


class ModelOptFp4LinearMethod(LinearMethodBase):
    """NVFP4 linear method using CUTLASS FP4 GEMM."""

    def __init__(self, quant_config: ModelOptFp4Config):
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
        if not self.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError(
                "NVFP4 quantization was selected, "
                " dynamic quantization is not supported."
            )
        if input_size_per_partition % 16 != 0:
            raise ValueError(
                f"Unsupported model when input features size is {input_size_per_partition}, not multiple of 16, for NVFP4 quantization."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_nvfp4_serialized
            else params_dtype
        )

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

        input_scale = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        set_weight_attrs(input_scale, {"missing_param_init": "ones"})
        layer.register_parameter("input_scale", input_scale)

        weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale_2", weight_scale_2)

        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        input_scale_2 = layer.input_scale.max().to(torch.float32)
        weight_scale_2 = layer.weight_scale_2.max().to(torch.float32)

        copy_or_rebind_param(
            layer, "alpha", (input_scale_2 * weight_scale_2).to(torch.float32)
        )
        copy_or_rebind_param(
            layer, "input_scale_inv", (1 / input_scale_2).to(torch.float32)
        )

        layer.output_size_per_partition = layer.weight.shape[0]

        # Swap nibbles: (byte >> 4) | (byte << 4).
        w = layer.weight.data
        w_swapped = ((w >> 4) | (w << 4)).contiguous()
        weight, weights_padding_cols = pad_nvfp4_weight(w_swapped)
        layer.weights_padding_cols = weights_padding_cols
        copy_or_rebind_param(layer, "weight", weight)

        scales = layer.weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        assert scales.ndim == 3
        B, M, K = scales.shape
        M_padded = round_up(M, 128)
        K_padded = round_up(K, 4)
        padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype)
        padded_scales[:B, :M, :K] = scales
        # Blockwise interleave for CUTLASS TMA layout required by CUTLASS kernel
        padded_scales = padded_scales.reshape(
            B, M_padded // 128, 4, 32, K_padded // 4, 4
        )
        padded_scales = padded_scales.permute(0, 1, 4, 3, 2, 5)
        padded_scales = padded_scales.contiguous().cuda()
        padded_scales = (
            padded_scales.reshape(M_padded, K_padded)
            if scale_ndim == 2
            else padded_scales.reshape(B, M_padded, K_padded)
        )
        copy_or_rebind_param(layer, "weight_scale_interleaved", padded_scales)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_dtype = x.dtype
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])

        output_size = layer.output_size_per_partition
        output_shape = list(input_shape[:-1]) + [output_size]

        fp4_quantize = _get_fp4_quantize_op()
        if fp4_quantize is None:
            raise RuntimeError(
                "No FP4 quantization kernel available. Install flashinfer or sgl_kernel."
            )

        x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)
        weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
        x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

        w = layer.weight
        w_scale_interleaved = layer.weight_scale_interleaved

        if x_scale_interleaved.dtype == torch.uint8:
            x_scale_interleaved = x_scale_interleaved.view(torch.float8_e4m3fn)
        if w_scale_interleaved.dtype == torch.uint8:
            w_scale_interleaved = w_scale_interleaved.view(torch.float8_e4m3fn)
        fp4_gemm, flashinfer_backend = _get_fp4_gemm_op()
        if flashinfer_backend is not None:
            out = fp4_gemm(
                x_fp4,
                w.T,
                x_scale_interleaved,
                w_scale_interleaved.T,
                layer.alpha,
                output_dtype,
                backend=flashinfer_backend,
            )
        elif fp4_gemm is not None:
            out = fp4_gemm(
                x_fp4,
                w,
                x_scale_interleaved,
                w_scale_interleaved,
                layer.alpha,
                output_dtype,
            )
        else:
            raise RuntimeError(
                "No FP4 GEMM kernel available. Install flashinfer or sgl_kernel."
            )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
