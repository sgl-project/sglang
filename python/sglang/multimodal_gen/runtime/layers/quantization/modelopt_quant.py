# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/modelopt_quant.py
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_fp8 import ComfyFp8Config
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
    KitchenInt8Config,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    pad_nvfp4_activation_for_cutlass,
    pad_nvfp4_weight,
    slice_nvfp4_output,
)
from sglang.srt.layers.quantization.utils import (
    convert_to_channelwise,
    requantize_with_max_scale,
)
from sglang.srt.layers.utils.common import copy_or_rebind_param
from sglang.srt.utils.common import is_flashinfer_available, round_up

logger = logging.getLogger(__name__)

if is_flashinfer_available():
    import flashinfer
else:
    flashinfer = None


@lru_cache(maxsize=1)
def _get_fp4_quantize_op():
    return current_platform.get_modelopt_fp4_quantize_op()


@lru_cache(maxsize=1)
def _get_fp4_gemm_op():
    return current_platform.get_modelopt_fp4_gemm_op()


def _prepare_nvfp4_weight_bytes(
    weight: torch.Tensor, *, swap_weight_nibbles: bool
) -> torch.Tensor:
    """Normalize serialized NVFP4 bytes before padding for the runtime kernel."""
    if not swap_weight_nibbles:
        return weight.contiguous()
    return ((weight >> 4) | (weight << 4)).contiguous()


def _swizzled_nvfp4_scales_to_linear(scales: torch.Tensor) -> torch.Tensor:
    """Convert FlashInfer/CUTLASS-swizzled FP4 scales back to row-major layout."""
    scale_ndim = scales.ndim
    if scale_ndim == 2:
        scales = scales.unsqueeze(0)
    assert scales.ndim == 3

    B, M, K = scales.shape
    M_padded = round_up(M, 128)
    K_padded = round_up(K, 4)
    if M != M_padded or K != K_padded:
        padded = torch.zeros(
            (B, M_padded, K_padded), dtype=scales.dtype, device=scales.device
        )
        padded[:B, :M, :K] = scales
        scales = padded

    linear = scales.reshape(B, M_padded // 128, K_padded // 4, 32, 4, 4)
    linear = linear.permute(0, 1, 4, 3, 2, 5).contiguous()
    linear = linear.reshape(B, M_padded, K_padded)[:, :M, :K]
    return linear.squeeze(0) if scale_ndim == 2 else linear


def _prepare_nvfp4_swiglu_fusion_weights(
    layer: torch.nn.Module,
    weight: torch.Tensor,
    scales: torch.Tensor,
) -> None:
    from sglang.kernels.ops.quantization.nvfp4_gemm_swiglu_nvfp4_quant import (
        interleave_linear_and_gate,
        swizzle_blockscale_2d,
    )

    weight, weights_padding_cols = pad_nvfp4_weight(weight)
    if weights_padding_cols != 0:
        raise ValueError(
            "Fused NVFP4 SwiGLU does not support K-padded weights; "
            f"got weights_padding_cols={weights_padding_cols}."
        )
    if scales.ndim != 2:
        raise ValueError(
            "Fused NVFP4 SwiGLU expects a 2D weight scale, "
            f"got shape={tuple(scales.shape)}."
        )
    if scales.shape[0] != weight.shape[0]:
        raise ValueError(
            "Fused NVFP4 SwiGLU does not support N padding; "
            f"scale rows={scales.shape[0]} vs weight rows={weight.shape[0]}."
        )
    if weight.shape[0] % 128 != 0:
        raise ValueError(
            f"Fused NVFP4 SwiGLU requires FC1 N % 128 == 0, got N={weight.shape[0]}."
        )

    # FLUX.2 stores [gate; up].  The kernel consumes 64-row groups in
    # [up; gate] order and applies SiLU to the gate half.
    gate_weight, up_weight = weight.chunk(2, dim=0)
    weight_swiglu_interleaved = interleave_linear_and_gate(
        torch.cat((up_weight, gate_weight), dim=0), group_size=64, dim=0
    )
    gate_scale, up_scale = scales.chunk(2, dim=0)
    weight_scale_swiglu_interleaved = swizzle_blockscale_2d(
        interleave_linear_and_gate(
            torch.cat((up_scale, gate_scale), dim=0), group_size=64, dim=0
        )
    )
    for name, value in (
        ("weight_swiglu_interleaved", weight_swiglu_interleaved),
        ("weight_scale_swiglu_interleaved", weight_scale_swiglu_interleaved),
    ):
        value = value.detach()
        existing = layer._buffers.get(name)
        if (
            existing is not None
            and existing.shape == value.shape
            and existing.dtype == value.dtype
            and existing.device == value.device
        ):
            existing.copy_(value)
        elif name in layer._buffers:
            layer._buffers[name] = value
        else:
            layer.register_buffer(name, value, persistent=False)
    layer._swiglu_fusion_ready = True


def _require_flashinfer():
    if flashinfer is None:
        raise RuntimeError(
            "flashinfer is required for the diffusion NVFP4 FlashInfer path."
        )
    return flashinfer


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
            if self.is_layer_excluded(prefix) or self._is_packed_layer_excluded(prefix):
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

    def _is_packed_layer_excluded(self, prefix: str) -> bool:
        proj_name = prefix.rsplit(".", 1)[-1]
        shard_names = self.packed_modules_mapping.get(proj_name)
        if shard_names is None:
            return False

        base_prefix = prefix[: -len(proj_name)]
        shard_exclusions = [
            self.is_layer_excluded(base_prefix + shard_name)
            for shard_name in shard_names
        ]
        if any(shard_exclusions) and not all(shard_exclusions):
            raise ValueError(
                f"Detected some but not all shards of {prefix} are quantized. "
                "All shards of fused layers must have the same precision."
            )
        return all(shard_exclusions)


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
    def from_config(
        cls,
        config: Dict[str, Any],
        ignore_remap: Optional[Dict[str, str]] = None,
    ) -> ModelOptFp8Config:
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

        if ignore_remap and exclude_modules:
            exclude_modules = [ignore_remap.get(p, p) for p in exclude_modules]

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
        swap_weight_nibbles: bool = False,
        checkpoint_weight_scale_layout: str = "linear",
        checkpoint_uses_comfy_quantization: bool = False,
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
        self.swap_weight_nibbles = swap_weight_nibbles
        self.checkpoint_weight_scale_layout = checkpoint_weight_scale_layout
        self.checkpoint_uses_comfy_quantization = checkpoint_uses_comfy_quantization
        self._comfy_int8_config: KitchenInt8Config | None = None
        self._comfy_fp8_config: ComfyFp8Config | None = None

    def set_comfy_layer_markers(self, layer_markers: dict[str, dict[str, Any]]) -> None:
        unsupported = {
            str(marker.get("format")) for marker in layer_markers.values()
        } - {"nvfp4", "int8_tensorwise", "float8_e4m3fn"}
        if unsupported:
            raise ValueError(
                "NVFP4 checkpoints cannot dispatch companion Comfy formats: "
                + ", ".join(sorted(unsupported))
            )
        int8_markers = {
            prefix: marker
            for prefix, marker in layer_markers.items()
            if marker.get("format") == "int8_tensorwise"
        }
        self._comfy_int8_config = (
            KitchenInt8Config(layer_markers=int8_markers) if int8_markers else None
        )
        fp8_markers = {
            prefix: marker
            for prefix, marker in layer_markers.items()
            if marker.get("format") == "float8_e4m3fn"
        }
        self._comfy_fp8_config = ComfyFp8Config(fp8_markers) if fp8_markers else None

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
        swap_weight_nibbles = False

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
            swap_weight_nibbles = config.get(
                "swap_weight_nibbles",
                config.get("checkpoint_uses_packed_qkv", False),
            )
        else:
            # Nested format (hf_quant_config.json)
            try:
                quant_config = cls.get_from_keys(config, ["quantization"])
                quant_method = quant_config["quant_algo"]
                group_size = ModelOptFp4Config.common_group_size(config)
                exclude_modules = quant_config.get("exclude_modules", [])
                swap_weight_nibbles = quant_config.get(
                    "swap_weight_nibbles",
                    config.get(
                        "swap_weight_nibbles",
                        config.get("checkpoint_uses_packed_qkv", False),
                    ),
                )
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
            swap_weight_nibbles=swap_weight_nibbles,
            checkpoint_weight_scale_layout=config.get(
                "checkpoint_weight_scale_layout", "linear"
            ),
            checkpoint_uses_comfy_quantization=config.get(
                "checkpoint_uses_comfy_quantization", False
            ),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if (
            self._comfy_int8_config is not None
            and prefix in self._comfy_int8_config.layer_markers
        ):
            return self._comfy_int8_config.get_quant_method(layer, prefix)
        if (
            self._comfy_fp8_config is not None
            and prefix in self._comfy_fp8_config.layer_markers
        ):
            return self._comfy_fp8_config.get_quant_method(layer, prefix)
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
        # ROCm gfx942 (MI300X/MI325X) uses e4m3fnuz as its native fp8 dtype, so
        # scaled_fp8_quant() produces e4m3fnuz activations. ModelOpt checkpoints
        # ship e4m3fn weights. hipBLASLt rejects the resulting e4m3fn x e4m3fnuz
        # torch._scaled_mm with HIPBLAS_STATUS_NOT_SUPPORTED. Reinterpreting the
        # weights as e4m3fnuz and doubling the static scales is numerically
        # identical and keeps the GEMM on the native fp8 path.
        weight = layer.weight
        if is_fp8_fnuz():
            weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
            )
            copy_or_rebind_param(layer, "weight_scale", weight_scale)
            if input_scale is not None:
                copy_or_rebind_param(layer, "input_scale", input_scale)

        complete_shard_scales = (
            self.cutlass_fp8_supported
            and len(layer.logical_widths) > 1
            and bool(
                torch.all(
                    layer.weight_scale > torch.finfo(torch.float8_e4m3fn).min
                ).item()
            )
        )
        if complete_shard_scales:
            # CUTLASS accepts a scale per output channel. Preserve each
            # checkpoint shard's original FP8 values and scale instead of
            # requantizing all packed shards to the largest scale.
            quantized_weight = weight
            processed_weight_scale = convert_to_channelwise(
                layer.weight_scale, layer.logical_widths
            )
        else:
            processed_weight_scale, quantized_weight = requantize_with_max_scale(
                weight, layer.weight_scale, layer.logical_widths
            )
            if self.cutlass_fp8_supported:
                processed_weight_scale = convert_to_channelwise(
                    processed_weight_scale, layer.logical_widths
                )
        # Preserve the parameter subclass metadata while rebinding to the
        # transposed FP8 view expected by the runtime.
        layer.weight.data = quantized_weight.t().detach()
        layer.weight.requires_grad_(False)
        copy_or_rebind_param(layer, "weight_scale", processed_weight_scale)
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
    """NVFP4 linear method using the selected FP4 GEMM backend."""

    def __init__(self, quant_config: ModelOptFp4Config):
        # the FlashInfer FP4 kernels this method dispatches to are Blackwell-only.
        # without this the load succeeds and the failure surfaces much later as
        # an opaque CUDA error inside the kernel. an undetectable capability is
        # left alone rather than rejected, so this only ever converts a crash
        # into a message and never blocks a card that would have worked.
        capability = current_platform.get_device_capability()
        min_capability = quant_config.get_min_capability()
        if capability is not None and capability.to_int() < min_capability:
            raise RuntimeError(
                f"NVFP4 checkpoints need compute capability "
                f"{min_capability // 10}.{min_capability % 10} or newer "
                f"(Blackwell); this GPU is {capability.as_version_str()}. "
                f"Load an FP8 or BF16 checkpoint instead."
            )
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
        set_weight_attrs(weight_scale_2, {"missing_param_init": "ones"})
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
        set_weight_attrs(weight_scale, {"missing_param_init": "ones"})
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

        w = layer.weight.data
        w_swapped = _prepare_nvfp4_weight_bytes(
            w,
            swap_weight_nibbles=getattr(
                self.quant_config, "swap_weight_nibbles", False
            ),
        )
        scales = layer.weight_scale
        if (
            getattr(self.quant_config, "checkpoint_weight_scale_layout", "linear")
            == "swizzled"
        ):
            scales = _swizzled_nvfp4_scales_to_linear(scales)

        if getattr(layer, "_interleave_for_swiglu_fusion", False):
            # The regular GEMM path swizzles this tensor below.  The fused
            # GEMM+SwiGLU epilogue needs the logical row order so gate/up rows
            # can first be paired and then swizzled as one matrix.
            _prepare_nvfp4_swiglu_fusion_weights(
                layer, w_swapped, scales.detach().clone()
            )

        _, flashinfer_backend = _get_fp4_gemm_op()
        if flashinfer_backend == "trtllm":
            flashinfer_ops = _require_flashinfer()

            weight, _ = pad_nvfp4_weight(w_swapped, n_alignment=128, k_alignment=0)
            if scales.shape[0] != weight.shape[0]:
                pad_n = weight.shape[0] - scales.shape[0]
                scales = torch.nn.functional.pad(scales, (0, 0, 0, pad_n))

            scale_k = scales.shape[1]
            weights_padding_cols = 0
            if scale_k % 4 != 0:
                padded_scale_k = round_up(scale_k, 4)
                pad_scale_k = padded_scale_k - scale_k
                scales = torch.nn.functional.pad(scales, (0, pad_scale_k, 0, 0))
                pad_weight_k = pad_scale_k * 8
                weight = torch.nn.functional.pad(weight, (0, pad_weight_k, 0, 0))
                weights_padding_cols = pad_weight_k

            epilogue_tile_m = 128
            shuffled_scale_shape = scales.shape
            if not weight.is_cuda:
                weight = weight.cuda()
            if scales.device != weight.device:
                scales = scales.to(device=weight.device)
            weight = flashinfer_ops.shuffle_matrix_a(
                weight.view(torch.uint8), epilogue_tile_m
            )
            scales = (
                flashinfer_ops.shuffle_matrix_sf_a(
                    scales.view(torch.uint8), epilogue_tile_m
                )
                .reshape(shuffled_scale_shape)
                .view(torch.float8_e4m3fn)
            )

            layer.weights_padding_cols = weights_padding_cols
            copy_or_rebind_param(layer, "weight", weight)
            copy_or_rebind_param(layer, "weight_scale_interleaved", scales)
            return
        weight, weights_padding_cols = pad_nvfp4_weight(w_swapped)
        layer.weights_padding_cols = weights_padding_cols
        copy_or_rebind_param(layer, "weight", weight)

        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        assert scales.ndim == 3
        B, M, K = scales.shape
        M_padded = round_up(M, 128)
        K_padded = round_up(K, 4)
        padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype)
        padded_scales[:B, :M, :K] = scales

        # Every FP4 GEMM reachable here reads block scales in the 128x4 TMA layout;
        # trtllm, the one backend wanting its own shuffled layout, returned above.
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
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_size = layer.output_size_per_partition
        if getattr(layer, "_accepts_prequantized_fp4", False) and isinstance(x, tuple):
            x_fp4, x_scale_interleaved = x
            output_dtype = layer.params_dtype
            output_shape = [x_fp4.shape[0], output_size]
        else:
            if isinstance(x, tuple):
                raise TypeError(
                    "Prequantized NVFP4 input requires the linear layer to opt in."
                )
            output_dtype = x.dtype
            input_shape = x.shape
            x = x.view(-1, input_shape[-1])
            output_shape = list(input_shape[:-1]) + [output_size]

            fp4_quantize = _get_fp4_quantize_op()
            if fp4_quantize is None:
                raise RuntimeError(
                    "No FP4 quantization kernel available. Install flashinfer."
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
        if fp4_gemm is None:
            raise RuntimeError("No FP4 GEMM kernel available. Install flashinfer.")
        out = fp4_gemm(
            x_fp4,
            w.T,
            x_scale_interleaved,
            w_scale_interleaved.T,
            layer.alpha,
            output_dtype,
            backend=flashinfer_backend,
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


def apply_nvfp4_gemm_prequantized(
    layer: torch.nn.Module,
    x_fp4: torch.Tensor,
    x_scale_interleaved: torch.Tensor,
    output_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a ModelOpt NVFP4 GEMM from already packed activations and scales."""
    weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

    w = layer.weight
    w_scale_interleaved = layer.weight_scale_interleaved
    if x_scale_interleaved.dtype == torch.uint8:
        x_scale_interleaved = x_scale_interleaved.view(torch.float8_e4m3fn)
    if w_scale_interleaved.dtype == torch.uint8:
        w_scale_interleaved = w_scale_interleaved.view(torch.float8_e4m3fn)

    fp4_gemm, flashinfer_backend = _get_fp4_gemm_op()
    if fp4_gemm is None:
        raise RuntimeError("No FP4 GEMM kernel available. Install flashinfer.")
    out = fp4_gemm(
        x_fp4,
        w.T,
        x_scale_interleaved,
        w_scale_interleaved.T,
        layer.alpha,
        output_dtype,
        backend=flashinfer_backend,
    )
    out = slice_nvfp4_output(out, layer.output_size_per_partition)
    return out + bias if bias is not None else out


def apply_nvfp4_gemm_swiglu_quant(
    linear_in: torch.nn.Module,
    linear_out: torch.nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """Run FLUX.2 FC1 GEMM + SwiGLU + FC2-input NVFP4 quantization."""
    if not getattr(linear_in, "_swiglu_fusion_ready", False):
        raise RuntimeError("NVFP4 SwiGLU weights were not prepared for fusion.")
    if not getattr(linear_out, "_accepts_prequantized_fp4", False):
        raise RuntimeError("NVFP4 output projection did not opt into packed input.")

    fp4_quantize = _get_fp4_quantize_op()
    if fp4_quantize is None:
        raise RuntimeError("No FP4 quantization kernel available. Install flashinfer.")

    from sglang.kernels.ops.quantization.nvfp4_gemm_swiglu_nvfp4_quant import (
        nvfp4_gemm_swiglu_nvfp4_quant,
    )

    input_shape = x.shape
    x_2d = x.view(-1, input_shape[-1])
    x_fp4, x_scale_interleaved = fp4_quantize(x_2d, linear_in.input_scale_inv)
    if x_scale_interleaved.dtype == torch.uint8:
        x_scale_interleaved = x_scale_interleaved.view(torch.float8_e4m3fn)

    out_fp4, out_scale_interleaved = nvfp4_gemm_swiglu_nvfp4_quant(
        x_fp4,
        x_scale_interleaved,
        linear_in.weight_swiglu_interleaved,
        linear_in.weight_scale_swiglu_interleaved,
        linear_in.alpha,
        linear_out.input_scale_inv,
        enable_pdl=True,
    )
    out, _ = linear_out((out_fp4, out_scale_interleaved))
    return out.view(*input_shape[:-1], out.shape[-1])
