# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/modelopt_quant.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

import sglang.multimodal_gen.envs as envs
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.models.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.srt.layers.quantization.utils import is_layer_skipped, swizzle_blockscale
from sglang.srt.utils.custom_op import register_custom_op

try:
    if current_platform.is_sm120():
        from flashinfer import fp4_quantize
    else:
        from sgl_kernel import scaled_fp4_quant as fp4_quantize

except ImportError:
    fp4_quantize = None

try:
    from flashinfer import mm_fp4 as flashinfer_fp4_gemm

    enable_flashinfer_fp4_gemm = True
except ImportError:
    if current_platform.is_cuda():
        from sgl_kernel import cutlass_scaled_fp4_mm as cutlass_fp4_gemm
    enable_flashinfer_fp4_gemm = False


# Initialize logger for the module
logger = logging.getLogger(__name__)


def _sglang_fp4_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> torch.Tensor:
    M = input.shape[-2]
    N = int(out_features)
    return input.new_empty((M, N), dtype=out_dtype)


@register_custom_op(fake_impl=_sglang_fp4_gemm_fake)
def fp4_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out_features: int,
) -> torch.Tensor:
    backend = FLASHINFER_FP4_GEMM_BACKEND if FLASHINFER_FP4_GEMM_BACKEND else "cutlass"
    if enable_flashinfer_fp4_gemm:
        return flashinfer_fp4_gemm(
            input, weight, input_sf, weight_sf, alpha, out_dtype, backend=backend
        )
    else:
        return cutlass_fp4_gemm(input, weight, input_sf, weight_sf, alpha, out_dtype)


# Comment out as it can't re-register the same sgl_kernel::scaled_fp4_quant
# if (
#     current_platform.is_cuda()
#     and (not current_platform.is_sm120())
#     and (fp4_quantize is not None)
# ):

#     @register_fake_if_exists("sgl_kernel::scaled_fp4_quant")
#     def _sgl_kernel_scaled_fp4_quant_fake(
#         output, input, output_scale, input_global_scale
#     ):
#         return


FLASHINFER_FP4_GEMM_BACKEND = envs.SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND
# Supported activation schemes for the current configuration
ACTIVATION_SCHEMES = ["static"]


class ModelOptQuantConfig(QuantizationConfig):
    def __init__(
        self,
        exclude_modules: Optional[List[str]],
        packed_modules_mapping: Optional[Dict[str, List[str]]],
    ):
        super().__init__()
        self.packed_modules_mapping = packed_modules_mapping
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
            if is_layer_skipped(
                prefix, self.exclude_modules, self.packed_modules_mapping
            ) or self.is_layer_excluded(prefix):
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
        """Shared ModelOpt quantization method override logic."""
        if hf_quant_config is None:
            return None

        # Check if this is a ModelOpt config
        quant_algo = hf_quant_config.get("quant_algo", "").upper()

        # If user specified generic "modelopt", auto-detect the specific method
        if user_quant == "modelopt":
            if not ("NVFP4" in quant_algo or "FP4" in quant_algo):
                logger.warning(
                    f"Unsupported quant_algo '{quant_algo}' for user_quant 'modelopt'. Using the default 'modelopt_fp4' quant_algo."
                )

            # The hf_quant_config may be a parsed quant config, so we need to check the
            # quant_method.if hf_quant_config.get("quant_method", "") == "modelopt_fp4":
            return "modelopt_fp4"

        return None


class ModelOptFp4Config(ModelOptQuantConfig):
    """Config class for FP4."""

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool = False,
        group_size: int = None,
        exclude_modules: List[str] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__(exclude_modules, packed_modules_mapping)
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected nvfp4 checkpoint. Please note that the "
                "format is experimental and subject to change."
            )
        self.group_size = group_size

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
        # Handle two different config formats:
        # 1. hf_quant_config.json format: {"quantization": {"quant_algo": "NVFP4", ...}}
        # 2. config.json quantization_config format: {"quant_algo": "NVFP4", ...}
        # In future modelopt will deprecate hf_quant_config.json, and only keep config.json.
        # For legacy reasons, we keep hf_quant_config.json for now.

        # Initialize variables
        group_size = None
        exclude_modules = []

        # Try flat format first (config.json quantization_config - preferred format)
        quant_method = config.get("quant_algo")
        if quant_method is not None:
            group_size = config.get("group_size")
            # If group_size is not at top level, try to extract from config_groups
            if group_size is None:
                config_groups = config.get("config_groups", {})
                if config_groups:
                    # Get group_size from the first group's weights config
                    first_group = next(iter(config_groups.values()), {})
                    weights_config = first_group.get("weights", {})
                    group_size = weights_config.get("group_size")

            exclude_modules = config.get("ignore", [])
        else:
            # Fall back to nested format (hf_quant_config.json - legacy format)
            try:
                quant_config = cls.get_from_keys(config, ["quantization"])
                quant_method = quant_config["quant_algo"]
                group_size = ModelOptFp4Config.common_group_size(config)
                exclude_modules = quant_config.get("exclude_modules", [])
            except (ValueError, KeyError):
                raise ValueError(
                    "Cannot find 'quant_algo' in the model's quantization config. "
                    "Expected either flat format (config.json) or nested format (hf_quant_config.json)."
                )

        if not quant_method in ["NVFP4"]:
            raise ValueError(
                f"ModelOpt currently only supports: NVFP4 quantization in sglang diffusion. The provided quant_algo is {quant_method}. Please check the "
                "quantization config for your model's configuration."
            )
        is_checkpoint_nvfp4_serialized = "NVFP4" in quant_method

        if group_size is None or exclude_modules is None:
            logger.warning(
                f"group_size: {group_size}," f"exclude_modules: {exclude_modules}"
            )
            raise ValueError(
                "NVFP4 quantization requires group_size and exclude_modules "
                "specified in the quantization config"
            )
        return cls(
            is_checkpoint_nvfp4_serialized,
            group_size,
            exclude_modules,
            config.get("packed_modules_mapping"),
        )

    def is_layer_excluded(self, prefix: str):
        import regex as re

        fused_patterns = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"]
        prefix_split = prefix.split(".")
        for pattern in self.exclude_modules:
            regex_str = pattern.replace(".", r"\.").replace("*", r".*")
            pattern_split = pattern.split(".")
            if re.fullmatch(regex_str, prefix):
                return True
            elif (
                pattern_split[-1] in fused_patterns
                and pattern_split[-1] in prefix_split[-1]
            ):
                # Check if the last part of the excluded pattern is contained in the last part of the prefix
                # This handles fused modules like fused_qkv_a_proj_with_mqa that contain q_a_proj and kv_a_proj_with_mqa
                # e.g., model.layers.{i}.self_attn.{fused_weight_name}
                assert len(prefix_split) == 5 and len(pattern_split) == 5
                return True
        return False

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return self._get_quant_method(
            layer,
            prefix,
            Linear=ModelOptFp4LinearMethod,
        )


class ModelOptFp4LinearMethod(LinearMethodBase):
    """Linear method for NVFP4.
    Supports loading NVFP4 checkpoints with the following structure:

    |Tensor Name           | datatype      |  shape      |
    |----------------------------------------------------|
    |input_scale           | torch.float32 | scalar      |
    |weight                | NVFP4(SE2M1)  | [1, X, y/2] |
    |weight_scale          | FP8-E4M3      | [X, Y]      |
    |weight_scale_2        | torch.float32 | scalar      |

    The weights are quantized per block of 16 elements.
    Args: quant_config: The ModelOpt quantization config.
    """

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

        weight = ModelWeightParameter(
            data=torch.empty(
                # 2 fp4 data is packed in one uint8 in the input dimension
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
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        input_scale_2 = layer.input_scale.max().to(torch.float32)
        weight_scale_2 = layer.weight_scale_2.max().to(torch.float32)
        layer.input_scale = Parameter(input_scale_2, requires_grad=False)
        layer.weight_scale_2 = Parameter(weight_scale_2, requires_grad=False)
        layer.alpha = Parameter(
            layer.input_scale * layer.weight_scale_2, requires_grad=False
        )
        layer.input_scale_inv = Parameter(
            (1 / input_scale_2).to(torch.float32), requires_grad=False
        )

        if FLASHINFER_FP4_GEMM_BACKEND != "trtllm":
            # Pad and blockwise interleave weight_scale
            padded_scales = swizzle_blockscale(layer.weight_scale)
            layer.weight_scale_interleaved = Parameter(
                padded_scales, requires_grad=False
            )
        else:
            # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
            # FlashInfer provides nvfp4_quantize to quantize + shuffle the
            # layout but we use our own quantization so we have to call
            # shuffles ourselves.
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = layer.weight
            scale = layer.weight_scale
            epilogue_tile_m = 128
            weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)
            scale = (
                shuffle_matrix_sf_a(scale.view(torch.uint8), epilogue_tile_m)
                .reshape(scale.shape)
                .view(torch.float8_e4m3fn)
            )

            layer.weight_scale_interleaved = Parameter(scale, requires_grad=False)
            layer.weight = Parameter(weight, requires_grad=False)
            return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_dtype = x.dtype
        x_m, _ = x.shape
        w_n, _ = layer.weight.shape
        output_shape = [x_m, w_n]

        # Quantize BF16 or FP16 to (FP4 and interleaved block scale)
        x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)

        assert x_fp4.dtype == torch.uint8
        assert layer.weight.dtype == torch.uint8
        assert layer.weight_scale_interleaved.dtype == torch.float8_e4m3fn
        assert layer.alpha.dtype == torch.float32

        w = layer.weight
        w_scale_interleaved = layer.weight_scale_interleaved
        if enable_flashinfer_fp4_gemm:
            w = layer.weight.T
            w_scale_interleaved = layer.weight_scale_interleaved.T
        out = fp4_gemm(
            x_fp4,
            w,
            x_scale_interleaved,
            w_scale_interleaved,
            layer.alpha,
            output_dtype,
            w_n,
        )
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
