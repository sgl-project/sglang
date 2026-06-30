# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import regex as re
import torch
from torch.nn.parameter import Parameter

from sglang.srt.environ import envs
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import (
    is_flashinfer_cutedsl_v1_path,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.parameter import ModelWeightParameter, PerTensorScaleParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp4_utils import (
    fp4_quantize,
    get_fp4_gemm_runner_backend,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    apply_fp8_linear_bmm_flashinfer,
    cutlass_fp8_supported,
    is_blackwell_supported,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_moe_nvfp4_layer_for_marlin,
    prepare_nvfp4_layer_for_marlin,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import (
    convert_to_channelwise,
    is_layer_skipped,
    per_tensor_dequantize,
    requantize_with_max_scale,
    swizzle_blockscale,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils import alias_or_bind_derived_param, copy_or_rebind_param
from sglang.srt.utils.common import (
    get_device_capability,
    is_cuda,
    is_flashinfer_available,
    is_sm100_supported,
    is_sm120_supported,
    round_up,
)
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.utils.patch_torch import register_fake_if_exists

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.models.utils import WeightsMapper

try:
    from flashinfer import mm_fp4 as flashinfer_fp4_gemm
    from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_sf_a

    enable_flashinfer_fp4_gemm = True
except ImportError:
    enable_flashinfer_fp4_gemm = False
    reorder_rows_for_gated_act_gemm = None
    shuffle_matrix_a = None
    shuffle_matrix_sf_a = None

if is_cuda():
    try:
        from sglang.jit_kernel.nvfp4 import cutlass_scaled_fp4_mm as cutlass_fp4_gemm
    except ImportError:
        cutlass_fp4_gemm = None
else:
    cutlass_fp4_gemm = None

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
    fp4_backend = get_fp4_gemm_runner_backend()
    if fp4_backend.is_cutlass() and cutlass_fp4_gemm is not None:
        # flashinfer.fp4_quantize returns scale factors as uint8 (e4m3fn bits
        # stored in uint8 memory). The JIT kernel requires float8_e4m3fn dtype.
        if input_sf.dtype != torch.float8_e4m3fn:
            input_sf = input_sf.view(torch.float8_e4m3fn)
        if weight_sf.dtype != torch.float8_e4m3fn:
            weight_sf = weight_sf.view(torch.float8_e4m3fn)
        return cutlass_fp4_gemm(input, weight, input_sf, weight_sf, alpha, out_dtype)
    elif enable_flashinfer_fp4_gemm:
        # Use the remapping logic to convert SGLang backend names to FlashInfer API names
        backend = fp4_backend.get_flashinfer_backend()
        return flashinfer_fp4_gemm(
            input, weight, input_sf, weight_sf, alpha, out_dtype, backend=backend
        )
    else:
        return cutlass_fp4_gemm(input, weight, input_sf, weight_sf, alpha, out_dtype)


if is_cuda() and (not is_sm120_supported()) and (fp4_quantize is not None):

    @register_fake_if_exists("sgl_kernel::scaled_fp4_quant")
    def _sgl_kernel_scaled_fp4_quant_fake(
        output, input, output_scale, input_global_scale
    ):
        return


# FP4 GEMM alignment constant - CUTLASS/FlashInfer kernels require dimensions divisible by 32
FP4_GEMM_ALIGNMENT = 32


def round_up_to_multiple(x: int, m: int) -> int:
    """Round up x to the nearest multiple of m."""
    return (x + m - 1) // m * m


def pad_nvfp4_weight(
    weight: torch.Tensor,
    n_alignment: int = FP4_GEMM_ALIGNMENT,
    k_alignment: int = FP4_GEMM_ALIGNMENT,
) -> tuple[torch.Tensor, int]:
    """
    Pad packed NVFP4 weights to satisfy alignment constraints for FP4 GEMM kernels.

    Different backends have different alignment requirements:
    - CUTLASS/cuDNN: N % 32 == 0, K % 32 == 0
    - TRTLLM: N % 128 == 0 (for shuffle_matrix_sf_a), K padding handled separately

    Args:
        weight: Packed FP4 weight tensor of shape [N, K//2] (2 FP4 values per byte)
        n_alignment: Required alignment for N dimension (default 32, use 128 for TRTLLM)
        k_alignment: Required alignment for K dimension (default 32, use 0 to skip)

    Returns:
        Tuple of (padded_weight, weights_padding_cols) where weights_padding_cols
        is the number of columns added for K-dimension padding (in bytes).
    """
    weight_current_rows = weight.shape[0]  # N dimension
    weight_current_col_bytes = weight.shape[1]  # K//2 (packed)

    # Calculate padding for N dimension (rows)
    pad_rows = 0
    if n_alignment > 0 and weight_current_rows % n_alignment != 0:
        total_rows = round_up_to_multiple(weight_current_rows, n_alignment)
        pad_rows = total_rows - weight_current_rows

    # Calculate padding for K dimension (columns)
    # 2 FP4 items are packed per byte in the input dimension
    weight_current_col_elements = weight_current_col_bytes * 2
    pad_cols_bytes = 0
    if k_alignment > 0 and weight_current_col_elements % k_alignment != 0:
        total_cols = round_up_to_multiple(weight_current_col_elements, k_alignment)
        pad_cols = total_cols - weight_current_col_elements
        # pad_cols is in elements, but padding is in bytes (2 elements per byte)
        pad_cols_bytes = pad_cols // 2

    # Apply padding in a single operation if needed
    # For 2D tensor, pad argument is (pad_left, pad_right, pad_top, pad_bottom)
    if pad_rows > 0 or pad_cols_bytes > 0:
        weight = torch.nn.functional.pad(
            weight, (0, pad_cols_bytes, 0, pad_rows)
        ).contiguous()

    return weight, pad_cols_bytes


def pad_nvfp4_activation_for_cutlass(
    x_fp4: torch.Tensor,
    weights_padding_cols: int,
) -> torch.Tensor:
    """
    Pad packed FP4 activations to match the K-dimension padding applied to weights.

    Args:
        x_fp4: Packed FP4 activation tensor
        weights_padding_cols: Number of padding columns (in bytes) from weight padding

    Returns:
        Padded activation tensor
    """
    if weights_padding_cols > 0:
        return torch.nn.functional.pad(x_fp4, (0, weights_padding_cols)).contiguous()
    return x_fp4


def slice_nvfp4_output(
    out: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """
    Slice the output tensor to remove padding in N dimension if weight was padded.

    Args:
        out: Output tensor from FP4 GEMM
        output_size: Original output size before padding

    Returns:
        Sliced output tensor with padding removed
    """
    if out.shape[-1] != output_size:
        return out[..., :output_size].contiguous()
    return out


# TODO make it true by default when the DeepEP PR is merged
MOE_NVFP4_DISPATCH = envs.SGLANG_MOE_NVFP4_DISPATCH.get()
# Supported activation schemes for the current configuration
ACTIVATION_SCHEMES = ["static"]


_SUPPORTED_ACT_STRS = ("silu", "relu2", "gelu")


class ModelOptQuantConfig(QuantizationConfig):
    def __init__(
        self,
        kv_cache_quant_algo: Optional[str],
        exclude_modules: Optional[List[str]],
        packed_modules_mapping: Optional[Dict[str, List[str]]],
    ):
        super().__init__()
        self.packed_modules_mapping = packed_modules_mapping
        self.exclude_modules = exclude_modules or []
        self.kv_cache_quant_algo = kv_cache_quant_algo
        self.use_per_token_activation = False

    def _get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        *,
        Linear: type[LinearMethodBase],
        Moe: type[FusedMoEMethodBase],
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix, self.exclude_modules, self.packed_modules_mapping
            ) or self.is_layer_excluded(prefix):
                return UnquantizedLinearMethod()
            return Linear(self)
        elif self.kv_cache_quant_algo and isinstance(layer, RadixAttention):
            return ModelOptFp8KVCacheMethod(self)
        elif isinstance(layer, FusedMoE):
            # Check if MoE layer should be excluded from quantization
            # (e.g., MTP layers that have no quantization scales in checkpoint)
            if self.is_layer_excluded(prefix):
                # Falls back to default unquantized MoE
                return None
            return Moe(self)
        return None

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    def get_scaled_act_names(self) -> List[str]:
        return []

    def apply_weight_name_mapper(
        self, hf_to_sglang_mapper: WeightsMapper
    ):  # noqa: B027
        # Map excluded module patterns from HF layout to sglang layout.
        # Ref: HF hf_quant_config.json for nvidia/Kimi-K2.5-NVFP4
        # https://huggingface.co/nvidia/Kimi-K2.5-NVFP4/blob/main/hf_quant_config.json
        if self.exclude_modules:
            mapped = hf_to_sglang_mapper.apply_list(self.exclude_modules)
            expanded: List[str] = []
            for name in mapped:
                expanded.append(name)
                if name.startswith("language_model."):
                    expanded.append(name.removeprefix("language_model."))
            # Preserve order, drop duplicates.
            self.exclude_modules = list(dict.fromkeys(expanded))

    def is_layer_excluded(self, prefix: str) -> bool:
        """Check if a layer should be excluded from quantization.

        Handles:
        - Exact matches (e.g., "lm_head" matching prefix "lm_head")
        - Glob-style wildcards (e.g., "mtp*" matching "mtp_layers")
        - Part-by-part matching (split prefix on "." and check each part)
        - language_model. prefix stripping for vision-language models
        - Fused module patterns (e.g., "q_a_proj" in "fused_qkv_a_proj_with_mqa")
        """
        if not self.exclude_modules:
            return False

        # Build prefix variants: some models wrap layers under "language_model."
        prefixes_to_check = [prefix]
        if prefix.startswith("language_model."):
            prefixes_to_check.append(prefix.removeprefix("language_model."))

        # Fused module patterns: the exclude list may reference a sub-component
        # (e.g., "q_a_proj") that is fused into a combined parameter name
        # (e.g., "fused_qkv_a_proj_with_mqa"). We check if the last segment of
        # the exclude pattern is a substring of the last segment of the prefix.
        fused_patterns = {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"}

        for pattern in self.exclude_modules:
            # Convert glob-style wildcard to regex (e.g., "mtp*" -> "mtp.*")
            regex_str = pattern.replace(".", r"\.").replace("*", r".*")

            for pfx in prefixes_to_check:
                if re.fullmatch(regex_str, pfx):
                    return True
                # Part-by-part check: handles wildcards like "mtp*" matching
                pfx_parts = pfx.split(".")
                for part in pfx_parts:
                    if re.fullmatch(regex_str, part):
                        return True

            # Check fused patterns: if the last segment of the exclude pattern
            # is a known fused component, check if it appears in the prefix's
            # last segment (handles fused_qkv_a_proj_with_mqa containing q_a_proj)
            pattern_tail = pattern.rsplit(".", maxsplit=1)[-1]
            if pattern_tail in fused_patterns:
                for pfx in prefixes_to_check:
                    if pattern_tail in pfx.rsplit(".", maxsplit=1)[-1]:
                        return True

        return False


class ModelOptFp8Config(ModelOptQuantConfig):
    """Configuration for ModelOpt FP8 quantization, including serialization and compatibility checks."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        kv_cache_quant_method: Optional[str] = None,
        exclude_modules: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Args:
            is_checkpoint_fp8_serialized (bool): Indicates if the checkpoint uses serialized FP8 format.
        """
        super().__init__(kv_cache_quant_method, exclude_modules, packed_modules_mapping)
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.warning(
                "Detected ModelOpt FP8 checkpoint. The format is experimental and subject to change."
            )

    @classmethod
    def override_quantization_method(cls, hf_quant_config, user_quant):
        """Override quantization method based on the model's config."""
        return cls._modelopt_override_quantization_method(hf_quant_config, user_quant)

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89  # Minimum hardware capability (e.g., Hopper GPUs).

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ModelOptFp8Config:
        # Handle two different config formats:
        # 1. hf_quant_config.json format: {"quantization": {"quant_algo": "FP8", ...}}
        # 2. config.json quantization_config format: {"quant_algo": "FP8", ...}
        # In future modelopt will deprecate hf_quant_config.json, and only keep config.json.
        # For legacy reasons, we keep hf_quant_config.json for now.

        # Initialize variables
        kv_cache_quant_method = None
        exclude_modules = None

        # Try flat format first (config.json quantization_config - preferred format)
        quant_method = config.get("quant_algo")
        if quant_method is not None:
            # Flat format (config.json quantization_config)
            # Derive kv_cache quant from kv_cache_scheme dict
            kv_cache_scheme = config.get("kv_cache_scheme")
            if isinstance(kv_cache_scheme, dict):
                if (
                    kv_cache_scheme.get("type") == "float"
                    and kv_cache_scheme.get("num_bits") == 8
                ):
                    kv_cache_quant_method = "FP8"

            # Map 'ignore' field to 'exclude_modules'
            exclude_modules = config.get("ignore")
        else:
            # Fall back to nested format (hf_quant_config.json - will be deprecated)
            try:
                quantization_section = cls.get_from_keys(config, ["quantization"])
                quant_method = quantization_section.get("quant_algo")
                kv_cache_quant_method = quantization_section.get("kv_cache_quant_algo")
                exclude_modules = quantization_section.get("exclude_modules")
            except ValueError:
                raise ValueError(
                    "Cannot find 'quant_algo' in the model's quantization config. "
                    "Expected either flat format (config.json) or nested format (hf_quant_config.json)."
                )
        if quant_method is None:
            raise ValueError(
                "Cannot find 'quant_algo' in the model's quantization config. "
            )
        if "FP8" not in quant_method:
            raise ValueError(
                "ModelOptFp8Config only supports static FP8 quantization in SGLang. "
                "For FP4 quantization, use ModelOptFp4Config. "
                "Check the quantization config for your model's configuration."
            )

        return cls(
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            packed_modules_mapping=config.get("packed_modules_mapping"),
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        return self._get_quant_method(
            layer, prefix, Linear=ModelOptFp8LinearMethod, Moe=ModelOptFp8MoEMethod
        )


class ModelOptFp8LinearMethod(LinearMethodBase):
    """Linear method for ModelOpt static FP8 quantization.

    Supports loading FP8 checkpoints with static weight and activation scales.
    Future support may include dynamic scales.

    **Limitations**:
    1. Only supports per-tensor quantization due to `torch._scaled_mm` limitations.
    2. Only supports the `float8_e4m3fn` data type.

    Args:
        quant_config (ModelOptFp8Config): The ModelOpt quantization configuration.
    """

    def __init__(self, quant_config: ModelOptFp8Config):
        super().__init__()
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()
        self.enable_flashinfer_bmm = is_sm100_supported() and is_flashinfer_available()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: Optional[int],
        output_size: Optional[int],
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Creates and registers weights, weight scales, and input scales for FP8 quantization."""
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
            else params_dtype
        )

        # Set layer attributes
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # Register weight
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
            # Register weight and input scales
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
        """Requantizes weights after loading using the maximum scale."""
        max_w_scale, quantized_weight = requantize_with_max_scale(
            layer.weight, layer.weight_scale, layer.logical_widths
        )
        layer.weight = Parameter(quantized_weight.t(), requires_grad=False)
        if self.cutlass_fp8_supported and not self.enable_flashinfer_bmm:
            max_w_scale = convert_to_channelwise(max_w_scale, layer.logical_widths)
        layer.weight_scale = Parameter(max_w_scale, requires_grad=False)
        layer.input_scale = Parameter(layer.input_scale.max(), requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies FP8 linear transformation."""
        if self.enable_flashinfer_bmm and layer.input_scale is not None:
            return apply_fp8_linear_bmm_flashinfer(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                input_scale=layer.input_scale,
                bias=bias,
            )
        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
        )


class ModelOptFp8KVCacheMethod(BaseKVCacheMethod):
    """
    Handles loading FP8 kv-cache scaling factors from modelopt quantized checkpoints.
    """

    def __init__(self, quant_config: ModelOptFp8Config):
        super().__init__(quant_config)


class ModelOptMixedPrecisionConfig(ModelOptQuantConfig):
    """Configuration for ModelOpt MIXED_PRECISION checkpoints."""

    def __init__(
        self,
        kv_cache_quant_algo: Optional[str],
        exclude_modules: Optional[List[str]],
        packed_modules_mapping: Optional[Dict[str, List[str]]],
        quantized_layers: Dict[str, Dict[str, Any]],
        fp8_config: ModelOptFp8Config,
        nvfp4_config: ModelOptFp4Config,
    ) -> None:
        super().__init__(kv_cache_quant_algo, exclude_modules, packed_modules_mapping)
        self.quantized_layers = quantized_layers
        self.fp8_config = fp8_config
        self.nvfp4_config = nvfp4_config

    @classmethod
    def override_quantization_method(cls, hf_quant_config, user_quant):
        if hf_quant_config is None:
            return None
        if hf_quant_config.get("quant_method", "") == "modelopt_mixed":
            return "modelopt_mixed"
        return None

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_mixed"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return ModelOptFp4Config.get_min_capability()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ModelOptMixedPrecisionConfig:
        kv_cache_quant_algo = None
        exclude_modules = None
        quantized_layers = {}

        quant_algo = config.get("quant_algo")
        if quant_algo is not None:
            kv_cache_scheme = config.get("kv_cache_scheme")
            if isinstance(kv_cache_scheme, dict):
                if (
                    kv_cache_scheme.get("type") == "float"
                    and kv_cache_scheme.get("num_bits") == 8
                ):
                    kv_cache_quant_algo = "FP8"
                elif (
                    kv_cache_scheme.get("type") == "float"
                    and kv_cache_scheme.get("num_bits") == 4
                ):
                    kv_cache_quant_algo = "NVFP4"
                else:
                    kv_cache_quant_algo = "auto"
            exclude_modules = config.get("ignore")
            quantized_layers = config.get("quantized_layers", {})
        else:
            quantization_section = cls.get_from_keys(config, ["quantization"])
            quant_algo = quantization_section.get("quant_algo")
            kv_cache_quant_algo = quantization_section.get("kv_cache_quant_algo")
            exclude_modules = quantization_section.get("exclude_modules")
            quantized_layers = quantization_section.get("quantized_layers", {})

        if quant_algo != "MIXED_PRECISION":
            raise ValueError(
                "ModelOptMixedPrecisionConfig only supports MIXED_PRECISION checkpoints."
            )
        if not quantized_layers:
            raise ValueError(
                "MIXED_PRECISION quantization requires a non-empty quantized_layers map."
            )

        group_size = None
        for layer_info in quantized_layers.values():
            if layer_info.get("quant_algo", "").upper() == "NVFP4":
                group_size = layer_info.get("group_size", 16)
                break
        if group_size is None:
            group_size = 16

        packed_modules_mapping = config.get("packed_modules_mapping")
        fp8_config = ModelOptFp8Config(
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=kv_cache_quant_algo,
            exclude_modules=[],
            packed_modules_mapping=packed_modules_mapping,
        )
        nvfp4_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=kv_cache_quant_algo,
            exclude_modules=[],
            packed_modules_mapping=packed_modules_mapping,
            group_size=group_size,
        )

        return cls(
            kv_cache_quant_algo=kv_cache_quant_algo,
            exclude_modules=exclude_modules,
            packed_modules_mapping=packed_modules_mapping,
            quantized_layers=quantized_layers,
            fp8_config=fp8_config,
            nvfp4_config=nvfp4_config,
        )

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: WeightsMapper):
        super().apply_weight_name_mapper(hf_to_sglang_mapper)
        if self.quantized_layers:
            self.quantized_layers = hf_to_sglang_mapper.apply_dict(
                self.quantized_layers
            )

    def _resolve_quant_algo(self, prefix: str) -> Optional[str]:
        if prefix in self.quantized_layers:
            return self.quantized_layers[prefix]["quant_algo"].upper()

        proj_name = prefix.rsplit(".", 1)[-1]
        if self.packed_modules_mapping and proj_name in self.packed_modules_mapping:
            algos = set()
            base = prefix.rsplit(".", 1)[0]
            for shard_name in self.packed_modules_mapping[proj_name]:
                shard_prefix = f"{base}.{shard_name}"
                if shard_prefix in self.quantized_layers:
                    algos.add(self.quantized_layers[shard_prefix]["quant_algo"].upper())
            if len(algos) == 1:
                return algos.pop()
            if len(algos) > 1:
                raise ValueError(
                    f"Mixed quant_algo within fused layer {prefix}: {algos}. "
                    "All shards must use the same quantization."
                )

        prefix_dot = prefix + "."
        for key, info in self.quantized_layers.items():
            if key.startswith(prefix_dot):
                return info["quant_algo"].upper()

        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        quant_algo = self._resolve_quant_algo(prefix)

        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix, self.exclude_modules, self.packed_modules_mapping
            ) or self.is_layer_excluded(prefix):
                return UnquantizedLinearMethod()
            if quant_algo == "FP8":
                return ModelOptFp8LinearMethod(self.fp8_config)
            if quant_algo == "NVFP4":
                return ModelOptFp4LinearMethod(self.nvfp4_config)
            return UnquantizedLinearMethod()

        if self.kv_cache_quant_algo and isinstance(layer, RadixAttention):
            return ModelOptFp8KVCacheMethod(self.fp8_config)

        if isinstance(layer, FusedMoE):
            if self.is_layer_excluded(prefix):
                return None
            if quant_algo == "FP8":
                return ModelOptFp8MoEMethod(self.fp8_config)
            if quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoEMethod(self.nvfp4_config)
            return None

        return None


class ModelOptFp8MoEMethod(FusedMoEMethodBase):
    """MoE method for ModelOpt FP8.
    Supports loading FP8 checkpoints with static weight scale and activation scale.

    Args:
        quant_config: The ModelOpt quantization config.
    """

    def __init__(self, quant_config: ModelOptFp8Config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        # Use FP8 dtype if checkpoint is serialized, otherwise use the default dtype
        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
            else params_dtype
        )
        weight_loader = extra_weight_attrs.get("weight_loader")
        num_shards = 2 if layer.moe_runner_config.is_gated else 1
        intermediate_size = num_shards * intermediate_size_per_partition
        w13_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                intermediate_size,
                hidden_size,
                dtype=weight_dtype,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight", w13_weight)

        w2_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=weight_dtype,
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight", w2_weight)

        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALES - Per-tensor scaling for ModelOpts
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_scale_shape = (num_experts, num_shards)
            w13_weight_scale = PerTensorScaleParameter(
                data=torch.full(
                    w13_scale_shape,
                    torch.finfo(torch.float32).min,
                    dtype=torch.float32,
                ),
                weight_loader=weight_loader,
            )
            w2_weight_scale = PerTensorScaleParameter(
                data=torch.full(
                    (num_experts,), torch.finfo(torch.float32).min, dtype=torch.float32
                ),
                weight_loader=weight_loader,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

            # Set weight loader attributes for scales
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
            )

            # INPUT SCALES - Per-tensor scaling for ModelOpt
            w13_input_scale = PerTensorScaleParameter(
                data=torch.full((num_experts,), 1.0, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            w2_input_scale = PerTensorScaleParameter(
                data=torch.full((num_experts,), 1.0, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process FP8 MoE weights after loading from serialized checkpoint.

        Only supports pre-quantized checkpoints with FP8 weights and scales.
        """

        layer.w13_weight = Parameter(layer.w13_weight.data, requires_grad=False)
        layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)

        # Handle scale parameters
        if hasattr(layer, "w13_weight_scale") and layer.w13_weight_scale is not None:
            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max of the w1 and w3 scales then dequant and requant each expert.
            if layer.w13_weight_scale.dim() == 2:  # Shape: (num_experts, 2)
                # Get the maximum scale across w1 and w3 for each expert
                max_w13_scales = layer.w13_weight_scale.max(dim=1).values

                # Requantize each expert's weights using the combined scale
                # w13_weight has shape (num_experts, 2 * intermediate_size_per_partition, hidden_size)
                # where the first intermediate_size_per_partition rows are w1, the next are w3
                num_shards = 2 if layer.moe_runner_config.is_gated else 1
                intermediate_size_per_partition = (
                    layer.w13_weight.shape[1] // num_shards
                )
                for expert_id in range(layer.w13_weight.shape[0]):
                    start = 0
                    for shard_id in range(num_shards):  # (w1 and w3) or w13
                        # Dequantize using the original scale for this shard
                        dq_weight = per_tensor_dequantize(
                            layer.w13_weight[expert_id][
                                start : start + intermediate_size_per_partition, :
                            ],
                            layer.w13_weight_scale[expert_id][shard_id],
                        )
                        # Requantize using the combined max scale
                        (
                            layer.w13_weight[expert_id][
                                start : start + intermediate_size_per_partition, :
                            ],
                            _,
                        ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])

                        start += intermediate_size_per_partition

                # Update the scale parameter to be per-expert instead of per-shard
                layer.w13_weight_scale = Parameter(max_w13_scales, requires_grad=False)
            else:
                layer.w13_weight_scale = Parameter(
                    layer.w13_weight_scale.data, requires_grad=False
                )

        if hasattr(layer, "w2_weight_scale") and layer.w2_weight_scale is not None:
            layer.w2_weight_scale = Parameter(
                layer.w2_weight_scale.data, requires_grad=False
            )
        if hasattr(layer, "w13_input_scale") and layer.w13_input_scale is not None:
            layer.w13_input_scale = Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
        if hasattr(layer, "w2_input_scale") and layer.w2_input_scale is not None:
            layer.w2_input_scale = Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

        # Align FP8 weights to FlashInfer per-tensor kernel layout if enabled
        if get_moe_runner_backend().is_flashinfer_trtllm():
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                align_fp8_moe_weights_for_flashinfer_trtllm,
            )

            # ModelOpt FP8 stores weights in [Up, Gate] order, so we need to swap
            align_fp8_moe_weights_for_flashinfer_trtllm(layer, swap_w13_halves=True)
        elif get_moe_runner_backend().is_flashinfer_cutlass():
            assert (
                hasattr(layer, "w13_input_scale") and layer.w13_input_scale is not None
            )
            assert hasattr(layer, "w2_input_scale") and layer.w2_input_scale is not None
            assert (
                hasattr(layer, "w13_weight_scale")
                and layer.w13_weight_scale is not None
            )
            assert (
                hasattr(layer, "w2_weight_scale") and layer.w2_weight_scale is not None
            )

            input_scale = layer.w13_input_scale.to(torch.float32)
            activation_scale = layer.w2_input_scale.to(torch.float32)
            w13_weight_scale = layer.w13_weight_scale.to(torch.float32)
            w2_weight_scale = layer.w2_weight_scale.to(torch.float32)

            layer.fc1_dequant = Parameter(
                w13_weight_scale * input_scale, requires_grad=False
            )
            layer.fc2_quant = Parameter(
                activation_scale.reciprocal(), requires_grad=False
            )
            layer.fc2_dequant = Parameter(
                activation_scale * w2_weight_scale, requires_grad=False
            )
            layer.fc1_input_dequant = Parameter(input_scale, requires_grad=False)

            # flashinfer_cutlass kernel requires intermediate_size to be a
            # multiple of 16.  Pad weight tensors with zeros after loading.
            # For gated activations (swiglu), w13 is [Up, Gate] concatenated
            # along dim 1 — we must split, pad each half separately, and
            # re-concat so the kernel's half-split stays aligned.
            num_shards = 2 if layer.moe_runner_config.is_gated else 1
            isp = layer.w13_weight.shape[1] // num_shards
            if isp % 16 != 0:
                pad_amount = round_up(isp, 16) - isp
                w13_data = layer.w13_weight.data
                if num_shards == 2:
                    up_weight = w13_data[:, :isp, :]
                    gate_weight = w13_data[:, isp:, :]
                    layer.w13_weight = Parameter(
                        torch.cat(
                            [
                                torch.nn.functional.pad(
                                    up_weight, (0, 0, 0, pad_amount)
                                ),
                                torch.nn.functional.pad(
                                    gate_weight, (0, 0, 0, pad_amount)
                                ),
                            ],
                            dim=1,
                        ),
                        requires_grad=False,
                    )
                else:
                    layer.w13_weight = Parameter(
                        torch.nn.functional.pad(w13_data, (0, 0, 0, pad_amount)),
                        requires_grad=False,
                    )
                layer.w2_weight = Parameter(
                    torch.nn.functional.pad(layer.w2_weight.data, (0, pad_amount)),
                    requires_grad=False,
                )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_flashinfer_cutlass():
            import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass  # noqa: F401

            self.runner = MoeRunner(
                MoeRunnerBackend.FLASHINFER_CUTLASS, moe_runner_config
            )
        else:
            self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        # Fast path: TRT-LLM FP8 per-tensor MoE using BYPASSED TopK routing

        if (
            get_moe_runner_backend().is_flashinfer_trtllm()
            and TopKOutputChecker.format_is_bypassed(topk_output)
        ):
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmFp8MoeQuantInfo,
                fused_experts_none_to_flashinfer_trtllm_fp8,
                get_activation_type,
            )
            from sglang.srt.layers.moe.utils import RoutingMethodType

            _SUPPORTED_FP8_ACTIVATIONS = {"silu", "relu2"}
            assert self.moe_runner_config.activation in _SUPPORTED_FP8_ACTIVATIONS, (
                f"Only {_SUPPORTED_FP8_ACTIVATIONS} are supported for "
                f"flashinfer trtllm fp8 moe, got '{self.moe_runner_config.activation}'"
            )

            routing_method_type = getattr(
                layer, "routing_method_type", RoutingMethodType.Llama4
            )

            quant_info = FlashInferTrtllmFp8MoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                local_num_experts=layer.num_local_experts,
                intermediate_size=layer.w2_weight.shape[2],
                routing_method_type=routing_method_type,
                block_quant=False,
                w13_input_scale=layer.w13_input_scale,
                output1_scales_scalar=layer.output1_scales_scalar,
                output1_scales_gate_scalar=layer.output1_scales_gate_scalar,
                output2_scales_scalar=layer.output2_scales_scalar,
                use_routing_scales_on_input=True,
                activation_type=get_activation_type(
                    self.moe_runner_config.activation,
                    is_gated=self.moe_runner_config.is_gated,
                ),
            )

            return fused_experts_none_to_flashinfer_trtllm_fp8(
                dispatch_output, quant_info, self.moe_runner_config
            )

        if get_moe_runner_backend().is_flashinfer_cutlass():
            activation_str = self.moe_runner_config.activation
            assert activation_str in _SUPPORTED_ACT_STRS, (
                f"Activation {activation_str!r} is not supported for "
                f"flashinfer cutlass fp8 moe (supported: {_SUPPORTED_ACT_STRS})."
            )
            from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
                FlashInferCutlassMoeQuantInfo,
            )

            quant_info = FlashInferCutlassMoeQuantInfo(
                quant_type="fp8",
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                quant_scales=[
                    layer.fc1_dequant,
                    layer.fc2_quant,
                    layer.fc2_dequant,
                    layer.fc1_input_dequant,
                ],
                output_dtype=x.dtype,
                moe_ep_size=layer.moe_ep_size,
                moe_ep_rank=layer.moe_ep_rank,
                moe_tp_size=layer.moe_tp_size,
                moe_tp_rank=layer.moe_tp_rank,
                apply_routed_scaling_factor=not layer.should_fuse_routed_scaling_factor_in_topk,
            )
            return self.runner.run(dispatch_output, quant_info)

        quant_info = TritonMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            use_fp8_w8a8=True,
            per_channel_quant=False,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a13_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
        )

        return self.runner.run(dispatch_output, quant_info)


class ModelOptFp4Config(ModelOptQuantConfig):
    """Config class for FP4."""

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool = False,
        kv_cache_quant_algo: str = None,
        group_size: int = None,
        exclude_modules: List[str] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        use_per_token_activation: Optional[bool] = None,
    ) -> None:
        super().__init__(kv_cache_quant_algo, exclude_modules, packed_modules_mapping)
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected nvfp4 checkpoint. Please note that the "
                "format is experimental and subject to change."
            )
        self.group_size = group_size
        self.use_per_token_activation = (
            use_per_token_activation
            or envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.get()
        )

    @classmethod
    def override_quantization_method(cls, hf_quant_config, user_quant):
        """Override quantization method based on the model's config."""
        return cls._modelopt_override_quantization_method(hf_quant_config, user_quant)

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_fp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def common_group_size(cfg: dict) -> int:
        """Return the unique group_size across the config; raise if missing/mismatched."""
        sizes = set()

        # Top-level and 'quantization' block
        v = cfg.get("group_size")
        if isinstance(v, int):
            sizes.add(v)
        q = cfg.get("quantization")
        if isinstance(q, dict):
            v = q.get("group_size")
            if isinstance(v, int):
                sizes.add(v)

        # config_groups: accept group-level or nested dicts (e.g., weights/input_activations)
        for g in (cfg.get("config_groups") or {}).values():
            if isinstance(g, dict):
                v = g.get("group_size")
                if isinstance(v, int):
                    sizes.add(v)
                for sub in g.values():
                    if isinstance(sub, dict):
                        v = sub.get("group_size")
                        if isinstance(v, int):
                            sizes.add(v)

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
        kv_cache_quant_algo = None
        group_size = None
        exclude_modules = []

        # Try flat format first (config.json quantization_config - preferred format)
        quant_method = config.get("quant_algo")
        if quant_method is not None:
            # Flat format (config.json quantization_config)
            # Derive kv_cache_quant_algo from kv_cache_scheme dict
            kv_cache_scheme = config.get("kv_cache_scheme")
            if isinstance(kv_cache_scheme, dict):
                if (
                    kv_cache_scheme.get("type") == "float"
                    and kv_cache_scheme.get("num_bits") == 8
                ):
                    kv_cache_quant_algo = "FP8"
                else:
                    kv_cache_quant_algo = "auto"
            elif isinstance(kv_cache_scheme, str):
                scheme_name = kv_cache_scheme.strip().upper()
                if scheme_name in ("FP8", "FLOAT8"):
                    kv_cache_quant_algo = "FP8"
                elif scheme_name in ("FP4", "FLOAT4", "NVFP4"):
                    kv_cache_quant_algo = "NVFP4"
                else:
                    kv_cache_quant_algo = "auto"
            else:
                kv_cache_quant_algo = "auto"

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
                kv_cache_quant_algo = quant_config.get("kv_cache_quant_algo")
                if not kv_cache_quant_algo:
                    kv_cache_quant_algo = "auto"
                group_size = ModelOptFp4Config.common_group_size(config)
                exclude_modules = quant_config.get("exclude_modules", [])
            except (ValueError, KeyError):
                raise ValueError(
                    "Cannot find 'quant_algo' in the model's quantization config. "
                    "Expected either flat format (config.json) or nested format (hf_quant_config.json)."
                )

        if quant_method not in ["FP8", "NVFP4"]:
            raise ValueError(
                "ModelOpt currently only supports: FP8, NVFP4"
                " quantizations in sglang. Please check the "
                "quantization config for your model's configuration."
            )
        is_checkpoint_nvfp4_serialized = "NVFP4" in quant_method

        if group_size is None or exclude_modules is None:
            logger.warning(
                f"group_size: {group_size},"
                f"kv_cache_quant_algo: {kv_cache_quant_algo},"
                f"exclude_modules: {exclude_modules}"
            )
            raise ValueError(
                "NVFP4 quantization requires group_size and exclude_modules "
                "specified in the quantization config"
            )
        return cls(
            is_checkpoint_nvfp4_serialized,
            kv_cache_quant_algo,
            group_size,
            exclude_modules,
            config.get("packed_modules_mapping"),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        return self._get_quant_method(
            layer,
            prefix,
            Linear=ModelOptFp4LinearMethod,
            Moe=ModelOptNvFp4FusedMoEMethod,
        )


class HybridFp8NvFp4Config(Fp8Config):
    """FP8 (linear/attention/MTP MoE) + NVFP4 (FusedMoE) hybrid quantization.

    For checkpoints like nvidia/DeepSeek-V4-Pro-NVFP4 where
    config.json:quantization_config declares quant_method=fp8 and
    moe_quant_algo=NVFP4. FusedMoE layers route through
    ModelOptNvFp4FusedMoEMethod; linear / attention layers
    delegate to the inherited Fp8Config dispatch.
    """

    def __init__(self, fp8_config: Fp8Config, nvfp4_config: ModelOptFp4Config):
        # Inherit all of fp8_config's state without re-running its
        # validation / logging (already happened at fp8_config build time).
        self.__dict__.update(fp8_config.__dict__)
        self.nvfp4_config = nvfp4_config

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            if not self.nvfp4_config.is_layer_excluded(prefix):
                return ModelOptNvFp4FusedMoEMethod(self.nvfp4_config)
            # Fall back to MXFP4 for MTP MoE layers
            if self.is_fp4_experts:
                from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
                from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
                    Mxfp4FlashinferTrtllmMoEMethod,
                )

                return Mxfp4FlashinferTrtllmMoEMethod(Fp8MoEMethod(self), prefix=prefix)
        return super().get_quant_method(layer, prefix)

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: WeightsMapper):
        super().apply_weight_name_mapper(hf_to_sglang_mapper)
        self.nvfp4_config.apply_weight_name_mapper(hf_to_sglang_mapper)


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

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype
        layer.quant_config = self.quant_config
        if input_size_per_partition % 16 != 0:
            raise ValueError(
                "Unsupported model when in features size is not multiple of 16"
            )

        weight_dtype = (
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_nvfp4_serialized
            else params_dtype
        )

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

        # alpha / input_scale_inv stay as scalar Parameters. Aliasing them into
        # the [N_partitions] source slot breaks fused-QKV linears whose
        # downstream kernels assume scalar input scale.
        copy_or_rebind_param(
            layer, "alpha", (input_scale_2 * weight_scale_2).to(torch.float32)
        )
        copy_or_rebind_param(
            layer, "input_scale_inv", (1 / input_scale_2).to(torch.float32)
        )

        # Store original output size before any padding
        layer.output_size_per_partition = layer.weight.shape[0]

        if get_fp4_gemm_runner_backend().is_marlin():
            if self.quant_config.group_size != 16:
                raise ValueError(
                    f"NVFP4 Marlin requires group_size=16, got {self.quant_config.group_size}."
                )
            copy_or_rebind_param(layer, "input_global_scale", input_scale_2)
            copy_or_rebind_param(layer, "weight_global_scale", weight_scale_2)
            prepare_nvfp4_layer_for_marlin(layer)
            layer.weights_padding_cols = 0
            return

        if not is_blackwell_supported():
            raise ValueError(
                "ModelOpt NVFP4 native dense GEMM backends require SM100+. "
                "Use --fp4-gemm-backend marlin on SM80-SM90."
            )

        if get_fp4_gemm_runner_backend().is_flashinfer_trtllm():
            # FlashInfer TRTLLM FP4 GEMM requires a different weight layout.
            # FlashInfer provides nvfp4_quantize to quantize + shuffle the
            # layout but we use our own quantization so we have to call
            # shuffles ourselves.
            #
            # Alignment requirements:
            #   - shuffle_matrix_a: weight.shape[0] (N) % 32 == 0
            #   - shuffle_matrix_sf_a: scale.shape[0] (N) % 128 == 0, scale.shape[1] (K/16) % 4 == 0
            # We pad N to multiple of 128 and K/16 to multiple of 4.
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            # Pad weight N dimension to 128
            weight, _ = pad_nvfp4_weight(
                layer.weight.data, n_alignment=128, k_alignment=0
            )
            # Pad scale N dimension to match weight
            scale = layer.weight_scale
            if scale.shape[0] != weight.shape[0]:
                pad_n = weight.shape[0] - scale.shape[0]
                scale = torch.nn.functional.pad(scale, (0, 0, 0, pad_n))

            # Pad K dimension: scale K/16 must be multiple of 4
            scale_k = scale.shape[1]  # K/16
            weights_padding_cols = 0
            if scale_k % 4 != 0:
                padded_scale_k = round_up_to_multiple(scale_k, 4)
                pad_scale_k = padded_scale_k - scale_k
                # Pad scale K/16 dimension
                scale = torch.nn.functional.pad(scale, (0, pad_scale_k, 0, 0))
                # Pad weight K/2 dimension correspondingly (K/2 = K/16 * 8)
                pad_weight_k = pad_scale_k * 8
                weight = torch.nn.functional.pad(weight, (0, pad_weight_k, 0, 0))
                # Store K padding for activation padding in apply()
                weights_padding_cols = pad_weight_k

            # Shuffle for TRTLLM layout
            epilogue_tile_m = 128
            shuffled_scale_shape = scale.shape
            weight = shuffle_matrix_a(weight.view(torch.uint8), epilogue_tile_m)
            scale = (
                shuffle_matrix_sf_a(scale.view(torch.uint8), epilogue_tile_m)
                .reshape(shuffled_scale_shape)
                .view(torch.float8_e4m3fn)
            )

            alias_or_bind_derived_param(
                layer, "weight_scale", "weight_scale_interleaved", scale
            )
            copy_or_rebind_param(layer, "weight", weight)
            layer.weights_padding_cols = weights_padding_cols
            return

        # Pad weights for CUTLASS/FlashInfer kernel alignment (K and N divisible by 32)
        weight, weights_padding_cols = pad_nvfp4_weight(layer.weight.data)
        layer.weights_padding_cols = weights_padding_cols
        copy_or_rebind_param(layer, "weight", weight)

        # Pad and blockwise interleave weight_scale
        scales = layer.weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        assert scales.ndim == 3
        B, M, K = scales.shape
        M_padded = round_up_to_multiple(M, 128)
        K_padded = round_up_to_multiple(K, 4)
        padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype)
        padded_scales[:B, :M, :K] = scales

        # Snapshot the raw (pre-swizzle) scale BEFORE alias_or_bind_derived_param
        # overwrites layer.weight_scale.data in-place via .copy_() on the broadcast
        # path. Without this, the swiglu side-channel below would read the swizzled
        # bytes when it later re-reads layer.weight_scale.
        raw_scale_snapshot = (
            (scales.squeeze(0) if scale_ndim == 2 else scales).detach().clone()
        )

        batches, rows, cols = padded_scales.shape
        assert rows % 128 == 0
        assert cols % 4 == 0
        padded_scales = padded_scales.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
        padded_scales = padded_scales.permute((0, 1, 4, 3, 2, 5))
        padded_scales = padded_scales.contiguous().cuda()
        padded_scales = (
            padded_scales.reshape(M_padded, K_padded)
            if scale_ndim == 2
            else padded_scales.reshape(B, M_padded, K_padded)
        )
        alias_or_bind_derived_param(
            layer, "weight_scale", "weight_scale_interleaved", padded_scales
        )

        if getattr(layer, "_interleave_for_swiglu_fusion", False):
            from sglang.srt.layers.quantization.nvfp4_gemm_swiglu_nvfp4_quant import (
                interleave_linear_and_gate,
                swizzle_blockscale_2d,
            )

            w = layer.weight.data
            assert weights_padding_cols == 0, (
                "_interleave_for_swiglu_fusion does not support K-padded weights; "
                f"got weights_padding_cols={weights_padding_cols}."
            )
            assert raw_scale_snapshot.shape[0] == w.shape[0], (
                "_interleave_for_swiglu_fusion requires no N-padding; "
                f"raw_scale rows={raw_scale_snapshot.shape[0]} vs weight rows={w.shape[0]}."
            )
            assert w.shape[0] % 128 == 0, (
                "_interleave_for_swiglu_fusion requires N % 128 == 0 (group_size=64 "
                f"with gate+up halves); got N={w.shape[0]}."
            )

            gate_w, up_w = w.chunk(2, dim=0)
            w_swiglu = interleave_linear_and_gate(
                torch.cat((up_w, gate_w), dim=0), group_size=64, dim=0
            )

            gate_s, up_s = raw_scale_snapshot.chunk(2, dim=0)
            w_scale_swiglu = swizzle_blockscale_2d(
                interleave_linear_and_gate(
                    torch.cat((up_s, gate_s), dim=0), group_size=64, dim=0
                )
            )

            layer.weight_swiglu_interleaved = w_swiglu
            layer.weight_scale_swiglu_interleaved = w_scale_swiglu

            # Keep the Parameter objects alive so weight reload can refill
            # them and re-run this hook; free their storage in the meantime.
            layer.weight.data = torch.empty(
                0, dtype=layer.weight.dtype, device=layer.weight.device
            )
            layer.weight_scale_interleaved.data = torch.empty(
                0,
                dtype=layer.weight_scale_interleaved.dtype,
                device=layer.weight_scale_interleaved.device,
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_fp4_gemm_runner_backend().is_marlin():
            return apply_fp4_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                weight_global_scale=layer.weight_global_scale,
                workspace=layer.workspace,
                size_n=layer.output_size_per_partition,
                size_k=layer.input_size_per_partition,
                bias=bias,
            )

        # `_accepts_prequantized_fp4` is the explicit opt-in so an accidental
        # tuple from unrelated code can't silently bypass quantization.
        if getattr(layer, "_accepts_prequantized_fp4", False) and isinstance(x, tuple):
            x_fp4, x_scale_interleaved = x
            x_m = x_fp4.shape[0]
            output_dtype = layer.params_dtype
        else:
            x_fp4, x_scale_interleaved = fp4_quantize(x, layer.input_scale_inv)
            x_m, _ = x.shape
            output_dtype = x.dtype

        output_size = layer.output_size_per_partition
        w_n, _ = layer.weight.shape
        output_shape = [x_m, output_size]

        assert x_fp4.dtype == torch.uint8
        assert layer.weight.dtype == torch.uint8
        assert layer.weight_scale_interleaved.dtype == torch.float8_e4m3fn
        assert layer.alpha.dtype == torch.float32

        # Pad activations to match weight K-dimension padding
        weights_padding_cols = getattr(layer, "weights_padding_cols", 0)
        x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_cols)

        w = layer.weight
        w_scale_interleaved = layer.weight_scale_interleaved
        if (
            enable_flashinfer_fp4_gemm
            and not get_fp4_gemm_runner_backend().is_cutlass()
        ):
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

        # Slice output to remove N-dimension padding
        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


def _compute_gemm1_alphas(
    w13_weight_scale_2: torch.Tensor,
    w13_input_scale: torch.Tensor,
    is_gated: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GEMM1 weight x input alphas for the gate (w1) and up (w3) halves of w13.

    w13 fuses the gate and up projections, which may carry separate NVFP4 weight
    scales stored as [num_experts, 2] (col 0 = gate, col 1 = up). A 1-D (or
    [num_experts, 1]) scale, and any non-gated layer, shares one scale across
    both halves; the col-1 read is guarded so those cases stay in bounds.

    Returns (g1_alphas, g1_alphas_up), equal for a shared scale. Single-alpha
    backends use g1_alphas; the TRT-LLM path also uses g1_alphas_up.
    """
    if is_gated and w13_weight_scale_2.dim() == 2 and w13_weight_scale_2.shape[1] >= 2:
        gate_scale = w13_weight_scale_2[:, 0]
        up_scale = w13_weight_scale_2[:, 1]
    else:
        gate_scale = w13_weight_scale_2.reshape(w13_weight_scale_2.shape[0])
        up_scale = gate_scale
    g1_alphas = (w13_input_scale * gate_scale).to(torch.float32)
    g1_alphas_up = (w13_input_scale * up_scale).to(torch.float32)
    return g1_alphas, g1_alphas_up


class ModelOptNvFp4FusedMoEMethod(FusedMoEMethodBase):
    """
       MoE Method for FP4 Quantization with Blockscales and PerTensorScales
    Args:
        quant_config: NVFP4 Quant Config
    """

    def __init__(self, quant_config: ModelOptFp4Config):
        self.quant_config = quant_config
        moe_runner_backend = get_moe_runner_backend()
        if moe_runner_backend.is_auto() and is_cuda():
            capability = get_device_capability()
            use_marlin_fallback = (8, 0) <= capability < (10, 0)
        else:
            use_marlin_fallback = moe_runner_backend.is_marlin()
        if not is_blackwell_supported() and not use_marlin_fallback:
            raise ValueError(
                "Current platform does not support NVFP4"
                " quantization with the selected MoE backend. Please use "
                "Blackwell and above, or use moe_runner_backend=marlin on SM80+."
            )
        self.enable_flashinfer_trtllm_moe = (
            get_moe_runner_backend().is_flashinfer_trtllm()
            or get_moe_runner_backend().is_flashinfer_trtllm_routed()
        )
        self._cache_permute_indices = {}

    @property
    def enable_flashinfer_cutlass_moe(self) -> bool:
        from sglang.srt.layers.moe import get_moe_runner_backend

        """Access the global enable_flashinfer_cutlass_moe setting."""
        return get_moe_runner_backend().is_flashinfer_cutlass()

    @property
    def enable_flashinfer_cutedsl_moe(self) -> bool:
        """Access the global enable_flashinfer_cutedsl_moe setting."""
        from sglang.srt.layers.moe import get_moe_runner_backend

        return get_moe_runner_backend().is_flashinfer_cutedsl()

    # ----- CuteDSL v1 vs v2 path helpers -----
    #
    # "v1": cutedsl + deepep low-latency.
    #   - MoeRunner fused func calls flashinfer_cutedsl_moe_masked
    #     (grouped_gemm_nt_masked).
    #   - Expects W13 in default [Gate, Up] order, NOT interleaved.
    #   - Uses swizzled blockscales directly (w13_blockscale_swizzled).
    #
    # "v2" (standard): cutedsl + none/flashinfer a2a.
    #   - MoeRunner fused func calls CuteDslMoEWrapper kernels.
    #   - Expects W13 in [Up, Gate] order, interleaved in 64-row chunks.
    #   - Uses MMA-layout blockscales (w13_blockscale_mma).

    @property
    def _is_cutedsl_v1_deepep(self) -> bool:
        """CuteDSL v1 + DeepEP low-latency path (masked grouped GEMM)."""
        return is_flashinfer_cutedsl_v1_path()

    @property
    def _is_cutedsl_v2_standard(self) -> bool:
        """CuteDSL v2 standard path (a2a=none or flashinfer, uses CuteDslMoEWrapper)."""
        return self.enable_flashinfer_cutedsl_moe and not self._is_cutedsl_v1_deepep

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        is_nvfp4_online = getattr(self.quant_config, "is_nvfp4_online", False)
        if not self.quant_config.is_checkpoint_nvfp4_serialized and not is_nvfp4_online:
            raise ValueError(
                "NVFP4 quantization was selected, "
                " dynamic quantization is not supported."
            )
        # `nvfp4_online` is not a serialized checkpoint format, but after the
        # online loader converts each expert it uses the same packed NVFP4
        # weights, block scales, and per-tensor scales as serialized ModelOpt
        # NVFP4 checkpoints. Reuse this layout and swap only the weight loader.
        if is_nvfp4_online:
            if not self.enable_flashinfer_trtllm_moe:
                raise ValueError(
                    "--quantization nvfp4_online supports only "
                    "--moe-runner-backend flashinfer_trtllm or "
                    "flashinfer_trtllm_routed."
                )

        # TODO(ch-wan): check if this is needed
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.params_dtype = params_dtype
        layer.quant_config = self.quant_config

        weight_dtype = torch.uint8
        weight_scale_dtype = torch.float8_e4m3fn
        weight_loader = extra_weight_attrs.get("weight_loader")
        if is_nvfp4_online:
            weight_loader = self.get_online_weight_loader(layer, weight_loader)
        # GEMM 1
        num_shards = 2 if layer.moe_runner_config.is_gated else 1

        w13_weight = ModelWeightParameter(
            data=torch.empty(
                layer.num_local_experts,
                num_shards * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight", w13_weight)

        # GEMM 2
        w2_weight = ModelWeightParameter(
            data=torch.empty(
                layer.num_local_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight", w2_weight)

        w13_weight_scale = ModelWeightParameter(
            data=torch.empty(
                layer.num_local_experts,
                num_shards * intermediate_size_per_partition,
                hidden_size // self.quant_config.group_size,
                dtype=weight_scale_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        # TRTLLM replaces blockscale_swizzled with an alias to weight_scale
        # during process_weights_after_loading, so skip the expensive
        # swizzle+allocate here to avoid GPU memory fragmentation
        if self.enable_flashinfer_trtllm_moe:
            layer.w13_blockscale_swizzled = None
        else:
            layer.w13_blockscale_swizzled = Parameter(
                swizzle_blockscale(layer.w13_weight_scale), requires_grad=False
            )

        w2_weight_scale = ModelWeightParameter(
            data=torch.empty(
                layer.num_local_experts,
                hidden_size,
                intermediate_size_per_partition // self.quant_config.group_size,
                dtype=weight_scale_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        if self.enable_flashinfer_trtllm_moe:
            layer.w2_blockscale_swizzled = None
        else:
            layer.w2_blockscale_swizzled = Parameter(
                swizzle_blockscale(layer.w2_weight_scale), requires_grad=False
            )

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )

        w13_weight_scale_shape = (
            (layer.num_local_experts, 2)
            if layer.moe_runner_config.is_gated
            else (layer.num_local_experts,)
        )
        w13_weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(w13_weight_scale_shape, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)

        w2_weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(layer.num_local_experts, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)

        if is_nvfp4_online and self.quant_config.is_checkpoint_fp8_serialized:
            # FP8 checkpoints usually store expert scales as weight_scale_inv.
            # Online NVFP4 consumes them in the loader and writes the generated
            # NVFP4 scales into w*_weight_scale / w*_weight_scale_2 instead.
            w13_source_weight_scale_inv = PerTensorScaleParameter(
                data=torch.empty(0, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter(
                "w13_weight_scale_inv", w13_source_weight_scale_inv
            )
            w2_source_weight_scale_inv = PerTensorScaleParameter(
                data=torch.empty(0, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter("w2_weight_scale_inv", w2_source_weight_scale_inv)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )

        w13_input_scale_shape = (layer.num_experts, num_shards)
        w13_input_scale = PerTensorScaleParameter(
            data=torch.empty(w13_input_scale_shape, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        w13_input_scale._sglang_require_global_experts = True
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = PerTensorScaleParameter(
            data=torch.empty(layer.num_experts, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        w2_input_scale._sglang_require_global_experts = True
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process FP4 MoE weights after loading from serialized checkpoint.

        Only supports pre-quantized checkpoints with FP8 weights and scales.
        """
        # GEMM1 scale processing is deferred until the input scale is known;
        # see _compute_gemm1_alphas, which splits w13's gate/up weight scales.
        moe_runner_backend = getattr(
            self, "_moe_runner_backend", get_moe_runner_backend()
        )
        if moe_runner_backend.is_marlin():
            # Marlin supports only a single shared w1/w3 weight scale, so collapse
            # the gate/up columns to the gate scale here. Other backends keep the
            # raw scale and split the halves later (see _compute_gemm1_alphas).
            if layer.moe_runner_config.is_gated:
                if layer.w13_weight_scale_2.dim() == 1:
                    # Some checkpoints store a shared scale for w1/w3.
                    w13_weight_scale_2 = layer.w13_weight_scale_2
                else:
                    if layer.w13_weight_scale_2.shape[1] >= 2 and not torch.allclose(
                        layer.w13_weight_scale_2[:, 0],
                        layer.w13_weight_scale_2[:, 1],
                    ):
                        logger.warning_once(
                            "w1_weight_scale_2 must match w3_weight_scale_2. "
                            "Accuracy may be affected."
                        )

                    w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0]
            else:
                w13_weight_scale_2 = layer.w13_weight_scale_2[:]
            copy_or_rebind_param(
                layer,
                "w13_weight_scale_2",
                w13_weight_scale_2.contiguous(),
            )
            prepare_moe_nvfp4_layer_for_marlin(layer)
            return

        # Calculate input scales based on strategy
        if self.enable_flashinfer_cutlass_moe or self.enable_flashinfer_trtllm_moe:
            w13_input_scale = layer.w13_input_scale.max().to(torch.float32)
            w2_input_scale = layer.w2_input_scale.max().to(torch.float32)
        elif self.enable_flashinfer_cutedsl_moe:
            # CuteDSL standard path uses a single scalar input scale (all experts).
            w13_input_scale = (
                layer.w13_input_scale.max()
                .to(torch.float32)
                .repeat(layer.w13_input_scale.shape[0])
            )
            w2_input_scale = layer.w2_input_scale

            def _slice_scale(w):
                assert w.shape == (layer.num_experts,)
                assert layer.moe_ep_size * layer.num_local_experts == layer.num_experts
                return w[
                    layer.moe_ep_rank
                    * layer.num_local_experts : (layer.moe_ep_rank + 1)
                    * layer.num_local_experts
                ]

            w13_input_scale = _slice_scale(w13_input_scale)
            w2_input_scale = _slice_scale(w2_input_scale)

            if MOE_NVFP4_DISPATCH:
                assert torch.all(w13_input_scale == w13_input_scale[0])
                w13_input_scale = w13_input_scale[0]
        else:
            w13_input_scale = layer.w13_input_scale.max(dim=-1).values.to(torch.float32)
            w2_input_scale = layer.w2_input_scale

        if self.quant_config.use_per_token_activation:
            # FlashInfer computes activation scales dynamically per token, so
            # the static checkpoint activation scale is intentionally neutral.
            w13_input_scale = torch.ones_like(w13_input_scale, dtype=torch.float32)
            w2_input_scale = torch.ones_like(w2_input_scale, dtype=torch.float32)

        # Create shared parameters. g1_alphas / g1_alphas_up are the gate (w1)
        # and up (w3) GEMM1 scales (equal for shared-scale checkpoints).
        g1_alphas, g1_alphas_up = _compute_gemm1_alphas(
            layer.w13_weight_scale_2,
            w13_input_scale,
            layer.moe_runner_config.is_gated,
        )
        copy_or_rebind_param(layer, "g1_alphas", g1_alphas)
        copy_or_rebind_param(layer, "g1_alphas_up", g1_alphas_up)
        copy_or_rebind_param(
            layer,
            "g2_alphas",
            (w2_input_scale * layer.w2_weight_scale_2).to(torch.float32),
        )
        copy_or_rebind_param(
            layer,
            "w13_input_scale_quant",
            (1 / w13_input_scale).to(torch.float32),
        )
        copy_or_rebind_param(
            layer,
            "w2_input_scale_quant",
            (1 / w2_input_scale).to(torch.float32),
        )

        swiglu_limit = layer.moe_runner_config.swiglu_limit
        if (
            swiglu_limit is not None
            and layer.moe_runner_config.is_gated
            and self.enable_flashinfer_trtllm_moe
        ):
            copy_or_rebind_param(
                layer,
                "gemm1_clamp_limit",
                (swiglu_limit / layer.g1_alphas).to(torch.float32),
            )

        # TODO: for flashinfer always do MOE_NVFP4_DISPATCH
        layer.dispatcher.set_quant_config(
            {
                "input_global_scale": (
                    layer.w13_input_scale_quant
                    if MOE_NVFP4_DISPATCH
                    or should_use_flashinfer_cutlass_moe_fp4_allgather()
                    else None
                )
            }
        )
        block_size = 16
        # Validate weight scales
        assert_dim = 2 if layer.moe_runner_config.is_gated else 1
        for name, weight_scale in [
            ("w13", layer.w13_weight_scale),
            ("w2", layer.w2_weight_scale),
        ]:
            # For NVFP4 TRTLLM we require one scale per 16 inputs (last dim == expected_blocks[name]).
            if get_moe_runner_backend().is_flashinfer_trtllm():
                expected_blocks = {
                    "w13": layer.w13_weight.shape[2] * 2 // block_size,
                    "w2": layer.w2_weight.shape[2] * 2 // block_size,
                }
                assert (
                    weight_scale.shape[-1] == expected_blocks[name]
                ), f"Expected {name}_weight_scale.dim(2) == {expected_blocks[name]}, got {weight_scale.shape[-1]}"
            else:
                if weight_scale.shape[assert_dim] % 4 != 0:
                    logger.warning(
                        "NVFP4 %s_weight_scale K' not multiple of 4: shape=%s, group_size=%s",
                        name,
                        tuple(weight_scale.shape),
                        getattr(self.quant_config, "group_size", None),
                    )
            assert (
                weight_scale.dtype == torch.float8_e4m3fn
            ), f"{name} Weight Blockscale must be represented as FP8-E4M3"

        # Weight processing based on strategy
        if (
            self.enable_flashinfer_trtllm_moe
            and reorder_rows_for_gated_act_gemm is not None
            and shuffle_matrix_sf_a is not None
        ):
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                align_fp4_moe_weights_for_flashinfer_trtllm,
            )

            # FlashInfer TRTLLM processing - handles both w13 and w2
            align_fp4_moe_weights_for_flashinfer_trtllm(layer)
            # TRTLLM doesn't read *_blockscale_swizzled; alias to free the
            # placeholders from create_weights.
            layer.w13_blockscale_swizzled = layer.w13_weight_scale
            layer.w2_blockscale_swizzled = layer.w2_weight_scale

        else:
            # CUTLASS processing - handle w13 and w2 separately

            if self._is_cutedsl_v2_standard and layer.moe_runner_config.is_gated:
                # CuteDSL v2 only: interleave the two logical W13 halves in
                # 64-row chunks for the fused SwiGLU GEMM1 layout expected by
                # CuteDslMoEWrapper.  The v1 (deepep) path uses
                # grouped_gemm_nt_masked which expects plain contiguous halves.
                from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
                    interleave_w13_halves,
                )

                layer.w13_weight = Parameter(
                    interleave_w13_halves(
                        layer.w13_weight.view(torch.uint8), group_size=64, dim=1
                    ).contiguous(),
                    requires_grad=False,
                )
                layer.w13_weight_scale = Parameter(
                    interleave_w13_halves(
                        layer.w13_weight_scale, group_size=64, dim=1
                    ).contiguous(),
                    requires_grad=False,
                )

            # Process w13 weights
            w13_blockscale_swizzled = swizzle_blockscale(layer.w13_weight_scale)
            alias_or_bind_derived_param(
                layer,
                "w13_weight_scale",
                "w13_blockscale_swizzled",
                w13_blockscale_swizzled,
            )

            w13_weight = layer.w13_weight
            intermediate_size_pad = w13_blockscale_swizzled.size(1) - w13_weight.size(1)
            if intermediate_size_pad:
                # padding gated activations will require to split w1 and w3
                # and pad them individually
                assert not layer.moe_runner_config.is_gated, (
                    "The intermediate size required padding, "
                    "but padding is also implemented for gated activations"
                )

                copy_or_rebind_param(
                    layer,
                    "w13_weight",
                    torch.nn.functional.pad(
                        w13_weight, (0, 0, 0, intermediate_size_pad)
                    ),
                )
                copy_or_rebind_param(
                    layer,
                    "w2_weight",
                    torch.nn.functional.pad(
                        layer.w2_weight, (0, intermediate_size_pad // 2, 0, 0)
                    ),
                )
                copy_or_rebind_param(
                    layer,
                    "w2_weight_scale",
                    torch.nn.functional.pad(
                        layer.w2_weight_scale, (0, intermediate_size_pad // 16)
                    ),
                )

            # Process w2 weights
            w2_blockscale_swizzled = swizzle_blockscale(layer.w2_weight_scale)
            alias_or_bind_derived_param(
                layer,
                "w2_weight_scale",
                "w2_blockscale_swizzled",
                w2_blockscale_swizzled,
            )

            if self._is_cutedsl_v2_standard:
                # CuteDSL v2 only: convert blockscales to MMA layout for
                # CuteDslMoEWrapper.  The v1 (deepep) path uses the
                # swizzled blockscales directly via flashinfer_cutedsl_moe_masked.
                from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

                from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
                    _FP4_SF_VEC_SIZE,
                )

                sf_vec_size = _FP4_SF_VEC_SIZE
                num_local_experts = layer.w13_weight.shape[0]
                w13_m = layer.w13_weight.shape[1]
                w13_k = layer.w13_weight.shape[2] * 2
                w2_m = layer.w2_weight.shape[1]
                w2_k = layer.w2_weight.shape[2] * 2
                layer.w13_blockscale_mma = Parameter(
                    convert_sf_to_mma_layout(
                        layer.w13_blockscale_swizzled.contiguous()
                        .view(torch.uint8)
                        .reshape(-1),
                        m=w13_m,
                        k=w13_k,
                        num_groups=num_local_experts,
                        sf_vec_size=sf_vec_size,
                    ),
                    requires_grad=False,
                )
                layer.w2_blockscale_mma = Parameter(
                    convert_sf_to_mma_layout(
                        layer.w2_blockscale_swizzled.contiguous()
                        .view(torch.uint8)
                        .reshape(-1),
                        m=w2_m,
                        k=w2_k,
                        num_groups=num_local_experts,
                        sf_vec_size=sf_vec_size,
                    ),
                    requires_grad=False,
                )

            # Both flashinfer cutlass and regular cutlass use same processing for w2

            # Set up CUTLASS MoE parameters (reuse to keep CUDA graph stable)
            device = layer.w13_weight.device
            inter_size = layer.w2_weight.shape[2] * 2
            hidden_size = layer.w13_weight.shape[2] * 2
            existing_params = getattr(layer, "cutlass_moe_params", None)
            if (
                existing_params is None
                or existing_params.cutlass_moe_type != CutlassMoEType.BlockscaledFP4
                or existing_params.num_experts != layer.num_experts
                or existing_params.intermediate_size_per_partition != inter_size
                or existing_params.hidden_size != hidden_size
                or existing_params.device != device
            ):
                layer.cutlass_moe_params = CutlassMoEParams(
                    CutlassMoEType.BlockscaledFP4,
                    device,
                    num_experts=layer.num_experts,  # global num experts
                    intermediate_size_per_partition=inter_size,  # n
                    hidden_size=hidden_size,
                )  # k

    @property
    def load_up_proj_weight_first(self) -> bool:
        # Load W13 as [Up, Gate] for FlashInfer CUTLASS and CuteDSL v2 kernels.
        # The CuteDSL v1 (deepep) path uses [Gate, Up] -- do NOT flip.
        return self.moe_runner_config.is_gated and (
            self.enable_flashinfer_cutlass_moe or self._is_cutedsl_v2_standard
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        moe_runner_backend = get_moe_runner_backend()

        if moe_runner_backend.is_auto():
            if is_cuda() and (8, 0) <= get_device_capability() < (10, 0):
                moe_runner_backend = MoeRunnerBackend.MARLIN
            else:
                # TRTLLM is currently the most performant and tested FP4 MoE
                # backend, so use it as the default.
                moe_runner_backend = MoeRunnerBackend.FLASHINFER_TRTLLM

        self._moe_runner_backend = moe_runner_backend

        if moe_runner_backend.is_flashinfer_cutedsl():
            import sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl  # noqa: F401 – triggers @register_fused_func

        if moe_runner_backend.is_flashinfer_cutlass():
            import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass  # noqa: F401

        # The plain CUTLASS backend uses the direct cutlass_moe_fp4 fused path
        # (see apply()), not a registered MoeRunner fused func, so skip creating
        # a MoeRunner for it -- constructing one would fail the fused-func check.
        if not moe_runner_backend.is_cutlass():
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)

    def apply(
        self,
        layer: FusedMoE,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        # Note: dispatch_output may be a DeepEPLLDispatchOutput (no topk_output
        # attribute -- topk_ids/topk_weights live directly on the dispatch
        # tuple). Defer per-attribute access to the branches that actually
        # consume them.
        activation = self.moe_runner_config.activation
        moe_runner_backend = getattr(
            self, "_moe_runner_backend", get_moe_runner_backend()
        )

        assert (
            activation in _SUPPORTED_ACT_STRS
        ), f"{activation=} not in supported {_SUPPORTED_ACT_STRS}"
        moe_runner_config = self.moe_runner_config

        if moe_runner_backend.is_marlin():
            from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo

            expert_map = None
            global_num_experts = -1
            if hasattr(layer, "dispatcher") and hasattr(
                layer.dispatcher, "local_expert_mapping"
            ):
                expert_map = layer.dispatcher.local_expert_mapping
                if expert_map is not None:
                    global_num_experts = self.moe_runner_config.num_experts

            quant_info = MarlinMoeQuantInfo(
                w13_qweight=layer.w13_weight,
                w2_qweight=layer.w2_weight,
                w13_scales=layer.w13_weight_scale,
                w2_scales=layer.w2_weight_scale,
                w13_g_idx_sort_indices=None,
                w2_g_idx_sort_indices=None,
                weight_bits=4,
                w13_global_scale=layer.w13_weight_scale_2,
                w2_global_scale=layer.w2_weight_scale_2,
                expert_map=expert_map,
                global_num_experts=global_num_experts,
            )
            return self.runner.run(dispatch_output, quant_info)

        # FlashInfer TRTLLM FP4 path
        if self.enable_flashinfer_trtllm_moe and hasattr(layer, "g1_scale_c"):
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmFp4MoeQuantInfo,
            )
            from sglang.srt.layers.moe.utils import RoutingMethodType

            # Determine routing method type based on layer configuration
            routing_method_type = getattr(
                layer, "routing_method_type", RoutingMethodType.Default
            )

            gemm1_clamp = getattr(layer, "gemm1_clamp_limit", None)
            quant_info = FlashInferTrtllmFp4MoeQuantInfo(
                w13_weight=layer.w13_weight.data,
                w2_weight=layer.w2_weight.data,
                w13_weight_scale=layer.w13_weight_scale.data,
                w2_weight_scale=layer.w2_weight_scale.data,
                g1_scale_c=layer.g1_scale_c.data,
                g1_alphas=layer.g1_alphas.data,
                g2_alphas=layer.g2_alphas.data,
                w13_input_scale_quant=layer.w13_input_scale_quant,
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                local_num_experts=layer.num_local_experts,
                intermediate_size_per_partition=layer.intermediate_size_per_partition,
                routing_method_type=routing_method_type,
                use_per_token_activation=self.quant_config.use_per_token_activation,
                gemm1_clamp_limit=gemm1_clamp.data if gemm1_clamp is not None else None,
            )

            return self.runner.run(dispatch_output, quant_info)

        if self.enable_flashinfer_cutedsl_moe:
            from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
                CuteDslFp4MoeQuantInfo,
                ensure_cutedsl_wrapper,
            )

            if self._is_cutedsl_v1_deepep:
                # v1 path: DeepEP low-latency + flashinfer_cutedsl_moe_masked.
                # Weights are [Gate, Up] (non-interleaved) with swizzled blockscales.
                quant_info = CuteDslFp4MoeQuantInfo(
                    w13_weight=layer.w13_weight,
                    w2_weight=layer.w2_weight,
                    w13_weight_sf=layer.w13_blockscale_swizzled,
                    w2_weight_sf=layer.w2_blockscale_swizzled,
                    w1_alpha=layer.g1_alphas,
                    w2_alpha=layer.g2_alphas,
                    a1_scale=layer.w13_input_scale_quant,
                    a2_scale=layer.w2_input_scale_quant,
                    use_nvfp4_dispatch=MOE_NVFP4_DISPATCH,
                    down_gemm_overlap_args=getattr(
                        self.runner, "down_gemm_overlap_args", None
                    ),
                )
                return self.runner.run(dispatch_output, quant_info)

            # v2 standard path (a2a=none/flashinfer): uses CuteDslMoEWrapper
            # with [Up, Gate] interleaved weights and MMA blockscales.
            ensure_cutedsl_wrapper(layer)
            w1_alpha, fc2_input_scale, w2_alpha = layer._cutedsl_scales
            quant_info = CuteDslFp4MoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_weight_sf=getattr(
                    layer, "w13_blockscale_mma", layer.w13_blockscale_swizzled
                ),
                w2_weight_sf=getattr(
                    layer, "w2_blockscale_mma", layer.w2_blockscale_swizzled
                ),
                w1_alpha=w1_alpha,
                w2_alpha=w2_alpha,
                a1_scale=layer._cutedsl_input_scale,
                a2_scale=fc2_input_scale,
                wrapper=layer._cutedsl_wrapper,
            )
            return self.runner.run(dispatch_output, quant_info)

        if self.enable_flashinfer_cutlass_moe:
            from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
                FlashInferCutlassMoeQuantInfo,
            )

            assert (
                not moe_runner_config.apply_router_weight_on_input
            ), "apply_router_weight_on_input is not supported for Flashinfer"
            quant_info = FlashInferCutlassMoeQuantInfo(
                quant_type="fp4",
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                output_dtype=torch.bfloat16,
                quant_scales=[
                    layer.w13_input_scale_quant,
                    layer.w13_blockscale_swizzled,
                    layer.g1_alphas,
                    layer.w2_input_scale_quant,
                    layer.w2_blockscale_swizzled,
                    layer.g2_alphas,
                ],
                moe_ep_size=layer.moe_ep_size,
                moe_ep_rank=layer.moe_ep_rank,
                moe_tp_size=layer.moe_tp_size,
                moe_tp_rank=layer.moe_tp_rank,
                apply_routed_scaling_factor=False,
            )
            return self.runner.run(dispatch_output, quant_info)

        from sglang.srt.layers.moe.cutlass_moe import cutlass_moe_fp4

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        output = cutlass_moe_fp4(
            a=x,
            a1_gscale=layer.w13_input_scale_quant,
            w1_fp4=layer.w13_weight,
            w1_blockscale=layer.w13_blockscale_swizzled,
            w1_alphas=layer.g1_alphas,
            a2_gscale=layer.w2_input_scale_quant,
            w2_fp4=layer.w2_weight,
            w2_blockscale=layer.w2_blockscale_swizzled,
            w2_alphas=layer.g2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            params=layer.cutlass_moe_params,
            apply_router_weight_on_input=moe_runner_config.apply_router_weight_on_input,
            no_combine=moe_runner_config.no_combine,
        ).to(x.dtype)
        # Scale by routed_scaling_factor is fused into select_experts.
        return StandardCombineInput(hidden_states=output)
