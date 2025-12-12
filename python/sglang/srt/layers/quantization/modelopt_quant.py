# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py
from __future__ import annotations

import logging
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.parameter import ModelWeightParameter, PerTensorScaleParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
    is_blackwell_supported,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import (
    convert_to_channelwise,
    is_layer_skipped,
    per_tensor_dequantize,
    requantize_with_max_scale,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils.common import (
    direct_register_custom_op,
    get_bool_env_var,
    is_cuda,
    is_sm120_supported,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import register_fake_if_exists

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

try:
    if is_sm120_supported():
        from flashinfer import fp4_quantize
    else:
        from sgl_kernel import scaled_fp4_quant as fp4_quantize

except ImportError:
    fp4_quantize = None

try:
    from flashinfer import mm_fp4 as fp4_gemm
    from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_sf_a

    enable_flashinfer_fp4_gemm = True
except ImportError:
    if is_cuda():
        from sgl_kernel import cutlass_scaled_fp4_mm as fp4_gemm
    else:
        fp4_gemm = None
    enable_flashinfer_fp4_gemm = False
    reorder_rows_for_gated_act_gemm = None
    shuffle_matrix_a = None
    shuffle_matrix_sf_a = None

try:
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType
except ImportError:
    flashinfer_cutlass_fused_moe = None

    # Define a minimal ActivationType enum if flashinfer is not available
    class ActivationType(IntEnum):
        Swiglu = 3
        Relu2 = 6


# Initialize logger for the module
logger = logging.getLogger(__name__)


def _sglang_fp4_gemm(
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
        return fp4_gemm(
            input, weight, input_sf, weight_sf, alpha, out_dtype, backend=backend
        )
    else:
        return fp4_gemm(input, weight, input_sf, weight_sf, alpha, out_dtype)


def _sglang_fp4_gemm_fake(
    input,
    weight,
    input_sf,
    weight_sf,
    alpha,
    out_dtype,
    out_features: int,
):
    M = input.shape[-2]
    N = int(out_features)
    return input.new_empty((M, N), dtype=out_dtype)


direct_register_custom_op(
    op_name="fp4_gemm",
    op_func=_sglang_fp4_gemm,
    mutates_args=[],
    fake_impl=_sglang_fp4_gemm_fake,
)


if is_cuda() and (not is_sm120_supported()) and (fp4_quantize is not None):

    @register_fake_if_exists("sgl_kernel::scaled_fp4_quant")
    def _sgl_kernel_scaled_fp4_quant_fake(
        output, input, output_scale, input_global_scale
    ):
        return


CUTEDSL_MOE_SCALAR_INPUT_SCALE = get_bool_env_var(
    "SGLANG_CUTEDSL_MOE_SCALAR_INPUT_SCALE", "true"
)

# TODO make it true by default when the DeepEP PR is merged
MOE_NVFP4_DISPATCH = envs.SGLANG_MOE_NVFP4_DISPATCH.get()
FLASHINFER_FP4_GEMM_BACKEND = envs.SGLANG_FLASHINFER_FP4_GEMM_BACKEND.get()
# Supported activation schemes for the current configuration
ACTIVATION_SCHEMES = ["static"]

ACT_STR_TO_TYPE_MAP = {
    "silu": ActivationType.Swiglu,  # This is the default
    "relu2": ActivationType.Relu2,
}


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
            return Moe(self)
        return None

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    def get_scaled_act_names(self) -> List[str]:
        return []


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
            # For kv_cache, check if kv_cache_scheme exists and extract algo
            kv_cache_scheme = config.get("kv_cache_scheme")
            if (
                kv_cache_scheme
                and kv_cache_scheme.get("type") == "float"
                and kv_cache_scheme.get("num_bits") == 8
            ):
                kv_cache_quant_method = "FP8"

            # Map 'ignore' field to 'exclude_modules'
            exclude_modules = config.get("ignore")
        else:
            # Fall back to nested format (hf_quant_config.json - legacy format)
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

    def is_layer_excluded(self, prefix: str) -> bool:
        if len(self.exclude_modules) == 0:
            return False
        return any(
            module in prefix
            or (
                prefix.startswith("language_model.")
                and module in prefix.removeprefix("language_model.")
            )
            for module in self.exclude_modules
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

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
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
        # cutlass sgl-kernel only supports per-channel scale
        if self.cutlass_fp8_supported:
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
            from flashinfer import reorder_rows_for_gated_act_gemm, shuffle_matrix_a

            # 1) Swap W13 halves: [Up, Gate] -> [Gate, Up] expected by FI
            num_experts, two_n, hidden = layer.w13_weight.shape
            inter = two_n // 2
            w13_swapped = (
                layer.w13_weight.reshape(num_experts, 2, inter, hidden)
                .flip(dims=[1])
                .reshape(num_experts, two_n, hidden)
            )

            # 2) Reorder rows for fused gated activation (W13)
            w13_interleaved = [
                reorder_rows_for_gated_act_gemm(w13_swapped[i])
                for i in range(num_experts)
            ]
            w13_interleaved = torch.stack(w13_interleaved).reshape(
                num_experts, two_n, hidden
            )

            # 3) Shuffle weights for transposed MMA output (both W13, W2)
            epilogue_tile_m = 128
            w13_shuffled = [
                shuffle_matrix_a(w13_interleaved[i].view(torch.uint8), epilogue_tile_m)
                for i in range(num_experts)
            ]
            w2_shuffled = [
                shuffle_matrix_a(layer.w2_weight[i].view(torch.uint8), epilogue_tile_m)
                for i in range(num_experts)
            ]

            layer.w13_weight = Parameter(
                torch.stack(w13_shuffled).view(torch.float8_e4m3fn),
                requires_grad=False,
            )
            layer.w2_weight = Parameter(
                torch.stack(w2_shuffled).view(torch.float8_e4m3fn),
                requires_grad=False,
            )

        # Precompute and register per-expert output scaling factors for FI MoE
        if get_moe_runner_backend().is_flashinfer_trtllm():
            # Note: w13_input_scale and w2_input_scale are scalar Parameters post-reduction
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

            output1_scales_scalar = (
                w13_weight_scale * input_scale * (1.0 / activation_scale)
            )
            output1_scales_gate_scalar = w13_weight_scale * input_scale
            output2_scales_scalar = activation_scale * w2_weight_scale

            layer.output1_scales_scalar = Parameter(
                output1_scales_scalar, requires_grad=False
            )
            layer.output1_scales_gate_scalar = Parameter(
                output1_scales_gate_scalar, requires_grad=False
            )
            layer.output2_scales_scalar = Parameter(
                output2_scales_scalar, requires_grad=False
            )
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

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        # Fast path: TRT-LLM FP8 per-tensor MoE using BYPASSED TopK routing
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        if (
            get_moe_runner_backend().is_flashinfer_trtllm()
            and TopKOutputChecker.format_is_bypassed(topk_output)
        ):
            router_logits = topk_output.router_logits
            topk_config = topk_output.topk_config

            # Constraints
            assert (
                self.moe_runner_config.activation == "silu"
            ), "Only silu is supported for flashinfer fp8 moe"

            from flashinfer import RoutingMethodType
            from flashinfer.fused_moe import trtllm_fp8_per_tensor_scale_moe

            correction_bias = (
                None
                if topk_config.correction_bias is None
                else topk_config.correction_bias
            )
            # Pre-quantize activations to FP8 per-tensor using provided input scale
            x_fp8, _ = scaled_fp8_quant(x, layer.w13_input_scale)

            use_routing_scales_on_input = True
            routed_scaling_factor = self.moe_runner_config.routed_scaling_factor

            # Enforce Llama4 routing for ModelOpt FP8 MoE for now.
            # TODO(brayden): support other routing methods
            assert topk_config.top_k == 1, "ModelOpt FP8 MoE requires top_k==1"
            assert (
                not topk_config.num_expert_group
            ), "ModelOpt FP8 MoE does not support expert grouping"
            assert (
                not topk_config.topk_group
            ), "ModelOpt FP8 MoE does not support grouped top-k"
            routing_method_type = RoutingMethodType.Llama4

            # FlashInfer TRTLLM requires routing_logits (and bias) to be bfloat16
            routing_logits_cast = router_logits.to(torch.bfloat16)
            routing_bias_cast = (
                None if correction_bias is None else correction_bias.to(torch.bfloat16)
            )

            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                # FIXME: there is a bug in the trtllm_fp8_block_scale_moe.
                # It ignored the `output`` argument. https://github.com/flashinfer-ai/flashinfer/blob/da01b1bd8f9f22aec8c0eea189ad54860b034947/flashinfer/fused_moe/core.py#L1323-L1325
                # so we put the whole function under the ``use_symmetric_memory`` context manager.
                # If the bug is fixed, we can only put the output tensor allocation under the context manager.
                output = trtllm_fp8_per_tensor_scale_moe(
                    routing_logits=routing_logits_cast,
                    routing_bias=routing_bias_cast,
                    hidden_states=x_fp8,
                    gemm1_weights=layer.w13_weight,
                    output1_scales_scalar=layer.output1_scales_scalar,
                    output1_scales_gate_scalar=layer.output1_scales_gate_scalar,
                    gemm2_weights=layer.w2_weight,
                    output2_scales_scalar=layer.output2_scales_scalar,
                    num_experts=layer.num_experts,
                    top_k=topk_config.top_k,
                    n_group=0,
                    topk_group=0,
                    intermediate_size=layer.w2_weight.shape[2],
                    local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                    local_num_experts=layer.num_local_experts,
                    routed_scaling_factor=(
                        routed_scaling_factor
                        if routed_scaling_factor is not None
                        else 1.0
                    ),
                    use_routing_scales_on_input=use_routing_scales_on_input,
                    tile_tokens_dim=None,
                    routing_method_type=routing_method_type,
                )

            from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

            return StandardCombineInput(hidden_states=output)

        if get_moe_runner_backend().is_flashinfer_cutlass():
            activation = ACT_STR_TO_TYPE_MAP[self.moe_runner_config.activation]
            assert (
                (
                    activation is ActivationType.Relu2
                    and not self.moe_runner_config.is_gated
                )
                or activation is ActivationType.Swiglu
                and self.moe_runner_config.is_gated
            ), "Only Relu2 non-gated or Swiglu gated are supported for flashinfer cutlass fp8 moe"
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
            x_fp8, _ = scaled_fp8_quant(x, layer.w13_input_scale)
            output_dtype = x.dtype
            original_col = x.shape[1]
            x_sf = None

            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                symm_output = torch.empty(
                    x.shape[0], original_col, dtype=output_dtype, device=x.device
                )
            output = flashinfer_cutlass_fused_moe(
                output=symm_output,
                input=x_fp8,
                token_selected_experts=topk_ids.to(torch.int),
                token_final_scales=topk_weights,
                fc1_expert_weights=layer.w13_weight,
                fc2_expert_weights=layer.w2_weight,
                output_dtype=output_dtype,
                input_sf=x_sf,
                quant_scales=[
                    layer.fc1_dequant,
                    layer.fc2_quant,
                    layer.fc2_dequant,
                    layer.fc1_input_dequant,
                ],
                ep_size=layer.moe_ep_size,
                ep_rank=layer.moe_ep_rank,
                tp_size=layer.moe_tp_size,
                tp_rank=layer.moe_tp_rank,
                tune_max_num_tokens=next_power_of_2(x.shape[0]),
                activation_type=activation,
            )[0]

            from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

            return StandardCombineInput(hidden_states=output)

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
    ) -> None:
        super().__init__(kv_cache_quant_algo, exclude_modules, packed_modules_mapping)
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected nvfp4 checkpoint. Please note that the "
                "format is experimental and subject to change."
            )
        self.group_size = group_size

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
        return 100

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
            # Note: FP4 models in config.json format may not have all the detailed fields
            # that are present in hf_quant_config.json, so we need to handle defaults
            kv_cache_quant_algo = config.get("kv_cache_quant_algo")
            if not kv_cache_quant_algo:
                # For config.json format, derive from kv_cache_scheme if available
                kv_cache_scheme = config.get("kv_cache_scheme")
                if (
                    kv_cache_scheme
                    and kv_cache_scheme.get("type") == "float"
                    and kv_cache_scheme.get("num_bits") == 8
                ):
                    kv_cache_quant_algo = "FP8"
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

        if not quant_method in ["FP8", "NVFP4"]:
            raise ValueError(
                f"ModelOpt currently only supports: FP8, NVFP4"
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
            Moe=ModelOptNvFp4FusedMoEMethod,  # FlashInferFP4MoE needs the same quantization method but with compatible attribute handling
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

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        if input_size_per_partition % 16 != 0:
            raise ValueError(
                "Unsupported model when in features size is " "not multiple of 16"
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
        layer.input_scale = Parameter(input_scale_2, requires_grad=False)
        layer.weight_scale_2 = Parameter(weight_scale_2, requires_grad=False)
        layer.alpha = Parameter(
            layer.input_scale * layer.weight_scale_2, requires_grad=False
        )
        layer.input_scale_inv = Parameter(
            (1 / input_scale_2).to(torch.float32), requires_grad=False
        )
        if FLASHINFER_FP4_GEMM_BACKEND == "trtllm":
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
        # Pad and blockwise interleave weight_scale
        scales = layer.weight_scale
        scale_ndim = scales.ndim
        if scale_ndim == 2:
            scales = scales.unsqueeze(0)
        assert scales.ndim == 3
        B, M, K = scales.shape
        round_up_multiple = lambda x, m: (x + m - 1) // m * m
        M_padded = round_up_multiple(M, 128)
        K_padded = round_up_multiple(K, 4)
        padded_scales = torch.zeros((B, M_padded, K_padded), dtype=scales.dtype)
        padded_scales[:B, :M, :K] = scales
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
        layer.weight_scale_interleaved = Parameter(padded_scales, requires_grad=False)

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
        # TODO(shuw@nvidia.com)
        # Remove the default after flashinfer bumped to 0.5.1
        backend = (
            FLASHINFER_FP4_GEMM_BACKEND if FLASHINFER_FP4_GEMM_BACKEND else "cutlass"
        )
        out = _sglang_fp4_gemm(
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


class ModelOptNvFp4FusedMoEMethod(FusedMoEMethodBase):
    """
       MoE Method for FP4 Quantization with Blockscales and PerTensorScales
    Args:
        quant_config: NVFP4 Quant Config
    """

    def __init__(self, quant_config: ModelOptFp4Config):
        self.quant_config = quant_config
        if not is_blackwell_supported():
            raise ValueError(
                "Current platform does not support NVFP4"
                " quantization. Please use Blackwell and"
                " above."
            )
        self.enable_flashinfer_trtllm_moe = (
            get_moe_runner_backend().is_flashinfer_trtllm()
        )
        self._cache_permute_indices = {}

    @property
    def enable_flashinfer_cutlass_moe(self) -> bool:
        from sglang.srt.layers.moe import get_moe_runner_backend

        """Access the global enable_flashinfer_cutlass_moe setting."""
        return get_moe_runner_backend().is_flashinfer_cutlass()

    @property
    def enable_flashinfer_cutedsl_moe(self) -> bool:
        from sglang.srt.layers.moe import get_moe_runner_backend

        """Access the global enable_flashinfer_cutedsl_moe setting."""
        return get_moe_runner_backend().is_flashinfer_cutedsl()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if not self.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError(
                "NVFP4 quantization was selected, "
                " dynamic quantization is not supported."
            )

        # TODO(ch-wan): check if this is needed
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.params_dtype = params_dtype
        layer.quant_config = self.quant_config

        weight_dtype = torch.uint8
        weight_scale_dtype = torch.float8_e4m3fn
        weight_loader = extra_weight_attrs.get("weight_loader")
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

        # Only use `swizzle_blockscale` for shapes, not for real content
        layer.w13_blockscale_swizzled = Parameter(
            self.swizzle_blockscale(layer.w13_weight_scale), requires_grad=False
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

        layer.w2_blockscale_swizzled = Parameter(
            self.swizzle_blockscale(layer.w2_weight_scale), requires_grad=False
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

    def swizzle_blockscale(self, scale: torch.Tensor):
        assert scale.dtype == torch.float8_e4m3fn
        # Pad and blockwise interleave weight_scale
        scale_ndim = scale.ndim
        if scale.ndim == 2:
            scale = scale.unsqueeze(0)
        assert scale.ndim == 3
        B, M, K = scale.shape
        round_up_multiple = lambda x, m: (x + m - 1) // m * m
        M_padded = round_up_multiple(M, 128)
        K_padded = round_up_multiple(K, 4)
        padded_scale = torch.zeros((B, M_padded, K_padded), dtype=scale.dtype)
        padded_scale[:B, :M, :K] = scale
        batches, rows, cols = padded_scale.shape
        assert rows % 128 == 0
        assert cols % 4 == 0
        padded_scale = padded_scale.reshape(batches, rows // 128, 4, 32, cols // 4, 4)
        swizzled_scale = padded_scale.permute((0, 1, 4, 3, 2, 5))
        swizzled_scale = swizzled_scale.contiguous().cuda()
        return (
            swizzled_scale.reshape(M_padded, K_padded)
            if scale_ndim == 2
            else swizzled_scale.reshape(B, M_padded, K_padded)
        )

    def prepare_static_weights_for_kernel(
        self,
        # args_dequant,
        # args,
        gemm1_weights,
        gemm2_weights,
        gemm1_scales_linear_fp4_bytes,
        gemm2_scales_linear_fp4_bytes,
        hidden_size,
        intermediate_size,
        num_experts,
    ):
        from flashinfer import nvfp4_block_scale_interleave
        from flashinfer.fused_moe.core import (
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )

        """Prepare quantized weights for kernel (done offline with weights)."""
        epilogue_tile_m = 128  # FIXME: this depends on the kernel internals

        # Convert quantized weights to proper formats
        gemm1_weights_fp4 = gemm1_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 2
        )  # packed fp4
        gemm1_scales_linear_fp4 = gemm1_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, 2 * intermediate_size, hidden_size // 16
        )  # fp8 scaling factors

        gemm2_weights_fp4 = gemm2_weights.view(torch.float8_e4m3fn).reshape(
            num_experts, hidden_size, intermediate_size // 2
        )  # packed fp4
        gemm2_scales_linear_fp4 = gemm2_scales_linear_fp4_bytes.view(
            torch.float8_e4m3fn
        ).reshape(
            num_experts, hidden_size, intermediate_size // 16
        )  # fp8 scaling factors

        gemm1_weights_fp4_shuffled = []
        gemm1_scales_fp4_shuffled = []
        gemm2_weights_fp4_shuffled = []
        gemm2_scales_fp4_shuffled = []
        for i in range(num_experts):
            # Calculate the permute indices for the following:
            # 1. Reorder rows of W1 and scales for fused gated activation
            # 2. Shuffle weights and scaling factors for transposed mma output
            # for both w3_w1 and w2 weights and scale factors
            permute_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                gemm1_weights_fp4[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm1_weights_fp4_shuffled.append(
                gemm1_weights_fp4[i]
                .view(torch.uint8)[permute_indices.to(gemm1_weights_fp4.device)]
                .contiguous()
            )

            permute_sf_indices = _maybe_get_cached_w3_w1_permute_indices(
                self._cache_permute_indices,
                gemm1_scales_linear_fp4[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm1_scales_fp4_shuffled.append(
                nvfp4_block_scale_interleave(
                    gemm1_scales_linear_fp4[i]
                    .view(torch.uint8)[
                        permute_sf_indices.to(gemm1_scales_linear_fp4.device)
                    ]
                    .contiguous()
                )
            )

            permute_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                gemm2_weights_fp4[i].view(torch.uint8),
                epilogue_tile_m,
            )
            gemm2_weights_fp4_shuffled.append(
                gemm2_weights_fp4[i]
                .view(torch.uint8)[permute_indices.to(gemm2_weights_fp4.device)]
                .contiguous()
            )

            permute_sf_indices = get_w2_permute_indices_with_cache(
                self._cache_permute_indices,
                gemm2_scales_linear_fp4[i].view(torch.uint8),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )
            gemm2_scales_fp4_shuffled.append(
                nvfp4_block_scale_interleave(
                    gemm2_scales_linear_fp4[i]
                    .view(torch.uint8)[
                        permute_sf_indices.to(gemm2_scales_linear_fp4.device)
                    ]
                    .contiguous()
                )
            )

        # Stack weights for all experts
        gemm1_weights_fp4_shuffled = torch.stack(gemm1_weights_fp4_shuffled)
        gemm1_scales_fp4_shuffled = (
            torch.stack(gemm1_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, 2 * intermediate_size, hidden_size // 16)
        )

        gemm2_weights_fp4_shuffled = torch.stack(gemm2_weights_fp4_shuffled)
        gemm2_scales_fp4_shuffled = (
            torch.stack(gemm2_scales_fp4_shuffled)
            .view(torch.float8_e4m3fn)
            .reshape(num_experts, hidden_size, intermediate_size // 16)
        )
        return (
            gemm1_weights_fp4_shuffled,
            gemm1_scales_fp4_shuffled,
            gemm2_weights_fp4_shuffled,
            gemm2_scales_fp4_shuffled,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process FP4 MoE weights after loading from serialized checkpoint.

        Only supports pre-quantized checkpoints with FP8 weights and scales.
        """

        # GEMM 1 scale processing
        if layer.moe_runner_config.is_gated:
            if not torch.allclose(
                layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
            ):
                logger.warning_once(
                    "w1_weight_scale_2 must match w3_weight_scale_2. "
                    "Accuracy may be affected."
                )

            w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0]
        else:
            w13_weight_scale_2 = layer.w13_weight_scale_2[:]
        layer.w13_weight_scale_2 = Parameter(w13_weight_scale_2, requires_grad=False)

        # Calculate input scales based on strategy
        if self.enable_flashinfer_cutlass_moe or self.enable_flashinfer_trtllm_moe:
            w13_input_scale = layer.w13_input_scale.max().to(torch.float32)
            w2_input_scale = layer.w2_input_scale.max().to(torch.float32)
        elif self.enable_flashinfer_cutedsl_moe:
            # All-expert-one-input-scale is mathematically different from default per-expert-input-scale
            # Thus we allow users to switch the flag to do thorough testing
            if CUTEDSL_MOE_SCALAR_INPUT_SCALE:
                w13_input_scale = (
                    layer.w13_input_scale.max()
                    .to(torch.float32)
                    .repeat(layer.w13_input_scale.shape[0])
                )
            else:
                w13_input_scale = layer.w13_input_scale.max(dim=1).values.to(
                    torch.float32
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

        # Create shared parameters
        layer.g1_alphas = Parameter(
            (w13_input_scale * w13_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )
        layer.g2_alphas = Parameter(
            (w2_input_scale * layer.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )
        layer.w13_input_scale_quant = Parameter(
            (1 / w13_input_scale).to(torch.float32), requires_grad=False
        )
        layer.w2_input_scale_quant = Parameter(
            (1 / w2_input_scale).to(torch.float32), requires_grad=False
        )

        layer.dispatcher.set_quant_config(
            {
                "input_global_scale": (
                    layer.w13_input_scale_quant if MOE_NVFP4_DISPATCH else None
                )
            }
        )
        # Validate weight scales
        assert_dim = 2 if layer.moe_runner_config.is_gated else 1
        for name, weight_scale in [
            ("w13", layer.w13_weight_scale),
            ("w2", layer.w2_weight_scale),
        ]:
            assert (
                weight_scale.shape[assert_dim] % 16 == 0
            ), f"Expected {name}_weight_scale.dim({assert_dim}) to be divisible by 16"
            assert (
                weight_scale.dtype == torch.float8_e4m3fn
            ), f"{name} Weight Blockscale must be represented as FP8-E4M3"

        # Weight processing based on strategy
        if (
            self.enable_flashinfer_trtllm_moe
            and reorder_rows_for_gated_act_gemm is not None
            and shuffle_matrix_sf_a is not None
        ):
            # FlashInfer TRTLLM processing - handles both w13 and w2
            (
                gemm1_weights_fp4_shuffled,
                gemm1_scales_fp4_shuffled,
                gemm2_weights_fp4_shuffled,
                gemm2_scales_fp4_shuffled,
            ) = self.prepare_static_weights_for_kernel(
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                layer.w2_weight.size(-2),  # hidden_size
                layer.w13_weight.size(-2) // 2,  # intermediate_size
                layer.w13_weight.size(0),  # num_experts
            )

            # Set flashinfer parameters
            layer.gemm1_weights_fp4_shuffled = Parameter(
                gemm1_weights_fp4_shuffled, requires_grad=False
            )
            layer.gemm2_weights_fp4_shuffled = Parameter(
                gemm2_weights_fp4_shuffled, requires_grad=False
            )
            layer.gemm1_scales_fp4_shuffled = Parameter(
                gemm1_scales_fp4_shuffled, requires_grad=False
            )
            layer.gemm2_scales_fp4_shuffled = Parameter(
                gemm2_scales_fp4_shuffled, requires_grad=False
            )

            # Additional parameter needed for TRT-LLM
            layer.g1_scale_c = Parameter(
                (layer.w2_input_scale_quant * layer.g1_alphas).to(torch.float32),
                requires_grad=False,
            )

            # Clean up weights that won't be used by TRT-LLM
            del (
                layer.w2_weight,
                layer.w2_weight_scale,
                layer.w13_weight,
                layer.w13_weight_scale,
            )

        else:
            # CUTLASS processing - handle w13 and w2 separately

            # Process w13 weights
            w13_blockscale_swizzled = self.swizzle_blockscale(layer.w13_weight_scale)
            del layer.w13_weight_scale
            layer.w13_blockscale_swizzled.data.copy_(w13_blockscale_swizzled)

            w13_weight = layer.w13_weight
            intermediate_size_pad = w13_blockscale_swizzled.size(1) - w13_weight.size(1)
            if intermediate_size_pad:
                # padding gated activations will require to split w1 and w3
                # and pad them individually
                assert not layer.moe_runner_config.is_gated, (
                    "The intermediate size required padding, "
                    "but padding is also implemented for gated activations"
                )

                layer.w13_weight = Parameter(
                    torch.nn.functional.pad(
                        w13_weight, (0, 0, 0, intermediate_size_pad)
                    ),
                    requires_grad=False,
                )
                layer.w2_weight = Parameter(
                    torch.nn.functional.pad(
                        layer.w2_weight, (0, intermediate_size_pad // 2, 0, 0)
                    ),
                    requires_grad=False,
                )
                layer.w2_weight_scale = Parameter(
                    torch.nn.functional.pad(
                        layer.w2_weight_scale, (0, intermediate_size_pad // 16)
                    ),
                    requires_grad=False,
                )
                layer.w2_blockscale_swizzled = Parameter(
                    self.swizzle_blockscale(layer.w2_weight_scale), requires_grad=False
                )

            layer.w13_weight = Parameter(layer.w13_weight.data, requires_grad=False)

            # Process w2 weights
            w2_blockscale_swizzled = self.swizzle_blockscale(layer.w2_weight_scale)
            del layer.w2_weight_scale
            layer.w2_blockscale_swizzled.data.copy_(w2_blockscale_swizzled)

            # Both flashinfer cutlass and regular cutlass use same processing for w2

            # Set up CUTLASS MoE parameters
            device = layer.w13_weight.device
            layer.cutlass_moe_params = CutlassMoEParams(
                CutlassMoEType.BlockscaledFP4,
                device,
                num_experts=layer.num_experts,  # global num experts
                intermediate_size_per_partition=layer.w2_weight.shape[2] * 2,  # n
                hidden_size=layer.w13_weight.shape[2] * 2,
            )  # k

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        return self.enable_flashinfer_cutlass_moe and self.moe_runner_config.is_gated

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: FusedMoE,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        x = dispatch_output.hidden_states
        x_sf = dispatch_output.hidden_states_scale
        topk_output = dispatch_output.topk_output

        activation = self.moe_runner_config.activation

        assert (
            activation in ACT_STR_TO_TYPE_MAP
        ), f"{activation=} missing from {ACT_STR_TO_TYPE_MAP.keys()=}"
        moe_runner_config = self.moe_runner_config

        # Check if this is a FlashInferFP4MoE layer that should handle its own forward
        if hasattr(layer, "gemm1_weights_fp4_shuffled"):
            # This layer was processed with flashinfer TRTLLM - delegate to its own forward
            from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

            return StandardCombineInput(hidden_states=layer.forward(x, topk_output))

        if self.enable_flashinfer_cutlass_moe:
            assert (
                not moe_runner_config.apply_router_weight_on_input
            ), "apply_router_weight_on_input is not supported for Flashinfer"
            # TRTLLM Cutlass moe takes in activations in BF16/Half/nvfp4 precision
            # and fp4 quantized weights loaded from the checkpoint
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

            output_dtype = torch.bfloat16

            # If x_sf is not None, x is FP4 packed (half size), so we need * 2
            # If x_sf is None, x is not packed, so output_col = x.shape[1]
            output_col = x.shape[1]
            if x_sf is not None and layer.moe_runner_config.is_gated:
                output_col *= 2
            with use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            ):
                symm_output = torch.empty(
                    x.shape[0],
                    output_col,
                    dtype=output_dtype,
                    device=x.device,
                )

            output = flashinfer_cutlass_fused_moe(
                output=symm_output,
                input=x,
                token_selected_experts=topk_ids.to(torch.int),
                token_final_scales=topk_weights,
                fc1_expert_weights=layer.w13_weight.view(torch.long),
                fc2_expert_weights=layer.w2_weight.view(torch.long),
                output_dtype=output_dtype,
                input_sf=x_sf,
                quant_scales=[
                    layer.w13_input_scale_quant,
                    layer.w13_blockscale_swizzled.view(torch.int32),
                    layer.g1_alphas,
                    layer.w2_input_scale_quant,
                    layer.w2_blockscale_swizzled.view(torch.int32),
                    layer.g2_alphas,
                ],
                ep_size=layer.moe_ep_size,
                ep_rank=layer.moe_ep_rank,
                tp_size=layer.moe_tp_size,
                tp_rank=layer.moe_tp_rank,
                tune_max_num_tokens=next_power_of_2(x.shape[0]),
                activation_type=ACT_STR_TO_TYPE_MAP[activation],
            )[0]

            from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

            return StandardCombineInput(hidden_states=output)

        from sglang.srt.layers.moe.cutlass_moe import cutlass_moe_fp4

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
        ).to(x.dtype)
        # Scale by routed_scaling_factor is fused into select_experts.
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        return StandardCombineInput(hidden_states=output)

    def apply_without_routing_weights(
        self,
        layer: FusedMoE,
        x: tuple[torch.Tensor, Optional[torch.Tensor]],
        masked_m: torch.Tensor,
        moe_runner_config: MoeRunnerConfig,
    ) -> torch.Tensor:
        assert (
            moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        assert self.enable_flashinfer_cutedsl_moe, "only support flashinfer cutedsl moe"
        assert (
            not moe_runner_config.apply_router_weight_on_input
        ), "apply_router_weight_on_input is not supported for Flashinfer"

        from sglang.srt.layers.moe.flashinfer_cutedsl_moe import (
            flashinfer_cutedsl_moe_masked,
        )

        down_gemm_overlap_args: Optional[DownGemmOverlapArgs] = getattr(
            layer, "down_gemm_overlap_args", None
        )

        out = flashinfer_cutedsl_moe_masked(
            hidden_states=x,
            input_global_scale=(
                None if MOE_NVFP4_DISPATCH else layer.w13_input_scale_quant
            ),
            w1=layer.w13_weight,
            w1_blockscale=layer.w13_blockscale_swizzled,
            w1_alpha=layer.g1_alphas,
            w2=layer.w2_weight,
            a2_global_scale=layer.w2_input_scale_quant,
            w2_blockscale=layer.w2_blockscale_swizzled,
            w2_alpha=layer.g2_alphas,
            masked_m=masked_m,
            **(
                dict(
                    down_sm_count=down_gemm_overlap_args.num_sms,
                    down_signals=down_gemm_overlap_args.signal,
                    down_start_event=down_gemm_overlap_args.start_event,
                )
                if down_gemm_overlap_args is not None
                else {}
            ),
        )
        return out
