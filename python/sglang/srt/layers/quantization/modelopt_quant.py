# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
from sglang.srt.layers.parameter import ModelWeightParameter, PerTensorScaleParameter
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
    is_sm100_supported,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.utils import (
    convert_to_channelwise,
    is_layer_skipped,
    per_tensor_dequantize,
    requantize_with_max_scale,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import get_bool_env_var, is_cuda, next_power_of_2

if is_cuda():
    from sgl_kernel import cutlass_scaled_fp4_mm, scaled_fp4_quant

try:
    from flashinfer import fp4_quantize as fp4_quantize
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
except ImportError:
    flashinfer_cutlass_fused_moe = None

ENABLE_TRTLMM_GEN_MOE = get_bool_env_var("SGLANG_ENABLE_TRTLLM_GEN_MOE", "false")

if ENABLE_TRTLMM_GEN_MOE:
    # import tensorrt_llm to register ops
    import tensorrt_llm  # noqa
    from tensorrt_llm._torch.modules.fused_moe import RoutingMethodType
    from tensorrt_llm.quantization.utils.fp4_utils import (
        float4_sf_dtype,
        get_reorder_rows_for_gated_act_gemm_row_indices,
        get_shuffle_matrix_a_row_indices,
        get_shuffle_matrix_sf_a_row_indices,
    )

# Initialize logger for the module
logger = logging.getLogger(__name__)

# Supported activation schemes for the current configuration
ACTIVATION_SCHEMES = ["static"]


class ModelOptFp8Config(QuantizationConfig):
    """Configuration for ModelOpt FP8 quantization, including serialization and compatibility checks."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        kv_cache_quant_method: Optional[str] = None,
        exclude_modules: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            is_checkpoint_fp8_serialized (bool): Indicates if the checkpoint uses serialized FP8 format.
        """
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        self.kv_cache_quant_method = kv_cache_quant_method
        self.exclude_modules = exclude_modules
        if is_checkpoint_fp8_serialized:
            logger.warning(
                "Detected ModelOpt FP8 checkpoint. The format is experimental and subject to change."
            )

    @classmethod
    def get_name(cls) -> str:
        return "modelopt"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89  # Minimum hardware capability (e.g., Hopper GPUs).

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelOptFp8Config":
        quant_method = cls.get_from_keys(config, ["quantization"]).get("quant_algo")
        kv_cache_quant_method = cls.get_from_keys(config, ["quantization"]).get(
            "kv_cache_quant_algo"
        )
        exclude_modules = cls.get_from_keys(config, ["quantization"]).get(
            "exclude_modules"
        )

        if "FP8" not in quant_method:
            raise ValueError(
                "ModelOpt only supports static FP8 quantization in SGLang. "
                "Check the `hf_quant_config.json` file for your model's configuration."
            )

        return cls(
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if self.exclude_modules and any(
            module in prefix
            or (
                prefix.startswith("language_model.")
                and module in prefix.removeprefix("language_model.")
            )
            for module in self.exclude_modules
        ):
            return None

        if isinstance(layer, LinearBase):
            return ModelOptFp8LinearMethod(self)
        if self.kv_cache_quant_method and isinstance(layer, RadixAttention):
            return ModelOptFp8KVCacheMethod(self)

        # Add MoE support
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, FusedMoE):
            return ModelOptFp8MoEMethod(self)

        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


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


class ModelOptFp8MoEMethod:
    """MoE method for ModelOpt FP8.
    Supports loading FP8 checkpoints with static weight scale and activation scale.

    Args:
        quant_config: The ModelOpt quantization config.
    """

    def __new__(cls, *args, **kwargs):
        """
        Dynamic class composition pattern.

        This allows us to effectively "inject" FusedMoEMethodBase as a parent class
        at runtime while avoiding circular import issues.
        """
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoEMethodBase

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config: ModelOptFp8Config):
        self.quant_config = quant_config
        self.cutlass_fp8_supported = cutlass_fp8_supported()

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
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

        w13_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts, 2 * intermediate_size, hidden_size, dtype=weight_dtype
            ),
            input_dim=2,
            output_dim=1,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight", w13_weight)

        w2_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts, hidden_size, intermediate_size, dtype=weight_dtype
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
            w13_weight_scale = PerTensorScaleParameter(
                data=torch.full(
                    (num_experts, 2),
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
                from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant

                # Get the maximum scale across w1 and w3 for each expert
                max_w13_scales = layer.w13_weight_scale.max(dim=1).values

                # Requantize each expert's weights using the combined scale
                # w13_weight has shape (num_experts, 2 * intermediate_size, hidden_size)
                # where the first intermediate_size rows are w1, the next are w3
                intermediate_size = layer.w13_weight.shape[1] // 2
                for expert_id in range(layer.w13_weight.shape[0]):
                    start = 0
                    for shard_id in range(2):  # w1 and w3
                        # Dequantize using the original scale for this shard
                        dq_weight = per_tensor_dequantize(
                            layer.w13_weight[expert_id][
                                start : start + intermediate_size, :
                            ],
                            layer.w13_weight_scale[expert_id][shard_id],
                        )
                        # Requantize using the combined max scale
                        (
                            layer.w13_weight[expert_id][
                                start : start + intermediate_size, :
                            ],
                            _,
                        ) = scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])

                        start += intermediate_size

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

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
        from sglang.srt.layers.moe.topk import select_experts

        # Expert selection
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )

        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=inplace,
            activation=activation,
            use_fp8_w8a8=True,
            per_channel_quant=False,  # ModelOpt uses per-tensor quantization
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            no_combine=no_combine,
        )


class ModelOptFp4Config(QuantizationConfig):
    """Config class for FP4."""

    def __init__(
        self,
        is_checkpoint_nvfp4_serialized: bool = False,
        kv_cache_quant_algo: str = None,
        group_size: int = None,
        exclude_modules: List[str] = None,
    ) -> None:
        self.is_checkpoint_nvfp4_serialized = is_checkpoint_nvfp4_serialized
        if is_checkpoint_nvfp4_serialized:
            logger.warning(
                "Detected nvfp4 checkpoint. Please note that the "
                "format is experimental and subject to change."
            )
        self.group_size = group_size
        self.kv_cache_quant_algo = kv_cache_quant_algo
        self.exclude_modules = exclude_modules

    @classmethod
    def get_name(cls) -> str:
        return "modelopt_fp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half, torch.float8_e4m3fn]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["hf_quant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelOptFp4Config":
        quant_config = cls.get_from_keys(config, ["quantization"])
        quant_method = quant_config["quant_algo"]
        if not quant_method in ["FP8", "NVFP4"]:
            raise ValueError(
                f"ModelOpt currently only supports: FP8, NVFP4"
                " quantizations in sglang. Please check the "
                "`hf_quant_config.json` file for your model's "
                "quant configuration."
            )
        is_checkpoint_nvfp4_serialized = "NVFP4" in quant_method
        kv_cache_quant_algo = quant_config["kv_cache_quant_algo"]
        if not kv_cache_quant_algo:
            kv_cache_quant_algo = "auto"
        group_size = quant_config["group_size"]
        exclude_modules = quant_config["exclude_modules"]
        if not (group_size and kv_cache_quant_algo and exclude_modules):
            logger.warning(
                f"group_size: {group_size},"
                f"kv_cache_quant_algo: {kv_cache_quant_algo},"
                f"exclude_modules: {exclude_modules}"
            )
            raise ValueError(
                "NVFP4 quantization requires group size and "
                "kv_cache_quant_algo specified in "
                "hf_quant_config.json"
            )
        return cls(
            is_checkpoint_nvfp4_serialized,
            kv_cache_quant_algo,
            group_size,
            exclude_modules,
        )

    def is_layer_excluded(self, prefix: str, exclude_modules: list):
        import regex as re

        for pattern in exclude_modules:
            regex_str = pattern.replace(".", r"\.").replace("*", r".*")
            if re.fullmatch(regex_str, prefix):
                return True
        return False

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.exclude_modules) or self.is_layer_excluded(
                prefix, self.exclude_modules
            ):
                return UnquantizedLinearMethod()
            return ModelOptFp4LinearMethod(self)
        if self.kv_cache_quant_algo and isinstance(layer, RadixAttention):
            return ModelOptFp8KVCacheMethod(self)
        elif isinstance(layer, FusedMoE):
            return ModelOptNvFp4FusedMoEMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


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

        if ENABLE_TRTLMM_GEN_MOE:
            self.kernel = torch.ops.trtllm.nvfp4_gemm
            self.sf_view_dtype = float4_sf_dtype
        else:
            self.kernel = cutlass_scaled_fp4_mm
            self.sf_view_dtype = torch.float8_e4m3fn

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
            padded_scales.reshape(M, K)
            if scale_ndim == 2
            else padded_scales.reshape(B, M, K)
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
        x_fp4, x_scale_interleaved = scaled_fp4_quant(x, layer.input_scale_inv)

        assert x_fp4.dtype == torch.uint8
        assert x_scale_interleaved.dtype == torch.float8_e4m3fn
        assert layer.weight.dtype == torch.uint8
        assert layer.weight_scale_interleaved.dtype == torch.float8_e4m3fn
        assert layer.alpha.dtype == torch.float32

        out = self.kernel(
            x_fp4,
            layer.weight,
            x_scale_interleaved.view(self.sf_view_dtype),
            layer.weight_scale_interleaved.view(self.sf_view_dtype),
            layer.alpha,
            output_dtype,
        )
        if bias is not None:
            out = out + bias
        return out.view(*output_shape)


class ModelOptNvFp4FusedMoEMethod:
    """
       MoE Method for FP4 Quantization with Blockscales and PerTensorScales
    Args:
        quant_config: NVFP4 Quant Config
    """

    _cache_permute_indices = {}
    weight_dtype = torch.uint8
    weight_scale_dtype = torch.float8_e4m3fn

    def __new__(cls, *args, **kwargs):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoEMethodBase

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config: ModelOptFp4Config):
        self.quant_config = quant_config
        if not is_sm100_supported():
            raise ValueError(
                "Current platform does not support NVFP4"
                " quantization. Please use Blackwell and"
                " above."
            )
        self.enable_flashinfer_moe = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,  # exactly num_experts_per_partition or local_num_experts
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

        layer.local_num_experts = num_experts
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.params_dtype = params_dtype
        layer.quant_config = self.quant_config
        weight_loader = extra_weight_attrs.get("weight_loader")
        # GEMM 1
        w13_weight = ModelWeightParameter(
            data=torch.empty(
                layer.local_num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // 2,
                dtype=self.weight_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight", w13_weight)

        # GEMM 2
        w2_weight = ModelWeightParameter(
            data=torch.empty(
                layer.local_num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // 2,
                dtype=self.weight_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight", w2_weight)

        assert (
            self.quant_config.group_size == 16
        ), f"NVFP4 MoE only supports group_size=16, but got {self.quant_config.group_size}"

        w13_weight_scale = ModelWeightParameter(
            data=torch.empty(
                layer.local_num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.quant_config.group_size,
                dtype=self.weight_scale_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = ModelWeightParameter(
            data=torch.empty(
                layer.local_num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.quant_config.group_size,
                dtype=self.weight_scale_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        )

        w13_weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(layer.local_num_experts, 2, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)

        w2_weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(layer.local_num_experts, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )

        w13_input_scale = PerTensorScaleParameter(
            data=torch.empty(layer.local_num_experts, 2, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = PerTensorScaleParameter(
            data=torch.empty(layer.local_num_experts, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def swizzle_blockscale(self, scale: torch.tensor):
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
            swizzled_scale.reshape(M, K)
            if scale_ndim == 2
            else swizzled_scale.reshape(B, M, K)
        )

    # https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc0/tensorrt_llm/_torch/modules/fused_moe/quantization.py#L1094
    def trtllm_gen_maybe_get_cached_w3_w1_permute_indices(
        self,
        dst_w3_w1_weight: torch.Tensor,
        epilogue_tile_m: int,
        num_elts_per_sf: Union[None, int] = None,
    ) -> torch.Tensor:
        if dst_w3_w1_weight.shape not in self._cache_permute_indices:
            # Get permute indices and chain them together
            permute0 = get_reorder_rows_for_gated_act_gemm_row_indices(dst_w3_w1_weight)
            if num_elts_per_sf is None:
                permute1 = get_shuffle_matrix_a_row_indices(
                    dst_w3_w1_weight, epilogue_tile_m=epilogue_tile_m
                )
            else:
                permute1 = get_shuffle_matrix_sf_a_row_indices(
                    dst_w3_w1_weight,
                    epilogue_tile_m=epilogue_tile_m,
                    num_elts_per_sf=num_elts_per_sf,
                )
            # Memoize permute indices as recompute is **very** costly
            self._cache_permute_indices[dst_w3_w1_weight.shape] = permute0[permute1].to(
                dst_w3_w1_weight.device
            )
        permute_indices = self._cache_permute_indices[dst_w3_w1_weight.shape]
        return permute_indices

    # https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc0/tensorrt_llm/_torch/modules/fused_moe/quantization.py#L1094
    def trtllm_gen_maybe_get_cached_w2_permute_indices(
        self,
        dst_w2_weight: torch.Tensor,
        epilogue_tile_m: int,
        num_elts_per_sf: Union[None, int] = None,
    ) -> torch.Tensor:
        if dst_w2_weight.shape not in self._cache_permute_indices:
            if num_elts_per_sf is None:
                permute_indices = get_shuffle_matrix_a_row_indices(
                    dst_w2_weight, epilogue_tile_m
                ).to(dst_w2_weight.device)
            else:
                permute_indices = get_shuffle_matrix_sf_a_row_indices(
                    dst_w2_weight,
                    epilogue_tile_m=epilogue_tile_m,
                    num_elts_per_sf=num_elts_per_sf,
                ).to(dst_w2_weight.device)
            # Memoize permute indices as recompute is **very** costly
            self._cache_permute_indices[dst_w2_weight.shape] = permute_indices
        permute_indices = self._cache_permute_indices[dst_w2_weight.shape]
        return permute_indices

    # https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc0/tensorrt_llm/_torch/modules/fused_moe/quantization.py#L1094
    def trtllm_gen_process_expert_w3_w1_weight(self, layer: torch.nn.Module):
        for expert_idx in range(layer.local_num_experts):
            dst_w3_w1_weight = layer.w13_weight.data[expert_idx]  # (E, imm * 2, hidden)

            # FIXME: this depends on the kernel internals
            epilogue_tile_m = 128

            # Get permute indices
            permute_indices = self.trtllm_gen_maybe_get_cached_w3_w1_permute_indices(
                dst_w3_w1_weight, epilogue_tile_m
            )

            # Shuffle the weight according to permute indices
            processed_w31_weight_shard = torch.ops.trtllm.shuffle_matrix(
                dst_w3_w1_weight, permute_indices.to(dst_w3_w1_weight.device)
            )

            # Copy the result into device buffer
            dst_w3_w1_weight.copy_(
                processed_w31_weight_shard.view(dst_w3_w1_weight.dtype)
            )

    # https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc0/tensorrt_llm/_torch/modules/fused_moe/quantization.py#L1094
    def trtllm_gen_process_expert_w2_weight(self, layer: torch.nn.Module):
        for expert_idx in range(layer.local_num_experts):
            dst_w2_weight = layer.w2_weight.data[expert_idx]  # (E, hidden, imm)

            # FIXME: this depends on the kernel internals
            epilogue_tile_m = 128

            # Get permuted indices
            permute_indices = self.trtllm_gen_maybe_get_cached_w2_permute_indices(
                dst_w2_weight, epilogue_tile_m
            )

            # Shuffle the weight according to permute indices
            processed_w2_weight = torch.ops.trtllm.shuffle_matrix(
                dst_w2_weight, permute_indices.to(dst_w2_weight.device)
            )

            # Copy the result into device buffer
            dst_w2_weight.copy_(processed_w2_weight.view(dst_w2_weight.dtype))

    # https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc0/tensorrt_llm/_torch/modules/fused_moe/quantization.py#L1094
    def trtllm_gen_process_expert_w3_w1_weight_scale_nvfp4(
        self, layer: torch.nn.Module
    ):
        for expert_idx in range(layer.local_num_experts):
            dst_w3_w1_weight_scale = layer.w13_weight_scale.data[expert_idx]  # (E, ...)
            orig_shape = dst_w3_w1_weight_scale.shape

            # trtllm-gen specific block scales preprocessing logics
            epilogue_tile_m = 128  # FIXME

            # Get permute indices
            permute_indices = self.trtllm_gen_maybe_get_cached_w3_w1_permute_indices(
                dst_w3_w1_weight_scale.view(float4_sf_dtype),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )

            # Shuffle the weight according to permute indices
            w3_w1_weight_scale = torch.ops.trtllm.shuffle_matrix(
                dst_w3_w1_weight_scale.view(float4_sf_dtype), permute_indices
            )

            # Assert should only be removed during debugging
            assert (
                w3_w1_weight_scale.is_cuda
            ), "w3_w1_weight_scale.is_cuda should be true or suffer from slow speed"
            # Interleave the weight.
            # TODO: use torch.ops.trllm.nvfp4_block_scale_interleave after trtllm 1.0.0
            processed_w3_w1_weight_scale = (
                torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(
                    w3_w1_weight_scale.view(float4_sf_dtype).reshape(orig_shape)
                )
            )
            # Copy the result into device buffer
            dst_w3_w1_weight_scale.copy_(
                processed_w3_w1_weight_scale.view(self.weight_scale_dtype).reshape(
                    orig_shape
                )
            )

    # https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc0/tensorrt_llm/_torch/modules/fused_moe/quantization.py#L1094
    def trtllm_gen_process_expert_w2_weight_scale_nvfp4(
        self,
        layer: torch.nn.Module,
    ):
        for expert_idx in range(layer.local_num_experts):
            dst_w2_weight_scale = layer.w2_weight_scale.data[expert_idx]  # (E, ...)

            orig_shape = dst_w2_weight_scale.shape

            # trtllm-gen specific block scales preprocessing logics
            epilogue_tile_m = 128  # FIXME: read from kernel

            # Assert should only be removed during debugging
            assert (
                dst_w2_weight_scale.is_cuda
            ), "dst_w2_weight_scale.is_cuda should be true or suffer from slow speed"

            # Get permute indices
            permute_indices = self.trtllm_gen_maybe_get_cached_w2_permute_indices(
                dst_w2_weight_scale.view(float4_sf_dtype),
                epilogue_tile_m,
                num_elts_per_sf=16,
            )

            # Shuffle the weight according to permute indices
            w_shuffled = torch.ops.trtllm.shuffle_matrix(
                dst_w2_weight_scale.view(dtype=float4_sf_dtype), permute_indices
            )
            # Interleave the weight.
            # TODO: use torch.ops.trllm.nvfp4_block_scale_interleave after trtllm 1.0.0
            processed_w2_weight_scale = (
                torch.ops.tensorrt_llm.nvfp4_block_scale_interleave(w_shuffled)
            )
            # Copy the result into device buffer
            dst_w2_weight_scale.copy_(
                processed_w2_weight_scale.view(self.weight_scale_dtype).reshape(
                    orig_shape
                )
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # GEMM 1
        if not torch.allclose(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ):
            logger.warning_once(
                "w1_weight_scale_2 must match w3_weight_scale_2. "
                "Accuracy may be affected."
            )
        # W13 input scale and global scales
        w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0]
        layer.w13_weight_scale_2 = Parameter(w13_weight_scale_2, requires_grad=False)
        if self.enable_flashinfer_moe or ENABLE_TRTLMM_GEN_MOE:
            w13_input_scale = layer.w13_input_scale.max().to(torch.float32)
        else:
            w13_input_scale = layer.w13_input_scale.max(dim=1).values.to(torch.float32)
        layer.g1_alphas = Parameter(
            (w13_input_scale * w13_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )
        # This is for quantization, so we need to invert it.
        layer.w13_input_scale_quant = Parameter(
            (1 / w13_input_scale).to(torch.float32), requires_grad=False
        )

        # W13 weight and weight scales
        assert (
            layer.w13_weight_scale.shape[2] % 16 == 0
        ), "Expected weight_scale.dim(1) to be divisible by 16"
        assert (
            layer.w13_weight_scale.dtype == torch.float8_e4m3fn
        ), "Weight Blockscale must be represented as FP8-E4M3"
        layer.w13_weight = Parameter(layer.w13_weight.data, requires_grad=False)
        if not ENABLE_TRTLMM_GEN_MOE:
            w13_blockscale_swizzled = self.swizzle_blockscale(layer.w13_weight_scale)
            layer.w13_weight_scale = Parameter(
                w13_blockscale_swizzled, requires_grad=False
            )
        else:
            layer.w13_weight_scale = Parameter(
                layer.w13_weight_scale.data, requires_grad=False
            )
            self.trtllm_gen_process_expert_w3_w1_weight(layer)
            self.trtllm_gen_process_expert_w3_w1_weight_scale_nvfp4(layer)

        # GEMM 2
        # W2 input scale and global scales
        if self.enable_flashinfer_moe or ENABLE_TRTLMM_GEN_MOE:
            w2_input_scale = layer.w2_input_scale.max().to(torch.float32)
        else:
            w2_input_scale = layer.w2_input_scale
        layer.g2_alphas = Parameter(
            (w2_input_scale * layer.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )
        # This is for quantization, so we need to invert it.
        layer.w2_input_scale_quant = Parameter(
            (1 / w2_input_scale).to(torch.float32), requires_grad=False
        )

        # W2 weight and weight scales
        assert (
            layer.w2_weight_scale.shape[2] % 16 == 0
        ), "Expected weight_scale.dim(1) to be divisible by 16"
        assert (
            layer.w2_weight_scale.dtype == torch.float8_e4m3fn
        ), "Weight Blockscale must be represented as FP8-E4M3"
        layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)
        if not ENABLE_TRTLMM_GEN_MOE:
            w2_blockscale_swizzled = self.swizzle_blockscale(layer.w2_weight_scale)
            layer.w2_weight_scale = Parameter(
                w2_blockscale_swizzled, requires_grad=False
            )
        else:
            layer.w2_weight_scale = Parameter(
                layer.w2_weight_scale.data, requires_grad=False
            )
            self.trtllm_gen_process_expert_w2_weight(layer)
            self.trtllm_gen_process_expert_w2_weight_scale_nvfp4(layer)

        # trtllm gen moe param
        if ENABLE_TRTLMM_GEN_MOE:
            layer.g1_scale_c = Parameter(
                (layer.w2_input_scale_quant * layer.g1_alphas).to(torch.float32),
                requires_grad=False,
            )

        # cutlass moe param struct
        device = layer.w13_weight.device
        layer.cutlass_moe_params = CutlassMoEParams(
            CutlassMoEType.BlockscaledFP4,
            device,
            num_experts=layer.local_num_experts,  # local num experts
            intermediate_size_per_partition=layer.w2_weight.shape[2] * 2,  # n
            hidden_size=layer.w13_weight.shape[2] * 2,
        )  # k

    @property
    def load_up_proj_weight_first(self) -> bool:
        # FlashInfer CUTLASS kernel assumes [Up, Gate] Proj as W13
        return self.enable_flashinfer_moe

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        inplace: bool = True,
        no_combine: bool = False,
        routed_scaling_factor: Optional[float] = None,
        ep_rank: Optional[int] = None,
        ep_size: Optional[int] = None,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> torch.Tensor:

        assert activation == "silu", "Only SiLU activation is supported."

        if ENABLE_TRTLMM_GEN_MOE:
            assert (
                self.enable_flashinfer_moe
            ), "must enable flashinfer moe for POC convenience"
            assert use_grouped_topk, "must be true for deepseek_v3"
            assert correction_bias is not None, "must appear for deepseek_v3"
            assert topk_group is not None, "must appear for deepseek_v3"
            assert routed_scaling_factor is not None, "must appear for deepseek_v3"
            assert num_expert_group is not None, "must appear for deepseek_v3"
            assert not apply_router_weight_on_input, "unsupported"
            assert custom_routing_function is None, "unsupported"
            assert (
                num_fused_shared_experts is None or num_fused_shared_experts == 0
            ), "unsupported"
            do_finalize = True
            scale_factor_use_ue8m0 = False
            is_scale_factor_swizzled = False  # use linear layout here
            hidden_states_fp4, hidden_states_scale_linear_fp4 = (
                torch.ops.trtllm.fp4_quantize(
                    x,
                    layer.w13_input_scale_quant,
                    16,
                    scale_factor_use_ue8m0,
                    is_scale_factor_swizzled,
                )
            )

            # FIXME: tile_tokens_dim is hardcoded for now
            tile_tokens_dim = 8

            # https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc1/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py#L195
            outputs = torch.ops.trtllm.fp4_block_scale_moe_runner(
                # NOTE: router out_dtype should be float32!
                # https://github.com/NVIDIA/TensorRT-LLM/blob/v1.0.0rc1/tensorrt_llm/_torch/models/modeling_deepseekv3.py#L378
                router_logits.to(torch.float32),
                correction_bias,
                hidden_states_fp4,
                hidden_states_scale_linear_fp4.view(torch.float8_e4m3fn),
                layer.w13_weight,
                layer.w13_weight_scale.view(torch.float8_e4m3fn),
                layer.w2_weight,
                layer.w2_weight_scale.view(torch.float8_e4m3fn),
                layer.g1_scale_c.data,
                layer.g1_alphas.data,
                layer.g2_alphas.data,
                layer.local_num_experts * ep_size,
                top_k,
                num_expert_group,
                topk_group,
                layer.intermediate_size_per_partition,
                layer.local_num_experts
                * ep_rank,  # self.slot_start # local_expert_start;  use ep_rank if stride!=1
                layer.local_num_experts,  # local_expert_size
                routed_scaling_factor,
                tile_tokens_dim,
                RoutingMethodType.DeepSeekV3,
                do_finalize=do_finalize,
            )
            return outputs[0]

        from sglang.srt.layers.moe.topk import select_experts

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            custom_routing_function=custom_routing_function,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )

        if self.enable_flashinfer_moe:
            assert (
                not apply_router_weight_on_input
            ), "apply_router_weight_on_input is not supported for Flashinfer"
            # TRTLLM Cutlass moe takes in activations in BF16/Half/nvfp4 precision
            # and fp4 quantized weights loaded from the checkpoint
            output = flashinfer_cutlass_fused_moe(
                x,
                topk_ids.to(torch.int),
                topk_weights,
                layer.w13_weight.view(torch.long),
                layer.w2_weight.view(torch.long),
                x.dtype,
                quant_scales=[
                    layer.w13_input_scale_quant,
                    layer.w13_weight_scale.view(torch.int32),
                    layer.g1_alphas,
                    layer.w2_input_scale_quant,
                    layer.w2_weight_scale.view(torch.int32),
                    layer.g2_alphas,
                ],
                ep_size=ep_size,
                ep_rank=ep_rank,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tune_max_num_tokens=next_power_of_2(x.shape[0]),
            )
            return output[0]

        from sglang.srt.layers.moe.cutlass_moe import cutlass_moe_fp4

        return cutlass_moe_fp4(
            a=x,
            a1_gscale=layer.w13_input_scale_quant,
            w1_fp4=layer.w13_weight,
            w1_blockscale=layer.w13_weight_scale,
            w1_alphas=layer.g1_alphas,
            a2_gscale=layer.w2_input_scale_quant,
            w2_fp4=layer.w2_weight,
            w2_blockscale=layer.w2_weight_scale,
            w2_alphas=layer.g2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            params=layer.cutlass_moe_params,
            apply_router_weight_on_input=apply_router_weight_on_input,
        ).to(x.dtype)
