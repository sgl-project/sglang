# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py

import logging
from typing import Any, Callable, Dict, List, Optional

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
    requantize_with_max_scale,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.utils import is_cuda

if is_cuda():
    from sgl_kernel import cutlass_scaled_fp4_mm, scaled_fp4_quant

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
            module in prefix for module in self.exclude_modules
        ):
            return None

        if isinstance(layer, LinearBase):
            return ModelOptFp8LinearMethod(self)
        if self.kv_cache_quant_method and isinstance(layer, RadixAttention):
            return ModelOptFp8KVCacheMethod(self)

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
        x_fp4, x_scale_interleaved = scaled_fp4_quant(x, 1 / layer.input_scale)

        assert x_fp4.dtype == torch.uint8
        assert x_scale_interleaved.dtype == torch.float8_e4m3fn
        assert layer.weight.dtype == torch.uint8
        assert layer.weight_scale_interleaved.dtype == torch.float8_e4m3fn
        assert layer.alpha.dtype == torch.float32

        out = cutlass_scaled_fp4_mm(
            x_fp4,
            layer.weight,
            x_scale_interleaved,
            layer.weight_scale_interleaved,
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

        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        layer.quant_config = self.quant_config
        weight_dtype = torch.uint8
        weight_scale_dtype = torch.float8_e4m3fn
        weight_loader = extra_weight_attrs.get("weight_loader")
        # GEMM 1
        w13_weight = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
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
                num_experts,
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
                num_experts,
                2 * intermediate_size_per_partition,
                # 2 fp4 items are packed in the input dimension
                hidden_size // self.quant_config.group_size,
                dtype=weight_scale_dtype,
            ),
            input_dim=1,
            output_dim=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = ModelWeightParameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                # 2 fp4 items are packed in the input dimension
                intermediate_size_per_partition // self.quant_config.group_size,
                dtype=weight_scale_dtype,
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
            data=torch.empty(num_experts, 2, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_weight_scale_2", w13_weight_scale_2)

        w2_weight_scale_2 = PerTensorScaleParameter(
            data=torch.empty(num_experts, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w2_weight_scale_2", w2_weight_scale_2)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )

        w13_input_scale = PerTensorScaleParameter(
            data=torch.empty(num_experts, 2, dtype=torch.float32),
            weight_loader=weight_loader,
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = PerTensorScaleParameter(
            data=torch.empty(num_experts, dtype=torch.float32),
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        # GEMM 1
        if not torch.allclose(
            layer.w13_weight_scale_2[:, 0], layer.w13_weight_scale_2[:, 1]
        ):
            logger.warning_once(
                "w1_weight_scale_2 must match w3_weight_scale_2. "
                "Accuracy may be affected."
            )

        w13_weight_scale_2 = layer.w13_weight_scale_2[:, 0]
        layer.w13_weight_scale_2 = Parameter(w13_weight_scale_2, requires_grad=False)

        w13_input_scale = layer.w13_input_scale.max(dim=1).values.to(torch.float32)
        layer.g1_alphas = Parameter(
            (w13_input_scale * w13_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )

        assert (
            layer.w13_weight_scale.shape[2] % 16 == 0
        ), "Expected weight_scale.dim(1) to be divisible by 16"
        assert (
            layer.w13_weight_scale.dtype == torch.float8_e4m3fn
        ), "Weight Blockscale must be represented as FP8-E4M3"
        w13_blockscale_swizzled = self.swizzle_blockscale(layer.w13_weight_scale)

        layer.w13_blockscale_swizzled = Parameter(
            w13_blockscale_swizzled, requires_grad=False
        )

        # This is for quantization, so we need to invert it.
        layer.w13_input_scale_quant = Parameter(
            (1 / w13_input_scale).to(torch.float32), requires_grad=False
        )

        layer.w13_weight = Parameter(layer.w13_weight.data, requires_grad=False)

        # GEMM 2
        layer.g2_alphas = Parameter(
            (layer.w2_input_scale * layer.w2_weight_scale_2).to(torch.float32),
            requires_grad=False,
        )

        # This is for quantization, so we need to invert it.
        layer.w2_input_scale_quant = Parameter(
            (1 / layer.w2_input_scale).to(torch.float32), requires_grad=False
        )

        assert (
            layer.w2_weight_scale.shape[2] % 16 == 0
        ), "Expected weight_scale.dim(1) to be divisible by 16"
        assert (
            layer.w2_weight_scale.dtype == torch.float8_e4m3fn
        ), "Weight Blockscale must be represented as FP8-E4M3"
        w2_blockscale_swizzled = self.swizzle_blockscale(layer.w2_weight_scale)

        layer.w2_blockscale_swizzled = Parameter(
            w2_blockscale_swizzled, requires_grad=False
        )
        layer.w2_weight = Parameter(layer.w2_weight.data, requires_grad=False)

        device = layer.w13_weight.device
        layer.cutlass_moe_params = CutlassMoEParams(
            CutlassMoEType.BlockscaledFP4,
            device,
            num_experts=layer.num_experts,
            intermediate_size_per_partition=layer.w2_weight.shape[2] * 2,  # n
            hidden_size=layer.w13_weight.shape[2] * 2,
        )  # k

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

        assert activation == "silu", "Only SiLU activation is supported."

        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
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

        from sglang.srt.layers.moe.cutlass_moe import cutlass_moe_fp4

        return cutlass_moe_fp4(
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
            apply_router_weight_on_input=apply_router_weight_on_input,
        ).to(x.dtype)
