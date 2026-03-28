# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe import (
    MoeRunner,
    MoeRunnerBackend,
    MoeRunnerConfig,
    get_moe_runner_backend,
)
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.parameter import ModelWeightParameter, PerTensorScaleParameter
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    cutlass_fp8_supported,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.layers.quantization.modelopt.modelopt import ModelOptQuantConfig
from sglang.srt.layers.quantization.modelopt.schemes.modelopt_fp4 import (
    ACT_STR_TO_TYPE_MAP,
    ActivationType,
    flashinfer_cutlass_fused_moe,
)
from sglang.srt.layers.quantization.utils import (
    convert_to_channelwise,
    per_tensor_dequantize,
    requantize_with_max_scale,
)
from sglang.srt.utils.common import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)


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
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmFp8MoeQuantInfo,
                fused_experts_none_to_flashinfer_trtllm_fp8,
            )
            from sglang.srt.layers.moe.utils import RoutingMethodType

            topk_config = topk_output.topk_config

            # Constraints for ModelOpt FP8 MoE
            assert (
                self.moe_runner_config.activation == "silu"
            ), "Only silu is supported for flashinfer fp8 moe"

            # Enforce Llama4 routing for ModelOpt FP8 MoE for now.
            # TODO(brayden): support other routing methods
            assert topk_config.top_k == 1, "ModelOpt FP8 MoE requires top_k==1"
            assert (
                not topk_config.num_expert_group
            ), "ModelOpt FP8 MoE does not support expert grouping"
            assert (
                not topk_config.topk_group
            ), "ModelOpt FP8 MoE does not support grouped top-k"

            quant_info = FlashInferTrtllmFp8MoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                local_num_experts=layer.num_local_experts,
                intermediate_size=layer.w2_weight.shape[2],
                routing_method_type=RoutingMethodType.Llama4,
                block_quant=False,
                w13_input_scale=layer.w13_input_scale,
                output1_scales_scalar=layer.output1_scales_scalar,
                output1_scales_gate_scalar=layer.output1_scales_gate_scalar,
                output2_scales_scalar=layer.output2_scales_scalar,
                use_routing_scales_on_input=True,
            )

            return fused_experts_none_to_flashinfer_trtllm_fp8(
                dispatch_output, quant_info, self.moe_runner_config
            )

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
