from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import acext
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.distributed.parallel_state import (
    get_moe_expert_parallel_rank,
    get_moe_expert_parallel_world_size,
)
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.utils import is_ppu, set_weight_attrs

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.models.utils import WeightsMapper


class MixedPrecisionW4Config(QuantizationConfig):
    """Config class for W4 Mixed Precision quantization.

    This quantization method supports:
    - INT4 weights for expert layers (using TensorRT-LLM unpack)
    - INT8 for other layers
    - Mixed precision activation schemes
    """

    def __init__(
        self,
        is_checkpoint_int8_serialized: bool = False,
        weight_block_size: List[int] = None,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        int8_channelwise_layers: Optional[list[str]] = None,
    ) -> None:
        super().__init__()
        self.is_checkpoint_int8_serialized = is_checkpoint_int8_serialized
        self.weight_block_size = weight_block_size
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        self.packed_modules_mapping = packed_modules_mapping or {}
        self.int8_channelwise_layers = int8_channelwise_layers or []

        # W4 quantization parameters
        self.weight_bits = 4
        self.pack_factor = 8 // self.weight_bits

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_name(self) -> str:
        return "mixed_precision_w4"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> MixedPrecisionW4Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_int8_serialized = "mixed_precision_w4" in quant_method
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        activation_scheme = cls.get_from_keys_or(
            config, ["activation_scheme"], "dynamic"
        )
        ignored_layers = cls.get_from_keys_or(
            config, ["ignore", "ignored_layers", "modules_to_not_convert"], None
        )
        packed_modules_mapping = cls.get_from_keys_or(
            config, ["packed_modules_mapping"], None
        )
        int8_channelwise_layers = cls.get_from_keys_or(
            config, ["int8_channelwise_layers"], None
        )
        return cls(
            is_checkpoint_int8_serialized=is_checkpoint_int8_serialized,
            weight_block_size=weight_block_size,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            packed_modules_mapping=packed_modules_mapping,
            int8_channelwise_layers=int8_channelwise_layers,
        )

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: WeightsMapper):
        if self.ignored_layers:
            self.ignored_layers = list(
                dict.fromkeys(hf_to_sglang_mapper.apply_list(self.ignored_layers))
            )
        if self.int8_channelwise_layers:
            self.int8_channelwise_layers = list(
                dict.fromkeys(
                    hf_to_sglang_mapper.apply_list(self.int8_channelwise_layers)
                )
            )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8LinearMethod
        from sglang.srt.layers.quantization.w8a8_int8 import (
            W8A8Int8Config,
            W8A8Int8LinearMethod,
        )

        # int8 channelwise
        if should_ignore_layer(
            prefix,
            ignore=self.int8_channelwise_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return W8A8Int8Config().get_quant_method(layer, prefix)

        if isinstance(layer, LinearBase):
            if should_ignore_layer(
                prefix,
                ignore=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if self.weight_block_size is not None:
                return BlockInt8LinearMethod(self)
            else:
                return W8A8Int8LinearMethod(self)
        elif isinstance(layer, FusedMoE):
            if should_ignore_layer(
                prefix,
                ignore=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod()
            return W4AInt8MoEMethod(self)
        return None


class W4AInt8MoEMethod(FusedMoEMethodBase):
    """MoE method for W4 quantization.
    Supports INT4 weights for expert layers with TensorRT-LLM unpacking.
    """

    def __init__(self, quant_config):
        assert is_ppu(), f"W4AInt8 MoE only supported on ppu now"
        self.quant_config = quant_config

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

        # INT4 packed weights for w13 (gate_up_proj) - column parallel
        w13_weight = torch.nn.Parameter(
            data=torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.quant_config.pack_factor,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # INT4 packed weights for w2 (down_proj) - row parallel
        w2_weight = torch.nn.Parameter(
            data=torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        """for quant model with acext a8w4 preprocessing"""
        w13_weight = layer.w13_weight
        w13_weight_shape = w13_weight.shape
        w13_weight = w13_weight.view(
            w13_weight_shape[0],
            w13_weight_shape[2] * self.quant_config.pack_factor,
            w13_weight_shape[1] // self.quant_config.pack_factor,
        )
        # [E, hidden_size, 2 * intermediate // 2]
        layer.w13_weight = Parameter(w13_weight, requires_grad=False)

        w2_weight = layer.w2_weight
        w2_weight_shape = w2_weight.shape
        w2_weight = w2_weight.view(
            w2_weight_shape[0],
            w2_weight_shape[2] * self.quant_config.pack_factor,
            w2_weight_shape[1] // self.quant_config.pack_factor,
        )
        # [E, intermediate, hidden_size // 2]
        layer.w2_weight = Parameter(w2_weight, requires_grad=False)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.acext import (
            _acext_supports_silu_after_clamp,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output
        local_topk_ids = topk_ids
        output = torch.empty_like(x)
        expanded_source_row_to_dest_size = acext.pad_to_multiple_of_16(
            num_tokens=x.shape[0], topk=topk_ids.shape[1]
        )
        Q_type = acext.get_enum_from_booleans(
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            use_fp8_w8a16=False,
            use_int8_w4a8=True,
        )

        ep_size = get_moe_expert_parallel_world_size()
        ep_rank = get_moe_expert_parallel_rank()
        local_e = layer.w13_weight.shape[0]
        if get_moe_expert_parallel_world_size() > 1:
            local_topk_ids = torch.where(
                topk_ids != -1,
                topk_ids + local_e * ep_rank,
                topk_ids,
            )

        if envs.SGLANG_SAIL_ACEXT_MOE_DEBUG.get():
            logging.info(
                f"[sglang][acext][DEBUG] CASE SHAPE: "
                f"M{x.shape[0]}_E{layer.w13_weight.shape[0]}_"
                f"H{layer.w13_weight.shape[1]}_In{layer.w13_weight.shape[2]}_"
                f"topk{topk_ids.shape[1]}"
            )

        # Pick the SwiGLU clamp limit from two mutually-exclusive fields, which
        # also encodes the clamp order:
        #   - gemm1_clamp_limit (Step3.5): silu then clamp -> is_silu_after_clamp=False
        #   - swiglu_limit (DeepSeek V4): clamp then silu -> is_silu_after_clamp=True
        gemm1_clamp_limit = self.moe_runner_config.gemm1_clamp_limit
        swiglu_limit = self.moe_runner_config.swiglu_limit
        is_silu_after_clamp = True
        if gemm1_clamp_limit is not None:
            act_limit = gemm1_clamp_limit
            is_silu_after_clamp = False  # Step3.5: silu then clamp
        else:
            act_limit = swiglu_limit

        # act_limit + is_silu_after_clamp must only be passed when the acext
        # whl supports them (>= 1.1.0), so gate on the pip metadata version to
        # stay compatible with older acext runtimes.
        fused_kwargs = dict(
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
        )
        if _acext_supports_silu_after_clamp():
            fused_kwargs["act_limit"] = act_limit
            fused_kwargs["is_silu_after_clamp"] = is_silu_after_clamp
        elif act_limit is not None:
            logger.warning(
                f"Current acext version does not support SwiGLU clamp (act_limit={act_limit}). "
                f"Clamping is skipped. Please upgrade acext to v1.1.0 or later "
                f"to avoid potential precision issues."
            )

        acext.fusedmoe_wrapper(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            local_topk_ids,
            output,
            expanded_source_row_to_dest_size,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            None,
            None,
            None,
            None,
            ep_rank,
            ep_size,
            Q_type,
            **fused_kwargs,
        )

        return StandardCombineInput(hidden_states=output)
