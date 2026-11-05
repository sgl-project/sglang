from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.amx_utils import _amx_process_weight_after_loading
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.moe.utils import get_moe_weight_sizes
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import set_weight_attrs, use_intel_amx_backend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

__all__ = ["CompressedTensorsW8A16Fp8MoE"]

logger = logging.getLogger(__name__)


class CompressedTensorsW8A16Fp8MoE(CompressedTensorsMoEScheme):
    """CPU-only fallback for fp8 MoE: weights stay fp8 (per-channel scale),
    activations are not quantized (weight-only dequant happens inside the
    fused_experts_cpu kernel via CPUQuantMethod.FP8_W8A16).
    """

    def __init__(self, weight_quant, input_quant):
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        assert input_quant is None or input_quant.dynamic
        self.static_input_scales = False

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

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

        params_dtype = torch.float8_e4m3fn

        w13_up_dim, w2_down_dim, weight_padded = get_moe_weight_sizes(
            intermediate_size_per_partition,
            is_aiter_moe=False,
            is_concat=True,
            is_packed=False,
        )
        extra_weight_attrs.update({"weight_padded": weight_padded})

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(num_experts, w13_up_dim, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, w2_down_dim, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES (per-channel, CHANNEL strategy only)
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, w13_up_dim, 1, dtype=torch.float32),
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

        # input_quant is dynamic (or absent): no input scale is registered/loaded.
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module | FusedMoE) -> None:
        # Weight-only path: fp8 weight + per-channel scale are dequantized on
        # the fly in apply_weights; pack the weights to VNNI layout once here
        # so apply_weights doesn't repack them on every call.
        _amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        # No MoeRunner backend supports this CPU weight-only path; apply_weights
        # below implements the forward pass directly.
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.amx_utils import CPUQuantMethod
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        moe_runner_config = self.moe_runner_config

        x, topk_weights = apply_topk_weights_cpu(
            moe_runner_config.apply_router_weight_on_input, topk_weights, x
        )

        # block_size_N == 1 selects the per-channel scale layout in
        # fused_experts_cpu; block_size_K is not meaningful there (w13's
        # reduction dim is hidden_size, w2's is intermediate_size) and is
        # ignored by the kernel, so it is left as a placeholder.
        output = torch.ops.sgl_kernel.fused_experts_cpu(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            False,  # inplace: must be False, see [Note] in fused_experts.
            CPUQuantMethod.FP8_W8A16,
            layer.w13_weight_scale,  # w1_scale
            layer.w2_weight_scale,  # w2_scale
            None,  # w1_zp
            None,  # w2_zp
            [1, 1],  # block_size (per-channel; block_size_K unused)
            None,  # w1_bias
            None,  # w2_bias
            None,  # alpha
            None,  # limit
            use_intel_amx_backend(layer),  # is_vnni
            moe_runner_config.activation,
        )
        return StandardCombineInput(hidden_states=output)
