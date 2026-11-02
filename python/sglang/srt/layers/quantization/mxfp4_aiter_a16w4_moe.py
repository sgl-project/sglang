"""MXFP4 expert backend running AITER's a16w4 (bf16 activation x MXFP4 weight) MoE.

``Fp8Config`` selects this for FP4-expert checkpoints under
``--moe-runner-backend aiter``. The checkpoint's separated ``[gate; up]`` w13
rows are preshuffled and consumed with ``GateMode.SEPARATED``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import Module

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.layers.moe.utils import RoutingMethodType


class Mxfp4AiterA16W4MoEMethod:
    """MXFP4 (E8M0 scales) MoE via aiter fused_moe with bf16 activations."""

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        if with_bias:
            raise NotImplementedError(
                "Mxfp4AiterA16W4MoEMethod does not support MoE GEMM biases."
            )
        # shuffle_scale_a16w4 passes fp32 scales through untouched, so keep E8M0.
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            fp4_scale_dtype=torch.float8_e8m0fnu,
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self, layer: Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        # The native A16W4/A8W4 SiLU epilogue does not apply an activation
        # clamp. A positive swiglu_limit can also override our separated
        # weight layout in the shared runner. Reject either config spelling
        # before runner setup rather than silently changing the model.
        if any(
            limit is not None and limit != 0
            for limit in (
                moe_runner_config.swiglu_limit,
                moe_runner_config.gemm1_clamp_limit,
            )
        ):
            raise ValueError(
                "Native MXFP4 A16W4/A8W4 does not support activation clamps; "
                "swiglu_limit and gemm1_clamp_limit must be unset or 0."
            )
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.AITER, moe_runner_config)

    def get_moe_runner_backend(self, layer):
        return self.runner.runner_backend

    def get_moe_routing_method_type(
        self, layer: Module, requested_method: RoutingMethodType | None
    ) -> RoutingMethodType | None:
        return requested_method

    def process_weights_after_loading(self, layer: Module) -> None:
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        num_experts = layer.w13_weight.shape[0]
        fp4_dtype = torch.float4_e2m1fn_x2

        # gate_up=False keeps the [gate; up] row order that GateMode.SEPARATED reads.
        layer.w13_weight.data = shuffle_weight_a16w4(
            layer.w13_weight.data.view(fp4_dtype), 16, False
        )
        layer.w2_weight.data = shuffle_weight_a16w4(
            layer.w2_weight.data.view(fp4_dtype), 16, False
        )
        w13_scale = layer.w13_weight_scale_inv.data
        w2_scale = layer.w2_weight_scale_inv.data
        layer.w13_weight_scale_inv = torch.nn.Parameter(
            shuffle_scale_a16w4(
                w13_scale.view(-1, w13_scale.shape[-1]), num_experts, False
            ),
            requires_grad=False,
        )
        layer.w2_weight_scale_inv = torch.nn.Parameter(
            shuffle_scale_a16w4(
                w2_scale.view(-1, w2_scale.shape[-1]), num_experts, False
            ),
            requires_grad=False,
        )

        # aiter fused_moe reads these off w1/w2 to pick the preshuffled
        # SEPARATED a16w4 kernels; the .data swap above drops them.
        for weight in (layer.w13_weight, layer.w2_weight):
            weight.is_shuffled = True
            weight.gate_up_interleave = False

        layer.dispatcher.set_quant_config(
            {"weight_dtype": layer.w13_weight.dtype, "use_mxfp8": False}
        )

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from aiter.ops.flydsl.moe_common import GateMode

        from sglang.srt.layers.moe.moe_runner.aiter import (
            AiterMoeQuantInfo,
            AiterQuantType,
        )

        quant_info = AiterMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            quant_type=AiterQuantType.PER_1X32,
            w13_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            expert_mask=layer.dispatcher.expert_mask_gpu,
            swiglu_limit=self.moe_runner_config.swiglu_limit or 0.0,
            fused_moe_kwargs={"gate_mode": GateMode.SEPARATED.value},
        )
        return self.runner.run(dispatch_output, quant_info)
