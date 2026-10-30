# SPDX-License-Identifier: Apache-2.0

# Portable dynamic-token W8A8 INT8 MoE scheme (Triton / DeepGEMM) for SAIL/CUDA,
# the counterpart to the NPU-specific NPUCompressedTensorsW8A8Int8DynamicMoE.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import is_sail, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

__all__ = ["CompressedTensorsW8A8Int8MoE"]

logger = logging.getLogger(__name__)

_is_sail = is_sail()


class CompressedTensorsW8A8Int8MoE(CompressedTensorsMoEScheme):
    """W8A8 INT8 dynamic-per-token fused MoE scheme for CUDA / SAIL.

    Only ``NPUCompressedTensorsW8A8Int8DynamicMoE`` exists for the
    compressed-tensors dynamic-token W8A8 MoE branch, so the branch raised
    ``NotImplementedError`` everywhere else. This scheme keeps the identical
    weight layout (per-expert INT8 weights + per-channel FP32 weight scales,
    dynamic per-token activation) but drives the portable Triton / DeepGEMM MoE
    runner the same way ``W8A8Int8MoEMethod`` in ``quantization/w8a8_int8.py``
    does.
    """

    def __init__(self, weight_quant, input_quant):
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
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
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        params_dtype = torch.int8

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES (per-channel)
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES: dynamic per-token -> no persistent input scale.
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(
            layer.w13_weight.data, requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight.data, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        from sglang.srt.layers import deep_gemm_wrapper

        self.moe_runner_config = moe_runner_config
        # Other platforms keep the existing Triton INT8 path; SAIL DeepEP must use
        # a compatible runner. The backend selection mirrors
        # w8a8_int8.W8A8Int8MoEMethod.
        moe_runner_backend = MoeRunnerBackend.TRITON
        if _is_sail:
            requested_backend = get_moe_runner_backend()
            if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and (
                get_moe_a2a_backend().is_deepep()
                or get_moe_a2a_backend().is_mooncake()
                or requested_backend.is_deep_gemm()
                or (requested_backend.is_auto() and envs.SGLANG_SAIL_DEEPGEMM_MOE.get())
            ):
                moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
                import sglang.srt.layers.moe.moe_runner.sail_deepgemm_moe  # noqa: F401
            elif get_moe_a2a_backend().is_deepep() or requested_backend.is_deep_gemm():
                raise ValueError(
                    "SAIL INT8 DeepEP requires an available DeepGEMM backend"
                )
        self.runner = MoeRunner(moe_runner_backend, moe_runner_config)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        if self.runner.runner_backend.is_deep_gemm():
            from sglang.srt.layers.moe.moe_runner.deep_gemm import DeepGemmMoeQuantInfo

            quant_info = DeepGemmMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                use_int8=True,
                per_channel_quant=True,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                block_shape=None,
            )
        else:
            quant_info = TritonMoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                use_int8_w8a8=True,
                per_channel_quant=True,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
            )
        return self.runner.run(dispatch_output, quant_info)
