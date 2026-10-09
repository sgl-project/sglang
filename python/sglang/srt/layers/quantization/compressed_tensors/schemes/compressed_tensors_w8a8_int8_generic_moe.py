# Migrated from v0.5.18_rel/glm-5.3-flash dirty WIP:
#   sglang/srt/layers/quantization/compressed_tensors/schemes/
#     compressed_tensors_w8a8_int8_generic_moe.py  (213 lines, class
#     CompressedTensorsW8A8Int8MoE).
#
# The upstream cherry-pick only shipped the NPU-specific
# NPUCompressedTensorsW8A8Int8DynamicMoE; on PPU/CUDA the dynamic-token
# W8A8 MoE branch raised NotImplementedError.  This file adds a portable
# (Triton / DeepGEMM / acext) counterpart validated on PPU GLM-5.3-Flash
# INT8 W8A8 MoE workloads.
#
# API drift from v0.5.18 → v0.5.21:
#   - MoeRunnerBackend moved from sglang.srt.layers.moe to
#     sglang.srt.layers.moe.utils.
#   - get_moe_runner_backend / get_moe_a2a_backend moved to
#     sglang.srt.layers.moe.utils.
#   - is_ppu lives in sglang.srt.utils.common (re-exported via utils).
#   - DeepGemmMoeQuantInfo grew extra optional fields (is_fp4_experts,
#     use_mxfp8); the W8A8 fields used here are unchanged.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.environ import envs
from sglang.srt.layers.moe import MoeRunner, MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    get_moe_a2a_backend,
    get_moe_runner_backend,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsMoEScheme,
)
from sglang.srt.utils import is_ppu, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

__all__ = ["CompressedTensorsW8A8Int8MoE"]

logger = logging.getLogger(__name__)

_is_ppu = is_ppu()


class CompressedTensorsW8A8Int8MoE(CompressedTensorsMoEScheme):
    """Generic (CUDA / PPU) W8A8 INT8 dynamic-per-token fused MoE scheme.

    The upstream cherry-pick only shipped ``NPUCompressedTensorsW8A8Int8DynamicMoE``
    (Ascend-only), so on PPU/CUDA the compressed-tensors dynamic-token W8A8 MoE
    branch raised ``NotImplementedError``. This scheme reuses the identical
    weight layout (per-expert INT8 weights + per-channel FP32 weight scales,
    dynamic per-token activation) but drives the portable Triton / DeepGEMM MoE
    runner exactly like ``W8A8Int8MoEMethod`` in ``quantization/w8a8_int8.py``.
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
        moe_runner_backend = get_moe_runner_backend()
        # Only PPU deepep / deepgemm support int8 fused MoE natively.
        if (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and _is_ppu
            and (
                get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake()
            )
        ):
            moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
        elif moe_runner_backend.is_auto():
            if (
                envs.SGLANG_SAIL_DEEPGEMM_MOE.get()
                and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            ):
                moe_runner_backend = MoeRunnerBackend.DEEP_GEMM
            elif envs.SGLANG_SAIL_USE_ACEXT_CUDA.get():
                moe_runner_backend = MoeRunnerBackend.ACEXT
            else:
                moe_runner_backend = MoeRunnerBackend.TRITON

        if moe_runner_backend.is_acext():
            import sglang.srt.layers.moe.moe_runner.acext  # noqa: F401
        if moe_runner_backend.is_deep_gemm():
            import sglang.srt.layers.moe.moe_runner.ppu_deepgemm_moe  # noqa: F401

        if (
            moe_runner_backend.is_acext()
            or moe_runner_backend.is_triton()
            or moe_runner_backend.is_deep_gemm()
        ):
            self.runner = MoeRunner(moe_runner_backend, moe_runner_config)
        else:
            # Fall back to triton for any unsupported auto-resolved backend.
            self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

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
