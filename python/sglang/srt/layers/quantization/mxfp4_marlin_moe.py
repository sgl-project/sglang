from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn import Module

from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import log_info_on_rank0
from sglang.srt.utils.common import is_sm90_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


class Mxfp4MarlinMoEMethod:
    """MXFP4 (E8M0 scales) MoE quantization method using the Marlin backend."""

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_moe_runner(self, layer, moe_runner_config):
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Delegate to the underlying FP8 method for weight creation —
        # the raw weight shapes are the same; only post-loading processing differs.
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        from sglang.srt.layers.quantization.marlin_utils import (
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_moe_mxfp4_layer_for_marlin,
        )

        # Let the FP8 base method handle ROCm normalization, etc.
        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        if not is_sm90_supported():
            raise RuntimeError(
                "DeepSeekV4 MXFP4 Marlin fallback requires Hopper/SM90 or above."
            )
        if not check_moe_marlin_supports_layer(layer, 32):
            raise RuntimeError(
                "Current DeepSeekV4 MoE layer does not satisfy Marlin constraints."
            )

        # NOTE: the Marlin MoE runner consumes w13 in the checkpoint's
        # native ``[w1; w3]`` order -- see ``silu_and_mul`` in
        # fused_marlin_moe.py which expects ``gate = intermediate[:, :N]``
        # (first half) and ``up = intermediate[:, N:]`` (second half).
        # Unlike the flashinfer trtllm_fp4 kernel (which wants [w3, w1]),
        # we must *not* call ``reorder_w1w3_to_w3w1`` here.

        log_info_on_rank0(
            logger,
            f"Preparing DeepSeekV4 MXFP4 experts for Marlin backend "
            f"(layer: {self.prefix})...",
        )
        prepare_moe_mxfp4_layer_for_marlin(layer)
        layer._dsv4_mxfp4_backend = "marlin"

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")

        quant_info = MarlinMoeQuantInfo(
            w13_qweight=layer.w13_weight,
            w2_qweight=layer.w2_weight,
            w13_scales=layer.w13_weight_scale_inv,
            w2_scales=layer.w2_weight_scale_inv,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            is_k_full=True,
        )
        runner_output = self.runner.run(dispatch_output, quant_info=quant_info)

        return StandardCombineInput(hidden_states=runner_output.hidden_states)
