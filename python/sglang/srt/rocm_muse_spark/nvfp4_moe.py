"""Corrected TRT-LLM FP4 weight alignment for the Muse Spark routed experts.

``ModelOptNvFp4FusedMoEMethod.process_weights_after_loading`` shuffles the
packed FP4 weights by calling
``sglang.srt.layers.moe.moe_runner.flashinfer_trtllm.align_fp4_moe_weights_for_flashinfer_trtllm``.
The public aligner has two accuracy bugs for a gated (SwiGLU) MoE, so
``MuseSparkNvFp4FusedMoEMethod`` substitutes the version below before
delegating to the public method.

Only the aligner lives here; weight allocation, scale processing and the GEMMs
all stay on the public path.
"""

from __future__ import annotations

import logging
from typing import cast

import torch

from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod

logger = logging.getLogger(__name__)


def _align_fp4_moe_weights_for_flashinfer_trtllm(layer) -> None:
    """Drop-in replacement for ``align_fp4_moe_weights_for_flashinfer_trtllm``.

    Fixes two accuracy bugs in the public aligner:

    1. It never reorders the gated halves. The TRT-LLM kernel reads the fused
       GEMM1 weight as ``[w3(up), w1(gate)]`` while the checkpoint stores
       ``[w1, w3]``, so ``reorder_w1w3_to_w3w1`` has to run before the shuffle.
       Without it the model is fluent but wrong.
    2. It derives ``g1_scale_c`` from the w1/gate global scale. ``g1_scale_c``
       rescales the gated intermediate before the GEMM2 input quantization, so
       it must use the w3/up-projection global scale (column 1).

    The shuffled weights are written to separate ``gemm{1,2}_*_fp4_shuffled``
    parameters rather than in place, which is what the forward reads.
    """
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
        reorder_w1w3_to_w3w1,
    )
    from sglang.srt.layers.utils import copy_or_rebind_param

    w13_weight = cast(torch.Tensor, layer.w13_weight)
    w2_weight = cast(torch.Tensor, layer.w2_weight)
    w13_weight_scale = cast(torch.Tensor, layer.w13_weight_scale)
    w2_weight_scale = cast(torch.Tensor, layer.w2_weight_scale)

    # Reorder [w1(gate), w3(up)] -> [w3(up), w1(gate)] for TRT-LLM.
    if layer.moe_runner_config.is_gated:
        w13_weight, w13_weight_scale = reorder_w1w3_to_w3w1(
            w13_weight, w13_weight_scale
        )

    (
        gemm1_weights_fp4_shuffled,
        gemm1_scales_fp4_shuffled,
        gemm2_weights_fp4_shuffled,
        gemm2_scales_fp4_shuffled,
    ) = prepare_static_weights_for_trtllm_fp4_moe(
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w2_weight.size(-2),  # hidden_size
        w13_weight.size(-2) // 2,  # intermediate_size
        w13_weight.size(0),  # num_experts
    )

    copy_or_rebind_param(
        layer, "gemm1_weights_fp4_shuffled", gemm1_weights_fp4_shuffled
    )
    copy_or_rebind_param(
        layer, "gemm2_weights_fp4_shuffled", gemm2_weights_fp4_shuffled
    )
    copy_or_rebind_param(layer, "gemm1_scales_fp4_shuffled", gemm1_scales_fp4_shuffled)
    copy_or_rebind_param(layer, "gemm2_scales_fp4_shuffled", gemm2_scales_fp4_shuffled)

    # Compute g1_scale_c from the w3/up global scale, not w1/gate.
    w2_input_scale_quant = cast(torch.Tensor, layer.w2_input_scale_quant)
    w13_weight_scale_2_raw = layer.w13_weight_scale_2

    if (
        layer.moe_runner_config.is_gated
        and w13_weight_scale_2_raw.dim() == 2
        and w13_weight_scale_2_raw.shape[1] >= 2
    ):
        w3_weight_scale_2 = w13_weight_scale_2_raw[:, 1]
        w13_input_scale = (1.0 / layer.w13_input_scale_quant).to(torch.float32)
        g1_scale_c = (w13_input_scale * w3_weight_scale_2 * w2_input_scale_quant).to(
            torch.float32
        )
        logger.debug(
            "Using split w1/w3 global scales for g1_scale_c: w1_scale=%s, w3_scale=%s",
            w13_weight_scale_2_raw[:, 0].tolist()[:3],
            w3_weight_scale_2.tolist()[:3],
        )
    else:
        # Unified scale (1D, or w1 == w3): the public derivation is correct.
        g1_alphas = cast(torch.Tensor, layer.g1_alphas)
        g1_scale_c = (w2_input_scale_quant * g1_alphas).to(torch.float32)

    copy_or_rebind_param(layer, "g1_scale_c", g1_scale_c)


class MuseSparkNvFp4FusedMoEMethod(ModelOptNvFp4FusedMoEMethod):
    """Prepare Muse FP4 experts in SGLang's standard post-load phase."""

    @torch.no_grad()
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.enable_flashinfer_trtllm_moe:
            import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm as trtllm

            trtllm.align_fp4_moe_weights_for_flashinfer_trtllm = (
                _align_fp4_moe_weights_for_flashinfer_trtllm
            )
        super().process_weights_after_loading(layer)

        if self.enable_flashinfer_trtllm_moe:
            del layer.w13_weight, layer.w2_weight
            del layer.w13_weight_scale, layer.w2_weight_scale


__all__ = [
    "MuseSparkNvFp4FusedMoEMethod",
    "_align_fp4_moe_weights_for_flashinfer_trtllm",
]
