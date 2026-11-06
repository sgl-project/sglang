"""Muse-specific gated alignment for FlashInfer's TRT-LLM FP4 MoE kernel."""

from __future__ import annotations

from typing import cast

import torch

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
    _compute_gemm1_alphas,
)
from sglang.srt.layers.utils import copy_or_rebind_param


def _align_fp4_moe_weights_for_flashinfer_trtllm(layer) -> None:
    """Align Muse's ``[w1, w3]`` checkpoint layout in-place.

    The kernel permutation interleaves gated rows but does not exchange the
    checkpoint's logical gate/up halves. Muse therefore performs the explicit
    ``[w1, w3] -> [w3, w1]`` reorder before the standard shuffle. The aligned
    tensors replace the canonical parameters so the model keeps only one copy.
    """
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
        reorder_w1w3_to_w3w1,
    )
    from sglang.srt.layers.utils import copy_or_rebind_param

    w13_weight = cast(torch.Tensor, layer.gate_up_proj)
    w2_weight = cast(torch.Tensor, layer.down_proj)
    w13_weight_scale = cast(torch.Tensor, layer.gate_up_proj_weight_scale)
    w2_weight_scale = cast(torch.Tensor, layer.down_proj_weight_scale)

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
        w2_weight.size(-2),
        w13_weight.size(-2) // 2,
        w13_weight.size(0),
        is_gated=layer.moe_runner_config.is_gated,
    )

    copy_or_rebind_param(layer, "gate_up_proj", gemm1_weights_fp4_shuffled.contiguous())
    copy_or_rebind_param(layer, "down_proj", gemm2_weights_fp4_shuffled.contiguous())
    copy_or_rebind_param(
        layer, "gate_up_proj_weight_scale", gemm1_scales_fp4_shuffled.contiguous()
    )
    copy_or_rebind_param(
        layer, "down_proj_weight_scale", gemm2_scales_fp4_shuffled.contiguous()
    )

    # The up-projection (w3), not the gate (w1), supplies TRT-LLM's extra
    # GEMM1-output dequantization factor before GEMM2 requantization.
    w2_input_scale_quant = cast(torch.Tensor, layer.w2_input_scale_quant)
    w13_weight_scale_2 = cast(torch.Tensor, layer.gate_up_proj_weight_scale_2)
    if (
        layer.moe_runner_config.is_gated
        and w13_weight_scale_2.dim() == 2
        and w13_weight_scale_2.shape[1] >= 2
    ):
        w13_input_scale = (1.0 / layer.w13_input_scale_quant).to(torch.float32)
        g1_scale_c = (
            w13_input_scale * w13_weight_scale_2[:, 1] * w2_input_scale_quant
        ).to(torch.float32)
    else:
        g1_alphas = cast(torch.Tensor, layer.g1_alphas)
        g1_scale_c = (w2_input_scale_quant * g1_alphas).to(torch.float32)
    copy_or_rebind_param(layer, "g1_scale_c", g1_scale_c.contiguous())


class MuseSparkNvFp4FusedMoEMethod(ModelOptNvFp4FusedMoEMethod):
    """Prepare Muse FP4 experts without process-global mutation or duplication."""

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        """Allocate the serialized HF expert tensors without renaming them."""
        n, h, i = num_experts, hidden_size, intermediate_size_per_partition
        group = self.quant_config.group_size
        for name, shape, dtype in (
            ("gate_up_proj", (n, 2 * i, h // 2), torch.uint8),
            ("down_proj", (n, h, i // 2), torch.uint8),
            ("gate_up_proj_weight_scale", (n, 2 * i, h // group), torch.float8_e4m3fn),
            ("down_proj_weight_scale", (n, h, i // group), torch.float8_e4m3fn),
            ("gate_up_proj_weight_scale_2", (n, 2), torch.float32),
            ("down_proj_weight_scale_2", (n,), torch.float32),
            ("gate_up_proj_input_scale", (n, 2), torch.float32),
            ("down_proj_input_scale", (n,), torch.float32),
        ):
            layer.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.empty(shape, dtype=dtype), requires_grad=False
                ),
            )

    # The base method's Marlin branch, and prepare_moe_nvfp4_layer_for_marlin
    # underneath it, address the experts by their pre-rename names.
    _MARLIN_PARAM_ALIASES = (
        ("w13_weight", "gate_up_proj"),
        ("w2_weight", "down_proj"),
        ("w13_weight_scale", "gate_up_proj_weight_scale"),
        ("w2_weight_scale", "down_proj_weight_scale"),
        ("w13_weight_scale_2", "gate_up_proj_weight_scale_2"),
        ("w2_weight_scale_2", "down_proj_weight_scale_2"),
    )

    def _process_weights_for_marlin(self, layer):
        for alias, native in self._MARLIN_PARAM_ALIASES:
            setattr(layer, alias, getattr(layer, native))
        result = super().process_weights_after_loading(layer)
        # Marlin repacks into freshly allocated parameters. Drop the native
        # names so the unpacked originals are freed instead of being kept
        # alive at full size alongside the repacked copies.
        for _, native in self._MARLIN_PARAM_ALIASES:
            if native in layer._parameters:
                del layer._parameters[native]
            elif hasattr(layer, native):
                delattr(layer, native)
        return result

    def process_weights_after_loading(self, layer):
        """Prepare the TRT-LLM layout from native HF parameters in place."""
        backend = getattr(self, "_moe_runner_backend", None)
        if backend is not None and backend.is_marlin():
            return self._process_weights_for_marlin(layer)
        input_scale = layer.gate_up_proj_input_scale.max().float()
        output_scale = layer.down_proj_input_scale.max().float()
        if self.quant_config.use_per_token_activation:
            input_scale = torch.ones_like(input_scale)
            output_scale = torch.ones_like(output_scale)
        gate_alpha, up_alpha = _compute_gemm1_alphas(
            layer.gate_up_proj_weight_scale_2, input_scale, True
        )
        copy_or_rebind_param(layer, "g1_alphas", gate_alpha)
        copy_or_rebind_param(layer, "g1_alphas_up", up_alpha)
        copy_or_rebind_param(
            layer, "g2_alphas", output_scale * layer.down_proj_weight_scale_2
        )
        copy_or_rebind_param(layer, "w13_input_scale_quant", input_scale.reciprocal())
        copy_or_rebind_param(layer, "w2_input_scale_quant", output_scale.reciprocal())
        for weight, scale in (
            (layer.gate_up_proj, layer.gate_up_proj_weight_scale),
            (layer.down_proj, layer.down_proj_weight_scale),
        ):
            if (
                scale.dtype != torch.float8_e4m3fn
                or scale.shape[-1] != weight.shape[-1] * 2 // 16
            ):
                raise ValueError(
                    "Muse NVFP4 requires one FP8 scale per 16 weight elements"
                )
        self._align_fp4_moe_weights_for_flashinfer_trtllm(layer)

    def _align_fp4_moe_weights_for_flashinfer_trtllm(self, layer) -> None:
        if getattr(layer, "use_megamoe", False):
            from sglang.srt.layers.muse_spark.muse_spark_megamoe import (
                build_nvfp4_megamoe_weights,
            )

            # Capture scale metadata while gate/up order is still canonical.
            build_nvfp4_megamoe_weights(layer)
        _align_fp4_moe_weights_for_flashinfer_trtllm(layer)
        if getattr(layer, "use_megamoe", False):
            from sglang.srt.layers.muse_spark.muse_spark_megamoe import (
                bind_megamoe_shared_weights,
            )

            bind_megamoe_shared_weights(layer, "nvfp4xnvfp4")


__all__ = [
    "MuseSparkNvFp4FusedMoEMethod",
    "_align_fp4_moe_weights_for_flashinfer_trtllm",
]
