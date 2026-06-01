"""sgl_flashinfer_trtllm-specific bits of ``FusedMoEWithLoRA``.

This file holds the two trtllm-specific code blocks that used to be inlined
inside the ``FusedMoEWithLoRA`` class in ``lora/layers.py``:

  - :func:`init_sgl_flashinfer_trtllm_lora` — builds the FP8 block-scale
    ``FlashInferTrtllmFp8MoeQuantInfo`` and stores it on the layer instance.
    Called from ``FusedMoEWithLoRA.__init__`` when the runner backend is the
    sgl_flashinfer_trtllm MoE.
  - :func:`dispatch_sgl_flashinfer_trtllm_lora` — dispatches the LoRA fused
    experts call. Called from ``FusedMoEWithLoRA.run`` for the same backend.

Keeping them here means ``lora/layers.py`` only has tiny ``if backend == ...:
init(self, base_layer)`` / ``dispatch(...)`` injection points for the new
trtllm path instead of ~70 lines of inlined logic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput


def init_sgl_flashinfer_trtllm_lora(layer, base_layer) -> None:
    """Build and store the trtllm FP8 LoRA quant info on the layer.

    Sets ``layer._lora_runner = None`` (trtllm path doesn't use ``MoeRunner``)
    and ``layer._quant_info`` to a fully-populated
    ``FlashInferTrtllmFp8MoeQuantInfo`` (or ``FlashInferTrtllmFp4MoeQuantInfo`` for
    NVFP4 / modelopt checkpoints like Kimi-K2.5-NVFP4).
    """
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmFp8MoeQuantInfo,
        get_activation_type,
    )
    from sglang.srt.layers.moe.utils import RoutingMethodType

    # ---- NVFP4 (modelopt) path ----
    # The fp4 weight loader sets ``g1_scale_c`` on the FusedMoE layer (see
    # ModelOptNvFp4FusedMoEMethod.apply). Mirror the non-LoRA construction in
    # modelopt_quant.py (~L2099) so the fp4 LoRA dispatch gets the same payload.
    # NOTE(w13 layout): the decomposed op runs the gate_up projection as a *non-gated*
    # Gemm2-style GEMM and lets the activation kernel do the SwiGLU split (silu(first)*second),
    # so w13 must be [Gate, Up] with shuffle_matrix_a but WITHOUT reorder_rows_for_gated_act_gemm.
    # The TRTLLM fp4 path uses load_up_proj_weight_first=False (=> [Gate, Up]); verify the
    # processed w13 layout against this at e2e (acc gate) and re-prep if mismatched.
    if hasattr(base_layer, "g1_scale_c"):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            FlashInferTrtllmFp4MoeQuantInfo,
        )

        layer._lora_runner = None
        layer._quant_info = FlashInferTrtllmFp4MoeQuantInfo(
            w13_weight=base_layer.w13_weight.data,
            w2_weight=base_layer.w2_weight.data,
            w13_weight_scale=base_layer.w13_weight_scale.data,
            w2_weight_scale=base_layer.w2_weight_scale.data,
            g1_scale_c=base_layer.g1_scale_c.data,
            g1_alphas=base_layer.g1_alphas.data,
            g2_alphas=base_layer.g2_alphas.data,
            w13_input_scale_quant=base_layer.w13_input_scale_quant,
            global_num_experts=int(base_layer.num_experts),
            local_expert_offset=int(base_layer.moe_ep_rank)
            * int(base_layer.num_local_experts),
            local_num_experts=int(base_layer.num_local_experts),
            intermediate_size_per_partition=int(
                base_layer.intermediate_size_per_partition
            ),
            routing_method_type=int(
                getattr(base_layer, "routing_method_type", None)
                or RoutingMethodType.DeepSeekV3
            ),
        )
        return

    quant_method = base_layer.quant_method
    quant_config = getattr(quant_method, "quant_config", None)
    weight_block_size = getattr(quant_config, "weight_block_size", None)
    if weight_block_size is None:
        weight_block_size = getattr(quant_method, "weight_block_size", None)
    use_mxfp8 = bool(getattr(quant_config, "use_mxfp8", False))
    assert getattr(quant_method, "block_quant", False), (
        "sgl_flashinfer_trtllm LoRA currently requires FP8 block quant."
    )
    assert not use_mxfp8, (
        "sgl_flashinfer_trtllm LoRA currently targets the non-MX FP8 Qwen path."
    )
    assert weight_block_size is not None, (
        "sgl_flashinfer_trtllm LoRA needs the FP8 weight block size."
    )
    w13_weight_scale = getattr(base_layer, "w13_weight_scale_inv", None)
    if w13_weight_scale is None:
        w13_weight_scale = getattr(base_layer, "w13_weight_scale", None)
    w2_weight_scale = getattr(base_layer, "w2_weight_scale_inv", None)
    if w2_weight_scale is None:
        w2_weight_scale = getattr(base_layer, "w2_weight_scale", None)
    assert w13_weight_scale is not None and w2_weight_scale is not None

    layer._lora_runner = None
    layer._quant_info = FlashInferTrtllmFp8MoeQuantInfo(
        w13_weight=base_layer.w13_weight,
        w2_weight=base_layer.w2_weight,
        global_num_experts=int(base_layer.num_experts),
        local_expert_offset=int(base_layer.moe_ep_rank)
        * int(base_layer.num_local_experts),
        local_num_experts=int(base_layer.num_local_experts),
        intermediate_size=base_layer.w2_weight.shape[2],
        routing_method_type=int(
            getattr(base_layer, "routing_method_type", None)
            or RoutingMethodType.DeepSeekV3
        ),
        block_quant=True,
        use_mxfp8=False,
        weight_block_k=weight_block_size[1],
        w13_weight_scale_inv=w13_weight_scale,
        w2_weight_scale_inv=w2_weight_scale,
        activation_type=get_activation_type(
            base_layer.moe_runner_config.activation,
            is_gated=base_layer.moe_runner_config.is_gated,
        ),
    )


def dispatch_sgl_flashinfer_trtllm_lora(
    dispatch_output, quant_info, base_layer, lora_info
) -> "StandardCombineInput":
    """Call the trtllm fused-experts LoRA function for a single layer.

    Looked up at call time so the install-time monkey-patch in
    :mod:`sglang.srt.lora.trtllm_moe` (the two-stream override) takes effect.
    """
    import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm as ft
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmFp4MoeQuantInfo,
    )

    # Resolve the fused-experts fn on the module at CALL TIME so the install-time
    # two-stream monkey-patch (sglang.srt.lora.trtllm_moe) takes effect. Route by
    # quant dtype: NVFP4 -> fp4 LoRA op, else the FP8 path.
    if isinstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo):
        fused_fn = ft.fused_experts_none_to_sgl_flashinfer_trtllm_fp4_lora
    else:
        fused_fn = ft.fused_experts_none_to_sgl_flashinfer_trtllm_fp8_lora

    return fused_fn(
        dispatch_output,
        quant_info,
        base_layer.moe_runner_config,
        lora_info,
    )
