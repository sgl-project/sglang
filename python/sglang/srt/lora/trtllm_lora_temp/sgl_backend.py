"""Registers the ``experimental_sgl_trtllm`` MoE fused-func.

``MoeRunner.__init__`` requires a registered fused-func at CONSTRUCTION time even
for the LoRA case, because LoRA is attached *after* the MoE layer is built (so
``lora_enabled`` is False inside ``MoeRunner.__init__``). At run time the runner
skips this for the LoRA path; the no-LoRA path delegates entirely to the upstream
flashinfer_trtllm dispatch (all quant types), so no-LoRA is identical to the stock backend.

Registration fires at model-build time via a one-line import of this module in
``moe_runner/flashinfer_trtllm.py`` (the module already imported there for the
trtllm weight-prep). Keeping the dispatch body here keeps that file otherwise
pristine; the sgl FP8 LoRA dispatch lives in ``sgl_fp8_moe.py`` (used only by the LoRA path).
"""

from sglang.srt.layers.moe.moe_runner.base import register_fused_func


@register_fused_func("none", "experimental_sgl_trtllm")
def fused_experts_none_to_experimental_sgl_trtllm(
    dispatch_output, quant_info, runner_config
):
    # No-LoRA on the experimental_sgl_trtllm backend == upstream flashinfer_trtllm for EVERY
    # quant type (FP8 / NVFP4 / bf16). When LoRA is disabled the runner calls this fused-func,
    # so delegating entirely to upstream keeps the no-LoRA path byte-identical to the stock
    # backend. The new sgl kernels (sgl_fp8_moe, trtllm_*_routed_moe_lora) run ONLY on the LoRA
    # dispatch (lora_dispatch.py), never here.
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        fused_experts_none_to_flashinfer_trtllm,
    )

    return fused_experts_none_to_flashinfer_trtllm(
        dispatch_output, quant_info, runner_config
    )
