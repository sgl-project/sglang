"""Register the no-LoRA alias for ``experimental_sgl_marlin``."""

from sglang.srt.layers.moe.moe_runner.base import register_fused_func


@register_fused_func("none", "experimental_sgl_marlin")
def fused_experts_none_to_experimental_sgl_marlin(
    dispatch_output, quant_info, runner_config
):
    from sglang.srt.layers.moe.moe_runner.marlin import fused_experts_none_to_marlin

    return fused_experts_none_to_marlin(dispatch_output, quant_info, runner_config)
