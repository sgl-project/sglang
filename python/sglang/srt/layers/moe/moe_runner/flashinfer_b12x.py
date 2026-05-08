from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


def ensure_b12x_wrapper(layer: torch.nn.Module) -> None:
    """Lazily create a B12xMoEWrapper on the layer.

    Created lazily because it depends on EP/TP-resolved shapes. inference_mode(False)
    ensures the pre-allocated CUDA-graph buffers are normal tensors, since inference
    tensors cannot be inplace-updated during later CUDA graph capture.
    """
    if getattr(layer, "_b12x_wrapper", None) is not None:
        return

    try:
        from flashinfer import B12xMoEWrapper
    except ImportError as e:
        raise ImportError(
            "flashinfer_b12x backend requires FlashInfer with B12x support."
        ) from e

    from sglang.srt.server_args import get_global_server_args

    assert layer.intermediate_size_per_partition > 0, (
        f"B12x MoE: intermediate_size_per_partition must be > 0, "
        f"got {layer.intermediate_size_per_partition}. Check EP/TP configuration."
    )

    server_args = get_global_server_args()
    use_cuda_graph = server_args is not None and not server_args.disable_cuda_graph
    max_num_tokens = max(
        getattr(server_args, "cuda_graph_max_bs", None) or 512,
        getattr(server_args, "chunked_prefill_size", None) or 8192,
    )
    top_k = layer.top_k if layer.top_k is not None else layer.moe_runner_config.top_k

    with torch.inference_mode(False):
        layer._b12x_wrapper = B12xMoEWrapper(
            num_experts=layer.num_experts,
            top_k=top_k,
            hidden_size=layer.hidden_size,
            intermediate_size=layer.intermediate_size_per_partition,
            use_cuda_graph=use_cuda_graph,
            max_num_tokens=max_num_tokens,
            num_local_experts=layer.num_local_experts,
            output_dtype=torch.bfloat16,
            device=str(layer.w13_weight.device),
            activation=layer.moe_runner_config.activation,
        )


@dataclass
class B12xFp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer B12x FP4 MoE kernels."""

    wrapper: Any

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor

    w13_weight_sf: torch.Tensor
    w2_weight_sf: torch.Tensor

    w1_alpha: torch.Tensor
    w2_alpha: torch.Tensor

    fc2_input_scale: torch.Tensor


@register_fused_func("none", "flashinfer_b12x")
def fused_experts_none_to_flashinfer_b12x_fp4(
    dispatch_output: "StandardDispatchOutput",
    quant_info: B12xFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> "StandardCombineInput":
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert runner_config.activation in ("silu", "relu2"), (
        f"B12x MoE supports activation in ('silu', 'relu2'), "
        f"got {runner_config.activation!r}."
    )
    assert (
        dispatch_output.hidden_states_scale is None
    ), "B12x MoE expects bf16 input; pre-quantized FP4 dispatch is not supported."

    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)

    output = quant_info.wrapper.run(
        x=dispatch_output.hidden_states,
        w1_weight=quant_info.w13_weight,
        w1_weight_sf=quant_info.w13_weight_sf,
        w2_weight=quant_info.w2_weight,
        w2_weight_sf=quant_info.w2_weight_sf,
        token_selected_experts=topk_output.topk_ids,
        token_final_scales=topk_output.topk_weights,
        w1_alpha=quant_info.w1_alpha,
        w2_alpha=quant_info.w2_alpha,
        fc2_input_scale=quant_info.fc2_input_scale,
    )

    return StandardCombineInput(hidden_states=output)
