from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple

import torch

from sglang.srt.distributed.parallel_state import get_moe_ep_group
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


def _ue8m0_bytes_to_float(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype != torch.uint8:
        raise TypeError(
            f"Expected raw MXFP4 scale tensor with dtype=torch.uint8, got {scale.dtype}."
        )
    return torch.bitwise_left_shift(scale.to(torch.int32), 23).view(torch.float32)


def _assert_zero_bias(name: str, bias: torch.Tensor | None) -> None:
    if bias is None:
        return
    if torch.count_nonzero(bias).item() != 0:
        raise NotImplementedError(
            f"DeepGEMM Mega MoE does not support non-zero {name}."
        )


def _validate_deep_gemm_mega_layer(layer: torch.nn.Module) -> None:
    runner_config = layer.moe_runner_config
    if runner_config.activation != "silu" or not runner_config.is_gated:
        raise NotImplementedError(
            "DeepGEMM Mega MoE currently supports only gated SwiGLU MoE."
        )
    if getattr(layer, "num_fused_shared_experts", 0) != 0:
        raise NotImplementedError(
            "DeepGEMM Mega MoE does not support shared-expert fusion."
        )
    if runner_config.params_dtype not in (None, torch.bfloat16):
        raise NotImplementedError(
            "DeepGEMM Mega MoE currently supports only bfloat16 outputs."
        )
    if layer.hidden_size % 128 != 0:
        raise ValueError(
            f"DeepGEMM Mega MoE requires hidden_size % 128 == 0, got {layer.hidden_size}."
        )
    if layer.intermediate_size_per_partition % 128 != 0:
        raise ValueError(
            "DeepGEMM Mega MoE requires intermediate_size_per_partition % 128 == 0, "
            f"got {layer.intermediate_size_per_partition}."
        )
    if layer.num_experts % layer.moe_ep_size != 0:
        raise ValueError(
            f"DeepGEMM Mega MoE requires num_experts % moe_ep_size == 0, got "
            f"{layer.num_experts=} {layer.moe_ep_size=}."
        )
    if layer.w13_weight.dtype != torch.uint8 or layer.w2_weight.dtype != torch.uint8:
        raise TypeError("DeepGEMM Mega MoE expects serialized MXFP4 packed weights.")
    if (
        layer.w13_weight_scale.dtype != torch.uint8
        or layer.w2_weight_scale.dtype != torch.uint8
    ):
        raise TypeError(
            "DeepGEMM Mega MoE expects serialized MXFP4 E8M0 scale tensors."
        )
    _assert_zero_bias("w13_weight_bias", getattr(layer, "w13_weight_bias", None))
    _assert_zero_bias("w2_weight_bias", getattr(layer, "w2_weight_bias", None))


def _transform_grouped_mxfp4_weight_scales(
    weight_scale: torch.Tensor,
    *,
    mn: int,
    k: int,
    num_groups: int,
) -> torch.Tensor:
    import deep_gemm

    return deep_gemm.transform_sf_into_required_layout(
        _ue8m0_bytes_to_float(weight_scale.contiguous()),
        mn=mn,
        k=k,
        recipe=(1, 32),
        num_groups=num_groups,
    )


def prepare_deep_gemm_mega_weights(layer: torch.nn.Module) -> None:
    if hasattr(layer, "_deep_gemm_mega_l1_weights") and hasattr(
        layer, "_deep_gemm_mega_l2_weights"
    ):
        return

    if not deep_gemm_wrapper.DEEPGEMM_MEGA_AVAILABLE:
        raise RuntimeError("DeepGEMM Mega MoE is not available in this environment.")

    _validate_deep_gemm_mega_layer(layer)

    num_groups = layer.num_local_experts
    l1_weight = layer.w13_weight.detach().contiguous().view(torch.int8)
    l2_weight = layer.w2_weight.detach().contiguous().view(torch.int8)

    l1_scale = _transform_grouped_mxfp4_weight_scales(
        layer.w13_weight_scale.detach(),
        mn=l1_weight.shape[1],
        k=l1_weight.shape[2] * 2,
        num_groups=num_groups,
    )
    l2_scale = _transform_grouped_mxfp4_weight_scales(
        layer.w2_weight_scale.detach(),
        mn=l2_weight.shape[1],
        k=l2_weight.shape[2] * 2,
        num_groups=num_groups,
    )

    (
        layer._deep_gemm_mega_l1_weights,
        layer._deep_gemm_mega_l2_weights,
    ) = deep_gemm_wrapper.transform_weights_for_mega_moe(
        (l1_weight, l1_scale),
        (l2_weight, l2_scale),
    )


@dataclass
class DeepGemmMegaMoeRuntime:
    symm_buffer: Any
    max_num_tokens_per_rank: int
    device_group: Any
    transformed_l1_weights: Tuple[torch.Tensor, torch.Tensor]
    transformed_l2_weights: Tuple[torch.Tensor, torch.Tensor]


@dataclass
class DeepGemmMegaMoeQuantInfo(MoeQuantInfo):
    runtime: DeepGemmMegaMoeRuntime
    activation_clamp: float | None = None
    fast_math: bool = True


def ensure_deep_gemm_mega_runtime(layer: torch.nn.Module) -> DeepGemmMegaMoeRuntime:
    runtime = getattr(layer, "_deep_gemm_mega_runtime", None)
    if runtime is not None:
        return runtime

    prepare_deep_gemm_mega_weights(layer)

    server_args = get_global_server_args()
    max_num_tokens_per_rank = max(
        getattr(server_args, "cuda_graph_max_bs", None) or 512,
        getattr(server_args, "chunked_prefill_size", None) or 8192,
    )
    device_group = get_moe_ep_group().device_group

    with torch.inference_mode(False):
        symm_buffer = deep_gemm_wrapper.get_symm_buffer_for_mega_moe(
            group=device_group,
            num_experts=layer.num_experts,
            num_max_tokens_per_rank=max_num_tokens_per_rank,
            num_topk=(
                layer.top_k
                if layer.top_k is not None
                else layer.moe_runner_config.top_k
            ),
            hidden=layer.hidden_size,
            intermediate_hidden=layer.intermediate_size_per_partition,
            use_fp8_dispatch=True,
            activation="swiglu",
        )

    runtime = DeepGemmMegaMoeRuntime(
        symm_buffer=symm_buffer,
        max_num_tokens_per_rank=symm_buffer.num_max_tokens_per_rank,
        device_group=device_group,
        transformed_l1_weights=layer._deep_gemm_mega_l1_weights,
        transformed_l2_weights=layer._deep_gemm_mega_l2_weights,
    )
    layer._deep_gemm_mega_runtime = runtime
    return runtime


@register_fused_func("none", "deep_gemm_mega")
def fused_experts_none_to_deep_gemm_mega_mxfp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: DeepGemmMegaMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if runner_config.activation != "silu" or not runner_config.is_gated:
        raise NotImplementedError(
            "DeepGEMM Mega MoE currently supports only gated SwiGLU MoE."
        )

    hidden_states = dispatch_output.hidden_states.contiguous()
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)

    num_tokens = hidden_states.shape[0]
    if num_tokens == 0:
        return StandardCombineInput(
            hidden_states=torch.empty(
                0,
                hidden_states.shape[1],
                device=hidden_states.device,
                dtype=torch.bfloat16,
            )
        )

    runtime = quant_info.runtime
    if num_tokens > runtime.max_num_tokens_per_rank:
        raise ValueError(
            "DeepGEMM Mega MoE symmetric buffer is too small for this forward: "
            f"{num_tokens} > {runtime.max_num_tokens_per_rank}."
        )

    x_q, x_s = sglang_per_token_group_quant_fp8(
        hidden_states,
        32,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    topk_ids = topk_output.topk_ids.to(torch.int64)
    topk_weights = topk_output.topk_weights.to(torch.float32)

    symm_buffer = runtime.symm_buffer
    expected_scale_shape = (num_tokens, symm_buffer.x_sf.shape[1])
    if x_s.dtype != symm_buffer.x_sf.dtype or x_s.shape != expected_scale_shape:
        raise ValueError(
            "DeepGEMM Mega MoE expected packed UE8M0 activation scales with "
            f"shape={expected_scale_shape} and dtype={symm_buffer.x_sf.dtype}, "
            f"got shape={tuple(x_s.shape)} dtype={x_s.dtype}."
        )
    symm_buffer.x[:num_tokens].copy_(x_q)
    symm_buffer.x_sf[:num_tokens].copy_(x_s)
    symm_buffer.topk_idx[:num_tokens].copy_(topk_ids)
    symm_buffer.topk_weights[:num_tokens].copy_(topk_weights)

    output = torch.empty(
        (num_tokens, hidden_states.shape[1]),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    deep_gemm_wrapper.fp8_fp4_mega_moe(
        output,
        runtime.transformed_l1_weights,
        runtime.transformed_l2_weights,
        symm_buffer,
        recipe=(1, 1, 32),
        activation="swiglu",
        activation_clamp=quant_info.activation_clamp,
        fast_math=quant_info.fast_math,
    )
    return StandardCombineInput(hidden_states=output)
