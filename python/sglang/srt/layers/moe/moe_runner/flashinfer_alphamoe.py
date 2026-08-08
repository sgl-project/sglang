"""FlashInfer AlphaMoE W8A8 adapter for exact SM100/SM103 GPUs.

This backend is deliberately narrow.  It consumes native fine-grained FP8
checkpoints with 128x128 FP32 block scales, a gated SiLU expert, raw FP32
router logits, and ``ep_size == 1``.  Gate/up rows are converted once from the
checkpoint's canonical ``[gate; up]`` layout to AlphaMoE's 8-row interleave.

NVFP4/ModelOpt checkpoints are not accepted.  Their global activation and
weight scales are part of the quantization contract, while the AlphaMoE W8A8
kernel API accepts only FP8 tensors and per-128x128 FP32 block scales.  Dropping
those global scales would silently produce incorrect output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


ALPHAMOE_BLOCK_M = 8
ALPHAMOE_WEIGHT_BLOCK = (128, 128)
ALPHAMOE_MAX_EXPERTS = 512
ALPHAMOE_MAX_TOP_K = 16


def validate_alphamoe_runner_contract(
    *,
    ep_size: int,
    a2a_is_none: bool,
    num_fused_shared_experts: int,
    with_bias: bool,
    is_gated: bool,
    activation: str,
    apply_router_weight_on_input: bool,
    no_combine: bool,
    gemm1_alpha: float | None,
    gemm1_clamp_limit: float | None,
    swiglu_limit: float | None,
    params_dtype: torch.dtype,
    top_k: int | None,
    num_experts: int,
) -> None:
    """Reject runner features outside the frozen AlphaMoE kernel contract."""

    if ep_size != 1 or not a2a_is_none:
        raise ValueError(
            "flashinfer_alphamoe requires ep_size=1 and moe_a2a_backend=none"
        )
    if num_fused_shared_experts != 0:
        raise ValueError("flashinfer_alphamoe requires shared-expert fusion disabled")
    if (
        with_bias
        or not is_gated
        or activation != "silu"
        or apply_router_weight_on_input
        or no_combine
        or gemm1_alpha is not None
        or gemm1_clamp_limit is not None
        or swiglu_limit is not None
    ):
        raise ValueError(
            "flashinfer_alphamoe supports only combined, unbiased gated SiLU "
            "experts without input-side router weights or activation clamps"
        )
    if params_dtype != torch.bfloat16:
        raise ValueError("flashinfer_alphamoe currently requires BF16 activations")
    if top_k is None or not 1 <= top_k <= ALPHAMOE_MAX_TOP_K:
        raise ValueError(
            f"flashinfer_alphamoe requires top_k in [1, {ALPHAMOE_MAX_TOP_K}]"
        )
    if top_k > num_experts:
        raise ValueError(
            f"flashinfer_alphamoe top_k={top_k} exceeds experts={num_experts}"
        )


class AlphaMoeRoutePlanCache:
    """Own reusable route plans and output buffers for eager and graph runs."""

    def __init__(self) -> None:
        self._graph_workspaces: dict[tuple[int, ...], tuple[Any, torch.Tensor]] = {}
        self._eager_workspaces: dict[
            tuple[int | None, int],
            tuple[tuple[int, ...], tuple[Any, torch.Tensor]],
        ] = {}

    def get(
        self,
        logits: torch.Tensor,
        *,
        hidden_size: int,
        top_k: int,
        block_m: int,
    ) -> tuple[Any, torch.Tensor]:
        from flashinfer.fused_moe.alphamoe_fused_router import (
            allocate_alphamoe_route_plan,
        )

        stream_id = int(torch.cuda.current_stream(logits.device).cuda_stream)
        key = (
            logits.shape[0],
            logits.shape[1],
            hidden_size,
            top_k,
            block_m,
            logits.device.index,
            stream_id,
        )
        from sglang.srt.model_executor.runner import get_is_capture_mode

        eager_key = (logits.device.index, stream_id)
        is_capture = get_is_capture_mode()
        if is_capture:
            workspace = self._graph_workspaces.get(key)
        else:
            eager_entry = self._eager_workspaces.get(eager_key)
            workspace = (
                None if eager_entry is None or eager_entry[0] != key else eager_entry[1]
            )
        if workspace is None:
            plan = allocate_alphamoe_route_plan(
                logits,
                top_k=top_k,
                block_m=block_m,
                has_shared_expert=False,
            )
            out = torch.empty(
                (logits.shape[0], hidden_size),
                dtype=torch.bfloat16,
                device=logits.device,
            )
            workspace = (plan, out)
            if is_capture:
                self._graph_workspaces[key] = workspace
            else:
                # Bound eager memory: retain only the latest shape on each stream.
                self._eager_workspaces[eager_key] = (key, workspace)
        return workspace


class FlashInferAlphaMoeFp8QuantInfo(MoeQuantInfo):
    """Weights and persistent workspace consumed by the fused AlphaMoE path."""

    def __init__(
        self,
        *,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_weight_scale_inv: torch.Tensor,
        w2_weight_scale_inv: torch.Tensor,
        route_plan_cache: AlphaMoeRoutePlanCache,
    ) -> None:
        self.w13_weight = w13_weight
        self.w2_weight = w2_weight
        self.w13_weight_scale_inv = w13_weight_scale_inv
        self.w2_weight_scale_inv = w2_weight_scale_inv
        self.route_plan_cache = route_plan_cache


def deinterleave_alphamoe_gated_rows(
    tensor: torch.Tensor, *, rows_per_chunk: int
) -> torch.Tensor:
    """Invert AlphaMoE's alternating gate/up row-chunk layout."""

    if tensor.ndim != 3:
        raise ValueError(f"expected a rank-3 tensor, got shape {tuple(tensor.shape)}")
    num_experts, rows, columns = tensor.shape
    if rows_per_chunk <= 0 or rows % (2 * rows_per_chunk) != 0:
        raise ValueError(
            f"row count {rows} must be divisible by 2 * rows_per_chunk "
            f"({2 * rows_per_chunk})"
        )
    chunks = tensor.reshape(
        num_experts, rows // (2 * rows_per_chunk), 2, rows_per_chunk, columns
    )
    gate = chunks[:, :, 0].reshape(num_experts, rows // 2, columns)
    up = chunks[:, :, 1].reshape(num_experts, rows // 2, columns)
    return torch.cat((gate, up), dim=1).contiguous()


def restore_alphamoe_fp8_weights_for_loading(
    w13_weight: torch.Tensor,
    w13_weight_scale_inv: torch.Tensor,
) -> None:
    """Restore canonical checkpoint layout before a hot weight reload."""

    canonical_weight = deinterleave_alphamoe_gated_rows(w13_weight, rows_per_chunk=8)
    canonical_scale = deinterleave_alphamoe_gated_rows(
        w13_weight_scale_inv, rows_per_chunk=1
    )
    w13_weight.copy_(canonical_weight)
    w13_weight_scale_inv.copy_(canonical_scale)


def validate_alphamoe_w8a8_weights(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale_inv: torch.Tensor,
    w2_weight_scale_inv: torch.Tensor,
    *,
    block_shape: list[int] | tuple[int, int],
    top_k: int,
    use_mxfp8: bool,
    is_fp4_expert: bool,
) -> tuple[int, int, int]:
    """Validate the exact tensor ABI and return ``(experts, hidden, inter)``."""

    if use_mxfp8 or is_fp4_expert:
        raise ValueError(
            "flashinfer_alphamoe supports only W8A8 block FP8. NVFP4/ModelOpt "
            "global scales and MXFP8 scale encodings are not representable by "
            "the AlphaMoE W8A8 API."
        )
    if tuple(block_shape) != ALPHAMOE_WEIGHT_BLOCK:
        raise ValueError(
            "flashinfer_alphamoe requires FP32 128x128 block scales; "
            f"got block_shape={tuple(block_shape)}"
        )
    if not 1 <= top_k <= ALPHAMOE_MAX_TOP_K:
        raise ValueError(
            f"flashinfer_alphamoe top_k must be in [1, {ALPHAMOE_MAX_TOP_K}], "
            f"got {top_k}"
        )

    tensors = {
        "w13_weight": w13_weight,
        "w2_weight": w2_weight,
        "w13_weight_scale_inv": w13_weight_scale_inv,
        "w2_weight_scale_inv": w2_weight_scale_inv,
    }
    for name, tensor in tensors.items():
        if tensor.ndim != 3:
            raise ValueError(f"{name} must be rank 3, got shape {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if (
        w13_weight.dtype != torch.float8_e4m3fn
        or w2_weight.dtype != torch.float8_e4m3fn
    ):
        raise TypeError(
            "flashinfer_alphamoe requires torch.float8_e4m3fn expert weights"
        )
    if (
        w13_weight_scale_inv.dtype != torch.float32
        or w2_weight_scale_inv.dtype != torch.float32
    ):
        raise TypeError(
            "flashinfer_alphamoe requires FP32 per-128x128 weight block scales"
        )

    num_experts, gate_up_rows, hidden_size = w13_weight.shape
    if not 1 <= num_experts <= ALPHAMOE_MAX_EXPERTS:
        raise ValueError(
            "flashinfer_alphamoe fused router supports at most "
            f"{ALPHAMOE_MAX_EXPERTS} experts, got {num_experts}"
        )
    if gate_up_rows % 2 != 0:
        raise ValueError(f"w13 gate/up row count must be even, got {gate_up_rows}")
    intermediate_size = gate_up_rows // 2
    if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
        raise ValueError(
            "flashinfer_alphamoe requires hidden_size and the TP-sharded "
            "intermediate_size to be divisible by 128; got "
            f"hidden_size={hidden_size}, intermediate_size={intermediate_size}"
        )

    expected_w2 = (num_experts, hidden_size, intermediate_size)
    expected_w13_scale = (
        num_experts,
        gate_up_rows // 128,
        hidden_size // 128,
    )
    expected_w2_scale = (
        num_experts,
        hidden_size // 128,
        intermediate_size // 128,
    )
    expected_shapes = {
        "w2_weight": expected_w2,
        "w13_weight_scale_inv": expected_w13_scale,
        "w2_weight_scale_inv": expected_w2_scale,
    }
    for name, expected in expected_shapes.items():
        actual = tuple(tensors[name].shape)
        if actual != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual}")
    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts}")
    return num_experts, hidden_size, intermediate_size


def interleave_alphamoe_fp8_weights_for_runtime(
    w13_weight: torch.Tensor,
    w13_weight_scale_inv: torch.Tensor,
) -> None:
    """Apply FlashInfer's canonical offline interleave without rebinding params."""

    from flashinfer.fused_moe import alphamoe_interleave_gated_weights

    interleaved_weight, interleaved_scale = alphamoe_interleave_gated_weights(
        w13_weight, w13_weight_scale_inv
    )
    w13_weight.copy_(interleaved_weight)
    w13_weight_scale_inv.copy_(interleaved_scale)


def warmup_alphamoe_jit_modules() -> None:
    """Build both JIT modules before any CUDA graph capture."""

    from flashinfer.fused_moe.alphamoe_fused_router import (
        get_alphamoe_fused_router_module,
    )
    from flashinfer.fused_moe.alphamoe_sm100 import get_alphamoe_sm100_module

    get_alphamoe_fused_router_module()
    get_alphamoe_sm100_module()


def _validate_topk_contract(dispatch_output: StandardDispatchOutput) -> None:
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    topk_output = dispatch_output.topk_output
    if not TopKOutputChecker.format_is_bypassed(topk_output):
        raise TypeError("flashinfer_alphamoe requires raw bypassed router logits")
    topk_config = topk_output.topk_config
    if (
        not topk_config.renormalize
        or topk_config.scoring_func != "softmax"
        or topk_config.use_grouped_topk
        or topk_config.correction_bias is not None
        or topk_config.custom_routing_function is not None
        or topk_config.num_fused_shared_experts != 0
        or topk_config.routed_scaling_factor is not None
        or topk_config.apply_routed_scaling_factor_on_output
    ):
        raise ValueError(
            "flashinfer_alphamoe supports only top-k selected-logit softmax "
            "routing without groups, correction bias, fused shared experts, "
            "or top-k-side routed scaling"
        )


@register_fused_func("none", "flashinfer_alphamoe")
def fused_experts_none_to_flashinfer_alphamoe(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferAlphaMoeFp8QuantInfo,
    config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Fuse raw-logit routing, W8A8 activation quantization, and AlphaMoE."""

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states = dispatch_output.hidden_states.contiguous()
    if hidden_states.shape[0] == 0:
        return StandardCombineInput(hidden_states=torch.empty_like(hidden_states))
    _validate_topk_contract(dispatch_output)

    from flashinfer.fused_moe import (
        alphamoe_fp8_block_scale_aligned_moe,
        alphamoe_fused_router,
    )

    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8

    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "flashinfer_alphamoe currently requires BF16 hidden states, got "
            f"{hidden_states.dtype}"
        )

    topk_output = dispatch_output.topk_output
    logits = topk_output.router_logits.float().contiguous()
    top_k = topk_output.topk_config.top_k
    if top_k != config.top_k:
        raise ValueError(f"top-k mismatch: router={top_k}, runner={config.top_k}")

    plan, out = quant_info.route_plan_cache.get(
        logits,
        hidden_size=hidden_states.shape[1],
        top_k=top_k,
        block_m=ALPHAMOE_BLOCK_M,
    )
    plan = alphamoe_fused_router(
        logits,
        top_k=top_k,
        block_m=ALPHAMOE_BLOCK_M,
        has_shared_expert=False,
        plan=plan,
        skip_check=True,
    )
    hidden_states_fp8, hidden_states_scale = per_token_group_quant_fp8(
        hidden_states,
        group_size=128,
        column_major_scales=False,
    )
    out.zero_()
    result = alphamoe_fp8_block_scale_aligned_moe(
        hidden_states_fp8,
        hidden_states_scale,
        quant_info.w13_weight,
        quant_info.w13_weight_scale_inv,
        quant_info.w2_weight,
        quant_info.w2_weight_scale_inv,
        plan.sorted_token_ids,
        plan.expert_ids,
        plan.num_tokens_post_padded,
        plan.topk_weights,
        top_k=top_k,
        block_m=ALPHAMOE_BLOCK_M,
        routed_scaling_factor=(
            1.0
            if config.routed_scaling_factor is None
            else config.routed_scaling_factor
        ),
        out=out,
    )
    return StandardCombineInput(hidden_states=result)


__all__ = [
    "AlphaMoeRoutePlanCache",
    "FlashInferAlphaMoeFp8QuantInfo",
    "deinterleave_alphamoe_gated_rows",
    "fused_experts_none_to_flashinfer_alphamoe",
    "interleave_alphamoe_fp8_weights_for_runtime",
    "restore_alphamoe_fp8_weights_for_loading",
    "validate_alphamoe_runner_contract",
    "validate_alphamoe_w8a8_weights",
    "warmup_alphamoe_jit_modules",
]
