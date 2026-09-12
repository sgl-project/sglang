"""FlashInfer AlphaMoE adapters for exact SM100/SM103 GPUs.

This backend is deliberately narrow.  It consumes native fine-grained FP8
checkpoints with 128x128 FP32 block scales, a gated SiLU expert, raw FP32
router logits, and ``ep_size == 1``.  Gate/up rows are converted once from the
checkpoint's canonical ``[gate; up]`` layout to AlphaMoE's 8-row interleave.
The admission gate is currently frozen to the validated
Qwen3-Next-80B-A3B-Instruct-FP8 TP4 expert geometry.  The exported kernel's
long-K source coordinate does not satisfy the strict FP8 error contract, so a
broader model-facing dispatch would risk silent numerical errors.

The NVFP4 path is separate from W8A8 routing.  It keeps the model's native
SGLang TopK, consumes checkpoint-canonical packed weights and linear per-16
E4M3 scales, and passes ModelOpt's three per-expert output scales through the
extended PR #4340 ABI.  It is admitted only for the real GLM-5.2-NVFP4 TP4
geometry used by the GB300 E2E test.
"""

from __future__ import annotations

import json
import logging
import os
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
ALPHAMOE_VALIDATED_EXPERTS = 512
ALPHAMOE_VALIDATED_HIDDEN_SIZE = 2048
ALPHAMOE_VALIDATED_INTERMEDIATE_SIZE = 128
ALPHAMOE_VALIDATED_TOP_K = 10
ALPHAMOE_NVFP4_VALIDATED_EXPERTS = 256
ALPHAMOE_NVFP4_VALIDATED_HIDDEN_SIZE = 6144
ALPHAMOE_NVFP4_VALIDATED_INTERMEDIATE_SIZE = 512
ALPHAMOE_NVFP4_VALIDATED_TOP_K = 8

logger = logging.getLogger(__name__)
_traced_nvfp4_shapes: set[tuple[int, ...]] = set()


def validate_alphamoe_runner_contract(
    *,
    tp_size: int,
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

    if tp_size != 4:
        raise ValueError("flashinfer_alphamoe requires moe_tp_size=4")
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
    if top_k != ALPHAMOE_VALIDATED_TOP_K or num_experts != ALPHAMOE_VALIDATED_EXPERTS:
        raise ValueError(
            "flashinfer_alphamoe is currently admitted only for the validated "
            "Qwen3-Next TP4 routing geometry: "
            f"experts={ALPHAMOE_VALIDATED_EXPERTS}, "
            f"top_k={ALPHAMOE_VALIDATED_TOP_K}; got experts={num_experts}, "
            f"top_k={top_k}"
        )


def validate_alphamoe_nvfp4_runner_contract(
    *,
    tp_size: int,
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
    """Admit only the registered GLM-5.2-NVFP4 TP4 execution contract."""

    if tp_size != 4:
        raise ValueError("flashinfer_alphamoe NVFP4 requires moe_tp_size=4")
    if ep_size != 1 or not a2a_is_none:
        raise ValueError(
            "flashinfer_alphamoe NVFP4 requires ep_size=1 and " "moe_a2a_backend=none"
        )
    if num_fused_shared_experts != 0:
        raise ValueError(
            "flashinfer_alphamoe NVFP4 requires shared-expert fusion disabled"
        )
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
            "flashinfer_alphamoe NVFP4 supports only combined, unbiased gated "
            "SiLU experts without input-side router weights or activation clamps"
        )
    if params_dtype != torch.bfloat16:
        raise ValueError("flashinfer_alphamoe NVFP4 requires BF16 activations")
    if (
        top_k != ALPHAMOE_NVFP4_VALIDATED_TOP_K
        or num_experts != ALPHAMOE_NVFP4_VALIDATED_EXPERTS
    ):
        raise ValueError(
            "flashinfer_alphamoe NVFP4 is currently admitted only for the real "
            "GLM-5.2-NVFP4 TP4 routing geometry: "
            f"experts={ALPHAMOE_NVFP4_VALIDATED_EXPERTS}, "
            f"top_k={ALPHAMOE_NVFP4_VALIDATED_TOP_K}; got "
            f"experts={num_experts}, top_k={top_k}"
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


class FlashInferAlphaMoeNvFp4QuantInfo(MoeQuantInfo):
    """Canonical NVFP4 weights and ModelOpt scales consumed by PR #4340."""

    def __init__(
        self,
        *,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w2_weight_scale: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        input_scale_quant: torch.Tensor,
    ) -> None:
        self.w13_weight = w13_weight
        self.w2_weight = w2_weight
        self.w13_weight_scale = w13_weight_scale
        self.w2_weight_scale = w2_weight_scale
        self.output1_scale_gate_scalar = output1_scale_gate_scalar
        self.output1_scale_scalar = output1_scale_scalar
        self.output2_scale_scalar = output2_scale_scalar
        self.input_scale_quant = input_scale_quant


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
    if (
        num_experts != ALPHAMOE_VALIDATED_EXPERTS
        or hidden_size != ALPHAMOE_VALIDATED_HIDDEN_SIZE
        or intermediate_size != ALPHAMOE_VALIDATED_INTERMEDIATE_SIZE
        or top_k != ALPHAMOE_VALIDATED_TOP_K
    ):
        raise ValueError(
            "flashinfer_alphamoe is currently admitted only for the validated "
            "Qwen3-Next-80B-A3B-Instruct-FP8 TP4 expert geometry "
            f"(experts={ALPHAMOE_VALIDATED_EXPERTS}, "
            f"hidden_size={ALPHAMOE_VALIDATED_HIDDEN_SIZE}, "
            f"TP-sharded intermediate_size="
            f"{ALPHAMOE_VALIDATED_INTERMEDIATE_SIZE}, "
            f"top_k={ALPHAMOE_VALIDATED_TOP_K}); got experts={num_experts}, "
            f"hidden_size={hidden_size}, intermediate_size={intermediate_size}, "
            f"top_k={top_k}. Broader/long-K dispatch remains disabled until "
            "the exported kernel passes the strict FP8 error gate."
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


def validate_alphamoe_nvfp4_weights(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    *,
    output1_scale_gate_scalar: torch.Tensor,
    output1_scale_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    top_k: int,
) -> tuple[int, int, int]:
    """Validate the checkpoint-canonical PR #4340 ABI for real GLM-5.2 TP4."""

    tensors = {
        "w13_weight": w13_weight,
        "w2_weight": w2_weight,
        "w13_weight_scale": w13_weight_scale,
        "w2_weight_scale": w2_weight_scale,
        "output1_scale_gate_scalar": output1_scale_gate_scalar,
        "output1_scale_scalar": output1_scale_scalar,
        "output2_scale_scalar": output2_scale_scalar,
    }
    device = w13_weight.device
    for name, tensor in tensors.items():
        if tensor.device != device or tensor.device.type != "cuda":
            raise ValueError(
                f"{name} must share the CUDA device {device}, got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if w13_weight.dtype != torch.uint8 or w2_weight.dtype != torch.uint8:
        raise TypeError("flashinfer_alphamoe NVFP4 requires packed uint8 weights")
    if (
        w13_weight_scale.dtype != torch.float8_e4m3fn
        or w2_weight_scale.dtype != torch.float8_e4m3fn
    ):
        raise TypeError(
            "flashinfer_alphamoe NVFP4 requires linear E4M3 per-16 weight scales"
        )
    for name, tensor in (
        ("output1_scale_gate_scalar", output1_scale_gate_scalar),
        ("output1_scale_scalar", output1_scale_scalar),
        ("output2_scale_scalar", output2_scale_scalar),
    ):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be FP32, got {tensor.dtype}")

    if w13_weight.ndim != 3 or w2_weight.ndim != 3:
        raise ValueError("flashinfer_alphamoe NVFP4 weights must be rank 3")
    num_experts, gate_up_rows, packed_hidden = w13_weight.shape
    hidden_size = packed_hidden * 2
    if gate_up_rows % 2 != 0:
        raise ValueError(f"w13 gate/up rows must be even, got {gate_up_rows}")
    intermediate_size = gate_up_rows // 2
    geometry = (num_experts, hidden_size, intermediate_size, top_k)
    expected_geometry = (
        ALPHAMOE_NVFP4_VALIDATED_EXPERTS,
        ALPHAMOE_NVFP4_VALIDATED_HIDDEN_SIZE,
        ALPHAMOE_NVFP4_VALIDATED_INTERMEDIATE_SIZE,
        ALPHAMOE_NVFP4_VALIDATED_TOP_K,
    )
    if geometry != expected_geometry:
        raise ValueError(
            "flashinfer_alphamoe NVFP4 is admitted only for the real "
            "GLM-5.2-NVFP4 TP4 expert geometry "
            f"(E,H,I_local,top_k)={expected_geometry}; got {geometry}"
        )

    expected_shapes = {
        "w2_weight": (num_experts, hidden_size, intermediate_size // 2),
        "w13_weight_scale": (
            num_experts,
            2 * intermediate_size,
            hidden_size // 16,
        ),
        "w2_weight_scale": (
            num_experts,
            hidden_size,
            intermediate_size // 16,
        ),
        "output1_scale_gate_scalar": (num_experts,),
        "output1_scale_scalar": (num_experts,),
        "output2_scale_scalar": (num_experts,),
    }
    for name, expected in expected_shapes.items():
        actual = tuple(tensors[name].shape)
        if actual != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual}")
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


def warmup_alphamoe_nvfp4_jit_module() -> None:
    """Build PR #4340 before SGLang starts CUDA graph capture."""

    from flashinfer.fused_moe.alphamoe_nvfp4_sm100 import (
        get_alphamoe_nvfp4_sm100_module,
    )

    get_alphamoe_nvfp4_sm100_module()


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


def _fused_experts_none_to_flashinfer_alphamoe_fp8(
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
    from sglang.srt.layers.moe.alphamoe_trace import record_alphamoe_kernel

    for kernel in ("alphamoe_fused_router", "alphamoe_fp8_block_scale_aligned_moe"):
        record_alphamoe_kernel(
            kernel=kernel,
            hidden_states=hidden_states,
            num_experts=quant_info.w13_weight.shape[0],
            intermediate_size=quant_info.w13_weight.shape[1] // 2,
            top_k=top_k,
            block_m=ALPHAMOE_BLOCK_M,
        )
    return StandardCombineInput(hidden_states=result)


def _trace_alphamoe_nvfp4_shape(
    *,
    hidden_states: torch.Tensor,
    quant_info: FlashInferAlphaMoeNvFp4QuantInfo,
    top_k: int,
) -> None:
    from sglang.srt.layers.moe.alphamoe_trace import record_alphamoe_kernel

    record_alphamoe_kernel(
        kernel="alphamoe_nvfp4_aligned_moe",
        hidden_states=hidden_states,
        num_experts=quant_info.w13_weight.shape[0],
        intermediate_size=quant_info.w13_weight.shape[1] // 2,
        top_k=top_k,
        block_m=ALPHAMOE_BLOCK_M,
    )
    if os.environ.get("SGLANG_FLASHINFER_ALPHAMOE_TRACE_SHAPES", "0") != "1":
        return
    # Validation may arm tracing only after server initialization, CUDA graph
    # capture, and warmup have finished.  This keeps startup-only padded graph
    # shapes out of evidence that is explicitly labeled as request-observed.
    arm_file = os.environ.get("SGLANG_FLASHINFER_ALPHAMOE_TRACE_ARM_FILE")
    if arm_file and not os.path.isfile(arm_file):
        return
    num_experts, gate_up_rows, _ = quant_info.w13_weight.shape
    key = (
        int(hidden_states.shape[0]),
        int(num_experts),
        int(hidden_states.shape[1]),
        int(gate_up_rows // 2),
        int(top_k),
        ALPHAMOE_BLOCK_M,
    )
    if key in _traced_nvfp4_shapes:
        return
    _traced_nvfp4_shapes.add(key)
    payload = {
        "schema": "sglang-flashinfer-alphamoe-nvfp4-runtime-shape-v1",
        "kernel": "flashinfer::alphamoe_nvfp4_aligned_moe",
        "M": key[0],
        "E": key[1],
        "H": key[2],
        "I_local": key[3],
        "top_k": key[4],
        "block_m": key[5],
        "shared_fused": False,
        "tp": 4,
        "ep": 1,
        "device": str(hidden_states.device),
    }
    logger.warning(
        "ALPHAMOE_NVFP4_RUNTIME_SHAPE %s", json.dumps(payload, sort_keys=True)
    )


def _fused_experts_none_to_flashinfer_alphamoe_nvfp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferAlphaMoeNvFp4QuantInfo,
    config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Run PR #4340 from the model's materialized, production TopK output."""

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states = dispatch_output.hidden_states.contiguous()
    if hidden_states.shape[0] == 0:
        return StandardCombineInput(hidden_states=torch.empty_like(hidden_states))
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "flashinfer_alphamoe NVFP4 requires BF16 hidden states, got "
            f"{hidden_states.dtype}"
        )
    topk_output = dispatch_output.topk_output
    if not TopKOutputChecker.format_is_standard(topk_output):
        raise TypeError(
            "flashinfer_alphamoe NVFP4 requires SGLang's materialized TopK output"
        )
    topk_ids = topk_output.topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_output.topk_weights.to(torch.float32).contiguous()
    top_k = int(topk_ids.shape[1])
    if top_k != config.top_k:
        raise ValueError(f"top-k mismatch: routing={top_k}, runner={config.top_k}")

    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    if fp4_quantize is None:
        raise RuntimeError("FlashInfer fp4_quantize is required for AlphaMoE NVFP4")
    hidden_states_fp4, hidden_states_scale = fp4_quantize(
        hidden_states,
        quant_info.input_scale_quant,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
        hidden_states.shape[0], hidden_states.shape[1] // 16
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        ALPHAMOE_BLOCK_M,
        quant_info.w13_weight.shape[0],
    )
    # SGLang sizes the sorted-id buffer for the maximum valid extent, while
    # PR #4340 launches every capacity block described by expert_ids and
    # therefore requires a complete inactive tail.  For the production
    # E=256/BM=8 plan this is at most one extra int32 slot.
    required_plan_capacity = expert_ids.numel() * ALPHAMOE_BLOCK_M
    if sorted_token_ids.numel() < required_plan_capacity:
        sorted_token_ids = torch.nn.functional.pad(
            sorted_token_ids,
            (0, required_plan_capacity - sorted_token_ids.numel()),
            value=topk_ids.numel(),
        )
    out = torch.zeros_like(hidden_states, dtype=torch.bfloat16)

    from flashinfer.fused_moe import alphamoe_nvfp4_aligned_moe

    alphamoe_nvfp4_aligned_moe(
        hidden_states_fp4,
        hidden_states_scale,
        quant_info.w13_weight,
        quant_info.w13_weight_scale,
        quant_info.w2_weight,
        quant_info.w2_weight_scale,
        output1_scale_gate_scalar=quant_info.output1_scale_gate_scalar,
        output1_scale_scalar=quant_info.output1_scale_scalar,
        output2_scale_scalar=quant_info.output2_scale_scalar,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        topk_weights=topk_weights,
        out=out,
        top_k=top_k,
        block_m=ALPHAMOE_BLOCK_M,
        # ModelOpt's standard TopK path has already applied GLM's routed 2.5.
        routed_scaling_factor=1.0,
    )
    # Emit evidence only after FlashInfer has accepted the full ABI and
    # launched the kernel.  Any asynchronous launch failure still fails E2E.
    _trace_alphamoe_nvfp4_shape(
        hidden_states=hidden_states,
        quant_info=quant_info,
        top_k=top_k,
    )
    return StandardCombineInput(hidden_states=out)


@register_fused_func("none", "flashinfer_alphamoe")
def fused_experts_none_to_flashinfer_alphamoe(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    config: MoeRunnerConfig,
) -> StandardCombineInput:
    # Idle DP ranks may not carry backend-specific quantization state.  Keep
    # the established empty-batch contract ahead of quant-info dispatch.
    if dispatch_output.hidden_states.shape[0] == 0:
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        return StandardCombineInput(
            hidden_states=torch.empty_like(dispatch_output.hidden_states)
        )
    if isinstance(quant_info, FlashInferAlphaMoeNvFp4QuantInfo):
        return _fused_experts_none_to_flashinfer_alphamoe_nvfp4(
            dispatch_output, quant_info, config
        )
    if isinstance(quant_info, FlashInferAlphaMoeFp8QuantInfo):
        return _fused_experts_none_to_flashinfer_alphamoe_fp8(
            dispatch_output, quant_info, config
        )
    raise TypeError(f"unsupported AlphaMoE quant info: {type(quant_info)}")


__all__ = [
    "AlphaMoeRoutePlanCache",
    "FlashInferAlphaMoeFp8QuantInfo",
    "FlashInferAlphaMoeNvFp4QuantInfo",
    "deinterleave_alphamoe_gated_rows",
    "fused_experts_none_to_flashinfer_alphamoe",
    "interleave_alphamoe_fp8_weights_for_runtime",
    "restore_alphamoe_fp8_weights_for_loading",
    "validate_alphamoe_runner_contract",
    "validate_alphamoe_nvfp4_runner_contract",
    "validate_alphamoe_nvfp4_weights",
    "validate_alphamoe_w8a8_weights",
    "warmup_alphamoe_nvfp4_jit_module",
    "warmup_alphamoe_jit_modules",
]
