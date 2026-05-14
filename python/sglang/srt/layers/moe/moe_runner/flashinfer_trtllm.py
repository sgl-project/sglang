from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

# Import to register custom ops for torch.compile compatibility
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.flashinfer_trtllm_moe import (
    trtllm_fp8_block_scale_moe_wrapper,
    trtllm_fp8_block_scale_routed_moe_wrapper,
    trtllm_fp8_per_tensor_scale_moe_wrapper,
)
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    scaled_fp8_quant,
)
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang.srt.utils.common import (
    is_cuda_alike,
    is_flashinfer_available,
    next_power_of_2,
)

logger = __import__("logging").getLogger(__name__)


def round_up_to_multiple(x: int, m: int) -> int:
    """Round up *x* to the nearest multiple of *m*."""
    return (x + m - 1) // m * m


if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

if is_flashinfer_available():
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize
elif is_cuda_alike():
    from sglang.jit_kernel.nvfp4 import scaled_fp4_quant as fp4_quantize
else:
    fp4_quantize = None

_flashinfer_trtllm_shuffle_row_indices_cache_mxfp8: dict[
    tuple, dict[str, torch.Tensor]
] = {}


def _is_gated(layer: Module) -> bool:
    """Return whether the MoE layer uses a gated activation (default True)."""
    is_gated = (
        getattr(layer, "moe_runner_config", None) and layer.moe_runner_config.is_gated
    )
    return True if is_gated is None else is_gated


def _align_fp8_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    is_gated: bool,
    min_alignment: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer TRTLLM FP8 kernels' alignment holds.

    Returns (w13, w2, padded_intermediate).
    """
    num_experts, hidden_size, intermediate = w2.shape

    padded_intermediate = round_up_to_multiple(intermediate, min_alignment)
    if padded_intermediate == intermediate:
        return w13, w2, intermediate

    logger.info(
        "FP8 MoE: padding intermediate size from %d to %d (alignment=%d)",
        intermediate,
        padded_intermediate,
        min_alignment,
    )

    up_mult = 2 if is_gated else 1
    padded_gate_up = up_mult * padded_intermediate

    padded_w13 = w13.new_zeros((num_experts, padded_gate_up, w13.shape[2]))
    padded_w13[:, : w13.shape[1], :] = w13

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate))
    padded_w2[:, :, :intermediate] = w2

    return padded_w13, padded_w2, padded_intermediate


def align_fp8_moe_weights_for_flashinfer_trtllm(
    layer: Module, swap_w13_halves: bool = False
) -> None:
    """Prepare FP8 MoE weights/scales for FlashInfer TRT-LLM kernels.

    Args:
        layer: The MoE layer to process.
        swap_w13_halves: If True, swap W13 halves from [Up, Gate] to [Gate, Up].
            This is needed for ModelOpt FP8 checkpoints which store weights in
            [Up, Gate] order, while regular FP8 checkpoints store them in [Gate, Up].
    """
    from flashinfer import shuffle_matrix_a

    is_gated = _is_gated(layer)

    w13_weight = cast(torch.Tensor, layer.w13_weight)
    w2_weight = cast(torch.Tensor, layer.w2_weight)
    num_experts, gate_up_dim, hidden = w13_weight.shape

    # Optionally swap W13 halves: [Up, Gate] -> [Gate, Up] (only for gated)
    if swap_w13_halves and is_gated:
        inter = gate_up_dim // 2
        w13_weight = (
            w13_weight.reshape(num_experts, 2, inter, hidden)
            .flip(dims=[1])
            .reshape(num_experts, gate_up_dim, hidden)
        )

    # Pad for kernel alignment (non-gated needs 128, gated needs 16)
    min_alignment = 16 if is_gated else 128
    w13_weight, w2_weight, _ = _align_fp8_moe_weights(
        w13_weight, w2_weight, is_gated, min_alignment
    )
    num_experts, gate_up_dim, hidden = w13_weight.shape

    epilogue_tile_m = 128

    if is_gated:
        from flashinfer import reorder_rows_for_gated_act_gemm

        w13_interleaved_list = [
            reorder_rows_for_gated_act_gemm(w13_weight[i]) for i in range(num_experts)
        ]
        w13_processed: torch.Tensor = torch.stack(w13_interleaved_list).reshape(
            num_experts, gate_up_dim, hidden
        )
    else:
        w13_processed = w13_weight

    # Shuffle weights for transposed MMA output (both W13, W2)
    w13_shuffled = [
        shuffle_matrix_a(w13_processed[i].view(torch.uint8), epilogue_tile_m)
        for i in range(num_experts)
    ]
    w2_shuffled = [
        shuffle_matrix_a(w2_weight[i].view(torch.uint8), epilogue_tile_m)
        for i in range(num_experts)
    ]

    layer.w13_weight = Parameter(
        torch.stack(w13_shuffled).view(torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.w2_weight = Parameter(
        torch.stack(w2_shuffled).view(torch.float8_e4m3fn),
        requires_grad=False,
    )

    # Precompute and register per-expert output scaling factors for FI MoE.
    # Note: w13_input_scale and w2_input_scale are scalar Parameters post-reduction.
    assert hasattr(layer, "w13_input_scale") and layer.w13_input_scale is not None
    assert hasattr(layer, "w2_input_scale") and layer.w2_input_scale is not None
    assert hasattr(layer, "w13_weight_scale") and layer.w13_weight_scale is not None
    assert hasattr(layer, "w2_weight_scale") and layer.w2_weight_scale is not None

    input_scale = cast(torch.Tensor, layer.w13_input_scale).to(torch.float32)
    activation_scale = cast(torch.Tensor, layer.w2_input_scale).to(torch.float32)
    w13_weight_scale = cast(torch.Tensor, layer.w13_weight_scale).to(torch.float32)
    w2_weight_scale = cast(torch.Tensor, layer.w2_weight_scale).to(torch.float32)

    # For gated (SwiGLU): g1_alphas = w1_scale * a1_scale, g1_scale_c = g1_alphas / a2_scale
    # For non-gated (Relu2): g1_scale_c = 1 / a2_scale (no gate dequant contribution)
    if is_gated:
        output1_scales_scalar = (
            w13_weight_scale * input_scale * (1.0 / activation_scale)
        )
    else:
        output1_scales_scalar = torch.ones_like(w13_weight_scale) * (
            1.0 / activation_scale
        )
    output1_scales_gate_scalar = w13_weight_scale * input_scale
    output2_scales_scalar = activation_scale * w2_weight_scale

    layer.output1_scales_scalar = Parameter(output1_scales_scalar, requires_grad=False)
    layer.output1_scales_gate_scalar = Parameter(
        output1_scales_gate_scalar, requires_grad=False
    )
    layer.output2_scales_scalar = Parameter(output2_scales_scalar, requires_grad=False)


def _align_mxfp8_moe_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    is_gated: bool,
    min_alignment: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer TRTLLM MXFP8 kernels' alignment holds.

    Returns (w13, w13_scale, w2, w2_scale, padded_intermediate).
    """
    num_experts, hidden_size, intermediate = w2.shape

    padded_intermediate = round_up_to_multiple(intermediate, min_alignment)
    if padded_intermediate == intermediate:
        return w13, w13_scale, w2, w2_scale, intermediate

    logger.info(
        "MXFP8 MoE: padding intermediate size from %d to %d (alignment=%d)",
        intermediate,
        padded_intermediate,
        min_alignment,
    )

    up_mult = 2 if is_gated else 1
    padded_gate_up = up_mult * padded_intermediate

    padded_w13 = w13.new_zeros((num_experts, padded_gate_up, w13.shape[2]))
    padded_w13[:, : w13.shape[1], :] = w13

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate))
    padded_w2[:, :, :intermediate] = w2

    padded_w13_scale = w13_scale.new_zeros(
        (num_experts, padded_gate_up, w13_scale.shape[2])
    )
    padded_w13_scale[:, : w13_scale.shape[1], :] = w13_scale

    # Scale's last dim tracks intermediate / block_size (MXFP8 block_size = 32)
    scale_block_k = intermediate // w2_scale.shape[2] if w2_scale.shape[2] > 0 else 32
    padded_w2_scale = w2_scale.new_zeros(
        (num_experts, hidden_size, padded_intermediate // scale_block_k)
    )
    padded_w2_scale[:, :, : w2_scale.shape[2]] = w2_scale

    return padded_w13, padded_w13_scale, padded_w2, padded_w2_scale, padded_intermediate


def align_mxfp8_moe_weights_for_flashinfer_trtllm(layer: Module) -> None:
    """Prepare MXFP8 MoE weights/scales for FlashInfer TRT-LLM kernels."""
    from flashinfer import block_scale_interleave
    from flashinfer.fused_moe.core import (
        get_reorder_rows_for_gated_act_gemm_row_indices,
    )
    from flashinfer.utils import (
        get_shuffle_matrix_a_row_indices,
        get_shuffle_matrix_sf_a_row_indices,
    )

    is_gated = _is_gated(layer)

    w13_weight = cast(torch.Tensor, layer.w13_weight).contiguous()
    w2_weight = cast(torch.Tensor, layer.w2_weight).contiguous()
    w13_scale = cast(torch.Tensor, layer.w13_weight_scale_inv).contiguous()
    w2_scale = cast(torch.Tensor, layer.w2_weight_scale_inv).contiguous()

    assert w13_scale.dtype == torch.uint8
    assert w2_scale.dtype == torch.uint8

    # Pad for kernel alignment (non-gated needs 128, gated needs 16)
    min_alignment = 16 if is_gated else 128
    w13_weight, w13_scale, w2_weight, w2_scale, _ = _align_mxfp8_moe_weights(
        w13_weight, w13_scale, w2_weight, w2_scale, is_gated, min_alignment
    )

    num_experts, gate_up_dim, _ = w13_weight.shape
    _, hidden_size, _ = w2_weight.shape
    epilogue_tile_m = 128

    # Reuse precomputed row-index transforms whenever shape/device are unchanged.
    w13_weight_u8 = w13_weight.view(torch.uint8)
    w2_weight_u8 = w2_weight.view(torch.uint8)
    cache_key = (
        gate_up_dim,
        hidden_size,
        w2_weight.shape[-1],
        w13_scale.shape[-1],
        w2_scale.shape[-1],
        epilogue_tile_m,
        (w13_weight.device.type, w13_weight.device.index),
        (w2_weight.device.type, w2_weight.device.index),
        (w13_scale.device.type, w13_scale.device.index),
        (w2_scale.device.type, w2_scale.device.index),
    )
    cache = _flashinfer_trtllm_shuffle_row_indices_cache_mxfp8.get(cache_key)
    if cache is None:
        if is_gated:
            reorder_row_indices = get_reorder_rows_for_gated_act_gemm_row_indices(
                w13_weight_u8[0]
            ).to(w13_weight.device)
        else:
            reorder_row_indices = torch.arange(
                gate_up_dim, device=w13_weight.device, dtype=torch.long
            )
        w13_shuffle_row_indices = get_shuffle_matrix_a_row_indices(
            w13_weight_u8[0], epilogue_tile_m
        ).to(w13_weight.device)
        w2_shuffle_row_indices = get_shuffle_matrix_a_row_indices(
            w2_weight_u8[0], epilogue_tile_m
        ).to(w2_weight.device)
        w13_scale_shuffle_row_indices = get_shuffle_matrix_sf_a_row_indices(
            w13_scale[0].reshape(gate_up_dim, -1), epilogue_tile_m
        ).to(w13_scale.device)
        w2_scale_shuffle_row_indices = get_shuffle_matrix_sf_a_row_indices(
            w2_scale[0].reshape(hidden_size, -1), epilogue_tile_m
        ).to(w2_scale.device)
        cache = {
            "reorder_row_indices": reorder_row_indices,
            "w13_shuffle_row_indices": w13_shuffle_row_indices,
            "w2_shuffle_row_indices": w2_shuffle_row_indices,
            "w13_scale_shuffle_row_indices": w13_scale_shuffle_row_indices,
            "w2_scale_shuffle_row_indices": w2_scale_shuffle_row_indices,
        }
        _flashinfer_trtllm_shuffle_row_indices_cache_mxfp8[cache_key] = cache

    reorder_row_indices = cache["reorder_row_indices"]
    w13_shuffle_row_indices = cache["w13_shuffle_row_indices"]
    w2_shuffle_row_indices = cache["w2_shuffle_row_indices"]
    w13_scale_shuffle_row_indices = cache["w13_scale_shuffle_row_indices"]
    w2_scale_shuffle_row_indices = cache["w2_scale_shuffle_row_indices"]

    w13_shuffled_u8 = torch.empty_like(w13_weight_u8)
    w2_shuffled_u8 = torch.empty_like(w2_weight_u8)
    w13_scale_shuffled = torch.empty_like(w13_scale)
    w2_scale_shuffled = torch.empty_like(w2_scale)

    for i in range(num_experts):
        w13_interleaved_u8 = w13_weight_u8[i].index_select(0, reorder_row_indices)
        w13_scale_interleaved = w13_scale[i].index_select(0, reorder_row_indices)

        w13_shuffled_u8[i].copy_(
            w13_interleaved_u8.index_select(0, w13_shuffle_row_indices)
        )
        w2_shuffled_u8[i].copy_(w2_weight_u8[i].index_select(0, w2_shuffle_row_indices))

        w13_scale_linear = w13_scale_interleaved.reshape(gate_up_dim, -1)
        w13_scale_shuffled[i].copy_(
            block_scale_interleave(
                w13_scale_linear.index_select(0, w13_scale_shuffle_row_indices)
            ).reshape_as(w13_scale_shuffled[i])
        )

        w2_scale_linear = w2_scale[i].reshape(hidden_size, -1)
        w2_scale_shuffled[i].copy_(
            block_scale_interleave(
                w2_scale_linear.index_select(0, w2_scale_shuffle_row_indices)
            ).reshape_as(w2_scale_shuffled[i])
        )

    # Keep parameter identities stable for CUDA graph capture reuse.
    copy_or_rebind_param(layer, "w13_weight", w13_shuffled_u8.view(torch.float8_e4m3fn))
    copy_or_rebind_param(layer, "w2_weight", w2_shuffled_u8.view(torch.float8_e4m3fn))
    copy_or_rebind_param(
        layer,
        "w13_weight_scale_inv",
        w13_scale_shuffled.contiguous(),
    )
    copy_or_rebind_param(
        layer,
        "w2_weight_scale_inv",
        w2_scale_shuffled.contiguous(),
    )
    layer.w13_weight_scale_inv.format_ue8m0 = True
    layer.w2_weight_scale_inv.format_ue8m0 = True


def _align_fp4_moe_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    is_gated: bool,
    min_alignment: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pad intermediate size so FlashInfer TRTLLM FP4 kernels' alignment holds.

    Returns (w13, w13_scale, w2, w2_scale, padded_intermediate).
    """
    num_experts, hidden_size, intermediate_packed = w2.shape
    intermediate = intermediate_packed * 2  # FP4 packs 2 values per byte

    padded_intermediate = round_up_to_multiple(intermediate, min_alignment)
    if padded_intermediate == intermediate:
        return w13, w13_scale, w2, w2_scale, intermediate

    logger.info(
        "FP4 MoE: padding intermediate size from %d to %d (alignment=%d)",
        intermediate,
        padded_intermediate,
        min_alignment,
    )

    up_mult = 2 if is_gated else 1
    padded_gate_up = up_mult * padded_intermediate

    padded_w13 = w13.new_zeros((num_experts, padded_gate_up, w13.shape[2]))
    padded_w13[:, : w13.shape[1], :] = w13

    padded_w2 = w2.new_zeros((num_experts, hidden_size, padded_intermediate // 2))
    padded_w2[:, :, : w2.shape[2]] = w2

    padded_w13_scale = w13_scale.new_zeros(
        (num_experts, padded_gate_up, w13_scale.shape[2])
    )
    padded_w13_scale[:, : w13_scale.shape[1], :] = w13_scale

    padded_w2_scale = w2_scale.new_zeros(
        (num_experts, hidden_size, padded_intermediate // 16)
    )
    padded_w2_scale[:, :, : w2_scale.shape[2]] = w2_scale

    return padded_w13, padded_w13_scale, padded_w2, padded_w2_scale, padded_intermediate


def align_fp4_moe_weights_for_flashinfer_trtllm(layer: Module) -> None:
    """Prepare FP4 MoE weights/scales for FlashInfer TRT-LLM kernels.

    This function handles the weight transformation needed for FP4 TRTLLM MoE:
    - Pads intermediate dimension for kernel alignment constraints
    - Reorders weights for gated activation GEMM
    - Shuffles weights and scales for transposed MMA output
    - Computes the output scale factors
    """
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
    )

    w13_weight = cast(torch.Tensor, layer.w13_weight)
    w2_weight = cast(torch.Tensor, layer.w2_weight)
    w13_weight_scale = cast(torch.Tensor, layer.w13_weight_scale)
    w2_weight_scale = cast(torch.Tensor, layer.w2_weight_scale)

    is_gated = layer.moe_runner_config.is_gated
    min_alignment = 16 if is_gated else 128

    # Pad for kernel alignment before shuffle/reorder
    w13_weight, w13_weight_scale, w2_weight, w2_weight_scale, intermediate_size = (
        _align_fp4_moe_weights(
            w13_weight,
            w13_weight_scale,
            w2_weight,
            w2_weight_scale,
            is_gated,
            min_alignment,
        )
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
        intermediate_size,  # padded intermediate_size
        w13_weight.size(0),  # num_experts
        is_gated=is_gated,
    )

    # Set flashinfer parameters in-place
    copy_or_rebind_param(layer, "w13_weight", gemm1_weights_fp4_shuffled.contiguous())
    copy_or_rebind_param(layer, "w2_weight", gemm2_weights_fp4_shuffled.contiguous())
    copy_or_rebind_param(
        layer, "w13_weight_scale", gemm1_scales_fp4_shuffled.contiguous()
    )
    copy_or_rebind_param(
        layer, "w2_weight_scale", gemm2_scales_fp4_shuffled.contiguous()
    )

    # Compute additional scaling factor needed for TRT-LLM.
    # For gated (SwiGLU): g1_scale_c = g1_alphas * a2_gscale
    # For non-gated (Relu2): g1_scale_c = a2_gscale (no gate dequant contribution)
    w2_input_scale_quant = cast(torch.Tensor, layer.w2_input_scale_quant)
    g1_alphas = cast(torch.Tensor, layer.g1_alphas)
    if layer.moe_runner_config.is_gated:
        g1_scale_c = (w2_input_scale_quant * g1_alphas).to(torch.float32)
    else:
        num_experts = g1_alphas.shape[0]
        g1_scale_c = (
            w2_input_scale_quant.to(torch.float32).expand(num_experts).contiguous()
        )
    copy_or_rebind_param(layer, "g1_scale_c", g1_scale_c)

    # Update intermediate_size_per_partition to reflect any padding applied
    layer.intermediate_size_per_partition = intermediate_size


def get_activation_type(activation: str) -> int:
    """Map SGLang activation string to FlashInfer ActivationType int value."""
    from flashinfer.fused_moe.core import ActivationType

    _ACTIVATION_STR_TO_TYPE = {
        "silu": ActivationType.Swiglu,
        "relu2": ActivationType.Relu2,
    }
    act = _ACTIVATION_STR_TO_TYPE.get(activation)
    if act is None:
        raise ValueError(
            f"Unsupported activation '{activation}' for TRTLLM MoE. "
            f"Expected one of {list(_ACTIVATION_STR_TO_TYPE.keys())}."
        )
    return act.value


@dataclass
class FlashInferTrtllmFp8MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer TRT-LLM FP8 MoE kernels."""

    # Weights
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor

    # Expert-parallel metadata
    global_num_experts: int
    local_expert_offset: int
    local_num_experts: int
    intermediate_size: int

    routing_method_type: int

    # Block-quant path
    block_quant: bool
    use_mxfp8: bool = False
    weight_block_k: int | None = None
    w13_weight_scale_inv: torch.Tensor | None = None
    w2_weight_scale_inv: torch.Tensor | None = None

    # Per-tensor path
    w13_input_scale: torch.Tensor | None = None
    output1_scales_scalar: torch.Tensor | None = None
    output1_scales_gate_scalar: torch.Tensor | None = None
    output2_scales_scalar: torch.Tensor | None = None
    use_routing_scales_on_input: bool = False

    # Activation type (None = kernel default / Swiglu)
    activation_type: int | None = None


def _pack_topk_for_flashinfer_routed(
    topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> torch.Tensor:
    """Pack routed top-k tensors into FlashInfer's int32 format."""
    packed_ids = topk_ids.to(torch.int32)
    packed_weights = topk_weights.to(torch.bfloat16)
    packed = (packed_ids << 16) | packed_weights.view(torch.int16).to(torch.int32)
    return packed


def fused_experts_none_to_flashinfer_trtllm_fp8(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmFp8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    use_routed_topk: bool = False,
) -> StandardCombineInput:
    from flashinfer.fused_moe import Fp8QuantizationType

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType

    _SUPPORTED_FP8_ACTIVATIONS = {"silu", "relu2"}
    assert runner_config.activation in _SUPPORTED_FP8_ACTIVATIONS, (
        f"Only {_SUPPORTED_FP8_ACTIVATIONS} are supported for FP8 MoE, "
        f"got '{runner_config.activation}'."
    )
    assert not runner_config.no_combine, "no_combine is not supported for flashinfer."

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    if TopKOutputChecker.format_is_bypassed(topk_output):
        router_logits = topk_output.router_logits
        topk_config = topk_output.topk_config
        correction_bias = (
            None
            if topk_config.correction_bias is None
            else topk_config.correction_bias.to(hidden_states.dtype)
        )
    else:
        router_logits = None
        topk_config = None
        correction_bias = None

    routing_method_type = quant_info.routing_method_type
    fp8_quantization_type = (
        Fp8QuantizationType.MxFp8
        if quant_info.use_mxfp8
        else Fp8QuantizationType.DeepSeekFp8
    )
    use_shuffled_weight = quant_info.use_mxfp8

    if quant_info.block_quant:
        assert quant_info.weight_block_k is not None
        assert quant_info.w13_weight_scale_inv is not None
        assert quant_info.w2_weight_scale_inv is not None

        if quant_info.use_mxfp8:
            assert quant_info.weight_block_k == 32
            from flashinfer import mxfp8_quantize

            a_q, a_sf = mxfp8_quantize(hidden_states, False)
            # FlashInfer TRT-LLM MxFP8 expects token-major activation scales:
            # [num_tokens, hidden_size // 32] (no transpose).
            a_sf_t = a_sf.view(torch.uint8).reshape(hidden_states.shape[0], -1)
        else:
            a_q, a_sf = per_token_group_quant_fp8(
                hidden_states, quant_info.weight_block_k
            )
            a_sf_t = a_sf.t().contiguous()

        # Allocate output inside symmetric memory context
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            symm_output = torch.empty(
                hidden_states.shape[0],
                hidden_states.shape[1],
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        # Move kernel call outside context manager to avoid graph breaks
        # during torch.compile for piecewise cuda graph.
        # Use custom op wrapper for torch.compile compatibility.
        if use_routed_topk:
            assert (
                runner_config.top_k is not None
            ), "runner_config.top_k is required for flashinfer_trtllm_routed."
            assert TopKOutputChecker.format_is_standard(topk_output)
            packed_topk_ids = _pack_topk_for_flashinfer_routed(
                topk_ids=topk_output.topk_ids,
                topk_weights=topk_output.topk_weights,
            )

            output = trtllm_fp8_block_scale_routed_moe_wrapper(
                topk_ids=packed_topk_ids,
                routing_bias=None,
                hidden_states=a_q,
                hidden_states_scale=a_sf_t,
                gemm1_weights=quant_info.w13_weight,
                gemm1_weights_scale=quant_info.w13_weight_scale_inv,
                gemm2_weights=quant_info.w2_weight,
                gemm2_weights_scale=quant_info.w2_weight_scale_inv,
                num_experts=quant_info.global_num_experts,
                top_k=runner_config.top_k,
                n_group=None,
                topk_group=None,
                intermediate_size=quant_info.intermediate_size,
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=quant_info.local_num_experts,
                routed_scaling_factor=(
                    runner_config.routed_scaling_factor
                    if runner_config.routed_scaling_factor is not None
                    else 1.0
                ),
                routing_method_type=(
                    RoutingMethodType.TopK
                    if routing_method_type == RoutingMethodType.DeepSeekV3
                    else routing_method_type
                ),
                use_shuffled_weight=use_shuffled_weight,
                tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
                fp8_quantization_type=int(fp8_quantization_type),
                activation_type=quant_info.activation_type,
            )
        else:
            assert TopKOutputChecker.format_is_bypassed(topk_output)

            output = trtllm_fp8_block_scale_moe_wrapper(
                routing_logits=(
                    router_logits.to(torch.float32)
                    if routing_method_type == RoutingMethodType.DeepSeekV3
                    else router_logits
                ),
                routing_bias=correction_bias,
                hidden_states=a_q,
                hidden_states_scale=a_sf_t,
                gemm1_weights=quant_info.w13_weight,
                gemm1_weights_scale=quant_info.w13_weight_scale_inv,
                gemm2_weights=quant_info.w2_weight,
                gemm2_weights_scale=quant_info.w2_weight_scale_inv,
                num_experts=quant_info.global_num_experts,
                top_k=topk_config.top_k,
                n_group=topk_config.num_expert_group,
                topk_group=topk_config.topk_group,
                intermediate_size=quant_info.intermediate_size,
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=quant_info.local_num_experts,
                routed_scaling_factor=(
                    runner_config.routed_scaling_factor
                    if runner_config.routed_scaling_factor is not None
                    else 1.0
                ),
                routing_method_type=routing_method_type,
                use_shuffled_weight=use_shuffled_weight,
                tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
                fp8_quantization_type=int(fp8_quantization_type),
                activation_type=quant_info.activation_type,
            )
        # TODO: Once https://github.com/flashinfer-ai/flashinfer/issues/2703 is fixed, pass output to moe kernel and remove this copy.
        symm_output.copy_(output)
        output = symm_output
    else:
        assert TopKOutputChecker.format_is_bypassed(topk_output)
        assert quant_info.w13_input_scale is not None
        assert quant_info.output1_scales_scalar is not None
        assert quant_info.output1_scales_gate_scalar is not None
        assert quant_info.output2_scales_scalar is not None

        a_q, _ = scaled_fp8_quant(hidden_states, quant_info.w13_input_scale)
        routing_bias_cast = (
            None if correction_bias is None else correction_bias.to(torch.bfloat16)
        )

        # Allocate output inside symmetric memory context
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            symm_output = torch.empty(
                hidden_states.shape[0],
                hidden_states.shape[1],
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )

        # Move kernel call outside context manager to avoid graph breaks
        # during torch.compile for piecewise cuda graph.
        # Use custom op wrapper for torch.compile compatibility.

        # The DeepSeekV3 routing method requires float32 router logits.
        if routing_method_type == RoutingMethodType.DeepSeekV3:
            router_logits = router_logits.to(torch.float32)
        else:
            router_logits = router_logits.to(torch.bfloat16)

        output = trtllm_fp8_per_tensor_scale_moe_wrapper(
            routing_logits=router_logits,
            routing_bias=routing_bias_cast,
            hidden_states=a_q,
            gemm1_weights=quant_info.w13_weight,
            output1_scales_scalar=quant_info.output1_scales_scalar,
            output1_scales_gate_scalar=quant_info.output1_scales_gate_scalar,
            gemm2_weights=quant_info.w2_weight,
            output2_scales_scalar=quant_info.output2_scales_scalar,
            num_experts=quant_info.global_num_experts,
            top_k=topk_config.top_k,
            n_group=topk_config.num_expert_group,
            topk_group=topk_config.topk_group,
            intermediate_size=int(quant_info.w2_weight.shape[2]),
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            routed_scaling_factor=(
                runner_config.routed_scaling_factor
                if runner_config.routed_scaling_factor is not None
                else 1.0
            ),
            use_routing_scales_on_input=False,
            routing_method_type=routing_method_type,
            tune_max_num_tokens=next_power_of_2(a_q.shape[0]),
            activation_type=quant_info.activation_type,
        )
        symm_output.copy_(output)
        output = symm_output

    return StandardCombineInput(hidden_states=output)


@dataclass
class FlashInferTrtllmFp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer TRT-LLM FP4 MoE kernels."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor

    # Scaling factors
    g1_scale_c: torch.Tensor
    g1_alphas: torch.Tensor
    g2_alphas: torch.Tensor
    w13_input_scale_quant: torch.Tensor

    # Expert-parallel metadata
    global_num_experts: int
    local_expert_offset: int
    local_num_experts: int
    intermediate_size_per_partition: int

    routing_method_type: int


def quantize_hidden_states_fp4(
    hidden_states: torch.Tensor,
    input_scale_quant: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize hidden states to FP4 for TRTLLM MoE.

    Global scale factor is set by ModelOptNvFp4FusedMoEMethod during weight loading.
    Only block scales are computed at runtime for efficiency.

    Returns (packed_fp4_uint8, scale_float8_e4m3fn_runtime)
    """

    # flashinfer.fp4_quantize returns (packed_uint8, scale_fp8)
    # Only the block scales are computed at runtime
    hs_fp4_bytes, hs_sf_bytes = fp4_quantize(
        hidden_states,
        input_scale_quant,
        16,  # sf_vec_size
        False,  # use_ue8m0
        False,  # is_sf_swizzled_layout
    )

    seq_len, hidden_size = hidden_states.shape
    hs_fp4 = hs_fp4_bytes.reshape(seq_len, hidden_size // 2)
    # TRT-LLM expects hidden state scales shaped as [seq_len, hidden_size // 16]
    hs_sf = hs_sf_bytes.view(torch.float8_e4m3fn).reshape(seq_len, hidden_size // 16)

    return hs_fp4, hs_sf


def fused_experts_none_to_flashinfer_trtllm_fp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    use_routed_topk: bool = False,
) -> StandardCombineInput:
    """FlashInfer TRTLLM FP4 MoE forward pass.

    This function handles the FP4 TRTLLM MoE path that was previously in
    ModelOptNvFp4FusedMoEMethod.apply.
    """
    from flashinfer.fused_moe import (
        trtllm_fp4_block_scale_moe,
        trtllm_fp4_block_scale_routed_moe,
    )

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType

    _SUPPORTED_FP4_ACTIVATIONS = {"silu", "relu2"}
    assert runner_config.activation in _SUPPORTED_FP4_ACTIVATIONS, (
        f"Only {_SUPPORTED_FP4_ACTIVATIONS} are supported for FP4 MoE, "
        f"got '{runner_config.activation}'."
    )

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    # Quantize hidden states to FP4
    hs_fp4, hs_scale_linear = quantize_hidden_states_fp4(
        hidden_states, quant_info.w13_input_scale_quant
    )
    hs_scale = hs_scale_linear.view(torch.float8_e4m3fn).reshape(
        *hs_scale_linear.shape[:-1], -1
    )
    activation_type = get_activation_type(runner_config.activation)

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        num_tokens = hs_fp4.shape[0]
        hidden_size = (
            hs_fp4.shape[-1] * 2 if hs_fp4.dtype == torch.uint8 else hs_fp4.shape[-1]
        )
        symm_output = torch.empty(
            num_tokens, hidden_size, dtype=hidden_states.dtype, device=hs_fp4.device
        )

    if use_routed_topk:
        assert TopKOutputChecker.format_is_standard(topk_output)

        packed_topk_ids = _pack_topk_for_flashinfer_routed(
            topk_output.topk_ids, topk_output.topk_weights
        )
        result = trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed_topk_ids,
            routing_bias=None,
            hidden_states=hs_fp4,
            hidden_states_scale=hs_scale,
            gemm1_weights=quant_info.w13_weight,
            gemm1_weights_scale=quant_info.w13_weight_scale.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=quant_info.w2_weight,
            gemm2_weights_scale=quant_info.w2_weight_scale.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=quant_info.g1_scale_c,
            output1_scale_gate_scalar=quant_info.g1_alphas,
            output2_scale_scalar=quant_info.g2_alphas,
            num_experts=quant_info.global_num_experts,
            top_k=topk_output.topk_ids.shape[1],
            n_group=0,
            topk_group=0,
            intermediate_size=quant_info.intermediate_size_per_partition,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=1,  # Unused, but must be 1 to pass validation.
            do_finalize=True,
            activation_type=activation_type,
            tune_max_num_tokens=next_power_of_2(hs_fp4.shape[0]),
            output=symm_output,
        )[0]
    else:
        assert TopKOutputChecker.format_is_bypassed(topk_output)

        router_logits = topk_output.router_logits
        topk_config = topk_output.topk_config
        routing_method_type = quant_info.routing_method_type

        # DeepSeekV3 style routing requires float32 router logits
        if routing_method_type == RoutingMethodType.DeepSeekV3:
            router_logits = router_logits.to(torch.float32)

        correction_bias = (
            None
            if topk_config.correction_bias is None
            else topk_config.correction_bias.to(hidden_states.dtype)
        )
        result = trtllm_fp4_block_scale_moe(
            routing_logits=router_logits,
            routing_bias=correction_bias,
            hidden_states=hs_fp4,
            hidden_states_scale=hs_scale,
            gemm1_weights=quant_info.w13_weight,
            gemm1_weights_scale=quant_info.w13_weight_scale.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=quant_info.w2_weight,
            gemm2_weights_scale=quant_info.w2_weight_scale.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=quant_info.g1_scale_c,
            output1_scale_gate_scalar=quant_info.g1_alphas,
            output2_scale_scalar=quant_info.g2_alphas,
            num_experts=quant_info.global_num_experts,
            top_k=topk_config.top_k,
            n_group=topk_config.num_expert_group,
            topk_group=topk_config.topk_group,
            intermediate_size=quant_info.intermediate_size_per_partition,
            local_expert_offset=quant_info.local_expert_offset,
            local_num_experts=quant_info.local_num_experts,
            routed_scaling_factor=runner_config.routed_scaling_factor,
            routing_method_type=(
                routing_method_type
                if routing_method_type is not None
                else RoutingMethodType.Default
            ),
            do_finalize=True,
            activation_type=activation_type,
            tune_max_num_tokens=next_power_of_2(hs_fp4.shape[0]),
            output=symm_output,
        )[0]

    return StandardCombineInput(hidden_states=result)


@dataclass
class FlashInferTrtllmBf16MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer TRT-LLM BF16 MoE kernels."""

    gemm1_weights: torch.Tensor
    gemm2_weights: torch.Tensor

    # Expert-parallel metadata
    global_num_experts: int
    local_expert_offset: int


def fused_experts_none_to_flashinfer_trtllm_bf16(
    dispatch_output: StandardDispatchOutput,
    quant_info: FlashInferTrtllmBf16MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    use_routed_topk: bool = False,
) -> StandardCombineInput:
    # lazy import
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.moe.utils import RoutingMethodType

    trtllm_bf16_routed_moe = None
    trtllm_bf16_moe = None
    if use_routed_topk:
        try:
            from flashinfer.fused_moe import trtllm_bf16_routed_moe
        except ImportError as e:
            raise ImportError(
                "Can't import trtllm_bf16_routed_moe from flashinfer. "
                "Please check flashinfer version to use bf16 with flashinfer_trtllm_routed backend."
            ) from e
    else:
        try:
            from flashinfer.fused_moe import trtllm_bf16_moe
        except ImportError as e:
            raise ImportError(
                "Can't import trtllm_bf16_moe from flashinfer. "
                "Please check flashinfer version to use bf16 with flashinfer_trtllm backend."
            ) from e

    _SUPPORTED_BF16_ACTIVATIONS = {"silu", "relu2"}
    assert runner_config.activation in _SUPPORTED_BF16_ACTIVATIONS, (
        f"Only {_SUPPORTED_BF16_ACTIVATIONS} are supported for flashinfer trtllm bf16 moe, "
        f"got '{runner_config.activation}'."
    )
    if not use_routed_topk:
        assert (
            dispatch_output.topk_output.topk_config.renormalize
        ), "Renormalize is required for flashinfer trtllm moe"
    assert (
        runner_config.num_fused_shared_experts == 0
    ), "Fused shared experts are not supported for flashinfer trtllm moe"
    activation_type = get_activation_type(runner_config.activation)

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        if use_routed_topk:
            assert (
                runner_config.top_k is not None
            ), "runner_config.top_k is required for flashinfer_trtllm_routed."
            assert TopKOutputChecker.format_is_standard(topk_output)
            routing_method_type = runner_config.routing_method_type
            if routing_method_type is None:
                routing_method_type = RoutingMethodType.Default
            elif routing_method_type == RoutingMethodType.DeepSeekV3:
                routing_method_type = RoutingMethodType.TopK

            packed_topk_ids = _pack_topk_for_flashinfer_routed(
                topk_ids=topk_output.topk_ids,
                topk_weights=topk_output.topk_weights,
            )
            final_hidden_states = trtllm_bf16_routed_moe(
                topk_ids=packed_topk_ids,
                hidden_states=hidden_states,
                gemm1_weights=quant_info.gemm1_weights,
                gemm2_weights=quant_info.gemm2_weights,
                num_experts=quant_info.global_num_experts,
                top_k=runner_config.top_k,
                n_group=None,
                topk_group=None,
                intermediate_size=runner_config.intermediate_size_per_partition,
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=runner_config.num_local_experts,
                routing_method_type=routing_method_type,
                routed_scaling_factor=(
                    runner_config.routed_scaling_factor
                    if runner_config.routed_scaling_factor is not None
                    else 1.0
                ),
                tune_max_num_tokens=next_power_of_2(hidden_states.shape[0]),
                activation_type=activation_type,
            )
        else:
            assert TopKOutputChecker.format_is_bypassed(topk_output)
            topk_config = topk_output.topk_config

            # Call the fused kernel
            final_hidden_states = trtllm_bf16_moe(
                routing_logits=topk_output.router_logits,
                routing_bias=topk_config.correction_bias,
                hidden_states=hidden_states,
                gemm1_weights=quant_info.gemm1_weights,
                gemm2_weights=quant_info.gemm2_weights,
                num_experts=quant_info.global_num_experts,
                top_k=topk_config.top_k,
                n_group=topk_config.num_expert_group,
                topk_group=topk_config.topk_group,
                intermediate_size=runner_config.intermediate_size_per_partition,
                local_expert_offset=quant_info.local_expert_offset,
                local_num_experts=runner_config.num_local_experts,
                routing_method_type=runner_config.routing_method_type,
                routed_scaling_factor=runner_config.routed_scaling_factor,
                tune_max_num_tokens=next_power_of_2(hidden_states.shape[0]),
                activation_type=activation_type,
            )

    return StandardCombineInput(hidden_states=final_hidden_states)


@register_fused_func("none", "flashinfer_trtllm")
def fused_experts_none_to_flashinfer_trtllm(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Dispatch to FP8 or FP4 FlashInfer TRT-LLM MoE based on quant_info type."""
    if isinstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch_output, quant_info, runner_config
        )
    if isinstance(quant_info, FlashInferTrtllmFp8MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_fp8(
            dispatch_output, quant_info, runner_config
        )
    if isinstance(quant_info, FlashInferTrtllmBf16MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_bf16(
            dispatch_output, quant_info, runner_config
        )
    raise TypeError(
        f"Unexpected quant_info type for flashinfer_trtllm: {type(quant_info)}"
    )


@register_fused_func("none", "flashinfer_trtllm_routed")
def fused_experts_none_to_flashinfer_trtllm_routed(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    if isinstance(quant_info, FlashInferTrtllmFp4MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,
        )
    if isinstance(quant_info, FlashInferTrtllmFp8MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_fp8(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,
        )
    if isinstance(quant_info, FlashInferTrtllmBf16MoeQuantInfo):
        return fused_experts_none_to_flashinfer_trtllm_bf16(
            dispatch_output,
            quant_info,
            runner_config,
            use_routed_topk=True,
        )
    raise TypeError(
        f"Unexpected quant_info type for flashinfer_trtllm_routed: {type(quant_info)}"
    )
