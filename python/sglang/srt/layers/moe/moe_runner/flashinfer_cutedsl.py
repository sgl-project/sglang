from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.utils.common import log_info_on_rank0, print_warning_once

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
        FlashinferDispatchOutput,
    )

logger = logging.getLogger(__name__)

_FP4_SF_VEC_SIZE = 16
_cutedsl_logged_scalarize: set = set()


# ---------------------------------------------------------------------------
# Weight / scale preparation utilities (called from modelopt_quant.py during
# process_weights_after_loading and lazy wrapper init)
# ---------------------------------------------------------------------------


def interleave_w13_halves(
    tensor: torch.Tensor, group_size: int = 64, dim: int = 1
) -> torch.Tensor:
    """Interleave the two logical W13 halves for CuteDSL's SwiGLU GEMM1 layout.

    The caller is responsible for loading W13 in the expected two-half order.
    This helper only rewrites the first and second halves into alternating
    `group_size` chunks along `dim`.
    """
    if tensor.shape[dim] % 2 != 0:
        raise ValueError(
            "Expected even size on interleave dimension for W13 half split."
        )
    split = tensor.shape[dim] // 2
    if split % group_size != 0:
        raise ValueError(
            f"Expected split dim divisible by group_size={group_size}, got {split}."
        )
    first_half = tensor.narrow(dim, 0, split)
    second_half = tensor.narrow(dim, split, split)
    first_half_groups = first_half.split(group_size, dim=dim)
    second_half_groups = second_half.split(group_size, dim=dim)
    interleaved = [
        item for pair in zip(first_half_groups, second_half_groups) for item in pair
    ]
    return torch.cat(interleaved, dim=dim)


def cutedsl_quant_scale_to_scalar(
    quant_scale: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Reduce per-expert quant-domain scale vector to a single scalar.

    The quant domain is the reciprocal of the raw checkpoint scale:
        quant_scale = 1 / raw_scale

    Returns min(quant_scale) = 1/max(raw_scale), which is the TRTLLM CuteDSL
    convention for global scalar activation scales (see TRTLLM quantization.py
    lines 2137-2141: fc2_input_scale = tmp_fc2_input_scale.max().reciprocal()).

    If quant_scale is already scalar (numel==1), returns it unchanged.
    """
    quant_scale = quant_scale.to(torch.float32)
    if quant_scale.numel() == 0:
        print_warning_once(
            f"CuteDSL got empty {name}; using 1.0 fallback.",
        )
        return torch.ones(1, device=quant_scale.device, dtype=torch.float32)
    if quant_scale.numel() == 1:
        return quant_scale.reshape(1)
    if name not in _cutedsl_logged_scalarize:
        log_info_on_rank0(
            logger,
            f"CuteDSL: reducing per-expert {name} to scalar via "
            "min(quant_scale) = 1/max(raw_scale), matching TRTLLM convention.",
        )
        _cutedsl_logged_scalarize.add(name)
    return quant_scale.min().reshape(1)


def resolve_cutedsl_standard_scales(
    layer: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve standard-path CuteDSL scales (baseline: scalar fc2/w13 input scales).

    Returns (w1_alpha, fc2_input_scale, w2_alpha, used_input_scale).
    used_input_scale is the scalarized w13 input scale for FP4 quantize and GEMM1.
    """

    def _to_fp32_tensor(x: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=ref.device)
        return x.to(device=ref.device, dtype=torch.float32)

    def _align_scale_to_alpha(
        scale: torch.Tensor, alpha: torch.Tensor, scale_name: str
    ) -> torch.Tensor:
        scale = scale.to(device=alpha.device, dtype=torch.float32)
        alpha = alpha.to(torch.float32)
        if scale.ndim == 0:
            return scale
        # Gated weight scales may be (num_experts, 2) with separate gate/up
        # columns. Collapse to 1D by taking the first column (gate == up for
        # well-formed checkpoints; mismatch is warned in process_weights_after_loading).
        if scale.ndim == 2 and scale.shape[1] <= 2:
            scale = scale[:, 0]
        if scale.numel() == alpha.numel():
            return scale
        if scale.numel() == 1:
            return scale.reshape(())

        # Some EP setups may carry global-per-expert scale vectors while alphas are
        # local-per-expert vectors. Slice to this rank's local expert range.
        num_local_experts = getattr(layer, "num_local_experts", None)
        num_experts = getattr(layer, "num_experts", None)
        moe_ep_rank = getattr(layer, "moe_ep_rank", 0)
        if (
            num_local_experts is not None
            and num_experts is not None
            and scale.numel() == num_experts
            and alpha.numel() == num_local_experts
        ):
            start = moe_ep_rank * num_local_experts
            end = start + num_local_experts
            return scale[start:end]

        raise ValueError(
            f"Unable to align {scale_name} shape={tuple(scale.shape)} "
            f"to alpha shape={tuple(alpha.shape)} for CuteDSL standard scale resolution."
        )

    def _resolve_w1_alpha_from_scalar_input_scale(
        used_input_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve GEMM1 alpha consistent with scalarized activation quant scale.

        CuteDSL pre-quantizes x with a single scalar (used_input_scale), but
        g1_alphas was derived with per-expert activation scales:
            g1_alphas[e] = (1/w13_isq[e]) * w13_ws2[e]
        Correct alpha for scalar quantization:
            w1_alpha[e] = w13_ws2[e] / used_input_scale
                        = g1_alphas[e] * w13_isq[e] / used_input_scale
        When w13_isq is already scalar, this is a no-op (ratio = 1).
        """
        eps = 1e-12
        scalar = torch.clamp(used_input_scale.to(torch.float32).reshape(()), min=eps)

        if hasattr(layer, "w13_weight_scale_2"):
            w13_weight_scale_2 = _align_scale_to_alpha(
                layer.w13_weight_scale_2, layer.g1_alphas, "w13_weight_scale_2"
            )
            return w13_weight_scale_2.to(torch.float32) / scalar

        w13_isq = _align_scale_to_alpha(
            layer.w13_input_scale_quant, layer.g1_alphas, "w13_input_scale_quant"
        )
        w13_isq = torch.clamp(_to_fp32_tensor(w13_isq, layer.g1_alphas), min=eps)
        return (layer.g1_alphas.to(torch.float32) * w13_isq / scalar).to(torch.float32)

    def _resolve_w2_alpha_from_scalar_fc2_input_scale(
        fc2_input_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve GEMM2 alpha consistent with scalarized FC2 input scale.

        CuteDSL standard path uses a scalar global scale for GEMM1 FP4 output
        quantization (`fc2_input_scale`). GEMM2 alpha must use the same scalar
        convention: alpha2 = w2_weight_scale_2 / fc2_input_scale.
        """
        eps = 1e-12
        fc2_input_scale = fc2_input_scale.to(torch.float32)
        fc2_scalar = torch.clamp(fc2_input_scale.reshape(-1)[:1], min=eps).reshape(())

        if hasattr(layer, "w2_weight_scale_2"):
            w2_weight_scale_2 = _align_scale_to_alpha(
                layer.w2_weight_scale_2, layer.g2_alphas, "w2_weight_scale_2"
            )
            w2_weight_scale_2 = w2_weight_scale_2.to(torch.float32)
            return w2_weight_scale_2 / fc2_scalar

        w2_q_for_w2 = _align_scale_to_alpha(
            layer.w2_input_scale_quant, layer.g2_alphas, "w2_input_scale_quant"
        )
        w2_q_for_w2 = torch.clamp(
            _to_fp32_tensor(w2_q_for_w2, layer.g2_alphas), min=eps
        )
        w2_weight_scale_2 = layer.g2_alphas.to(torch.float32) * w2_q_for_w2
        return w2_weight_scale_2 / fc2_scalar

    fc2_input_scale = cutedsl_quant_scale_to_scalar(
        layer.w2_input_scale_quant,
        name="w2_input_scale_quant",
    )
    w2_alpha = _resolve_w2_alpha_from_scalar_fc2_input_scale(fc2_input_scale)
    used_input_scale = cutedsl_quant_scale_to_scalar(
        layer.w13_input_scale_quant,
        name="w13_input_scale_quant",
    )
    w1_alpha = _resolve_w1_alpha_from_scalar_input_scale(used_input_scale)
    return w1_alpha, fc2_input_scale, w2_alpha, used_input_scale


def ensure_cutedsl_wrapper(layer: torch.nn.Module, num_tokens: int = 0) -> None:
    """Lazily create CuteDslMoEWrapper and resolve scales on first forward.

    Args:
        layer: The FusedMoE layer module.
        num_tokens: Current token count entering the MoE layer.  Used as
            the buffer size for the non-a2a (allgather) path, where the
            autotune dummy run passes req_to_token_pool.size * dp_size —
            the worst-case post-allgather batch.  For the a2a path this
            is ignored in favour of the dispatcher's workspace limit.

    The wrapper is created lazily (not in __init__ / create_weights) because
    it depends on final weight shapes and EP configuration.  The wrapper's
    CUDA-graph buffers are allocated inside CuteDslMoEWrapper.__init__, which
    typically runs during the autotune dummy forward under inference_mode().
    We wrap the creation in inference_mode(False) so that those pre-allocated
    buffers are normal tensors -- inference tensors cannot be inplace-updated
    during later CUDA graph capture, which runs outside inference_mode.
    """
    if getattr(layer, "_cutedsl_wrapper", None) is not None:
        return

    try:
        from flashinfer import CuteDslMoEWrapper
    except ImportError as e:
        raise ImportError(
            "flashinfer_cutedsl backend requires FlashInfer with CuteDSL support. "
            "Install with: pip install flashinfer"
        ) from e

    from sglang.srt.server_args import get_global_server_args

    assert layer.intermediate_size_per_partition > 0, (
        f"CuteDSL MoE: intermediate_size_per_partition must be > 0, "
        f"got {layer.intermediate_size_per_partition}. Check EP/TP configuration."
    )

    server_args = get_global_server_args()
    use_cuda_graph = server_args is not None and not server_args.disable_cuda_graph

    # Buffer size must cover the worst-case token count the MoE layer can see.
    # - A2A path: dispatch returns tensors flattened from
    #   [ep_size, max_tokens_per_rank, ...].
    # - Standard allgather path: dp_size * max local tokens per rank.
    dispatcher = getattr(layer, "dispatcher", None)
    if hasattr(dispatcher, "max_num_tokens"):
        max_num_tokens = dispatcher.max_num_tokens * getattr(dispatcher, "ep_size", 1)
    else:
        # Standard allgather path: num_tokens from the first forward is
        # req_to_token_pool.size * dp_size (the autotune dummy run's batch),
        # which is the worst-case post-allgather token count.
        max_num_tokens = max(num_tokens, 1)
    top_k = layer.top_k if layer.top_k is not None else layer.moe_runner_config.top_k
    # inference_mode(False) ensures the wrapper's pre-allocated CUDA-graph
    # buffers are normal tensors.  This call typically happens inside
    # _dummy_run which runs under inference_mode(); inference tensors cannot
    # be inplace-updated during later CUDA graph capture (which runs outside
    # inference_mode), so we must opt out here.
    with torch.inference_mode(False):
        layer._cutedsl_wrapper = CuteDslMoEWrapper(
            num_experts=layer.num_experts,
            top_k=top_k,
            hidden_size=layer.hidden_size,
            intermediate_size=layer.intermediate_size_per_partition,
            use_cuda_graph=use_cuda_graph,
            max_num_tokens=max_num_tokens,
            num_local_experts=layer.num_local_experts,
            local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
            output_dtype=layer.moe_runner_config.params_dtype,
            device=str(layer.w13_weight.device),
        )

    w1_alpha, fc2_input_scale, w2_alpha, used_input_scale = (
        resolve_cutedsl_standard_scales(layer)
    )
    layer._cutedsl_scales = (w1_alpha, fc2_input_scale, w2_alpha)
    layer._cutedsl_input_scale = used_input_scale


# ---------------------------------------------------------------------------
# Dataclass + fused function for moe_runner dispatch
# ---------------------------------------------------------------------------


@dataclass
class CuteDslFp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload for FlashInfer CuteDSL FP4 MoE kernels.

    Shared by the two CuteDSL runner entries:

    * "v2" standard path (a2a=``none``/``flashinfer``): consumed by the
      ``@register_fused_func("none", "flashinfer_cutedsl")`` entry, which
      drives ``CuteDslMoEWrapper.run``. Weights are ``[Up, Gate]``
      interleaved with MMA-layout blockscales. ``wrapper`` is set;
      ``w*_scale`` are scalarized.

    * "v1" DeepEP low-latency path (a2a=``deepep``): consumed by the
      ``@register_fused_func("deepep", "flashinfer_cutedsl")`` entry,
      which drives ``flashinfer_cutedsl_moe_masked``. Weights are
      ``[Gate, Up]`` non-interleaved with swizzled blockscales.
      ``wrapper`` is ``None``; ``w*_scale`` are per-expert.
    """

    # FP4 packed weights (uint8)
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor

    # Block-scale factors (MMA layout for v2, swizzled for v1)
    w13_weight_sf: torch.Tensor
    w2_weight_sf: torch.Tensor

    # Per-expert GEMM dequant alphas (scalarized for v2, per-expert for v1)
    w1_alpha: torch.Tensor
    w2_alpha: torch.Tensor

    # Activation quant scales (1 / raw_input_scale).
    #   - a1_scale: quantizes hidden_states before GEMM1
    #   - a2_scale: quantizes GEMM1 output before GEMM2 (a.k.a. fc2 input)
    a1_scale: torch.Tensor
    a2_scale: torch.Tensor

    # v2 only: lazily-created CuteDslMoEWrapper (``None`` on the v1 path).
    wrapper: Optional[Any] = None

    # v1 only: ``True`` when DeepEP pre-quantizes activations to NVFP4.
    use_nvfp4_dispatch: bool = False

    # v1 only: SBO down-GEMM overlap args.
    down_gemm_overlap_args: Optional["DownGemmOverlapArgs"] = None


@register_fused_func("none", "flashinfer_cutedsl")
def fused_experts_none_to_flashinfer_cutedsl_fp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    assert runner_config.activation == "silu", "Only silu is supported for CuteDSL MoE."
    assert quant_info.wrapper is not None, "CuteDSL v2 path requires CuteDslMoEWrapper."

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    x_fp4, x_sf = fp4_quantize(
        hidden_states,
        quant_info.a1_scale,
        sf_vec_size=_FP4_SF_VEC_SIZE,
        is_sf_swizzled_layout=False,
    )

    output = quant_info.wrapper.run(
        x=x_fp4,
        x_sf=x_sf,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        w1_weight=quant_info.w13_weight,
        w1_weight_sf=quant_info.w13_weight_sf,
        w1_alpha=quant_info.w1_alpha,
        fc2_input_scale=quant_info.a2_scale,
        w2_weight=quant_info.w2_weight,
        w2_weight_sf=quant_info.w2_weight_sf,
        w2_alpha=quant_info.w2_alpha,
    )

    return StandardCombineInput(hidden_states=output)


@register_fused_func("flashinfer", "flashinfer_cutedsl")
def fused_experts_flashinfer_to_flashinfer_cutedsl_fp4(
    dispatch_output: FlashinferDispatchOutput,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> FlashinferCombineInput:
    """CuteDSL fused func for flashinfer alltoall dispatcher.

    Two cases depending on whether the dispatcher did FP4 quantization:
    - bf16 input (SGLANG_MOE_NVFP4_DISPATCH=0): quantize with cutedsl's scale
    - FP4 input (SGLANG_MOE_NVFP4_DISPATCH=1): pass through (same fp4_quantize params)
    """
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    assert runner_config.activation == "silu", "Only silu is supported for CuteDSL MoE."
    assert quant_info.wrapper is not None, "CuteDSL v2 path requires CuteDslMoEWrapper."

    hidden_states = dispatch_output.hidden_states
    x_sf = dispatch_output.hidden_states_scale
    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    if x_sf is not None:
        # NVFP4 dispatch, inputs are already quantized.
        x_fp4 = hidden_states
    else:
        x_fp4, x_sf = fp4_quantize(
            hidden_states,
            quant_info.a1_scale,
            sf_vec_size=_FP4_SF_VEC_SIZE,
            is_sf_swizzled_layout=False,
        )

    output = quant_info.wrapper.run(
        x=x_fp4,
        x_sf=x_sf,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        w1_weight=quant_info.w13_weight,
        w1_weight_sf=quant_info.w13_weight_sf,
        w1_alpha=quant_info.w1_alpha,
        fc2_input_scale=quant_info.a2_scale,
        w2_weight=quant_info.w2_weight,
        w2_weight_sf=quant_info.w2_weight_sf,
        w2_alpha=quant_info.w2_alpha,
    )

    # Note: output contains routed expert results; shared_expert is handled separately

    # Write into pre-allocated workspace buffer if available
    if dispatch_output.moe_output is not None:
        dispatch_output.moe_output.copy_(output)
        output = dispatch_output.moe_output

    return FlashinferCombineInput(hidden_states=output)


@register_fused_func("deepep", "flashinfer_cutedsl")
def fused_experts_deepep_to_flashinfer_cutedsl_fp4(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.flashinfer_cutedsl_moe import (
        flashinfer_cutedsl_moe_masked,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    assert runner_config.activation == "silu", "Only silu is supported for CuteDSL MoE."
    assert (
        not runner_config.apply_router_weight_on_input
    ), "apply_router_weight_on_input is not supported for Flashinfer"

    hidden_states, hidden_states_scale, _, _, masked_m, _ = dispatch_output

    # flashinfer_cutedsl_moe_masked reinterprets scales as float8_e4m3fn.
    # Same-dtype .view is a no-op; only wider dtypes (e.g. int32-packed
    # UE8M0) need stride(-1)==1.
    if (
        quant_info.use_nvfp4_dispatch
        and hidden_states_scale is not None
        and hidden_states_scale.element_size() != 1
        and hidden_states_scale.stride(-1) != 1
    ):
        raise AssertionError(
            f"NVFP4 dispatch scale has stride(-1)={hidden_states_scale.stride(-1)}, "
            f"dtype={hidden_states_scale.dtype}; .view(float8_e4m3fn) requires stride(-1)==1. "
            "Try SGLANG_MOE_NVFP4_DISPATCH=0 or check DeepEP version."
        )

    overlap = quant_info.down_gemm_overlap_args
    output = flashinfer_cutedsl_moe_masked(
        hidden_states=(hidden_states, hidden_states_scale),
        input_global_scale=(
            None if quant_info.use_nvfp4_dispatch else quant_info.a1_scale
        ),
        w1=quant_info.w13_weight,
        w1_blockscale=quant_info.w13_weight_sf,
        w1_alpha=quant_info.w1_alpha,
        w2=quant_info.w2_weight,
        a2_global_scale=quant_info.a2_scale,
        w2_blockscale=quant_info.w2_weight_sf,
        w2_alpha=quant_info.w2_alpha,
        masked_m=masked_m,
        **(
            dict(
                down_sm_count=overlap.num_sms,
                down_signals=overlap.signal,
                down_start_event=overlap.start_event,
            )
            if overlap is not None
            else {}
        ),
    )

    return DeepEPLLCombineInput(
        hidden_states=output,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
    )
