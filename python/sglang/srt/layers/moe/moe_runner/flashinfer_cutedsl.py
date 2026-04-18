from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Union

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.utils.common import log_info_on_rank0, print_warning_once

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
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


def _create_cutedsl_wrapper(
    layer: torch.nn.Module,
    num_local_experts: int,
    local_expert_offset: int,
) -> "CuteDslMoEWrapper":
    """Create a CuteDslMoEWrapper with the given expert configuration.

    Wraps creation in inference_mode(False) so pre-allocated CUDA-graph
    buffers are normal tensors (inference tensors cannot be inplace-updated
    during later CUDA graph capture).
    """
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
    max_num_tokens = max(
        getattr(server_args, "cuda_graph_max_bs", None) or 512,
        getattr(server_args, "chunked_prefill_size", None) or 8192,
    )
    top_k = layer.top_k if layer.top_k is not None else layer.moe_runner_config.top_k

    with torch.inference_mode(False):
        wrapper = CuteDslMoEWrapper(
            num_experts=layer.num_experts,
            top_k=top_k,
            hidden_size=layer.hidden_size,
            intermediate_size=layer.intermediate_size_per_partition,
            use_cuda_graph=use_cuda_graph,
            max_num_tokens=max_num_tokens,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            output_dtype=layer.moe_runner_config.params_dtype,
            device=str(layer.w13_weight.device),
        )
    return wrapper


def ensure_cutedsl_wrapper(layer: torch.nn.Module) -> None:
    """Lazily create CuteDslMoEWrapper for EP (local experts) and resolve scales.

    Creates the wrapper for local experts with the EP-appropriate offset.
    For DWDP hybrid mode, use ensure_cutedsl_wrapper_dwdp() separately
    to create a second wrapper that covers all routed experts.
    """
    if getattr(layer, "_cutedsl_wrapper", None) is not None:
        return

    layer._cutedsl_wrapper = _create_cutedsl_wrapper(
        layer,
        num_local_experts=layer.num_local_experts,
        local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
    )

    w1_alpha, fc2_input_scale, w2_alpha, used_input_scale = (
        resolve_cutedsl_standard_scales(layer)
    )
    layer._cutedsl_scales = (w1_alpha, fc2_input_scale, w2_alpha)
    layer._cutedsl_input_scale = used_input_scale


def ensure_cutedsl_wrapper_dwdp(layer: torch.nn.Module) -> None:
    """Lazily create CuteDslMoEWrapper for DWDP (all routed experts, offset=0).

    Also ensures the EP wrapper and scales exist (via ensure_cutedsl_wrapper).
    """
    if getattr(layer, "_cutedsl_wrapper_dwdp", None) is not None:
        return

    # Ensure EP wrapper + scales are created first
    ensure_cutedsl_wrapper(layer)

    layer._cutedsl_wrapper_dwdp = _create_cutedsl_wrapper(
        layer,
        num_local_experts=layer._num_global_routed,
        local_expert_offset=0,
    )


# ---------------------------------------------------------------------------
# Dataclass + fused function for moe_runner dispatch
# ---------------------------------------------------------------------------


@dataclass
class CuteDslFp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer CuteDSL FP4 MoE kernels.

    Weight fields accept either a single Tensor (standard path) or a
    List[Tensor] (DWDP multi-B path where each entry corresponds to a
    different DWDP rank's expert weights).
    """

    # Lazily-created CuteDslMoEWrapper (stashed on layer)
    wrapper: Any

    # Weights (uint8 FP4 packed)
    w13_weight: Union[torch.Tensor, List[torch.Tensor]]
    w2_weight: Union[torch.Tensor, List[torch.Tensor]]

    # Block-scale factors
    w13_weight_sf: Union[torch.Tensor, List[torch.Tensor]]
    w2_weight_sf: Union[torch.Tensor, List[torch.Tensor]]

    # Per-expert GEMM scales
    w1_alpha: Union[torch.Tensor, List[torch.Tensor]]
    w2_alpha: Union[torch.Tensor, List[torch.Tensor]]

    # Intermediate quantization scale (fc2 input)
    fc2_input_scale: torch.Tensor

    # Activation quantization scale (scalarized)
    input_scale: torch.Tensor

    # Reference to the FusedMoE layer for the masked kernel path (DeepEP LL).
    # The masked kernel needs different weight formats (swizzled blockscale,
    # raw per-expert alphas) than the wrapper path (MMA blockscale, scalarized
    # alphas).  Optional — only populated when DeepEP A2A is active.
    layer: Any = None


def _cutedsl_fp4_core(
    hidden_states: torch.Tensor,
    topk_output,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> torch.Tensor:
    """Core CuteDSL FP4 compute: quantize activations + run CuteDSL wrapper."""
    from flashinfer import fp4_quantize

    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert runner_config.activation == "silu", "Only silu is supported for CuteDSL MoE."
    assert TopKOutputChecker.format_is_standard(topk_output)

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    x_fp4, x_sf = fp4_quantize(
        hidden_states,
        quant_info.input_scale,
        sf_vec_size=_FP4_SF_VEC_SIZE,
        is_sf_swizzled_layout=False,
    )

    # CuteDSL multi-B API accepts List[Tensor] natively for DWDP
    return quant_info.wrapper.run(
        x=x_fp4,
        x_sf=x_sf,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        w1_weight=quant_info.w13_weight,
        w1_weight_sf=quant_info.w13_weight_sf,
        w1_alpha=quant_info.w1_alpha,
        fc2_input_scale=quant_info.fc2_input_scale,
        w2_weight=quant_info.w2_weight,
        w2_weight_sf=quant_info.w2_weight_sf,
        w2_alpha=quant_info.w2_alpha,
    )


@register_fused_func("none", "flashinfer_cutedsl")
def fused_experts_none_to_flashinfer_cutedsl_fp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    output = _cutedsl_fp4_core(
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
        quant_info,
        runner_config,
    )
    return StandardCombineInput(hidden_states=output)


@register_fused_func("flashinfer", "flashinfer_cutedsl")
def fused_experts_flashinfer_to_flashinfer_cutedsl_fp4(
    dispatch_output,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
):
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import FlashinferCombineInput

    output = _cutedsl_fp4_core(
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
        quant_info,
        runner_config,
    )
    return FlashinferCombineInput(hidden_states=output)


@register_fused_func("deepep", "flashinfer_cutedsl")
def fused_experts_deepep_to_flashinfer_cutedsl_fp4(
    dispatch_output,
    quant_info: CuteDslFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
):
    """DeepEP LL + CuteDSL: use the masked grouped-GEMM kernel directly.

    This mirrors modelopt_quant.py:apply_without_routing_weights() — same
    weight attributes, same call signature to flashinfer_cutedsl_moe_masked().
    The masked kernel uses swizzled blockscale and raw per-expert alphas,
    which differ from the wrapper path's MMA blockscale and scalarized alphas.
    """
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.flashinfer_cutedsl_moe import (
        flashinfer_cutedsl_moe_masked,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    assert dispatch_output.format.name == "DEEPEP_LL", (
        f"DeepEP Normal + CuteDSL not supported; DWDP hybrid decode uses LL mode. "
        f"Got format: {dispatch_output.format}"
    )

    layer = quant_info.layer
    assert layer is not None, (
        "CuteDslFp4MoeQuantInfo.layer must be set for DeepEP LL path"
    )

    moe_nvfp4_dispatch = envs.SGLANG_MOE_NVFP4_DISPATCH.get()

    output = flashinfer_cutedsl_moe_masked(
        hidden_states=(dispatch_output.hidden_states, dispatch_output.hidden_states_scale),
        input_global_scale=(
            None if moe_nvfp4_dispatch else layer.w13_input_scale_quant
        ),
        w1=layer.w13_weight,
        w1_blockscale=layer.w13_blockscale_swizzled,
        w1_alpha=layer.g1_alphas,
        w2=layer.w2_weight,
        a2_global_scale=layer.w2_input_scale_quant,
        w2_blockscale=layer.w2_blockscale_swizzled,
        w2_alpha=layer.g2_alphas,
        masked_m=dispatch_output.masked_m,
    )

    return DeepEPLLCombineInput(
        hidden_states=output,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
    )
