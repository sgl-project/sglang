from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    FusedOpPool,
    MoeQuantInfo,
    MoeRunnerConfig,
)
from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
    resolve_cutedsl_standard_scales,
)
from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
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


_RUNNER_BACKEND_NAME = "flashinfer_cutedsl_sm120"
_B12X_PLAN_CACHE = {}
_B12X_OUTPUT_CACHE = {}


def _load_b12x_moe_fp4() -> Callable:
    try:
        from b12x.integration.tp_moe import b12x_moe_fp4
    except ImportError as e:
        raise ImportError(
            "flashinfer_cutedsl_sm120 requires b12x.integration.tp_moe.b12x_moe_fp4 "
            "to be importable."
        ) from e
    return b12x_moe_fp4


def ensure_cutedsl_sm120_runtime(layer: torch.nn.Module) -> None:
    if getattr(layer, "_cutedsl_sm120_ready", False):
        return

    if not torch.cuda.is_available():
        raise RuntimeError("flashinfer_cutedsl_sm120 requires CUDA.")

    major, _ = torch.cuda.get_device_capability(layer.w13_weight.device)
    if major < 12:
        raise RuntimeError(
            f"flashinfer_cutedsl_sm120 requires SM120+, got compute capability sm{major}x."
        )

    _load_b12x_moe_fp4()
    w1_alpha, fc2_input_scale, w2_alpha, used_input_scale = (
        resolve_cutedsl_standard_scales(layer)
    )
    layer._cutedsl_sm120_scales = (w1_alpha, fc2_input_scale, w2_alpha)
    layer._cutedsl_sm120_input_scale = used_input_scale
    layer._cutedsl_sm120_ready = True


@dataclass
class CuteDslSm120Fp4MoeQuantInfo(MoeQuantInfo):
    # FP4 packed expert weights.
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor

    # Expert blockscales in b12x-compatible layout.
    w13_weight_sf: torch.Tensor
    w2_weight_sf: torch.Tensor

    # Dequant alphas.
    w1_alpha: torch.Tensor
    w2_alpha: torch.Tensor

    # Reciprocal activation scales.
    a1_scale: torch.Tensor
    a2_scale: torch.Tensor


def _run_sm120_b12x(
    *,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    quant_info: CuteDslSm120Fp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    if runner_config.activation != "silu":
        raise AssertionError("Only silu is supported for flashinfer_cutedsl_sm120.")
    if runner_config.apply_router_weight_on_input:
        raise AssertionError(
            "apply_router_weight_on_input is not supported for flashinfer_cutedsl_sm120."
        )

    b12x_moe_fp4 = _load_b12x_moe_fp4()
    if output is None:
        device_idx = (
            hidden_states.device.index if hidden_states.device.index is not None else 0
        )
        output_cache_key = (
            device_idx,
            int(hidden_states.shape[0]),
            int(hidden_states.shape[1]),
            str(hidden_states.dtype),
        )
        output = _B12X_OUTPUT_CACHE.get(output_cache_key)
        if output is None:
            output = torch.empty_like(hidden_states)
            _B12X_OUTPUT_CACHE[output_cache_key] = output
    output.zero_()

    kwargs = dict(
        a=hidden_states,
        a1_gscale=quant_info.a1_scale,
        w1_fp4=quant_info.w13_weight,
        w1_blockscale=quant_info.w13_weight_sf,
        w1_alphas=quant_info.w1_alpha,
        a2_gscale=quant_info.a2_scale,
        w2_fp4=quant_info.w2_weight,
        w2_blockscale=quant_info.w2_weight_sf,
        w2_alphas=quant_info.w2_alpha,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation="silu",
        quant_mode="nvfp4",
        source_format="modelopt_nvfp4",
        input_scales_are_reciprocal=True,
        input_scales_static=True,
        apply_router_weight_on_input=False,
        output=output,
    )
    from b12x.integration import TPMoEScratchCaps, plan_tp_moe_scratch

    m = int(hidden_states.shape[0])
    k = int(hidden_states.shape[1])
    weight_E = int(quant_info.w13_weight.shape[0])
    n = int(quant_info.w2_weight.shape[2]) * 2
    num_topk = int(topk_ids.shape[1])
    device_idx = (
        hidden_states.device.index if hidden_states.device.index is not None else 0
    )
    cache_key = (
        device_idx,
        m,
        k,
        weight_E,
        n,
        num_topk,
        str(hidden_states.dtype),
    )
    plan = _B12X_PLAN_CACHE.get(cache_key)
    if plan is None:
        plan = plan_tp_moe_scratch(
            TPMoEScratchCaps(
                max_tokens=m,
                weight_E=weight_E,
                k=k,
                n=n,
                num_topk=num_topk,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
                core_token_counts=(m,),
                route_num_experts=0,
                quant_mode="nvfp4",
                activation="silu",
                apply_router_weight_on_input=False,
                source_format="modelopt_nvfp4",
                w13_layout="w13",
            )
        )
        _B12X_PLAN_CACHE[cache_key] = plan

    specs = plan.scratch_specs()
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=specs[idx].device)
        for idx, (shape, dtype) in enumerate(plan.shapes_and_dtypes())
    )

    binding = plan.bind(scratch=scratch, **kwargs)
    return b12x_moe_fp4(binding=binding)


def fused_experts_none_to_flashinfer_cutedsl_sm120_fp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: CuteDslSm120Fp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)

    topk_ids = topk_output.topk_ids
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    output = _run_sm120_b12x(
        hidden_states=dispatch_output.hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_output.topk_weights,
        quant_info=quant_info,
        runner_config=runner_config,
        output=None,
    )
    return StandardCombineInput(hidden_states=output)


def fused_experts_flashinfer_to_flashinfer_cutedsl_sm120_fp4(
    dispatch_output: FlashinferDispatchOutput,
    quant_info: CuteDslSm120Fp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> FlashinferCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
    )

    if dispatch_output.hidden_states_scale is not None:
        raise NotImplementedError(
            "flashinfer_cutedsl_sm120 currently expects BF16 dispatcher output "
            "(hidden_states_scale must be None)."
        )

    topk_output = dispatch_output.topk_output
    assert TopKOutputChecker.format_is_standard(topk_output)

    topk_ids = topk_output.topk_ids
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    output = _run_sm120_b12x(
        hidden_states=dispatch_output.hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_output.topk_weights,
        quant_info=quant_info,
        runner_config=runner_config,
        output=dispatch_output.moe_output,
    )
    return FlashinferCombineInput(hidden_states=output)


def fused_experts_deepep_to_flashinfer_cutedsl_sm120_fp4(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: CuteDslSm120Fp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> DeepEPLLCombineInput:
    raise NotImplementedError(
        "flashinfer_cutedsl_sm120 does not support deepep path yet. "
        "Use a2a=none/flashinfer."
    )


def _maybe_register_sm120_fused_funcs() -> None:
    available_runner_backends = {backend.value for backend in MoeRunnerBackend}
    if _RUNNER_BACKEND_NAME not in available_runner_backends:
        return

    FusedOpPool.register_fused_func(
        "none",
        _RUNNER_BACKEND_NAME,
        fused_experts_none_to_flashinfer_cutedsl_sm120_fp4,
    )
    FusedOpPool.register_fused_func(
        "flashinfer",
        _RUNNER_BACKEND_NAME,
        fused_experts_flashinfer_to_flashinfer_cutedsl_sm120_fp4,
    )
    FusedOpPool.register_fused_func(
        "deepep",
        _RUNNER_BACKEND_NAME,
        fused_experts_deepep_to_flashinfer_cutedsl_sm120_fp4,
    )


_maybe_register_sm120_fused_funcs()
