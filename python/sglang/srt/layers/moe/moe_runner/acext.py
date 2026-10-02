from __future__ import annotations

import logging
from importlib.metadata import version as get_package_version
from typing import TYPE_CHECKING, Optional

from packaging.version import Version

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import (
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    fused_experts_none_to_triton,
)
from sglang.srt.utils import is_ppu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

# acext runtime version is non-monotonic (acext==1.0.0 but acext.get_version()==1050100);
# use pip package version (>=1.1.0) instead to ensure act_limit support.
_ACEXT_SILU_AFTER_CLAMP_MIN_VERSION = Version("1.1.0")
_ACEXT_SILU_AFTER_CLAMP_SUPPORTED: Optional[bool] = None


def _acext_supports_silu_after_clamp() -> bool:
    global _ACEXT_SILU_AFTER_CLAMP_SUPPORTED
    if _ACEXT_SILU_AFTER_CLAMP_SUPPORTED is None:
        try:
            installed = Version(get_package_version("acext"))
            _ACEXT_SILU_AFTER_CLAMP_SUPPORTED = (
                installed >= _ACEXT_SILU_AFTER_CLAMP_MIN_VERSION
            )
        except Exception:
            # Metadata missing or unparsable: stay conservative to avoid
            # passing unsupported kwargs to older runtimes.
            _ACEXT_SILU_AFTER_CLAMP_SUPPORTED = False
    return _ACEXT_SILU_AFTER_CLAMP_SUPPORTED


@register_fused_func("none", "acext")
def fused_experts_none_to_acext(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    assert is_ppu(), (
        f"Only PPU support acext MoE backend, use other MoE backend on current platform please!"
    )

    from acext import (
        fusedmoe_wrapper,
        get_enum_from_booleans,
        get_fusedmoe_status_wrapper,
    )

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states = dispatch_output.hidden_states
    w1 = quant_info.w13_weight
    w2 = quant_info.w2_weight
    topk_output = dispatch_output.topk_output
    moe_runner_config = runner_config
    use_fp8_w8a8 = quant_info.use_fp8_w8a8
    use_int8_w8a8 = quant_info.use_int8_w8a8
    use_int8_w8a16 = quant_info.use_int8_w8a16
    use_int4_w4a16 = quant_info.use_int4_w4a16
    w1_scale = quant_info.w13_scale
    w2_scale = quant_info.w2_scale
    a1_scale = quant_info.a13_scale
    a2_scale = quant_info.a2_scale
    topk_weights, topk_ids, _ = topk_output
    routed_scaling_factor = moe_runner_config.routed_scaling_factor

    acext_cuda_debug = envs.SGLANG_SAIL_ACEXT_MOE_DEBUG.get()
    if acext_cuda_debug:
        logger.info(
            f"[sglang][acext][DEBUG] CASE SHAPE: M_{hidden_states.shape[0]}_E{w1.shape[0]}_H{w1.shape[2]}_In{w1.shape[1]}_topk{topk_ids.shape[1]}"
        )

    Q_type = get_enum_from_booleans(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        use_fp8_w8a16=False,
    )

    use_acext_impl = 1

    def pad_to_multiple_of_16(value):
        padding = (16 - value % 16) % 16
        return value + padding

    use_acext_impl = get_fusedmoe_status_wrapper(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        None,
        pad_to_multiple_of_16(int(hidden_states.shape[0] * topk_ids.shape[1])),
        w1_scale,
        w2_scale,
        None,
        None,
        a1_scale,
        a2_scale,
        0,
        1,
        Q_type,
    )
    if use_acext_impl != 0:
        logger.info(
            f"Get acext moe wrapper failed, fallback to triton fused_moe instead"
        )
        return fused_experts_none_to_triton(dispatch_output, quant_info, runner_config)

    # Pick the SwiGLU clamp limit from two mutually-exclusive fields, which
    # also encodes the clamp order:
    #   - gemm1_clamp_limit (Step3.5): silu then clamp -> is_silu_after_clamp=False
    #   - swiglu_limit (DeepSeek V4): clamp then silu -> is_silu_after_clamp=True
    gemm1_clamp_limit = moe_runner_config.gemm1_clamp_limit
    swiglu_limit = moe_runner_config.swiglu_limit
    is_silu_after_clamp = True
    if gemm1_clamp_limit is not None:
        act_limit = gemm1_clamp_limit
        is_silu_after_clamp = False  # Step3.5: silu then clamp
    else:
        act_limit = swiglu_limit

    # act_limit + is_silu_after_clamp must only be passed when the acext whl
    # supports them (>= 1.1.0), so gate on the pip metadata version to stay
    # compatible with older acext runtimes.
    fused_kwargs = dict(
        routed_scaling_factor=routed_scaling_factor,
    )
    if _acext_supports_silu_after_clamp():
        fused_kwargs["act_limit"] = act_limit
        fused_kwargs["is_silu_after_clamp"] = is_silu_after_clamp
    elif act_limit is not None:
        logger.warning(
            f"Current acext version does not support SwiGLU clamp (act_limit={act_limit}). "
            f"Clamping is skipped. Please upgrade acext to v1.1.0 or later "
            f"to avoid potential precision issues."
        )

    output = (
        hidden_states
        if runner_config.inplace
        else hidden_states.new_empty(hidden_states.shape)
    )

    fusedmoe_wrapper(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        output,
        pad_to_multiple_of_16(int(hidden_states.shape[0] * topk_ids.shape[1])),
        w1_scale,
        w2_scale,
        None,
        None,
        a1_scale,
        a2_scale,
        0,
        1,
        Q_type,
        **fused_kwargs,
    )

    return StandardCombineInput(hidden_states=output)
