"""FlashInfer CUTLASS MoE fused funcs.

This module owns the FlashInfer ``cutlass_fused_moe`` calls used by the
unquantized, ModelOpt FP8, ModelOpt NVFP4, and MXFP4 MoE paths.
Quantization methods prepare a small quant_info payload and route through
``MoeRunner``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import KernelBackend
from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    register_fused_func,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    _is_stream_capturing,
)
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import next_power_of_2

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.kernels.ops.moe.nvfp4_moe_sm120 import Nvfp4MoeWorkspace
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
        FlashinferDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class FlashInferCutlassMoeQuantInfo(MoeQuantInfo):
    """Payload for FlashInfer CUTLASS fused MoE.

    ``quant_type`` selects the input/weight conventions:
      - ``"bf16"``: unquantized weights, BF16/FP16 input, no quant scales.
      - ``"fp8"``: FP8 weights, FP8-quantized input, per-tensor scales.
      - ``"fp4"``: NVFP4 packed weights and optional NVFP4 packed input.
    """

    quant_type: str
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    quant_scales: Optional[list[torch.Tensor]] = None
    output_dtype: Optional[torch.dtype] = None
    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    moe_ep_size: int = 1
    moe_ep_rank: int = 0
    apply_routed_scaling_factor: bool = True
    g1_alpha_up: Optional[torch.Tensor] = None
    smallm_workspace: Optional[Nvfp4MoeWorkspace] = None
    smallm_global_routed_experts: Optional[int] = None
    smallm_local_routed_experts: Optional[int] = None
    smallm_local_expert_start: Optional[int] = None


@dataclass
class FlashInferCutlassMxfp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload for CUTLASS MXFP4 MoE.

    SM90 consumes W4A16-interleaved weights and scales. SM120 consumes packed
    MXFP4 weights and block-interleaved scales with MXFP8 activations.
    """

    # SM90 weights are interleaved; SM120 weights remain checkpoint-packed.
    w13_weight: torch.Tensor  # [E, 2*N, K/2]
    w2_weight: torch.Tensor  # [E, K, N/2]

    # E8M0 block scales in the layout selected by the quantization method.
    w13_weight_scale: torch.Tensor  # [E, 2*N, K/32]
    w2_weight_scale: torch.Tensor  # [E, K, N/32]

    # A non-None global scale selects the SM120 MXFP8 activation path.
    mxfp4_weight_global_scale: Optional[torch.Tensor] = None

    # Per-expert bias. GPT-OSS has both; DSv4 leaves both None.
    w13_bias: Optional[torch.Tensor] = None  # bf16 [E, 2*N]
    w2_bias: Optional[torch.Tensor] = None  # bf16 [E, K]

    # Optional per-expert SwiGLU overrides, fp32 [E].
    swiglu_alpha: Optional[torch.Tensor] = None
    swiglu_beta: Optional[torch.Tensor] = None
    swiglu_limit: Optional[torch.Tensor] = None

    # TP/EP topology (forwarded to the FlashInfer kernel)
    moe_tp_size: int = 1
    moe_tp_rank: int = 0
    moe_ep_size: int = 1
    moe_ep_rank: int = 0

    # GPT-OSS pads its input hidden dim up to the (pre-padded) loaded weight
    # width and trims the output back. DSv4 leaves this as ``None`` (no pad).
    padded_hidden: Optional[int] = None


def _flashinfer_cutlass_fused_moe():
    if not is_flashinfer_available():
        raise RuntimeError(
            "flashinfer_cutlass MoE runner backend requires flashinfer to be installed."
        )
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    return cutlass_fused_moe, ActivationType


def _activation_type(runner_config: MoeRunnerConfig):
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import get_activation_type

    _, ActivationType = _flashinfer_cutlass_fused_moe()
    activation = ActivationType(
        get_activation_type(
            runner_config.activation,
            is_gated=runner_config.is_gated,
        )
    )
    supported = {
        ActivationType.Swiglu,
        ActivationType.Geglu,
        ActivationType.Relu2,
        ActivationType.Identity,
    }
    assert activation in supported, (
        f"Activation {runner_config.activation!r} "
        f"(is_gated={runner_config.is_gated}) maps to {activation.name}, "
        "which is not supported by flashinfer cutlass moe."
    )
    return activation


def _maybe_apply_routed_scaling_factor(
    output: torch.Tensor,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> torch.Tensor:
    if (
        quant_info.apply_routed_scaling_factor
        and runner_config.routed_scaling_factor is not None
    ):
        output.mul_(runner_config.routed_scaling_factor)
    return output


def _prepare_input(
    dispatch_output,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.dtype, int]:
    x = dispatch_output.hidden_states
    x_sf = dispatch_output.hidden_states_scale

    if quant_info.quant_type == "fp8":
        assert quant_info.quant_scales is not None and len(quant_info.quant_scales) == 4
        x, _ = scaled_fp8_quant(x, quant_info.quant_scales[3])
        x_sf = None
        output_dtype = quant_info.output_dtype or dispatch_output.hidden_states.dtype
        output_col = dispatch_output.hidden_states.shape[1]
    elif quant_info.quant_type == "fp4":
        output_dtype = quant_info.output_dtype or torch.bfloat16
        output_col = x.shape[1]
        if x_sf is not None and runner_config.is_gated:
            output_col *= 2
    else:
        assert quant_info.quant_type == "bf16"
        output_dtype = quant_info.output_dtype or x.dtype
        output_col = x.shape[1]

    return x, x_sf, output_dtype, output_col


_logged_smallm_decisions: set[str] = set()


def _log_smallm_decision(message: str) -> None:
    if message in _logged_smallm_decisions:
        return
    _logged_smallm_decisions.add(message)
    logger.info("SM120 NVFP4 small-row MoE: %s", message)


def _smallm_ineligibility_reason(
    *,
    quant_info: FlashInferCutlassMoeQuantInfo,
    workspace: Optional[Nvfp4MoeWorkspace],
    x: torch.Tensor,
    x_sf: Optional[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: Optional[list[torch.Tensor]],
    runner_config: MoeRunnerConfig,
    output_supplied: bool,
    enable_alltoall: bool,
    capturing: Optional[bool] = None,
) -> Optional[str]:
    if quant_info.quant_type != "fp4":
        return "quantization is not NVFP4"
    if workspace is None:
        return "workspace is disabled or the layer shape is unsupported"
    if quant_info.g1_alpha_up is None:
        return "the up-projection alpha is unavailable"
    if any(
        value is None
        for value in (
            quant_info.smallm_global_routed_experts,
            quant_info.smallm_local_routed_experts,
            quant_info.smallm_local_expert_start,
        )
    ):
        return "expert topology is unavailable"
    if output_supplied or enable_alltoall:
        return "the all-to-all output contract requires CUTLASS"
    if x_sf is not None:
        return "the input is already quantized"
    if capturing is None:
        capturing = torch.cuda.is_available() and _is_stream_capturing(
            torch.cuda.current_stream(x.device)
        )
    if capturing and workspace.graph_capture_supported is not True:
        return "cooperative graph capture is unavailable"
    if not 0 < x.shape[0] <= workspace.max_tokens:
        return "token count is outside the small-row range"
    if (
        x.dtype != torch.bfloat16
        or topk_ids.dtype != torch.int32
        or topk_weights.dtype != torch.float32
    ):
        return "input or routing dtypes are unsupported"
    if not (
        x.is_contiguous() and topk_ids.is_contiguous() and topk_weights.is_contiguous()
    ):
        return "input or routing tensors are not contiguous"
    if (
        x.dim() != 2
        or topk_ids.dim() != 2
        or topk_weights.shape != topk_ids.shape
        or topk_ids.shape[0] != x.shape[0]
        or x.shape[1] != workspace.hidden_size
        or topk_ids.shape[1] != workspace.top_k
    ):
        return "input or routing shapes do not match the workspace"
    if not runner_config.is_gated or runner_config.activation not in (
        "silu",
        "swiglu",
    ):
        return "the activation is unsupported"
    if any(
        value is not None
        for value in (
            runner_config.gemm1_alpha,
            runner_config.gemm1_beta,
            runner_config.gemm1_clamp_limit,
            runner_config.swiglu_limit,
        )
    ):
        return "the activation modifiers are unsupported"
    if output_dtype != torch.bfloat16:
        return "the output dtype is unsupported"
    if quant_scales is None or len(quant_scales) != 6:
        return "the NVFP4 scale set is incomplete"
    if quant_info.moe_tp_size < 1 or quant_info.moe_ep_size < 1:
        return "the MoE parallel topology is invalid"

    global_routed = quant_info.smallm_global_routed_experts
    local_routed = quant_info.smallm_local_routed_experts
    local_start = quant_info.smallm_local_expert_start
    if (
        global_routed is None
        or local_routed is None
        or local_start is None
        or global_routed <= 0
        or local_routed <= 0
        or local_start < 0
        or not 0 <= quant_info.moe_ep_rank < quant_info.moe_ep_size
        or global_routed != local_routed * quant_info.moe_ep_size
        or local_start != quant_info.moe_ep_rank * local_routed
        or local_routed > quant_info.w13_weight.shape[0]
    ):
        return "the expert topology is not a uniform contiguous EP shard"
    return None


def _run_flashinfer_cutlass(
    *,
    dispatch_output,
    quant_info: FlashInferCutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    output: Optional[torch.Tensor] = None,
    enable_alltoall: bool = False,
) -> torch.Tensor:
    flashinfer_cutlass_fused_moe, _ = _flashinfer_cutlass_fused_moe()

    topk_output = dispatch_output.topk_output
    topk_weights = topk_output.topk_weights
    topk_ids = topk_output.topk_ids
    x, x_sf, output_dtype, output_col = _prepare_input(
        dispatch_output, quant_info, runner_config
    )

    workspace = quant_info.smallm_workspace
    quant_scales = quant_info.quant_scales
    capturing = False
    if workspace is not None and quant_info.quant_type == "fp4":
        capturing = torch.cuda.is_available() and _is_stream_capturing(
            torch.cuda.current_stream(x.device)
        )
    smallm_reason = None
    if workspace is not None:
        smallm_reason = _smallm_ineligibility_reason(
            quant_info=quant_info,
            workspace=workspace,
            x=x,
            x_sf=x_sf,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output_dtype=output_dtype,
            quant_scales=quant_scales,
            runner_config=runner_config,
            output_supplied=output is not None,
            enable_alltoall=enable_alltoall,
            capturing=capturing,
        )
    use_smallm = workspace is not None and smallm_reason is None
    if not use_smallm and workspace is not None and quant_info.quant_type == "fp4":
        _log_smallm_decision(f"using CUTLASS because {smallm_reason}")

    if output is None:
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            output = torch.empty(
                x.shape[0],
                output_col,
                dtype=output_dtype,
                device=x.device,
            )

    if use_smallm:
        launch_error = None
        try:
            launched = get_kernel("moe.nvfp4_fused_experts", KernelBackend.JIT)(
                x=x,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                w13_weight=quant_info.w13_weight,
                w2_weight=quant_info.w2_weight,
                w13_scale=quant_scales[1],
                w2_scale=quant_scales[4],
                input_scale_1=quant_scales[0],
                input_scale_2=quant_scales[3],
                g1_alpha=quant_scales[2],
                g1_alpha_up=quant_info.g1_alpha_up,
                g2_alpha=quant_scales[5],
                global_routed_experts=quant_info.smallm_global_routed_experts,
                local_routed_experts=quant_info.smallm_local_routed_experts,
                local_expert_start=quant_info.smallm_local_expert_start,
                output=output,
                workspace=workspace,
            )
        except Exception as error:
            if capturing:
                raise RuntimeError(
                    "SM120 NVFP4 cooperative launch failed during CUDA graph capture"
                ) from error
            launched = False
            launch_error = error
            error_text = str(error).strip()
            detail = error_text.splitlines()[0] if error_text else "no detail"
            _log_smallm_decision(
                "using CUTLASS because the SM120 JIT launch failed: "
                f"{type(error).__name__}: {detail}"
            )
        if launched:
            _log_smallm_decision("selected")
            return output
        if launch_error is None:
            if capturing:
                raise RuntimeError(
                    "SM120 NVFP4 cooperative launch failed during CUDA graph capture"
                )
            _log_smallm_decision(
                "using CUTLASS because the SM120 launch is unavailable"
            )

    if output is None:
        raise RuntimeError("FlashInfer CUTLASS MoE output allocation failed")

    w13_weight = quant_info.w13_weight
    w2_weight = quant_info.w2_weight
    if quant_info.quant_type == "fp4":
        w13_weight = w13_weight.view(torch.long)
        w2_weight = w2_weight.view(torch.long)
        if quant_scales is None or len(quant_scales) != 6:
            raise ValueError("NVFP4 CUTLASS MoE requires six quantization scales")
        quant_scales = [
            quant_scales[0],
            quant_scales[1].view(torch.int32),
            quant_scales[2],
            quant_scales[3],
            quant_scales[4].view(torch.int32),
            quant_scales[5],
        ]

    output = flashinfer_cutlass_fused_moe(
        output=output,
        input=x,
        token_selected_experts=topk_ids.to(torch.int),
        token_final_scales=topk_weights,
        fc1_expert_weights=w13_weight,
        fc2_expert_weights=w2_weight,
        output_dtype=output_dtype,
        input_sf=x_sf,
        quant_scales=quant_scales,
        ep_size=quant_info.moe_ep_size,
        ep_rank=quant_info.moe_ep_rank,
        tp_size=quant_info.moe_tp_size,
        tp_rank=quant_info.moe_tp_rank,
        tune_max_num_tokens=next_power_of_2(x.shape[0]),
        activation_type=_activation_type(runner_config),
        enable_alltoall=enable_alltoall,
        use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
    )[0]

    if quant_info.quant_type in ("bf16", "fp8"):
        _maybe_apply_routed_scaling_factor(output, quant_info, runner_config)
    return output


@register_fused_func("none", "flashinfer_cutlass")
def fused_experts_none_to_flashinfer_cutlass(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert isinstance(
        quant_info, FlashInferCutlassMoeQuantInfo
    ), f"Unexpected quant_info type for flashinfer_cutlass: {type(quant_info)}"
    assert (
        not runner_config.apply_router_weight_on_input
    ), "apply_router_weight_on_input is not supported for FlashInfer CUTLASS"

    output = _run_flashinfer_cutlass(
        dispatch_output=dispatch_output,
        quant_info=quant_info,
        runner_config=runner_config,
    )
    return StandardCombineInput(hidden_states=output)


@register_fused_func("flashinfer", "flashinfer_cutlass")
def fused_experts_flashinfer_to_flashinfer_cutlass(
    dispatch_output: FlashinferDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> FlashinferCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
    )

    assert isinstance(
        quant_info, FlashInferCutlassMoeQuantInfo
    ), f"Unexpected quant_info type for flashinfer_cutlass: {type(quant_info)}"
    assert (
        not runner_config.apply_router_weight_on_input
    ), "apply_router_weight_on_input is not supported for FlashInfer CUTLASS"

    output = _run_flashinfer_cutlass(
        dispatch_output=dispatch_output,
        quant_info=quant_info,
        runner_config=runner_config,
        output=dispatch_output.moe_output,
        enable_alltoall=True,
    )
    return FlashinferCombineInput(hidden_states=output)


@register_fused_func("none", "flashinfer_mxfp4")
def fused_experts_none_to_flashinfer_mxfp4(
    dispatch_output: StandardDispatchOutput,
    quant_info: MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Run the FlashInfer CUTLASS MXFP4 fused experts."""
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    assert isinstance(
        quant_info, FlashInferCutlassMxfp4MoeQuantInfo
    ), f"Unexpected quant_info type for flashinfer_mxfp4: {type(quant_info)}"

    flashinfer_cutlass_fused_moe, ActivationType = _flashinfer_cutlass_fused_moe()

    x = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output

    # Under ``--moe-runner-backend flashinfer_mxfp4`` topk may be in bypassed
    # form (the SM100 trtllm-gen path does routing internally). The CUTLASS
    # SM90 path needs explicit topk_ids / topk_weights; materialize here.
    if TopKOutputChecker.format_is_bypassed(topk_output):
        topk_output = topk_output.to_standard()
    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights

    # GPT-OSS: pad input hidden dim up to the loaded weight width. DSv4
    # leaves padded_hidden as None (or equal to origin_hidden), no pad.
    origin_hidden = x.shape[-1]
    padded_hidden = quant_info.padded_hidden
    do_pad = padded_hidden is not None and padded_hidden != origin_hidden
    if do_pad:
        x = torch.nn.functional.pad(
            x,
            (0, padded_hidden - origin_hidden),
            mode="constant",
            value=0.0,
        )

    weight_global_scale = quant_info.mxfp4_weight_global_scale
    use_mxfp8_act_scaling = weight_global_scale is not None
    input_sf = None
    fc1_expert_weights = quant_info.w13_weight
    fc2_expert_weights = quant_info.w2_weight
    if weight_global_scale is not None:
        from flashinfer import mxfp8_quantize

        x, input_sf = mxfp8_quantize(
            x,
            is_sf_swizzled_layout=True,
            alignment=32,
        )
        fc1_expert_weights = fc1_expert_weights.view(torch.int64)
        fc2_expert_weights = fc2_expert_weights.view(torch.int64)
        quant_scales = [
            quant_info.w13_weight_scale.view(torch.int32),
            weight_global_scale,
            quant_info.w2_weight_scale.view(torch.int32),
            weight_global_scale,
        ]
    else:
        quant_scales = [
            quant_info.w13_weight_scale.view(torch.int32),
            quant_info.w2_weight_scale.view(torch.int32),
        ]

    out_hidden = padded_hidden if do_pad else origin_hidden
    output_dtype = torch.bfloat16
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        out = torch.empty(x.shape[0], out_hidden, dtype=output_dtype, device=x.device)

    flashinfer_cutlass_fused_moe(
        input=x,
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_weights,
        fc1_expert_weights=fc1_expert_weights,
        fc2_expert_weights=fc2_expert_weights,
        output_dtype=output_dtype,
        quant_scales=quant_scales,
        input_sf=input_sf,
        fc1_expert_biases=quant_info.w13_bias,
        fc2_expert_biases=quant_info.w2_bias,
        swiglu_alpha=quant_info.swiglu_alpha,
        swiglu_beta=quant_info.swiglu_beta,
        swiglu_limit=quant_info.swiglu_limit,
        tp_size=quant_info.moe_tp_size,
        tp_rank=quant_info.moe_tp_rank,
        ep_size=quant_info.moe_ep_size,
        ep_rank=quant_info.moe_ep_rank,
        use_w4_group_scaling=not use_mxfp8_act_scaling,
        use_mxfp8_act_scaling=use_mxfp8_act_scaling,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=next_power_of_2(x.shape[0]),
        output=out,
        use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
    )

    if do_pad:
        out = out[:, :origin_hidden].contiguous()

    return StandardCombineInput(hidden_states=out)
