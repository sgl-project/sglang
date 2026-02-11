from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils.common import is_flashinfer_available, is_sm100_supported

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import DownGemmOverlapArgs
    from sglang.srt.layers.moe.token_dispatcher import DeepEPLLDispatchOutput
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput


@dataclass
class FlashInferCuteDslNvFp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by FlashInfer CuteDSL NVFP4 MoE kernels."""

    input_global_scale: torch.Tensor | None
    w13_weight: torch.Tensor
    w13_blockscale_swizzled: torch.Tensor
    g1_alphas: torch.Tensor
    w2_weight: torch.Tensor
    w2_input_scale_quant: torch.Tensor
    w2_blockscale_swizzled: torch.Tensor
    g2_alphas: torch.Tensor


@dataclass
class FlashInferCuteDslRunnerInput(RunnerInput):
    hidden_states: tuple[torch.Tensor, torch.Tensor | None]
    masked_m: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.FLASHINFER_CUTEDSL


@dataclass
class FlashInferCuteDslRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.FLASHINFER_CUTEDSL


class FlashInferCuteDslRunnerCore(MoeRunnerCore):
    """Runner core for FlashInfer CuteDSL masked MoE kernels (NVFP4 path)."""

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.FLASHINFER_CUTEDSL

    def run(
        self,
        runner_input: RunnerInput,
        quant_info: MoeQuantInfo,
        running_state: dict,
    ) -> RunnerOutput:
        if not is_flashinfer_available() or not is_sm100_supported():
            raise RuntimeError(
                "flashinfer_cutedsl MoE requires FlashInfer and SM100 support."
            )

        if not isinstance(runner_input, FlashInferCuteDslRunnerInput):
            raise TypeError(
                f"Unexpected runner_input type for flashinfer_cutedsl: {type(runner_input)}"
            )
        if not isinstance(quant_info, FlashInferCuteDslNvFp4MoeQuantInfo):
            raise TypeError(
                f"Unexpected quant_info type for flashinfer_cutedsl: {type(quant_info)}"
            )

        from sglang.srt.layers.moe.flashinfer_cutedsl_moe import (
            flashinfer_cutedsl_moe_masked,
        )

        down_gemm_overlap_args: DownGemmOverlapArgs | None = running_state.get(
            "down_gemm_overlap_args"
        )

        out = flashinfer_cutedsl_moe_masked(
            hidden_states=runner_input.hidden_states,
            input_global_scale=quant_info.input_global_scale,
            w1=quant_info.w13_weight,
            w1_blockscale=quant_info.w13_blockscale_swizzled,
            w1_alpha=quant_info.g1_alphas,
            w2=quant_info.w2_weight,
            a2_global_scale=quant_info.w2_input_scale_quant,
            w2_blockscale=quant_info.w2_blockscale_swizzled,
            w2_alpha=quant_info.g2_alphas,
            masked_m=runner_input.masked_m,
            **(
                dict(
                    down_sm_count=down_gemm_overlap_args.num_sms,
                    down_signals=down_gemm_overlap_args.signal,
                    down_start_event=down_gemm_overlap_args.start_event,
                )
                if down_gemm_overlap_args is not None
                else {}
            ),
        )
        return FlashInferCuteDslRunnerOutput(hidden_states=out)


@register_pre_permute("deepep_ll", "flashinfer_cutedsl")
def pre_permute_deepep_ll_to_flashinfer_cutedsl(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: FlashInferCuteDslNvFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> FlashInferCuteDslRunnerInput:
    hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, _ = (
        dispatch_output
    )

    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    return FlashInferCuteDslRunnerInput(
        hidden_states=(hidden_states, hidden_states_scale),
        masked_m=masked_m,
    )


@register_post_permute("flashinfer_cutedsl", "deepep_ll")
def post_permute_flashinfer_cutedsl_to_deepep_ll(
    runner_output: FlashInferCuteDslRunnerOutput,
    quant_info: FlashInferCuteDslNvFp4MoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> "DeepEPLLCombineInput":
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )
