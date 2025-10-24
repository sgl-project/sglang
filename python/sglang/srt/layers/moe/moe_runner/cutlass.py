from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from sglang.srt.layers.moe.cutlass_moe_params import CutlassMoEParams, CutlassMoEType
from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported
from sglang.srt.utils import is_cuda, is_sm90_supported

if is_cuda():
    from sgl_kernel import (
        apply_shuffle_mul_sum,
        cutlass_fp4_group_mm,
        fp8_blockwise_scaled_grouped_mm,
        prepare_moe_input,
        scaled_fp4_experts_quant,
        shuffle_rows,
        silu_and_mul,
    )

    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )


@dataclass
class CutlassRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    a_map: torch.Tensor
    c_map: torch.Tensor
    rep_primary: torch.Tensor
    rep_aux: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.CUTLASS


@dataclass
class CutlassRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.CUTLASS


@dataclass
class CutlassMoeQuantInfo(MoeQuantInfo):
    moe_type: CutlassMoEType
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    expert_offsets: torch.Tensor
    problem_sizes1: torch.Tensor
    problem_sizes2: torch.Tensor
    ab_strides_13: Optional[torch.Tensor] = None
    ab_strides_2: Optional[torch.Tensor] = None
    c_strides_13: Optional[torch.Tensor] = None
    c_strides_2: Optional[torch.Tensor] = None
    workspace: Optional[torch.Tensor] = None
    a_ptrs: Optional[torch.Tensor] = None
    b_ptrs: Optional[torch.Tensor] = None
    out_ptrs: Optional[torch.Tensor] = None
    a_scales_ptrs: Optional[torch.Tensor] = None
    b_scales_ptrs: Optional[torch.Tensor] = None
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    w1_blockscale: Optional[torch.Tensor] = None
    w2_blockscale: Optional[torch.Tensor] = None
    w1_alpha: Optional[torch.Tensor] = None
    w2_alpha: Optional[torch.Tensor] = None
    a1_gscale: Optional[torch.Tensor] = None
    a2_gscale: Optional[torch.Tensor] = None
    params: Optional[CutlassMoEParams] = None


class CutlassRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        if not is_cuda():
            raise RuntimeError("Cutlass runner requires CUDA support.")
        if not is_sm90_supported():
            raise RuntimeError("Cutlass runner requires NVIDIA SM90 or newer GPUs.")
        if not cutlass_fp8_supported():
            raise RuntimeError("CUTLASS FP8 kernels are not available on this system.")
        if self.config.activation not in ("silu", None):
            raise ValueError("Cutlass runner currently supports SiLU activation only.")

    def run(
        self,
        runner_input: CutlassRunnerInput,
        quant_info: CutlassMoeQuantInfo,
        running_state: Dict[str, torch.Tensor],
    ) -> CutlassRunnerOutput:
        moe_type = quant_info.moe_type
        out_dtype = runner_input.hidden_states.dtype
        device = runner_input.hidden_states.device

        rep_primary = runner_input.rep_primary
        rep_aux = runner_input.rep_aux
        num_tokens = runner_input.hidden_states.shape[0]
        hidden_size = runner_input.hidden_states.shape[1]
        topk = runner_input.topk_ids.shape[1]

        if moe_type == CutlassMoEType.BlockscaledFP8:

            intermediate_size = quant_info.w2_weight.size(1)
            num_experts = quant_info.w13_weight.size(0)

            c1 = torch.empty(
                (num_tokens * topk, intermediate_size * 2),
                device=device,
                dtype=out_dtype,
            )
            a_sf_layout = torch.empty(
                (num_experts, 5), device=device, dtype=torch.int32
            )
            w_sf_layout = torch.empty(
                (num_experts, 5), device=device, dtype=torch.int32
            )

            fp8_blockwise_scaled_grouped_mm(
                c1,
                quant_info.a_ptrs,
                quant_info.b_ptrs,
                quant_info.out_ptrs,
                quant_info.a_scales_ptrs,
                quant_info.b_scales_ptrs,
                rep_primary,
                quant_info.w13_weight,
                rep_aux,
                quant_info.w13_scale,
                quant_info.ab_strides_13,
                quant_info.ab_strides_13,
                quant_info.c_strides_13,
                a_sf_layout,
                w_sf_layout,
                quant_info.problem_sizes1,
                quant_info.expert_offsets[:-1],
                quant_info.workspace,
            )

            intermediate = torch.empty(
                (num_tokens * topk, intermediate_size), device=device, dtype=out_dtype
            )
            silu_and_mul(c1, intermediate)

            intermediate_q, a2_scale = sglang_per_token_group_quant_fp8(
                intermediate, 128
            )
            down_output = torch.empty(
                (num_tokens * topk, hidden_size), device=device, dtype=out_dtype
            )

            fp8_blockwise_scaled_grouped_mm(
                down_output,
                quant_info.a_ptrs,
                quant_info.b_ptrs,
                quant_info.out_ptrs,
                quant_info.a_scales_ptrs,
                quant_info.b_scales_ptrs,
                intermediate_q,
                quant_info.w2_weight,
                a2_scale,
                quant_info.w2_scale,
                quant_info.ab_strides_2,
                quant_info.ab_strides_2,
                quant_info.c_strides_2,
                a_sf_layout,
                w_sf_layout,
                quant_info.problem_sizes2,
                quant_info.expert_offsets[:-1],
                quant_info.workspace,
            )

        elif moe_type == CutlassMoEType.BlockscaledFP4:
            params = quant_info.params

            c1 = cutlass_fp4_group_mm(
                rep_primary,
                quant_info.w13_weight,
                rep_aux,
                quant_info.w1_blockscale,
                quant_info.w1_alpha,
                out_dtype,
                device,
                params.to_gemm1_args(),
            )

            intermediate = torch.empty(
                (num_tokens * topk, hidden_size // 2), device=device, dtype=out_dtype
            )
            silu_and_mul(c1, intermediate)

            intermediate_fp4, intermediate_blockscale = scaled_fp4_experts_quant(
                intermediate,
                quant_info.a2_gscale,
                params.expert_offsets,
                params.blockscale_offsets,
                topk,
            )

            down_output = cutlass_fp4_group_mm(
                intermediate_fp4,
                quant_info.w2_weight,
                intermediate_blockscale,
                quant_info.w2_blockscale,
                quant_info.w2_alpha,
                out_dtype,
                device,
                params.to_gemm2_args(),
            )
        # Optional no-combine path: return (M, topk, K) without reduction
        if self.config.no_combine:
            reordered = shuffle_rows(
                down_output,
                runner_input.c_map,
                (num_tokens * topk, hidden_size),
            ).view(num_tokens, topk, hidden_size)
            if not self.config.apply_router_weight_on_input:
                reordered.mul_(
                    runner_input.topk_weights.view(num_tokens, topk, 1).to(
                        reordered.dtype
                    )
                )
            return CutlassRunnerOutput(hidden_states=reordered)
        combined = self._combine(down_output, runner_input, quant_info)
        return CutlassRunnerOutput(hidden_states=combined)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.CUTLASS

    def _combine(
        self,
        down_output: torch.Tensor,
        runner_input: CutlassRunnerInput,
        quant_info: CutlassMoeQuantInfo,
    ) -> torch.Tensor:
        if quant_info.moe_type == CutlassMoEType.BlockscaledFP8:
            m, k = runner_input.hidden_states.shape
            result = torch.empty(
                (m, k),
                device=runner_input.hidden_states.device,
                dtype=runner_input.hidden_states.dtype,
            )
            apply_shuffle_mul_sum(
                down_output,
                result,
                runner_input.c_map,
                runner_input.topk_weights.to(result.dtype),
            )
            return result
        if quant_info.moe_type == CutlassMoEType.BlockscaledFP4:
            num_tokens = runner_input.hidden_states.shape[0]
            topk = runner_input.topk_ids.shape[1]
            hidden_size = (
                quant_info.params.hidden_size
                if quant_info.params
                else down_output.shape[1]
            )
            reordered = shuffle_rows(
                down_output,
                runner_input.c_map,
                (num_tokens * topk, hidden_size),
            )
            reordered = reordered.view(num_tokens, topk, hidden_size)
            if not self.config.apply_router_weight_on_input:
                reordered.mul_(
                    runner_input.topk_weights.view(num_tokens, topk, 1).to(
                        reordered.dtype
                    )
                )
            combined = reordered.sum(dim=1).to(runner_input.hidden_states.dtype)
            return combined

        raise NotImplementedError(
            f"Unsupported CUTLASS MoE type: {quant_info.moe_type}"
        )


@register_pre_permute("standard", "cutlass")
def pre_permute_standard_to_cutlass(
    dispatch_output: StandardDispatchOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> CutlassRunnerInput:
    hidden_states, topk_output = dispatch_output
    topk_weights, topk_ids, _ = topk_output

    device = hidden_states.device
    a_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)
    c_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)

    if quant_info.moe_type == CutlassMoEType.BlockscaledFP8:
        num_experts = quant_info.w13_weight.size(0)
        k = quant_info.w13_weight.size(1)
        n = quant_info.w2_weight.size(1)
        m = hidden_states.shape[0]

        prepare_moe_input(
            topk_ids,
            quant_info.expert_offsets,
            quant_info.problem_sizes1,
            quant_info.problem_sizes2,
            a_map,
            c_map,
            num_experts,
            n,
            k,
        )

        a_q, a1_scale = sglang_per_token_group_quant_fp8(hidden_states, 128)
        rep_a_q = shuffle_rows(a_q, a_map, (m * topk_ids.shape[1], k))
        rep_a1_scales = shuffle_rows(
            a1_scale, a_map, (m * topk_ids.shape[1], max(k // 128, 1))
        )

        rep_primary = rep_a_q
        rep_aux = rep_a1_scales

    elif quant_info.moe_type == CutlassMoEType.BlockscaledFP4:
        params = quant_info.params

        prepare_moe_input(
            topk_ids,
            params.expert_offsets,
            params.problem_sizes1,
            params.problem_sizes2,
            a_map,
            c_map,
            params.num_experts,
            params.intermediate_size_per_partition,
            params.hidden_size,
            params.blockscale_offsets,
        )

        rep_a_fp4, rep_a_blockscale = scaled_fp4_experts_quant(
            hidden_states,
            quant_info.a1_gscale,
            params.expert_offsets,
            params.blockscale_offsets,
            topk_ids.shape[1],
            expert_map=a_map,
        )

        rep_primary = rep_a_fp4
        rep_aux = rep_a_blockscale

    return CutlassRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        a_map=a_map,
        c_map=c_map,
        rep_primary=rep_primary,
        rep_aux=rep_aux,
    )


@register_post_permute("cutlass", "standard")
def post_permute_cutlass_to_standard(
    runner_output: CutlassRunnerOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> StandardCombineInput:
    hidden_states = runner_output.hidden_states
    if runner_config.routed_scaling_factor is not None:
        hidden_states.mul_(runner_config.routed_scaling_factor)
    return StandardCombineInput(hidden_states=hidden_states)
