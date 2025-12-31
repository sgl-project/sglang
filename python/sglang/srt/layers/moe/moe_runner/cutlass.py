from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

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
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import is_cuda, is_sm90_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

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


@dataclass
class CutlassRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    # Standard mode fields
    topk_weights: Optional[torch.Tensor] = None
    a_map: Optional[torch.Tensor] = None
    c_map: Optional[torch.Tensor] = None
    rep_primary: Optional[torch.Tensor] = None
    rep_aux: Optional[torch.Tensor] = None
    # DeepEP mode fields
    masked_m: Optional[torch.Tensor] = None
    expected_m: Optional[int] = None

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
    # W4A8 specific fields
    a_strides1: Optional[torch.Tensor] = None
    b_strides1: Optional[torch.Tensor] = None
    c_strides1: Optional[torch.Tensor] = None
    a_strides2: Optional[torch.Tensor] = None
    b_strides2: Optional[torch.Tensor] = None
    c_strides2: Optional[torch.Tensor] = None
    s_strides13: Optional[torch.Tensor] = None
    s_strides2: Optional[torch.Tensor] = None
    w13_input_scale: Optional[torch.Tensor] = None
    w2_input_scale: Optional[torch.Tensor] = None
    params: Optional[CutlassMoEParams] = None


class CutlassRunnerCore(MoeRunnerCore):
    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported

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
        from sglang.srt.layers.quantization.fp8_kernel import (  # Handle DeepEP modes
            sglang_per_token_group_quant_fp8,
        )

        # Common validation assertions for all CUTLASS MoE types
        assert (
            runner_input.topk_weights.shape == runner_input.topk_ids.shape
        ), "topk shape mismatch"
        assert runner_input.hidden_states.dtype in [
            torch.half,
            torch.bfloat16,
        ], "Invalid input dtype - must be half or bfloat16"
        assert (
            quant_info.w13_weight.shape[0] == quant_info.w2_weight.shape[0]
        ), "Expert number mismatch between w13 and w2"

        num_tokens = runner_input.hidden_states.shape[0]
        hidden_size = runner_input.hidden_states.shape[1]
        topk = runner_input.topk_ids.shape[1]

        # Standard mode
        moe_type = quant_info.moe_type

        if moe_type == CutlassMoEType.DeepEP_LL:
            down_output = cutlass_w4a8_moe_deepep_ll(
                runner_input.hidden_states,
                quant_info.w13_weight,
                quant_info.w2_weight,
                quant_info.w13_scale,
                quant_info.w2_scale,
                runner_input.topk_ids,
                runner_input.masked_m,
                quant_info.a_strides1,
                quant_info.b_strides1,
                quant_info.c_strides1,
                quant_info.a_strides2,
                quant_info.b_strides2,
                quant_info.c_strides2,
                quant_info.s_strides13,
                quant_info.s_strides2,
                quant_info.expert_offsets,
                quant_info.problem_sizes1,
                quant_info.problem_sizes2,
                quant_info.w13_input_scale,
                quant_info.w2_input_scale,
            )
            return CutlassRunnerOutput(hidden_states=down_output)

        elif moe_type == CutlassMoEType.DeepEP_Normal:
            down_output = cutlass_w4a8_moe_deepep_normal(
            runner_input.hidden_states,
            quant_info.w13_weight,
            quant_info.w2_weight,
            quant_info.w13_scale,
            quant_info.w2_scale,
            runner_input.topk_weights,
            runner_input.topk_ids,
            quant_info.a_strides1,
            quant_info.b_strides1,
            quant_info.c_strides1,
            quant_info.a_strides2,
            quant_info.b_strides2,
            quant_info.c_strides2,
            quant_info.s_strides13,
            quant_info.s_strides2,
            quant_info.expert_offsets,
            quant_info.problem_sizes1,
            quant_info.problem_sizes2,
            quant_info.w13_input_scale,
            quant_info.w2_input_scale,
        )
            return CutlassRunnerOutput(hidden_states=down_output)

        elif moe_type == CutlassMoEType.BlockscaledFP8:
            down_output = self.cutlass_fused_experts_fp8(
                a=runner_input.hidden_states,
                w1_q=quant_info.w13_weight,
                w2_q=quant_info.w2_weight,
                w1_scale=quant_info.w13_scale,
                w2_scale=quant_info.w2_scale,
                topk_weights=runner_input.topk_weights,
                topk_ids=runner_input.topk_ids,
                a1_strides=quant_info.ab_strides_13,
                c1_strides=quant_info.c_strides_13,
                a2_strides=quant_info.ab_strides_2,
                c2_strides=quant_info.c_strides_2,
                workspace=quant_info.workspace,
                a_ptrs=quant_info.a_ptrs,
                b_ptrs=quant_info.b_ptrs,
                out_ptrs=quant_info.out_ptrs,
                a_scales_ptrs=quant_info.a_scales_ptrs,
                b_scales_ptrs=quant_info.b_scales_ptrs,
                expert_offsets=quant_info.expert_offsets,
                problem_sizes1=quant_info.problem_sizes1,
                problem_sizes2=quant_info.problem_sizes2,
            )

        elif moe_type == CutlassMoEType.BlockscaledFP4:
            down_output = self.cutlass_moe_fp4(
                a=runner_input.hidden_states,
                a1_gscale=quant_info.a1_gscale,
                w1_fp4=quant_info.w13_weight,
                w1_blockscale=quant_info.w1_blockscale,
                w1_alphas=quant_info.w1_alpha,
                a2_gscale=quant_info.a2_gscale,
                w2_fp4=quant_info.w2_weight,
                w2_blockscale=quant_info.w2_blockscale,
                w2_alphas=quant_info.w2_alpha,
                topk_weights=runner_input.topk_weights,
                topk_ids=runner_input.topk_ids,
                params=quant_info.params,
                apply_router_weight_on_input=self.config.apply_router_weight_on_input,
            )

        elif moe_type == CutlassMoEType.W4A8:
            from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe

            down_output = cutlass_w4a8_moe(
                runner_input.hidden_states,
                quant_info.w13_weight,
                quant_info.w2_weight,
                quant_info.w13_scale,
                quant_info.w2_scale,
                runner_input.topk_weights,
                runner_input.topk_ids,
                quant_info.a_strides1,
                quant_info.b_strides1,
                quant_info.c_strides1,
                quant_info.a_strides2,
                quant_info.b_strides2,
                quant_info.c_strides2,
                quant_info.s_strides13,
                quant_info.s_strides2,
                quant_info.expert_offsets,
                quant_info.problem_sizes1,
                quant_info.problem_sizes2,
                quant_info.w13_input_scale,
                quant_info.w2_input_scale,
                apply_router_weight_on_input=self.config.apply_router_weight_on_input,
            )
            # W4A8 already returns combined output
            return CutlassRunnerOutput(hidden_states=down_output)
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

    def cutlass_w4a8_moe(
        a: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        a_strides1: torch.Tensor,
        b_strides1: torch.Tensor,
        c_strides1: torch.Tensor,
        a_strides2: torch.Tensor,
        b_strides2: torch.Tensor,
        c_strides2: torch.Tensor,
        s_strides13: torch.Tensor,
        s_strides2: torch.Tensor,
        expert_offsets: torch.Tensor,
        problem_sizes1: torch.Tensor,
        problem_sizes2: torch.Tensor,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        routed_scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        """
        This function computes a w4a8-quantized Mixture of Experts (MoE) layer
        using two sets of quantized weights, w1_q and w2_q, and top-k gating
        mechanism. The matrix multiplications are implemented with CUTLASS
        grouped gemm.

        Parameters:
        - a (torch.Tensor): The input tensor to the MoE layer.
            Shape: [M, K]
        - w1_q (torch.Tensor): The first set of int4-quantized expert weights.
            Shape: [num_experts, N * 2,  K // 2]
            (the weights are passed transposed and int4-packed)
        - w2_q (torch.Tensor): The second set of int4-quantized expert weights.
            Shape: [num_experts, K, N // 2]
            (the weights are passed transposed and int4-packed)
        - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
            Shape: [num_experts, K // 512, N * 8]
        - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
            Shape: [num_experts, N // 512, K * 4]
        - topk_weights (torch.Tensor): The weights of each token->expert mapping.
        - topk_ids (torch.Tensor): The ids of each token->expert mapping.
        - a_strides1 (torch.Tensor): The input strides of the first grouped gemm.
        - b_strides1 (torch.Tensor): The weights strides of the first grouped gemm.
        - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
        - a_strides2 (torch.Tensor): The input strides of the second grouped gemm.
        - b_strides2 (torch.Tensor): The weights strides of the second grouped gemm.
        - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
        - s_strides13 (torch.Tensor): The input and scale strides of the first grouped gemm.
        - s_strides2 (torch.Tensor): The scale strides of the second grouped gemm.
        - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
            Shape: scalar or [1, K]
        - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
            quantize the intermediate result between the gemms.
            Shape: scalar or [1, N]
        - apply_router_weight_on_input (bool): When true, the topk weights are
            applied directly on the inputs. This is only applicable when topk is 1.

        Returns:
        - torch.Tensor: The fp8 output tensor after applying the MoE layer.
        """
        assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
        assert w1_q.dtype == torch.int8
        assert w2_q.dtype == torch.int8
        assert a.shape[1] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
        assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
        assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
        assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
        assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"

        assert (
            a_strides1.shape[0] == w1_q.shape[0]
        ), "A Strides 1 expert number mismatch"
        assert (
            b_strides1.shape[0] == w1_q.shape[0]
        ), "B Strides 1 expert number mismatch"
        assert (
            a_strides2.shape[0] == w2_q.shape[0]
        ), "A Strides 2 expert number mismatch"
        assert (
            b_strides2.shape[0] == w2_q.shape[0]
        ), "B Strides 2 expert number mismatch"
        num_local_experts = w1_q.size(0)
        m = a.size(0)
        k = w1_q.size(2) * 2  # w1_q is transposed and packed
        n = w2_q.size(2) * 2  # w2_q is transposed and packed
        topk = topk_ids.size(1)

        if apply_router_weight_on_input:
            assert (
                topk == 1
            ), "apply_router_weight_on_input is only implemented for topk=1"

        device = a.device
        if get_moe_expert_parallel_world_size() > 1:
            topk_ids = torch.where(topk_ids == -1, num_local_experts, topk_ids)

        src2dst = cutlass_w4_run_moe_ep_preproess(
            topk_ids,
        )

        gateup_input = torch.empty(
            (m * topk, k),
            device=device,
            dtype=torch.float8_e4m3fn,
        )

        pre_reorder_for_cutlass_moe(
            a,
            gateup_input,
            src2dst,
            topk_ids,
            a1_scale,
            num_local_experts,
            topk,
            m,
            k,
        )

        # NOTE: a_map and c_map are not used in the get_cutlass_w4a8_moe_mm_data kernel,
        # they are kept to allow for a quick switch of the permutation logic
        # from the current triton kernel implementation to the cutlass-based one if needed.
        a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
        c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
        get_cutlass_w4a8_moe_mm_data(
            topk_ids,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            a_map,
            c_map,
            num_local_experts,
            n,
            k,
        )

        c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)
        c2 = torch.empty((m * topk, k), device=device, dtype=torch.bfloat16)

        cutlass_w4a8_moe_mm(
            c1,
            gateup_input,
            w1_q,
            a1_scale.float(),
            w1_scale,
            expert_offsets[:-1],
            problem_sizes1,
            a_strides1,
            b_strides1,
            c_strides1,
            s_strides13,
            128,
            topk,
        )

        intermediate_q = torch.empty(
            (m * topk, n), dtype=torch.float8_e4m3fn, device=device
        )
        silu_mul_static_tensorwise_quant_for_cutlass_moe(
            c1, intermediate_q, a2_scale.float(), expert_offsets[-1:], m * topk, n
        )

        cutlass_w4a8_moe_mm(
            c2,
            intermediate_q,
            w2_q,
            a2_scale.float(),
            w2_scale,
            expert_offsets[:-1],
            problem_sizes2,
            a_strides2,
            b_strides2,
            c_strides2,
            s_strides2,
            128,
            topk,
        )

        output = torch.empty_like(a)

        post_reorder_for_cutlass_moe(
            c2,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            num_local_experts,
            topk,
            m,
            k,
            routed_scaling_factor,
        )
        return output

    def cutlass_w4a8_moe_deepep_normal(
        a: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids_: torch.Tensor,
        a_strides1: torch.Tensor,
        b_strides1: torch.Tensor,
        c_strides1: torch.Tensor,
        a_strides2: torch.Tensor,
        b_strides2: torch.Tensor,
        c_strides2: torch.Tensor,
        s_strides13: torch.Tensor,
        s_strides2: torch.Tensor,
        expert_offsets: torch.Tensor,
        problem_sizes1: torch.Tensor,
        problem_sizes2: torch.Tensor,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        This function computes a w4a8-quantized Mixture of Experts (MoE) layer
        using two sets of quantized weights, w1_q and w2_q, and top-k gating
        mechanism. The matrix multiplications are implemented with CUTLASS
        grouped gemm.

        Parameters:
        - a (torch.Tensor): The input tensor to the MoE layer.
            Shape: [M, K]
        - w1_q (torch.Tensor): The first set of int4-quantized expert weights.
            Shape: [num_experts, N * 2,  K // 2]
            (the weights are passed transposed and int4-packed)
        - w2_q (torch.Tensor): The second set of int4-quantized expert weights.
            Shape: [num_experts, K, N // 2]
            (the weights are passed transposed and int4-packed)
        - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
            Shape: [num_experts, K // 512, N * 8]
        - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
            Shape: [num_experts, N // 512, K * 4]
        - topk_weights (torch.Tensor): The weights of each token->expert mapping.
        - a_strides1 (torch.Tensor): The input strides of the first grouped gemm.
        - b_strides1 (torch.Tensor): The weights strides of the first grouped gemm.
        - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
        - a_strides2 (torch.Tensor): The input strides of the second grouped gemm.
        - b_strides2 (torch.Tensor): The weights strides of the second grouped gemm.
        - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
        - s_strides13 (torch.Tensor): The input and scale strides of the first grouped gemm.
        - s_strides2 (torch.Tensor): The scale strides of the second grouped gemm.
        - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
            Shape: scalar or [1, K]
        - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
            quantize the intermediate result between the gemms.
            Shape: scalar or [1, N]
        - apply_router_weight_on_input (bool): When true, the topk weights are
            applied directly on the inputs. This is only applicable when topk is 1.

        Returns:
        - torch.Tensor: The fp8 output tensor after applying the MoE layer.
        """
        assert topk_weights.shape == topk_ids_.shape, "topk shape mismatch"
        assert w1_q.dtype == torch.int8
        assert w2_q.dtype == torch.int8
        assert a.shape[1] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
        assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
        assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
        assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
        assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"

        assert (
            a_strides1.shape[0] == w1_q.shape[0]
        ), "A Strides 1 expert number mismatch"
        assert (
            b_strides1.shape[0] == w1_q.shape[0]
        ), "B Strides 1 expert number mismatch"
        assert (
            a_strides2.shape[0] == w2_q.shape[0]
        ), "A Strides 2 expert number mismatch"
        assert (
            b_strides2.shape[0] == w2_q.shape[0]
        ), "B Strides 2 expert number mismatch"
        num_experts = w1_q.size(0)
        m = a.size(0)
        k = w1_q.size(2) * 2  # w1_q is transposed and packed
        n = w2_q.size(2) * 2  # w2_q is transposed and packed
        topk = topk_ids_.size(1)

        num_experts = w1_q.size(0)
        m = a.size(0)
        k = w1_q.size(2) * 2
        n = w2_q.size(2) * 2
        topk = topk_ids_.size(1)
        device = a.device

        reorder_topk_ids, src2dst, _ = deepep_run_moe_deep_preprocess(
            topk_ids_, num_experts
        )
        num_total_tokens = reorder_topk_ids.numel()
        gateup_input_pre_reorder = torch.empty(
            (int(num_total_tokens), a.shape[1]),
            device=device,
            dtype=a.dtype,
        )
        deepep_permute_triton_kernel[(a.shape[0],)](
            a,
            gateup_input_pre_reorder,
            src2dst,
            topk_ids_.to(torch.int64),
            None,
            topk,
            a.shape[1],
            BLOCK_SIZE=512,
        )
        gateup_input = torch.empty(
            gateup_input_pre_reorder.shape, dtype=torch.float8_e4m3fn, device=device
        )
        sgl_per_tensor_quant_fp8(
            gateup_input_pre_reorder, gateup_input, a1_scale.float(), True
        )
        del gateup_input_pre_reorder
        local_topk_ids = topk_ids_
        local_topk_ids = (
            torch.where(local_topk_ids == -1, num_experts, topk_ids_).to(torch.int32)
        ).contiguous()

        a_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)
        c_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)
        get_cutlass_w4a8_moe_mm_data(
            local_topk_ids,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            a_map,
            c_map,
            num_experts,
            n,
            k,
        )
        c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)
        c2 = torch.zeros((m * topk, k), device=device, dtype=torch.bfloat16)

        cutlass_w4a8_moe_mm(
            c1,
            gateup_input,
            w1_q,
            a1_scale.float(),
            w1_scale,
            expert_offsets[:-1],
            problem_sizes1,
            a_strides1,
            b_strides1,
            c_strides1,
            s_strides13,
            128,
            topk,
        )
        intermediate = torch.empty((m * topk, n), device=device, dtype=torch.bfloat16)
        silu_and_mul(c1, intermediate)

        intermediate_q = torch.empty(
            intermediate.shape, dtype=torch.float8_e4m3fn, device=device
        )
        sgl_per_tensor_quant_fp8(intermediate, intermediate_q, a2_scale.float(), True)

        cutlass_w4a8_moe_mm(
            c2,
            intermediate_q,
            w2_q,
            a2_scale.float(),
            w2_scale,
            expert_offsets[:-1],
            problem_sizes2,
            a_strides2,
            b_strides2,
            c_strides2,
            s_strides2,
            128,
            topk,
        )
        num_tokens = src2dst.shape[0] // topk
        output = torch.empty(
            (num_tokens, c2.shape[1]),
            device=c2.device,
            dtype=torch.bfloat16,
        )
        deepep_post_reorder_triton_kernel[(num_tokens,)](
            c2,
            output,
            src2dst,
            topk_ids_,
            topk_weights,
            topk,
            c2.shape[1],
            BLOCK_SIZE=512,
        )

        return output

    def cutlass_w4a8_moe_deepep_ll(
        a: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_ids_: torch.Tensor,
        masked_m: torch.Tensor,
        a_strides1: torch.Tensor,
        b_strides1: torch.Tensor,
        c_strides1: torch.Tensor,
        a_strides2: torch.Tensor,
        b_strides2: torch.Tensor,
        c_strides2: torch.Tensor,
        s_strides13: torch.Tensor,
        s_strides2: torch.Tensor,
        expert_offsets: torch.Tensor,
        problem_sizes1: torch.Tensor,
        problem_sizes2: torch.Tensor,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        This function computes a w4a8-quantized Mixture of Experts (MoE) layer
        using two sets of quantized weights, w1_q and w2_q, and top-k gating
        mechanism. The matrix multiplications are implemented with CUTLASS
        grouped gemm.

        Parameters:
        - a (torch.Tensor): The input tensor to the MoE layer.
            Shape: [num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, K]
        - w1_q (torch.Tensor): The first set of int4-quantized expert weights.
            Shape: [num_experts, N * 2,  K // 2]
            (the weights are passed transposed and int4-packed)
        - w2_q (torch.Tensor): The second set of int4-quantized expert weights.
            Shape: [num_experts, K, N // 2]
            (the weights are passed transposed and int4-packed)
        - w1_scale (torch.Tensor): The fp32 scale to dequantize w1_q.
            Shape: [num_experts, K // 512, N * 8]
        - w2_scale (torch.Tensor): The fp32 scale to dequantize w2_q.
            Shape: [num_experts, N // 512, K * 4]
        - topk_weights (torch.Tensor): The weights of each token->expert mapping.
        - a_strides1 (torch.Tensor): The input strides of the first grouped gemm.
        - b_strides1 (torch.Tensor): The weights strides of the first grouped gemm.
        - c_strides1 (torch.Tensor): The output strides of the first grouped gemm.
        - a_strides2 (torch.Tensor): The input strides of the second grouped gemm.
        - b_strides2 (torch.Tensor): The weights strides of the second grouped gemm.
        - c_strides2 (torch.Tensor): The output strides of the second grouped gemm.
        - s_strides13 (torch.Tensor): The input and scale strides of the first grouped gemm.
        - s_strides2 (torch.Tensor): The scale strides of the second grouped gemm.
        - a1_scale (Optional[torch.Tensor]): The optional fp32 scale to quantize a.
            Shape: scalar or [1, K]
        - a2_scale (Optional[torch.Tensor]): The optional fp32 scale to
            quantize the intermediate result between the gemms.
            Shape: scalar or [1, N]
        - apply_router_weight_on_input (bool): When true, the topk weights are
            applied directly on the inputs. This is only applicable when topk is 1.

        Returns:
        - torch.Tensor: The fp8 output tensor after applying the MoE layer.
        """
        assert w1_q.dtype == torch.int8
        assert w2_q.dtype == torch.int8
        assert a.shape[2] // 2 == w1_q.shape[2], "Hidden size mismatch w1"
        assert w1_q.shape[2] * 2 == w2_q.shape[1], "Hidden size mismatch w2"
        assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
        assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
        assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"

        assert (
            a_strides1.shape[0] == w1_q.shape[0]
        ), "A Strides 1 expert number mismatch"
        assert (
            b_strides1.shape[0] == w1_q.shape[0]
        ), "B Strides 1 expert number mismatch"
        assert (
            a_strides2.shape[0] == w2_q.shape[0]
        ), "A Strides 2 expert number mismatch"
        assert (
            b_strides2.shape[0] == w2_q.shape[0]
        ), "B Strides 2 expert number mismatch"
        num_experts = w1_q.size(0)
        m = a.size(1)
        k = w1_q.size(2) * 2  # w1_q is transposed and packed
        n = w2_q.size(2) * 2  # w2_q is transposed and packed
        topk = topk_ids_.size(1)

        device = a.device

        problem_sizes1, problem_sizes2 = deepep_ll_get_cutlass_w4a8_moe_mm_data(
            masked_m,
            problem_sizes1,
            problem_sizes2,
            num_experts,
            n,
            k,
        )

        gateup_input = torch.empty(a.shape, dtype=torch.float8_e4m3fn, device=device)
        sgl_per_tensor_quant_fp8(a, gateup_input, a1_scale.float(), True)
        c1 = torch.empty((num_experts, m, n * 2), device=device, dtype=torch.bfloat16)
        c2 = torch.empty((num_experts, m, k), device=device, dtype=torch.bfloat16)

        cutlass_w4a8_moe_mm(
            c1,
            gateup_input,
            w1_q,
            a1_scale.float(),
            w1_scale,
            expert_offsets[:-1],
            problem_sizes1,
            a_strides1,
            b_strides1,
            c_strides1,
            s_strides13,
            128,
            topk,
        )

        intermediate_q = torch.empty(
            (num_experts, m, n), device=a.device, dtype=torch.float8_e4m3fn
        )
        silu_and_mul_masked_post_per_tensor_quant_fwd(
            c1, intermediate_q, masked_m, a2_scale
        )
        cutlass_w4a8_moe_mm(
            c2,
            intermediate_q,
            w2_q,
            a2_scale.float(),
            w2_scale,
            expert_offsets[:-1],
            problem_sizes2,
            a_strides2,
            b_strides2,
            c_strides2,
            s_strides2,
            128,
            topk,
        )

        return c2

    def cutlass_fused_experts_fp8(
        a: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        a1_strides: torch.Tensor,
        c1_strides: torch.Tensor,
        a2_strides: torch.Tensor,
        c2_strides: torch.Tensor,
        workspace: torch.Tensor,
        a_ptrs: torch.Tensor,
        b_ptrs: torch.Tensor,
        out_ptrs: torch.Tensor,
        a_scales_ptrs: torch.Tensor,
        b_scales_ptrs: torch.Tensor,
        expert_offsets: torch.Tensor,
        problem_sizes1: torch.Tensor,
        problem_sizes2: torch.Tensor,
        use_fp8_blockscale: bool = True,
        output: Optional[torch.Tensor] = None,
        enable_es: Tuple[bool, bool] = (False, False),
    ) -> torch.Tensor:
        """Performs Fused MoE computation using CUTLASS-like kernels with FP8 weights and activations.

        This function implements a Mixture of Experts (MoE) layer with a SwiGLU/SiLU
        activation, leveraging custom kernels likely derived from CUTLASS principles
        for grouped matrix multiplication (`fp8_blockwise_scaled_grouped_mm`) and
        data preparation (`prepare_moe_input`, `silu_and_mul`).

        It handles per-token routing, quantizes input activations to FP8 with
        per-token scales, performs the expert computations using FP8 GEMMs with
        pre-quantized FP8 weights (per-block scales), applies the SiLU activation,
        and combines the results weighted by the router scores.

        Args:
            a (torch.Tensor): Input activations. Shape: `(m, k)`, where `m` is the total
                number of tokens and `k` is the hidden size. Expected dtype: `torch.half`
                or `torch.bfloat16`.
            w1_q (torch.Tensor): Pre-quantized FP8 weight tensor for the first GEMM
                (up-projection part of SwiGLU). Expected shape: `(E, k, n*2)`, where
                `E` is the number of experts, `k` is the hidden size, and `n*2` is the
                intermediate size (`I`). Expected dtype: `torch.float8_e4m3fn`.
                Note: This shape implies weights are stored as (num_experts, hidden_size, intermediate_size).
            w2_q (torch.Tensor): Pre-quantized FP8 weight tensor for the second GEMM
                (down-projection). Expected shape: `(E, n, k)`, where `n` is half the
                intermediate size (`I // 2`). Expected dtype: `torch.float8_e4m3fn`.
                Note: This shape implies weights are stored as (num_experts, intermediate_size // 2, hidden_size).
            w1_scale (torch.Tensor): Scales corresponding to `w1_q` (per-block scales).
                Shape: `(E, num_blocks_n, num_blocks_k)`. Dtype: `torch.float32`.
            w2_scale (torch.Tensor): Scales corresponding to `w2_q` (per-block scales).
                Shape: `(E, num_blocks_k, num_blocks_n)`. Dtype: `torch.float32`.
            topk_weights (torch.Tensor): Router weights for the selected top-k experts
                for each token. Shape: `(m, topk)`. Dtype should ideally match `a`.
            topk_ids (torch.Tensor): Indices of the selected top-k experts for each token.
                Shape: `(m, topk)`. Dtype: `torch.int32`.
            a1_strides (torch.Tensor): Stride information for the first GEMM's 'a' input.
                Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
                Note: Its exact usage within `fp8_blockwise_scaled_grouped_mm` needs clarification
                as it's passed as both a_stride and b_stride in the first call.
            c1_strides (torch.Tensor): Stride information for the first GEMM's 'c' output.
                Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
            a2_strides (torch.Tensor): Stride information for the second GEMM's 'a' input.
                Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
                Note: Its exact usage within `fp8_blockwise_scaled_grouped_mm` needs clarification
                as it's passed as both a_stride and b_stride in the second call.
            c2_strides (torch.Tensor): Stride information for the second GEMM's 'c' output.
                Passed directly to the underlying kernel. Expected shape `(E,)`, dtype `torch.int64`.
            workspace (torch.Tensor): Reusable workspace for the underlying kernel.
            a_ptrs (torch.Tensor): Pointers container for calculating offsets of the input activations for each expert.
            b_ptrs (torch.Tensor): Pointers container for calculating offsets of the input weights for each expert.
            out_ptrs (torch.Tensor): Pointers container for calculating offsets of the output activations for each expert.
            a_scales_ptrs (torch.Tensor): Pointers container for calculating offsets of the input scales for each expert.
            b_scales_ptrs (torch.Tensor): Pointers container for calculating offsets of the input scales for each expert.
            use_fp8_blockscale (bool, optional): Flag indicating usage of FP8 with
                block scaling. Currently, only `True` is supported. Defaults to `True`.
            output (torch.Tensor, optional): Output tensor. If not provided, a new tensor will be created.
            enable_es (tuple(bool, bool)): Flag indicating usage of expert specialization kernel for (up-projection, down-projection)
        Returns:
            torch.Tensor: The computed MoE layer output. Shape: `(m, k)`, dtype matches `a`.

        Raises:
            AssertionError: If input shapes, dtypes, or flags are inconsistent or unsupported.
            NotImplementedError: If CUDA is not available or `sgl_kernel` is not properly installed.
        """
        assert use_fp8_blockscale, "Only support fp8 blockscale for now"
        assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
        assert w1_q.dtype == torch.float8_e4m3fn
        assert w2_q.dtype == torch.float8_e4m3fn
        assert a.shape[1] == w1_q.shape[1], "Hidden size mismatch w1"
        assert w1_q.shape[2] == w2_q.shape[1] * 2, "Hidden size mismatch w2"
        assert w1_q.shape[0] == w2_q.shape[0], "Expert number mismatch"
        assert w1_q.shape[0] == w2_q.shape[0], "Weights expert number mismatch"
        assert w1_q.shape[0] == w1_scale.shape[0], "w1 scales expert number mismatch"
        assert w1_q.shape[0] == w2_scale.shape[0], "w2 scales expert number mismatch"
        assert a.dtype in [torch.half, torch.bfloat16], "Invalid output dtype"

        if is_cuda:
            from sglang.srt.layers.quantization.fp8_kernel import (
                sglang_per_token_group_quant_fp8,
            )
        es_up, es_down = enable_es
        out_dtype = a.dtype
        num_experts = w1_q.size(0)
        m = a.size(0)
        k = w1_q.size(1)
        n = w2_q.size(1)

        topk = topk_ids.size(1)
        device = a.device

        a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
        c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)

        prepare_moe_input(
            topk_ids,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            a_map,
            c_map,
            num_experts,
            n,
            k,
        )

        a_q, a1_scale = sglang_per_token_group_quant_fp8(a, 128)
        rep_a_q = shuffle_rows(a_q, a_map, (m * topk, k))
        rep_a1_scales = shuffle_rows(a1_scale, a_map, (m * topk, int(k / 128)))

        c1 = torch.empty((m * topk, n * 2), device=device, dtype=out_dtype)
        c2 = torch.empty((m * topk, k), device=device, dtype=out_dtype)

        a_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)
        w_sf_layout = torch.empty((num_experts, 5), device=device, dtype=torch.int)

        if is_sm90_supported() and es_up:
            es_fp8_blockwise_scaled_grouped_mm(
                c1,
                rep_a_q,
                w1_q,
                rep_a1_scales,
                w1_scale,
                a1_strides,
                a1_strides,
                c1_strides,
                problem_sizes1,
                expert_offsets[:-1],
                workspace,
            )
        else:
            fp8_blockwise_scaled_grouped_mm(
                c1,
                a_ptrs,
                b_ptrs,
                out_ptrs,
                a_scales_ptrs,
                b_scales_ptrs,
                rep_a_q,
                w1_q,
                rep_a1_scales,
                w1_scale,
                a1_strides,
                a1_strides,
                c1_strides,
                a_sf_layout,
                w_sf_layout,
                problem_sizes1,
                expert_offsets[:-1],
                workspace,
            )

        intermediate = torch.empty((m * topk, n), device=device, dtype=out_dtype)
        silu_and_mul(c1, intermediate)

        intemediate_q, a2_scale = sglang_per_token_group_quant_fp8(intermediate, 128)

        if is_sm90_supported() and es_down:
            es_fp8_blockwise_scaled_grouped_mm(
                c2,
                intemediate_q,
                w2_q,
                a2_scale,
                w2_scale,
                a2_strides,
                a2_strides,
                c2_strides,
                problem_sizes2,
                expert_offsets[:-1],
                workspace,
            )
        else:
            fp8_blockwise_scaled_grouped_mm(
                c2,
                a_ptrs,
                b_ptrs,
                out_ptrs,
                a_scales_ptrs,
                b_scales_ptrs,
                intemediate_q,
                w2_q,
                a2_scale,
                w2_scale,
                a2_strides,
                a2_strides,
                c2_strides,
                a_sf_layout,
                w_sf_layout,
                problem_sizes2,
                expert_offsets[:-1],
                workspace,
            )

        if output is None:
            output = torch.empty((m, k), device=device, dtype=out_dtype)

        apply_shuffle_mul_sum(c2, output, c_map, topk_weights.to(out_dtype))
        return output

    def cutlass_moe_fp4(
        a: torch.Tensor,
        a1_gscale: torch.Tensor,
        w1_fp4: torch.Tensor,
        w1_blockscale: torch.Tensor,
        w1_alphas: torch.Tensor,
        a2_gscale: torch.Tensor,
        w2_fp4: torch.Tensor,
        w2_blockscale: torch.Tensor,
        w2_alphas: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        params: CutlassMoEParams,
        apply_router_weight_on_input: bool = False,
    ):
        """
        MoE implementation for FP4 Inputs

        # Gemm 1
        a: Input tensor: [m, k] (half/bfloat16)
        a1_gscale: Activation scale per expert: [e]  (float32)
        w1(gate up) (not an argument to cutlass_moe_fp4): [e, 2 * n, k]
        w1_fp4: [e, 2 * n, k // 2], dtype: torch.uint8 (stacked fp4: E2M1)
        (Note: `n` is the up projection output dim, `k` is the input dim in
        full precision)
        w1_blockscale: [e, 2 * n, k // block_size] (float8_e4m3)
                    (Block size = 16 for NVFP4)

        # Gemm 2
        a2_gscale: Activation scale per expert: [e]
        w2(down projection) (not an argument to cutlass_moe_fp4): [e, k, n]
        w2_fp4: [e, k, n // 2], dtype: torch.uint8 (stacked E2M1)
        w2_blockscale: [e, k, n // block_size], dtype: float8_e4m3

        Strides for activations, weights and output in logical number of elements.
        The activations & output stride is the number of elements to the next row.
        The weights stride is the number of elements to the next row per expert.
        For example, if the weight is [e, n, k], then the b_stride is a tensor of
        shape [e] with each element being k. Similarly for activations, if the
        shape is [m, k], then the a_stride has shape [e] with each value k.
        Similarly for output, if the output is [m, n], then the c_stride is a
        tensor of shape [e] with each element being k.

        Note: cutlass_fp4_group_mm is designed to accept the strides of
        activations and weights to be the same, so it is passed in as a single
        tensor.
        ab_strides_13: [e] dtype: int64 [Gemm 1: Activation / Weight strides]
        ab_strides_2: [e] dtype: int64 [Gemm 2: Activation / Weight strides]
        c_strides_13: [e] dtype: int64 [Gemm 1: Output Strides]
        c_strides_2: [e] dtype: int64 [Gemm 1: Output Strides]

        topk_weights: [m, topk] dtype: float8
        topk_ids: [m, topk] dtype: float8

        m, n, k: Unquantized weight shapes, dtype: int
        e: number of experts for the current rank, dtype: int
        assumes that topk < k < n to satisfy - up/down projection expectations.
        """
        assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
        assert w1_fp4.dtype == torch.uint8, "weight 1 must be uint8"
        assert w2_fp4.dtype == torch.uint8, "weight 2 must be uint8"
        assert (
            w1_fp4.ndim == 3
            and w2_fp4.ndim == 3
            and w1_blockscale.ndim == 3
            and w2_blockscale.ndim == 3
        ), "All Weights must be of rank 3 for cutlass_moe_fp4"
        m_a, k_a = a.shape
        e_w1, nx2_w1, half_k_w1 = w1_fp4.shape
        e_w2, k_w2, half_n_w2 = w2_fp4.shape

        assert e_w1 == e_w2 and e_w1 == params.num_experts, (
            "Number of experts must match",
            " between weights.",
        )
        assert (
            k_a // 2 == half_k_w1 and params.hidden_size == k_w2
        ), "Hidden size mismatch between a, w1 and w2"
        assert (
            nx2_w1 == params.intermediate_size_per_partition * 2
            and half_n_w2 == params.intermediate_size_per_partition // 2
        ), "mismatch in " "expected `n`"
        assert 2 * half_k_w1 == k_w2, "Hidden size mismatch w2 and w1"
        assert a.dtype in [torch.half, torch.bfloat16], "Invalid input dtype"

        out_dtype = a.dtype
        num_topk = topk_ids.shape[1]
        device = a.device
        a_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
        c_map = torch.empty((topk_ids.numel()), dtype=torch.int32, device=device)
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
            a,
            a1_gscale,
            params.expert_offsets,
            params.blockscale_offsets,
            num_topk,
            expert_map=a_map,
        )
        c1 = cutlass_fp4_group_mm(
            rep_a_fp4,
            w1_fp4,
            rep_a_blockscale,
            w1_blockscale,
            w1_alphas,
            out_dtype,
            device,
            params.to_gemm1_args(),
        )
        del rep_a_fp4, rep_a_blockscale

        # hidden size dimension is split to one halfpytho sized tensor.
        intermediate = torch.empty(
            (m_a * num_topk, w1_fp4.shape[1] // 2), device=device, dtype=out_dtype
        )
        silu_and_mul(c1, intermediate)

        int_fp4, int_blockscale = scaled_fp4_experts_quant(
            intermediate,
            a2_gscale,
            params.expert_offsets,
            params.blockscale_offsets,
            num_topk,
        )
        c2 = cutlass_fp4_group_mm(
            int_fp4,
            w2_fp4,
            int_blockscale,
            w2_blockscale,
            w2_alphas,
            out_dtype,
            device,
            params.to_gemm2_args(),
        )
        del int_fp4, int_blockscale
        c2 = shuffle_rows(c2, c_map, (m_a * num_topk, params.hidden_size))
        c2 = c2.view(m_a, num_topk, params.hidden_size)
        if not apply_router_weight_on_input:
            c2 = c2 * topk_weights.view(m_a, num_topk, 1).to(out_dtype)
        return c2.sum(dim=1).to(out_dtype)


@register_pre_permute("standard", "cutlass")
def pre_permute_standard_to_cutlass(
    dispatch_output: StandardDispatchOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> CutlassRunnerInput:
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

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


@register_pre_permute("deepep_ll", "cutlass")
def pre_permute_deepep_ll_to_cutlass(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> CutlassRunnerInput:
    hidden_states, _, topk_ids, topk_weights, masked_m, expected_m = dispatch_output

    # Store for post_permute
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    return CutlassRunnerInput(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        masked_m=masked_m,
        expected_m=expected_m,
    )


@register_post_permute("cutlass", "deepep_ll")
def post_permute_cutlass_to_deepep_ll(
    runner_output: CutlassRunnerOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> DeepEPLLCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )


@register_pre_permute("deepep_normal", "cutlass")
def pre_permute_deepep_normal_to_cutlass(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> CutlassRunnerInput:
    (
        hidden_states,
        _,
        topk_ids,
        topk_weights,
        _,
    ) = dispatch_output

    # Store for post_permute
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    return CutlassRunnerInput(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )


@register_post_permute("cutlass", "deepep_normal")
def post_permute_cutlass_to_deepep_normal(
    runner_output: CutlassRunnerOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    return DeepEPNormalCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["topk_ids"],
        topk_weights=running_state["topk_weights"],
    )
