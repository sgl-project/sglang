from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Tuple

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
from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported
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
        cutlass_w4a8_moe_mm,
        es_fp8_blockwise_scaled_grouped_mm,
        fp8_blockwise_scaled_grouped_mm,
        get_cutlass_w4a8_moe_mm_data,
        prepare_moe_input,
        scaled_fp4_experts_quant,
        shuffle_rows,
        silu_and_mul,
    )
    from sgl_kernel.gemm import sgl_per_tensor_quant_fp8


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

        if not is_cuda():
            raise RuntimeError("Cutlass runner requires CUDA support.")
        if not is_sm90_supported():
            raise RuntimeError("Cutlass runner requires NVIDIA SM90 or newer GPUs.")
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
        if runner_input.topk_weights is not None:
            assert (
                runner_input.topk_weights.shape == runner_input.topk_ids.shape
            ), "topk shape mismatch"
        assert (
            quant_info.w13_weight.shape[0] == quant_info.w2_weight.shape[0]
        ), "Expert number mismatch between w13 and w2"

        num_tokens = runner_input.hidden_states.shape[0]
        hidden_size = runner_input.hidden_states.shape[1]
        topk = runner_input.topk_ids.shape[1]

        # Standard mode
        moe_type = quant_info.moe_type

        if moe_type == CutlassMoEType.DeepEP_LL:
            down_output = self.cutlass_w4a8_moe_deepep_ll(
                runner_input.rep_primary,  # Use preprocessed input
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
                running_state,
                quant_info.w13_input_scale,
                quant_info.w2_input_scale,
            )
            return CutlassRunnerOutput(hidden_states=down_output)

        elif moe_type == CutlassMoEType.DeepEP_Normal:
            down_output = self.cutlass_w4a8_moe_deepep_normal(
                a=runner_input.hidden_states,
                w1_q=quant_info.w13_weight,
                w2_q=quant_info.w2_weight,
                w1_scale=quant_info.w13_scale,
                w2_scale=quant_info.w2_scale,
                topk_ids_=runner_input.topk_ids,
                a_strides1=quant_info.a_strides1,
                b_strides1=quant_info.b_strides1,
                c_strides1=quant_info.c_strides1,
                a_strides2=quant_info.a_strides2,
                b_strides2=quant_info.b_strides2,
                c_strides2=quant_info.c_strides2,
                s_strides13=quant_info.s_strides13,
                s_strides2=quant_info.s_strides2,
                expert_offsets=quant_info.expert_offsets,
                problem_sizes1=quant_info.problem_sizes1,
                problem_sizes2=quant_info.problem_sizes2,
                a1_scale=quant_info.w13_input_scale,
                a2_scale=quant_info.w2_input_scale,
            )
            return CutlassRunnerOutput(hidden_states=down_output)

        elif moe_type == CutlassMoEType.BlockscaledFP8:
            if not cutlass_fp8_supported():
                raise RuntimeError(
                    "CUTLASS FP8 kernels are not available on this system."
                )
            down_output = self.cutlass_fused_experts_fp8(
                a=runner_input.rep_primary,  # Use preprocessed quantized input
                w1_q=quant_info.w13_weight,
                w2_q=quant_info.w2_weight,
                w1_scale=quant_info.w13_scale,
                w2_scale=quant_info.w2_scale,
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
                a=runner_input.rep_primary,  # Use preprocessed input
                a1_gscale=quant_info.a1_gscale,
                w1_fp4=quant_info.w13_weight,
                w1_blockscale=quant_info.w1_blockscale,
                w1_alphas=quant_info.w1_alpha,
                a2_gscale=quant_info.a2_gscale,
                w2_fp4=quant_info.w2_weight,
                w2_blockscale=quant_info.w2_blockscale,
                w2_alphas=quant_info.w2_alpha,
                topk_ids=runner_input.topk_ids,
                params=quant_info.params,
            )

        elif moe_type == CutlassMoEType.W4A8:
            down_output = self.cutlass_w4a8_moe(
                a=runner_input.rep_primary,  # Use preprocessed input
                w1_q=quant_info.w13_weight,
                w2_q=quant_info.w2_weight,
                w1_scale=quant_info.w13_scale,
                w2_scale=quant_info.w2_scale,
                topk_ids=runner_input.topk_ids,
                a_strides1=quant_info.a_strides1,
                b_strides1=quant_info.b_strides1,
                c_strides1=quant_info.c_strides1,
                a_strides2=quant_info.a_strides2,
                b_strides2=quant_info.b_strides2,
                c_strides2=quant_info.c_strides2,
                s_strides13=quant_info.s_strides13,
                s_strides2=quant_info.s_strides2,
                expert_offsets=quant_info.expert_offsets,
                problem_sizes1=quant_info.problem_sizes1,
                problem_sizes2=quant_info.problem_sizes2,
                a1_scale=quant_info.w13_input_scale,
                a2_scale=quant_info.w2_input_scale,
            )
            # Post-processing will be done in post_permute_cutlass_to_standard
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

        # Store combination info for post_permute function
        running_state["c_map"] = runner_input.c_map
        running_state["topk_weights"] = runner_input.topk_weights
        running_state["original_hidden_states_shape"] = runner_input.hidden_states.shape
        running_state["moe_type"] = quant_info.moe_type
        running_state["apply_router_weight_on_input"] = (
            self.config.apply_router_weight_on_input
        )

        return CutlassRunnerOutput(hidden_states=down_output)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.CUTLASS

    def cutlass_w4a8_moe(
        self,
        a: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
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
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.ep_moe.kernels import (
            silu_mul_static_tensorwise_quant_for_cutlass_moe,
        )

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

        Returns:
        - torch.Tensor: The fp8 output tensor after applying the MoE layer.
        """
        # Preprocessing is done in pre_permute_standard_to_cutlass
        # Input 'a' is already the preprocessed gateup_input
        gateup_input = a
        num_local_experts = w1_q.size(0)
        k = w1_q.size(2) * 2  # w1_q is transposed and packed
        n = w2_q.size(2) * 2  # w2_q is transposed and packed
        topk = topk_ids.size(1)
        m = gateup_input.shape[0] // topk  # number of tokens

        # The preprocessing (input reordering) is done in pre_permute_standard_to_cutlass
        # The problem sizes and GEMM operations remain here

        device = a.device

        c1 = torch.empty((m * topk, n * 2), device=device, dtype=torch.bfloat16)
        c2 = torch.empty((m * topk, k), device=device, dtype=torch.bfloat16)
        gateup_input = torch.empty(
            (m * topk, k),
            device=device,
            dtype=torch.float8_e4m3fn,
        )

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

        # Post-processing (reordering back to token order) is done in post_permute_cutlass_to_standard
        return c2

    def cutlass_w4a8_moe_deepep_normal(
        self,
        a: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
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
        - a (torch.Tensor): The preprocessed input tensor (already permuted and quantized to fp8).
            Shape: [M * topk, K]
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

        Returns:
        - torch.Tensor: The fp8 output tensor after applying the MoE layer.
        """
        # Input 'a' is already preprocessed (permuted and quantized to fp8)
        gateup_input = a
        n = w2_q.size(2) * 2  # w2_q is transposed and packed
        k = w1_q.size(2) * 2  # w1_q is transposed and packed

        topk = topk_ids_.size(1)
        m = gateup_input.shape[0] // topk  # number of tokens

        device = a.device
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

        return c2

    def cutlass_w4a8_moe_deepep_ll(
        self,
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
        running_state: Dict[str, torch.Tensor],
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.ep_moe.kernels import (
            silu_and_mul_masked_post_per_tensor_quant_fwd,
        )

        """
        This function computes a w4a8-quantized Mixture of Experts (MoE) layer
        using two sets of quantized weights, w1_q and w2_q, and top-k gating
        mechanism. The matrix multiplications are implemented with CUTLASS
        grouped gemm.

        Parameters:
        - a (torch.Tensor): The preprocessed FP8-quantized input tensor.
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

        Returns:
        - torch.Tensor: The fp8 output tensor after applying the MoE layer.
        """
        # Input 'a' is already preprocessed (quantized to FP8)
        gateup_input = a
        num_experts = w1_q.size(0)
        m = a.size(1)
        k = w1_q.size(2) * 2  # w1_q is transposed and packed
        n = w2_q.size(2) * 2  # w2_q is transposed and packed
        topk = topk_ids_.size(1)

        c1 = torch.empty((num_experts, m, n * 2), device=a.device, dtype=torch.bfloat16)
        c2 = torch.empty((num_experts, m, k), device=a.device, dtype=torch.bfloat16)

        cutlass_w4a8_moe_mm(
            c1,
            gateup_input,
            w1_q,
            a1_scale.float(),
            w1_scale,
            expert_offsets[:-1],
            running_state["problem_sizes1"],
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
            running_state["problem_sizes2"],
            a_strides2,
            b_strides2,
            c_strides2,
            s_strides2,
            128,
            topk,
        )

        return c2

    def cutlass_fused_experts_fp8(
        self,
        a: torch.Tensor,
        w1_q: torch.Tensor,
        w2_q: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
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
        enable_es: Tuple[bool, bool] = (False, False),
    ) -> torch.Tensor:
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        """Performs Fused MoE computation using CUTLASS-like kernels with FP8 weights and activations.

        This function implements a Mixture of Experts (MoE) layer with a SwiGLU/SiLU
        activation, leveraging custom kernels likely derived from CUTLASS principles
        for grouped matrix multiplication (`fp8_blockwise_scaled_grouped_mm`) and
        data preparation (`prepare_moe_input`, `silu_and_mul`).

        It expects preprocessed shuffled and quantized FP8 input activations with per-token scales,
        performs the expert computations using FP8 GEMMs with pre-quantized FP8 weights
        (per-block scales), applies the SiLU activation, and returns the raw expert outputs
        for combination.

        Args:
            a (torch.Tensor): Preprocessed shuffled and FP8-quantized input activations. Shape: `(m * topk, k)`,
                where `m` is the number of tokens and `k` is the hidden size. Expected dtype: `torch.half`
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
            enable_es (tuple(bool, bool)): Flag indicating usage of expert specialization kernel for (up-projection, down-projection)
        Returns:
            torch.Tensor: Raw expert outputs with shape `(m * topk, k)`, where each token's expert assignments are contiguous.

        Raises:
            AssertionError: If input shapes, dtypes, or flags are inconsistent or unsupported.
            NotImplementedError: If CUDA is not available or `sgl_kernel` is not properly installed.
        """
        # Input 'a' is already preprocessed (shuffled and quantized to FP8)
        assert use_fp8_blockscale, "Only support fp8 blockscale for now"

        es_up, es_down = enable_es
        out_dtype = torch.bfloat16  # Default output dtype for FP8 operations
        num_experts = w1_q.size(0)
        m = a.size(0) // topk_ids.size(1)  # Original number of tokens
        topk = topk_ids.size(1)
        k = w1_q.size(1)
        n = w2_q.size(1)

        device = a.device

        # Input 'a' is already quantized and shuffled
        rep_a_q = a
        rep_a1_scales = None  # Scales are handled differently in preprocessed input

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

        return c2

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
        topk_ids: torch.Tensor,
        params: CutlassMoEParams,
    ):
        """
        MoE implementation for FP4 Inputs

        # Gemm 1
        a: Preprocessed FP4 input tensor: [m * topk, k // 2] (uint8, FP4 quantized and reordered)
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
        # Input 'a' is already preprocessed (FP4 quantized and reordered)
        rep_a_fp4 = a
        rep_a_blockscale = None  # Blockscale is not needed for FP4 after preprocessing

        out_dtype = torch.bfloat16  # Default output dtype for FP4 operations
        num_topk = topk_ids.shape[1]
        device = a.device
        m_a = a.size(0) // num_topk  # Original number of tokens
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
        return CutlassRunnerOutput(hidden_states=c2)


@register_pre_permute("standard", "cutlass")
def pre_permute_standard_to_cutlass(
    dispatch_output: StandardDispatchOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> CutlassRunnerInput:
    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )
    topk_weights, topk_ids, _ = topk_output

    device = hidden_states.device
    a_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)
    c_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)

    if quant_info.moe_type == CutlassMoEType.BlockscaledFP8:
        # Validation assertions
        assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
        assert quant_info.w13_weight.dtype == torch.float8_e4m3fn
        assert quant_info.w2_weight.dtype == torch.float8_e4m3fn
        assert (
            hidden_states.shape[1] == quant_info.w13_weight.shape[1]
        ), "Hidden size mismatch w1"
        assert (
            quant_info.w13_weight.shape[2] == quant_info.w2_weight.shape[1] * 2
        ), "Hidden size mismatch w2"
        assert (
            quant_info.w13_weight.shape[0] == quant_info.w2_weight.shape[0]
        ), "Expert number mismatch"
        assert (
            quant_info.w13_weight.shape[0] == quant_info.w2_weight.shape[0]
        ), "Weights expert number mismatch"
        assert (
            quant_info.w13_weight.shape[0] == quant_info.w13_scale.shape[0]
        ), "w1 scales expert number mismatch"
        assert (
            quant_info.w13_weight.shape[0] == quant_info.w2_scale.shape[0]
        ), "w2 scales expert number mismatch"
        assert hidden_states.dtype in [
            torch.half,
            torch.bfloat16,
        ], "Invalid input dtype"

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

        # Do shuffling/reordering and quantization
        shuffled_input = shuffle_rows(hidden_states, a_map, (m * topk_ids.shape[1], k))

        # Quantize the shuffled input
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        a_q, a1_scale = sglang_per_token_group_quant_fp8(shuffled_input, 128)

        rep_primary = a_q
        rep_aux = a1_scale

    elif quant_info.moe_type == CutlassMoEType.BlockscaledFP4:
        params = quant_info.params

        # Store state for post_permute
        running_state["moe_type"] = CutlassMoEType.BlockscaledFP4
        running_state["original_hidden_states_shape"] = hidden_states.shape
        running_state["apply_router_weight_on_input"] = (
            runner_config.apply_router_weight_on_input
        )

        # Pre-permutation validation logic
        assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
        assert quant_info.w13_weight.dtype == torch.uint8, "weight 1 must be uint8"
        assert quant_info.w2_weight.dtype == torch.uint8, "weight 2 must be uint8"
        assert (
            quant_info.w13_weight.ndim == 3
            and quant_info.w2_weight.ndim == 3
            and quant_info.w1_blockscale.ndim == 3
            and quant_info.w2_blockscale.ndim == 3
        ), "All Weights must be of rank 3 for cutlass_moe_fp4"
        m_a, k_a = hidden_states.shape
        e_w1, nx2_w1, half_k_w1 = quant_info.w13_weight.shape
        e_w2, k_w2, half_n_w2 = quant_info.w2_weight.shape

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
        assert hidden_states.dtype in [
            torch.half,
            torch.bfloat16,
        ], "Invalid input dtype"

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

    elif quant_info.moe_type == CutlassMoEType.W4A8:
        from sglang.srt.distributed.parallel_state import (
            get_moe_expert_parallel_world_size,
        )
        from sglang.srt.layers.moe.ep_moe.kernels import (
            cutlass_w4_run_moe_ep_preproess,
            deepep_ll_get_cutlass_w4a8_moe_mm_data,
            deepep_permute_triton_kernel,
            deepep_run_moe_deep_preprocess,
            pre_reorder_for_cutlass_moe,
            silu_and_mul_masked_post_per_tensor_quant_fwd,
        )

        # Store state for post_permute
        running_state["moe_type"] = CutlassMoEType.W4A8
        running_state["original_hidden_states_shape"] = hidden_states.shape

        # Pre-permutation validation logic for W4A8
        assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
        assert quant_info.w13_weight.dtype == torch.int8
        assert quant_info.w2_weight.dtype == torch.int8
        assert (
            hidden_states.shape[1] // 2 == quant_info.w13_weight.shape[2]
        ), "Hidden size mismatch w1"
        assert (
            quant_info.w13_weight.shape[2] * 2 == quant_info.w2_weight.shape[1]
        ), "Hidden size mismatch w2"
        assert (
            quant_info.w13_weight.shape[0] == quant_info.w2_weight.shape[0]
        ), "Expert number mismatch"
        assert (
            quant_info.w13_weight.shape[0] == quant_info.w13_scale.shape[0]
        ), "w1 scales expert number mismatch"
        assert (
            quant_info.w13_weight.shape[0] == quant_info.w2_scale.shape[0]
        ), "w2 scales expert number mismatch"

        num_local_experts = quant_info.w13_weight.size(0)
        m = hidden_states.size(0)
        k = quant_info.w13_weight.size(2) * 2  # w1_q is transposed and packed
        n = quant_info.w2_weight.size(2) * 2  # w2_q is transposed and packed
        topk = topk_ids.size(1)

        running_state["topk"] = topk
        running_state["num_local_experts"] = num_local_experts

        if runner_config.apply_router_weight_on_input:
            assert (
                topk == 1
            ), "apply_router_weight_on_input is only implemented for topk=1"

        if get_moe_expert_parallel_world_size() > 1:
            topk_ids = torch.where(topk_ids == -1, num_local_experts, topk_ids)

        src2dst = cutlass_w4_run_moe_ep_preproess(topk_ids)
        running_state["src2dst"] = src2dst
        running_state["topk_ids"] = topk_ids
        running_state["topk_weights"] = topk_weights

        gateup_input = torch.empty(
            (m * topk, k),
            device=device,
            dtype=torch.float8_e4m3fn,
        )

        pre_reorder_for_cutlass_moe(
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            quant_info.a1_gscale,
            num_local_experts,
            topk,
            m,
            k,
        )

        # NOTE: a_map and c_map are not used in the get_cutlass_w4a8_moe_mm_data kernel,
        # they are kept to allow for a quick switch of the permutation logic
        # from the current triton kernel implementation to the cutlass-based one if needed.

        get_cutlass_w4a8_moe_mm_data(
            topk_ids,
            quant_info.expert_offsets,
            quant_info.problem_sizes1,
            quant_info.problem_sizes2,
            a_map,
            c_map,
            num_local_experts,
            n,
            k,
        )

        rep_primary = gateup_input
        rep_aux = None

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
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    down_output = runner_output.hidden_states
    moe_type = running_state["moe_type"]

    if moe_type == CutlassMoEType.BlockscaledFP8:
        num_tokens, hidden_size = running_state["original_hidden_states_shape"]
        result = torch.empty(
            (num_tokens, hidden_size),
            device=down_output.device,
            dtype=down_output.dtype,
        )
        apply_shuffle_mul_sum(
            down_output,
            result,
            running_state["c_map"],
            running_state["topk_weights"].to(result.dtype),
        )
        hidden_states = result
    elif moe_type == CutlassMoEType.BlockscaledFP4:
        num_tokens = running_state["original_hidden_states_shape"][0]
        topk = running_state["c_map"].shape[0] // num_tokens
        hidden_size = down_output.shape[1]

        reordered = shuffle_rows(
            down_output,
            running_state["c_map"],
            (num_tokens * topk, hidden_size),
        )
        reordered = reordered.view(num_tokens, topk, hidden_size)
        if not running_state["apply_router_weight_on_input"]:
            reordered.mul_(
                running_state["topk_weights"]
                .view(num_tokens, topk, 1)
                .to(reordered.dtype)
            )
        hidden_states = reordered.sum(dim=1).to(down_output.dtype)
    elif moe_type == CutlassMoEType.W4A8:
        from sglang.srt.layers.moe.ep_moe.kernels import post_reorder_for_cutlass_moe

        num_tokens = running_state["original_hidden_states_shape"][0]
        hidden_size = running_state["original_hidden_states_shape"][1]
        topk = running_state["topk"]

        result = torch.empty(
            (num_tokens, hidden_size),
            device=down_output.device,
            dtype=down_output.dtype,
        )

        post_reorder_for_cutlass_moe(
            down_output,
            result,
            running_state["src2dst"],
            running_state["topk_ids"],
            running_state["topk_weights"],
            running_state["num_local_experts"],
            topk,
            num_tokens,
            hidden_size,
            runner_config.routed_scaling_factor or 1.0,
        )

        hidden_states = result
    else:
        raise NotImplementedError(f"Unsupported CUTLASS MoE type: {moe_type}")

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
    from sglang.srt.layers.moe.ep_moe.kernels import (
        deepep_ll_get_cutlass_w4a8_moe_mm_data,
    )

    hidden_states, _, topk_ids, topk_weights, masked_m, expected_m = dispatch_output

    # Store for post_permute
    running_state["topk_ids"] = topk_ids
    running_state["topk_weights"] = topk_weights

    assert quant_info.w13_weight.dtype == torch.int8
    assert quant_info.w2_weight.dtype == torch.int8
    assert (
        hidden_states.shape[2] // 2 == quant_info.w13_weight.shape[2]
    ), "Hidden size mismatch w1"
    assert (
        quant_info.w13_weight.shape[2] * 2 == quant_info.w2_weight.shape[1]
    ), "Hidden size mismatch w2"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w2_weight.shape[0]
    ), "Expert number mismatch"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w13_scale.shape[0]
    ), "w1 scales expert number mismatch"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w2_scale.shape[0]
    ), "w2 scales expert number mismatch"

    assert (
        quant_info.a_strides1.shape[0] == quant_info.w13_weight.shape[0]
    ), "A Strides 1 expert number mismatch"
    assert (
        quant_info.b_strides1.shape[0] == quant_info.w13_weight.shape[0]
    ), "B Strides 1 expert number mismatch"
    assert (
        quant_info.a_strides2.shape[0] == quant_info.w2_weight.shape[0]
    ), "A Strides 2 expert number mismatch"
    assert (
        quant_info.b_strides2.shape[0] == quant_info.w2_weight.shape[0]
    ), "B Strides 2 expert number mismatch"
    num_experts = quant_info.w13_weight.size(0)
    m = hidden_states.size(1)
    k = quant_info.w13_weight.size(2) * 2  # w1_q is transposed and packed
    n = quant_info.w2_weight.size(2) * 2  # w2_q is transposed and packed
    topk = topk_ids.size(1)

    device = hidden_states.device

    problem_sizes1, problem_sizes2 = deepep_ll_get_cutlass_w4a8_moe_mm_data(
        masked_m,
        quant_info.problem_sizes1,
        quant_info.problem_sizes2,
        num_experts,
        n,
        k,
    )

    gateup_input = torch.empty(
        hidden_states.shape, dtype=torch.float8_e4m3fn, device=device
    )
    sgl_per_tensor_quant_fp8(
        hidden_states, gateup_input, quant_info.w13_input_scale.float(), True
    )

    # Store additional state for the main function
    running_state["problem_sizes1"] = problem_sizes1
    running_state["problem_sizes2"] = problem_sizes2

    return CutlassRunnerInput(
        hidden_states=gateup_input,  # Preprocessed and quantized input
        topk_ids=topk_ids,
        masked_m=masked_m,
        expected_m=expected_m,
        rep_primary=gateup_input,  # Use preprocessed quantized input
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
    from sglang.srt.layers.moe.ep_moe.kernels import (
        deepep_permute_triton_kernel,
        deepep_run_moe_deep_preprocess,
    )

    hidden_states, _, topk_ids_, topk_weights, _ = dispatch_output

    # Store state for post_permute
    running_state["topk_ids"] = topk_ids_
    running_state["topk_weights"] = topk_weights

    assert topk_weights.shape == topk_ids_.shape, "topk shape mismatch"
    assert quant_info.w13_weight.dtype == torch.int8
    assert quant_info.w2_weight.dtype == torch.int8
    assert (
        hidden_states.shape[1] // 2 == quant_info.w13_weight.shape[2]
    ), "Hidden size mismatch w1"
    assert (
        quant_info.w13_weight.shape[2] * 2 == quant_info.w2_weight.shape[1]
    ), "Hidden size mismatch w2"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w2_weight.shape[0]
    ), "Expert number mismatch"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w13_scale.shape[0]
    ), "w1 scales expert number mismatch"
    assert (
        quant_info.w13_weight.shape[0] == quant_info.w2_scale.shape[0]
    ), "w2 scales expert number mismatch"

    assert (
        quant_info.a_strides1.shape[0] == quant_info.w13_weight.shape[0]
    ), "A Strides 1 expert number mismatch"
    assert (
        quant_info.b_strides1.shape[0] == quant_info.w13_weight.shape[0]
    ), "B Strides 1 expert number mismatch"
    assert (
        quant_info.a_strides2.shape[0] == quant_info.w2_weight.shape[0]
    ), "A Strides 2 expert number mismatch"
    assert (
        quant_info.b_strides2.shape[0] == quant_info.w2_weight.shape[0]
    ), "B Strides 2 expert number mismatch"
    num_experts = quant_info.w13_weight.size(0)
    m = hidden_states.size(0)
    k = quant_info.w13_weight.size(2) * 2  # w1_q is transposed and packed
    n = quant_info.w2_weight.size(2) * 2  # w2_q is transposed and packed
    topk = topk_ids_.size(1)
    device = hidden_states.device

    reorder_topk_ids, src2dst, _ = deepep_run_moe_deep_preprocess(
        topk_ids_, num_experts
    )
    num_total_tokens = reorder_topk_ids.numel()
    gateup_input_pre_reorder = torch.empty(
        (int(num_total_tokens), hidden_states.shape[1]),
        device=device,
        dtype=hidden_states.dtype,
    )
    deepep_permute_triton_kernel[(hidden_states.shape[0],)](
        hidden_states,
        gateup_input_pre_reorder,
        src2dst,
        topk_ids_.to(torch.int64),
        None,
        topk,
        hidden_states.shape[1],
        BLOCK_SIZE=512,
    )
    gateup_input = torch.empty(
        gateup_input_pre_reorder.shape, dtype=torch.float8_e4m3fn, device=device
    )
    sgl_per_tensor_quant_fp8(
        gateup_input_pre_reorder, gateup_input, quant_info.w13_input_scale.float(), True
    )
    del gateup_input_pre_reorder
    local_topk_ids = topk_ids_
    local_topk_ids = (
        torch.where(local_topk_ids == -1, num_experts, topk_ids_).to(torch.int32)
    ).contiguous()

    a_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)
    c_map = torch.empty((local_topk_ids.numel()), dtype=torch.int32, device=device)

    quant_info.problem_sizes1.zero_()
    quant_info.problem_sizes2.zero_()
    # Safely zero offsets too if the kernel calculates them via cumsum
    quant_info.expert_offsets.zero_()

    get_cutlass_w4a8_moe_mm_data(
        local_topk_ids,
        quant_info.expert_offsets,
        quant_info.problem_sizes1,
        quant_info.problem_sizes2,
        a_map,
        c_map,
        num_experts,
        n,
        k,
    )

    # Store additional state for the main function
    running_state["src2dst"] = src2dst
    running_state["topk"] = topk
    running_state["a_map"] = a_map
    running_state["c_map"] = c_map

    return CutlassRunnerInput(
        hidden_states=gateup_input,
        topk_ids=topk_ids_,
        topk_weights=topk_weights,
        a_map=a_map,
        c_map=c_map,
    )


@register_post_permute("cutlass", "deepep_normal")
def post_permute_cutlass_to_deepep_normal(
    runner_output: CutlassRunnerOutput,
    quant_info: CutlassMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: Dict[str, torch.Tensor],
) -> DeepEPNormalCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput
    from sglang.srt.layers.moe.ep_moe.kernels import deepep_post_reorder_triton_kernel

    c2 = runner_output.hidden_states
    src2dst = running_state["src2dst"]
    topk_ids_ = running_state["topk_ids"]
    topk_weights = running_state["topk_weights"]
    topk = running_state["topk"]
    hidden_size = c2.shape[1]  # Use c2.shape[1] directly

    num_tokens = src2dst.shape[0] // topk
    output = torch.empty(
        (num_tokens, hidden_size),
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
        hidden_size,
        BLOCK_SIZE=512,
    )

    return DeepEPNormalCombineInput(
        hidden_states=output,
        topk_ids=topk_ids_,
        topk_weights=topk_weights,
    )
