"""NVFP4 MoE runner that exposes ``after_gate_up`` / ``after_down`` LoRA hooks
between the W13 and W2 group GEMMs of FlashInfer-CUTLASS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
)
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.lora.lora_moe_runners import LoRAHooks


_is_cuda = is_cuda()
_is_hip = is_hip()

# Use the JIT silu_and_mul: it carries the ``swap_halves`` flag (the AOT
# sgl_kernel build doesn't expose it yet) for Kimi-K2.5's ``[Up|Gate]`` W13.
if _is_cuda:
    from sglang.jit_kernel.activation import silu_and_mul
elif _is_hip:
    # SM100+ only; this import is here for symmetry and should never run.
    from vllm._custom_ops import silu_and_mul


@dataclass
class CutlassFp4MoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor  # [E, 2*N, K // 2] uint8
    w2_weight: torch.Tensor  # [E, K, N // 2]   uint8
    w13_blockscale_swizzled: torch.Tensor
    w2_blockscale_swizzled: torch.Tensor
    g1_alphas: torch.Tensor  # [E] fp32
    g2_alphas: torch.Tensor  # [E] fp32
    # [E]-expanded once at load time so the hot path doesn't broadcast.
    w13_input_scale_expanded: torch.Tensor  # [E] fp32
    w2_input_scale_expanded: torch.Tensor  # [E] fp32
    cutlass_moe_params: object
    num_local_experts: int
    hidden_size: int
    intermediate_size_per_partition: int
    moe_ep_rank: int
    # ``[Up|Gate]`` W13 layout (Kimi-K2.5): silu the second half, multiply the first.
    w13_swap_halves: bool


class CutlassFp4LoraRunnerCore:
    """LoRA-aware NVFP4 MoE forward on the FlashInfer-CUTLASS GEMM primitives."""

    def __init__(self, config: MoeRunnerConfig):
        # config is unused today; runner_config is passed per-call.
        pass

    def run_from_dispatch(
        self,
        dispatch_output: "StandardDispatchOutput",
        quant_info: CutlassFp4MoeQuantInfo,
        runner_config: MoeRunnerConfig,
        hooks: Optional["LoRAHooks"] = None,
        lora_info=None,
    ) -> "StandardCombineInput":
        from sgl_kernel import apply_shuffle_mul_sum, prepare_moe_input

        from sglang.srt.layers.moe.cutlass_moe import (
            cutlass_fp4_group_mm,
            scaled_fp4_experts_quant,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        # FP4 all-gather dispatch (gated by should_use_flashinfer_cutlass_moe_fp4_allgather)
        # delivers pre-packed FP4 input; this runner re-quantizes bf16 only.
        if getattr(dispatch_output, "hidden_states_scale", None) is not None:
            raise NotImplementedError(
                "CutlassFp4LoraRunnerCore does not support the FP4 all-gather "
                "dispatch path; disable it for LoRA configurations."
            )

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

        m_a = hidden_states.shape[0]
        num_topk = topk_ids.shape[1]
        out_dtype = hidden_states.dtype
        device = hidden_states.device
        E = quant_info.num_local_experts
        K = quant_info.hidden_size
        inter = quant_info.intermediate_size_per_partition
        # This LoRA runner injects between the gate/up projection and the
        # SwiGLU activation, so it only supports gated MoE weights.
        N = quant_info.w13_weight.shape[1]
        if N != inter * 2:
            raise NotImplementedError(
                "CutlassFp4LoraRunnerCore expects gated NVFP4 MoE weights with "
                f"w13 output dim {inter * 2}, but got {N}."
            )
        params = quant_info.cutlass_moe_params
        offsets = params.expert_offsets
        total_tokens = m_a * num_topk

        # StandardDispatcher hands flashinfer_cutlass global topk_ids; remap
        # to local. Non-local tokens go to local expert 0 with weight 0.
        local_offset = quant_info.moe_ep_rank * E
        local_ids = topk_ids.to(torch.int32) - local_offset
        non_local = (local_ids < 0) | (local_ids >= E)
        local_ids = local_ids.masked_fill(non_local, 0)
        local_weights = topk_weights.to(torch.float32).masked_fill(non_local, 0.0)

        a_map = torch.empty(total_tokens, dtype=torch.int32, device=device)
        c_map = torch.empty(total_tokens, dtype=torch.int32, device=device)
        prepare_moe_input(
            local_ids,
            offsets,
            params.problem_sizes1,
            params.problem_sizes2,
            a_map,
            c_map,
            E,
            inter,
            K,
            params.blockscale_offsets,
        )

        # ---- GEMM 1 (w13)
        rep_a_fp4, rep_a_blockscale = scaled_fp4_experts_quant(
            hidden_states,
            quant_info.w13_input_scale_expanded,
            offsets,
            params.blockscale_offsets,
            num_topk,
            expert_map=a_map,
        )
        gateup_flat = cutlass_fp4_group_mm(
            rep_a_fp4,
            quant_info.w13_weight,
            rep_a_blockscale,
            quant_info.w13_blockscale_swizzled,
            quant_info.g1_alphas,
            out_dtype,
            params.to_gemm1_args(),
        )

        # Hand the LoRA hooks the c_map so their kernels read/write expert-sorted
        # rows directly, skipping the token-major round-trips.
        if lora_info is not None:
            lora_info.c_map = c_map
            lora_info.sorted_layout = True

        # ---- LoRA w13 delta
        if hooks is not None and hooks.after_gate_up is not None:
            gateup_3d = gateup_flat.view(m_a, num_topk, N)
            hooks.after_gate_up(hidden_states, gateup_3d, topk_weights, topk_ids)

        # ---- silu + mul
        # ``w13_swap_halves=True`` selects the ``[up | gate]`` convention
        # (silu(second) * first) for FlashInfer-CUTLASS NVFP4 W13 loaders.
        intermediate = torch.empty(total_tokens, N // 2, dtype=out_dtype, device=device)
        silu_and_mul(gateup_flat, intermediate, swap_halves=quant_info.w13_swap_halves)

        # ---- GEMM 2 (w2)
        int_fp4, int_blockscale = scaled_fp4_experts_quant(
            intermediate,
            quant_info.w2_input_scale_expanded,
            offsets,
            params.blockscale_offsets,
            num_topk,
        )
        out_flat = cutlass_fp4_group_mm(
            int_fp4,
            quant_info.w2_weight,
            int_blockscale,
            quant_info.w2_blockscale_swizzled,
            quant_info.g2_alphas,
            out_dtype,
            params.to_gemm2_args(),
        )

        # ---- LoRA w2 delta. Sorted-layout: hook writes unweighted delta into
        # out_flat; router weighting happens once in the combine below.
        if hooks is not None and hooks.after_down is not None:
            out_3d_sorted_view = out_flat.view(m_a, num_topk, K)
            hooks.after_down(intermediate, out_3d_sorted_view, local_weights, topk_ids)

        # ---- combine: un-sort, weight (base + delta), sum. Router weights stay
        # fp32 to match FlashInfer-CUTLASS fused MoE's final accumulation.
        output = torch.empty((m_a, K), dtype=out_dtype, device=device)
        apply_shuffle_mul_sum(
            out_flat,
            output,
            c_map,
            (
                None
                if runner_config.apply_router_weight_on_input
                else local_weights.reshape(-1)
            ),
        )
        return StandardCombineInput(hidden_states=output)
