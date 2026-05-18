"""FlashInfer-CUTLASS NVFP4 MoE runner core with LoRA support.

Breaks open the upstream ``flashinfer_cutlass_fused_moe`` (a single black-box
kernel called from ``ModelOptNvFp4FusedMoEMethod.apply``) into its FP4 group
GEMMs so we can inject the standard ``after_gate_up`` / ``after_down`` LoRA
hooks between w13 and w2, mirroring ``MarlinLoraRunnerCore``.

Layout bridge: ``cutlass_fp4_group_mm`` writes expert-sorted ``[M*topk, *]``,
the hooks expect token-major ``[M, topk, *]``. We round-trip via
``shuffle_rows(.., c_map)`` (un-sort) and ``shuffle_rows(.., inv_c_map)``
(re-sort) around ``after_gate_up``; for ``after_down`` the un-sort doubles
as the combine permutation.
"""

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

if _is_cuda:
    from sgl_kernel import silu_and_mul
elif _is_hip:
    from vllm._custom_ops import silu_and_mul


@dataclass
class CutlassFp4MoeQuantInfo(MoeQuantInfo):
    """Quantization payload consumed by ``CutlassFp4LoraRunnerCore``."""

    w13_weight: torch.Tensor  # [E, 2 * N, K // 2] uint8
    w2_weight: torch.Tensor  # [E, K, N // 2]     uint8
    w13_blockscale_swizzled: torch.Tensor
    w2_blockscale_swizzled: torch.Tensor
    g1_alphas: torch.Tensor  # [E] fp32
    g2_alphas: torch.Tensor  # [E] fp32
    # ``[E]``-expanded once at load time so the hot path never has to expand.
    w13_input_scale_expanded: torch.Tensor  # [E] fp32
    w2_input_scale_expanded: torch.Tensor  # [E] fp32
    cutlass_moe_params: object
    num_local_experts: int
    hidden_size: int
    intermediate_size_per_partition: int
    moe_ep_rank: int
    moe_ep_size: int
    # FlashInfer-CUTLASS loads w13 as ``[up | gate]`` (Kimi-K2.5).
    w13_swap_halves: bool


class CutlassFp4LoraRunnerCore:
    """LoRA-aware NVFP4 MoE forward on the FlashInfer-CUTLASS GEMM primitives.

    Pipeline: EP local-id remap → FP4 GEMM 1 → un-sort → ``after_gate_up`` →
    re-sort → silu_and_mul → FP4 GEMM 2 → un-sort → ``after_down`` → combine.
    """

    def __init__(self, config: MoeRunnerConfig):
        self.config = config

    def run_from_dispatch(
        self,
        dispatch_output: "StandardDispatchOutput",
        quant_info: CutlassFp4MoeQuantInfo,
        runner_config: MoeRunnerConfig,
        hooks: Optional["LoRAHooks"] = None,
    ) -> "StandardCombineInput":
        from sgl_kernel import prepare_moe_input

        from sglang.srt.layers.moe.cutlass_moe import (
            cutlass_fp4_group_mm,
            scaled_fp4_experts_quant,
            shuffle_rows,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
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
        # to local for the per-rank GEMM. Non-local tokens go to local expert
        # 0 with weight 0 — they pay GEMM cost but contribute nothing.
        local_offset = quant_info.moe_ep_rank * E
        local_ids = topk_ids.to(torch.int32) - local_offset
        non_local = (local_ids < 0) | (local_ids >= E)
        local_ids = local_ids.masked_fill(non_local, 0)
        local_weights = topk_weights.masked_fill(non_local, 0.0)

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

        # ---- LoRA w13 delta
        # Un-sort to token-major with ``c_map`` (pair-index permutation), run
        # the hook, then re-sort with ``inv_c_map``. ``a_map`` is an M-index
        # and is NOT a valid inverse for the pair-index ``[M*topk, *]`` tensor
        # — using it silently corrupts every (token, k>0) row.
        if hooks is not None and hooks.after_gate_up is not None:
            inv_c_map = torch.empty_like(c_map)
            inv_c_map[c_map.long()] = torch.arange(
                total_tokens, device=c_map.device, dtype=c_map.dtype
            )
            gateup_token_major = shuffle_rows(gateup_flat, c_map, (total_tokens, N))
            gateup_3d = gateup_token_major.view(m_a, num_topk, N)
            hooks.after_gate_up(hidden_states, gateup_3d, topk_weights, topk_ids)
            gateup_flat = shuffle_rows(
                gateup_token_major.view(total_tokens, N),
                inv_c_map,
                (total_tokens, N),
            )

        # ---- silu + mul
        if quant_info.w13_swap_halves:
            # ``[up | gate]`` layout: silu(gate) * up = silu(second) * first.
            intermediate = (
                torch.nn.functional.silu(gateup_flat[:, N // 2 :].float())
                * gateup_flat[:, : N // 2].float()
            ).to(out_dtype)
        else:
            intermediate = torch.empty(
                total_tokens, N // 2, dtype=out_dtype, device=device
            )
            silu_and_mul(gateup_flat, intermediate)

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

        # Un-sort to token-major; needed by ``after_down`` and by the combine.
        out_3d = shuffle_rows(out_flat, c_map, (total_tokens, K)).view(m_a, num_topk, K)

        # Standard hook contract: the down hook adds a router-weighted delta
        # into an already-router-weighted per-expert output.
        if not runner_config.apply_router_weight_on_input:
            out_3d = out_3d * local_weights.view(m_a, num_topk, 1).to(out_dtype)

        # ---- LoRA w2 delta
        if hooks is not None and hooks.after_down is not None:
            intermediate_token_major = shuffle_rows(
                intermediate, c_map, (total_tokens, N // 2)
            )
            hooks.after_down(intermediate_token_major, out_3d, local_weights, topk_ids)

        return StandardCombineInput(hidden_states=out_3d.sum(dim=1))
