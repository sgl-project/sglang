"""FlashInfer-CUTLASS NVFP4 MoE runner core with LoRA support.

Hosts the LoRA-aware analogue of the upstream ``flashinfer_cutlass`` fused
MoE forward (see ``ModelOptNvFp4FusedMoEMethod.apply``).  That upstream
call is a single black-box kernel that performs the entire MoE forward
in one shot, leaving no hook between w13 and w2 to inject a LoRA delta.
We therefore break the forward open into the same FP4 group GEMMs the
primitive uses and insert standard LoRA hooks between them — mirroring
the structure of ``MarlinLoraRunnerCore``.

Layout bridge between the FP4 group GEMM and the LoRA hooks:

* ``cutlass_fp4_group_mm`` writes outputs in **expert-sorted** order
  (``[M*topk, *]`` permuted by ``a_map``).
* The standard ``after_gate_up`` / ``after_down`` hooks (built by
  ``build_lora_hooks``) expect **token-major** ``[M, topk, *]`` layout.

We bridge with ``shuffle_rows(out, c_map, ...)`` to un-sort to token-major
before the hook fires, then re-sort back with ``shuffle_rows(..., a_map)``
for the next FP4 GEMM.  For w2 the un-sort doubles as the combine
permutation we already needed, so only the w13 path pays the extra
``shuffle_rows`` round-trip.

The core is wired in ``MoeRunner.__init__`` under
``runner_backend.is_flashinfer_cutlass() and lora_enabled``.
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
    """Quantization payload consumed by ``CutlassFp4LoraRunnerCore``.

    Mirrors the inputs the upstream ``flashinfer_cutlass_fused_moe`` call
    pulls off the ``FusedMoE`` layer, plus cached ``[E]``-expanded
    per-expert input scales and a flag for the FlashInfer ``[up | gate]``
    w13 loader layout (Kimi-K2.5).
    """

    # Packed FP4 weights (uint8) — viewable as torch.long for the cutlass
    # primitive.
    w13_weight: torch.Tensor  # [E, 2 * N, K // 2] uint8
    w2_weight: torch.Tensor  # [E, K, N // 2]     uint8

    # Block-scale factors (uint8, swizzled).
    w13_blockscale_swizzled: torch.Tensor  # [E, ...]
    w2_blockscale_swizzled: torch.Tensor  # [E, ...]

    # Per-expert GEMM scales (fp32).
    g1_alphas: torch.Tensor  # [E]
    g2_alphas: torch.Tensor  # [E]

    # Per-expert input scales used as quant scales for ``scaled_fp4_experts_quant``.
    # Stored expanded to ``[E]`` once at load time; the original tensor on
    # the FusedMoE layer may be scalar.
    w13_input_scale_expanded: torch.Tensor  # [E] fp32
    w2_input_scale_expanded: torch.Tensor  # [E] fp32

    # ``cutlass_moe_params`` for this layer (reused across forwards so
    # CUDA-graph captures see the same buffer pointers).
    cutlass_moe_params: object

    # Topology.
    num_local_experts: int
    hidden_size: int
    intermediate_size_per_partition: int
    moe_ep_rank: int
    moe_ep_size: int

    # FlashInfer-CUTLASS ``[up | gate]`` w13 layout (Kimi-K2.5).
    w13_swap_halves: bool


class CutlassFp4LoraRunnerCore:
    """LoRA-aware NVFP4 MoE forward on the FlashInfer-CUTLASS GEMM primitives.

    Pipeline:

      1. Local-id / local-weight remap for EP (``a_map`` / ``c_map`` come
         from ``prepare_moe_input``).
      2. Base FP4 GEMM 1 (w13)            via ``cutlass_fp4_group_mm``.
      3. ``shuffle_rows(.., c_map)``      un-sort -> token-major.
      4. ``hooks.after_gate_up(...)``     standard LoRA delta (proven).
      5. ``shuffle_rows(.., a_map)``      re-sort -> expert-sorted.
      6. ``silu_and_mul``                 (or ``[up | gate]`` variant).
      7. Base FP4 GEMM 2 (w2)             via ``cutlass_fp4_group_mm``.
      8. ``shuffle_rows(.., c_map)``      un-sort -> token-major
                                          (also needed for combine).
      9. ``hooks.after_down(...)``        standard LoRA delta (proven).
      10. Combine (topk weighting + reduce).
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
        # ``prepare_moe_input`` expects the *un-doubled* intermediate size as
        # its ``n`` argument — it derives ``problem_sizes1`` (whose ``n`` is
        # ``2 * inter`` for stacked w13) internally.  ``N`` below is the
        # doubled w13 output width used for ``silu_and_mul`` / ``shuffle_rows``.
        inter = quant_info.intermediate_size_per_partition
        N = inter * 2  # gate + up

        params = quant_info.cutlass_moe_params
        offsets = params.expert_offsets
        total_tokens = m_a * num_topk

        # ---------------------------------------- local-id / local-weight remap
        # ``StandardDispatcher`` skips local-id remapping for flashinfer_cutlass
        # (it hands global topk_ids to the kernel which handles EP internally),
        # so we do it here.  Non-local tokens get routed to local expert 0 with
        # weight 0.0 — they pay GEMM cost but contribute zero to the combine.
        local_offset = quant_info.moe_ep_rank * E
        local_ids = topk_ids.to(torch.int32) - local_offset
        non_local = (local_ids < 0) | (local_ids >= E)
        local_ids = local_ids.masked_fill(non_local, 0)
        local_weights = topk_weights.masked_fill(non_local, 0.0)

        # ---------------------------------------- a_map / c_map / problem sizes
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
            inter,  # un-doubled intermediate; the kernel doubles for stacked w13
            K,
            params.blockscale_offsets,
        )

        # ---------------------------------------- GEMM 1 base (w13)
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
        del rep_a_fp4, rep_a_blockscale

        # ---------------------------------------- LoRA w13 delta via hook
        # The hook builder ``build_lora_hooks`` produces closures that operate
        # on token-major ``[M, topk, 2*N]``, matching what Marlin / Triton MoE
        # cores hand them.  ``cutlass_fp4_group_mm`` writes expert-sorted
        # output, so un-sort with ``c_map`` first, then re-sort with
        # ``a_map`` for the next GEMM.
        if hooks is not None and hooks.after_gate_up is not None:
            gateup_token_major = shuffle_rows(gateup_flat, c_map, (total_tokens, N))
            gateup_3d = gateup_token_major.view(m_a, num_topk, N)
            hooks.after_gate_up(hidden_states, gateup_3d, topk_weights, topk_ids)
            gateup_flat = shuffle_rows(
                gateup_token_major.view(total_tokens, N), a_map, (total_tokens, N)
            )
            del gateup_token_major, gateup_3d

        # ---------------------------------------- silu + mul
        if quant_info.w13_swap_halves:
            # ``[up | gate]`` base layout: silu(gate) * up = silu(second_half) * first_half.
            intermediate = (
                torch.nn.functional.silu(gateup_flat[:, N // 2 :].float())
                * gateup_flat[:, : N // 2].float()
            ).to(out_dtype)
        else:
            intermediate = torch.empty(
                total_tokens, N // 2, dtype=out_dtype, device=device
            )
            silu_and_mul(gateup_flat, intermediate)
        del gateup_flat

        # ---------------------------------------- GEMM 2 base (w2)
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
        del int_fp4, int_blockscale

        # ---------------------------------------- un-sort for hook + combine
        # ``shuffle_rows(.., c_map)`` un-sorts to token-major; this is needed
        # both by ``hooks.after_down`` and by the combine reduction below.
        out_token_major = shuffle_rows(out_flat, c_map, (total_tokens, K))
        out_3d = out_token_major.view(m_a, num_topk, K)
        del out_flat

        # ---------------------------------------- LoRA w2 delta via hook
        if hooks is not None and hooks.after_down is not None:
            # ``intermediate`` is the input to the w2 GEMM; pass it as the
            # "down input" the hook expects.  It is still in expert-sorted
            # layout, but the existing ``_add_lora_down_delta`` only uses it
            # for activation shape / dtype and routes via ``topk_ids`` —
            # safe to hand over as-is.
            hooks.after_down(intermediate, out_3d, topk_weights, topk_ids)

        del intermediate

        # ---------------------------------------- combine
        if not runner_config.apply_router_weight_on_input:
            out_3d = out_3d * local_weights.view(m_a, num_topk, 1).to(out_dtype)
        return StandardCombineInput(hidden_states=out_3d.sum(dim=1).to(out_dtype))
