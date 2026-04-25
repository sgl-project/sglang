"""
Pure-torch MoE ref for W4A16 acc investigation.

The body of ``torch_ref_cutlass_fused_moe`` adapts
``_compute_with_active_experts`` from flashinfer-sunrise PR #3084 at
    tests/moe/test_trtllm_cutlass_fused_moe.py
    (commit 77746b81, lines 2458-2491)
into a drop-in replacement for ``flashinfer.fused_moe.cutlass_fused_moe``:
identical signature, identical in-place output semantics. Only the subset of
kwargs actually needed to reproduce DSv4 W4A16 numerics is consumed; the rest
are accepted and ignored.

Expectation on weights: both ``fc1_expert_weights`` and ``fc2_expert_weights``
are already bf16 (caller dequanted FP4+UE8M0 up front — see
``DeepSeekW4A16MoEMethod.process_weights_after_loading`` under
``SGLANG_HACK_DEBUG_W4A16_USE_TORCH_REF=1``). ``quant_scales`` /
``use_w4_group_scaling`` are accepted for signature parity but unused.

Activation: we reproduce the kernel's behavior when only ``swiglu_limit`` is
passed (no ``swiglu_alpha``/``swiglu_beta``) — SiLU on the clamped gate,
symmetric clamp on the up. This matches the sglang ``w4a16_deepseek.py``
apply() kernel call.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F


def torch_ref_cutlass_fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: torch.Tensor,
    fc1_expert_weights: torch.Tensor,
    fc2_expert_weights: torch.Tensor,
    output_dtype: torch.dtype,
    quant_scales: Optional[List[torch.Tensor]] = None,
    fc1_expert_biases: Optional[torch.Tensor] = None,
    fc2_expert_biases: Optional[torch.Tensor] = None,
    input_sf: Optional[torch.Tensor] = None,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    output: Optional[torch.Tensor] = None,
    enable_alltoall: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4_group_scaling: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    use_packed_weights: bool = False,
    tune_max_num_tokens: int = 8192,
    enable_pdl: Optional[bool] = None,
    activation_type=None,
    swizzled_input_sf: bool = True,
) -> torch.Tensor:
    """Pure-torch drop-in for flashinfer ``cutlass_fused_moe`` (W4A16 path).

    Consumed args: ``input``, ``token_selected_experts``,
    ``token_final_scales``, ``fc1_expert_weights``, ``fc2_expert_weights``
    (both bf16), ``output_dtype``, ``swiglu_limit``, ``ep_size``, ``ep_rank``,
    ``output``. Everything else is ignored.
    """
    del (
        quant_scales,
        fc1_expert_biases,
        fc2_expert_biases,
        input_sf,
        swiglu_alpha,
        swiglu_beta,
        tp_size,
        tp_rank,
        cluster_size,
        cluster_rank,
        enable_alltoall,
        use_deepseek_fp8_block_scale,
        use_w4_group_scaling,
        use_mxfp8_act_scaling,
        min_latency_mode,
        use_packed_weights,
        tune_max_num_tokens,
        enable_pdl,
        activation_type,
        swizzled_input_sf,
    )

    assert fc1_expert_weights.dtype == torch.bfloat16, (
        f"torch-ref expects bf16 weights, got {fc1_expert_weights.dtype}"
    )
    assert fc2_expert_weights.dtype == torch.bfloat16, (
        f"torch-ref expects bf16 weights, got {fc2_expert_weights.dtype}"
    )

    num_tokens = input.shape[0]
    hidden = fc2_expert_weights.shape[1]
    num_local_experts = fc1_expert_weights.shape[0]
    local_expert_offset = ep_rank * num_local_experts

    if output is None:
        output = torch.empty(
            num_tokens, hidden, dtype=output_dtype, device=input.device
        )
    output.zero_()

    topk_ids_local = token_selected_experts.long() - local_expert_offset
    in_range = (topk_ids_local >= 0) & (topk_ids_local < num_local_experts)
    if not in_range.any():
        return output

    active_local = torch.unique(topk_ids_local[in_range])
    for eid_local in active_local.tolist():
        mask = (topk_ids_local == eid_local) & in_range
        tok_idx, nth = torch.where(mask)
        if tok_idx.numel() == 0:
            continue

        w31 = fc1_expert_weights[eid_local]
        w3, w1 = torch.chunk(w31, 2, dim=0)
        w2 = fc2_expert_weights[eid_local]

        expert_in = input[tok_idx]
        x1 = expert_in @ w1.t()
        x3 = expert_in @ w3.t()

        if swiglu_limit is not None:
            limit = swiglu_limit[eid_local]
            x1 = x1.clamp(max=limit)
            x3 = x3.clamp(min=-limit, max=limit)

        inter = F.silu(x1) * x3
        out = inter @ w2.t()

        weight = token_final_scales[tok_idx, nth, None].to(out.dtype)
        output.index_add_(0, tok_idx, (weight * out).to(output.dtype))

    return output
