"""Element-wise equivalence check between
    sglang's torch_ref_w4a16_moe_forward (debug_utils/w4a16_moe_ref_related.py)
and
    flashinfer's cutlass_fused_moe(use_w4_group_scaling=True, swiglu_limit=...)
on tiny random MXFP4 weights.

Purpose: before running a full DSv4 bench with
SGLANG_HACK_DEBUG_W4A16_USE_TORCH_REF=1 (which takes many hours per seed), we
want a fast smoke that the torch ref matches the kernel on tiny shapes. If
this smoke diverges, the bench-scale acc numbers are not comparable.

Run (needs CUDA + flashinfer PR #3084 installed):
    uv run python sunrise/verify_torch_ref_w4a16_moe.py
"""

from __future__ import annotations

import torch
from flashinfer.fused_moe import (
    cutlass_fused_moe,
    interleave_moe_scales_for_sm90_mixed_gemm,
    interleave_moe_weights_for_sm90_mixed_gemm,
)

from sglang.srt.debug_utils.w4a16_moe_ref_related import torch_ref_cutlass_fused_moe
from sglang.srt.layers.quantization.w4a16_deepseek import _dequant_mxfp4

# (batch_size, hidden_size, num_experts, top_k, intermediate_size, swiglu_limit)
# Shapes borrowed from flashinfer's W4A16_CORRECTNESS_CONFIGS so we stay inside
# the kernel's supported envelope. Swiglu limit is the DSv4 260415 value.
SHAPES = [
    (4, 128, 4, 2, 128, 10.0),
    (4, 768, 8, 2, 512, 10.0),
    (4, 2048, 8, 4, 1024, 10.0),
    (4, 4096, 8, 4, 2048, 10.0),
]


def _compute_routing(
    router_logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = probs.topk(top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(router_logits.dtype), topk_ids.to(torch.int64)


def _run_one(
    batch_size: int,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    swiglu_limit_val: float,
) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    m, k, e, n = batch_size, hidden_size, num_experts, intermediate_size

    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)

    w13_fp4 = torch.randint(
        0, 256, (e, 2 * n, k // 2), device=device, dtype=torch.uint8
    )
    w2_fp4 = torch.randint(
        0, 256, (e, k, n // 2), device=device, dtype=torch.uint8
    )
    w13_scale = torch.randint(
        118, 123, (e, 2 * n, k // 32), device=device, dtype=torch.uint8
    )
    w2_scale = torch.randint(
        118, 123, (e, k, n // 32), device=device, dtype=torch.uint8
    )

    router_logits = torch.randn(m, e, dtype=torch.bfloat16, device=device)
    topk_weights, topk_ids = _compute_routing(router_logits, top_k)

    swiglu_limit_tensor = torch.full(
        (e,), swiglu_limit_val, dtype=torch.float32, device=device
    )

    # --- Flashinfer kernel path ---
    w13_il = interleave_moe_weights_for_sm90_mixed_gemm(w13_fp4, "fp4")
    w2_il = interleave_moe_weights_for_sm90_mixed_gemm(w2_fp4, "fp4")
    w13_scale_il = interleave_moe_scales_for_sm90_mixed_gemm(w13_scale, group_size=32)
    w2_scale_il = interleave_moe_scales_for_sm90_mixed_gemm(w2_scale, group_size=32)

    flash_output = torch.zeros_like(x)
    cutlass_fused_moe(
        input=x,
        token_selected_experts=topk_ids.to(torch.int32).contiguous(),
        token_final_scales=topk_weights.to(torch.float32).contiguous(),
        fc1_expert_weights=w13_il,
        fc2_expert_weights=w2_il,
        output_dtype=torch.bfloat16,
        quant_scales=[w13_scale_il.view(torch.int32), w2_scale_il.view(torch.int32)],
        swiglu_limit=swiglu_limit_tensor,
        ep_size=1,
        ep_rank=0,
        tp_size=1,
        tp_rank=0,
        use_w4_group_scaling=True,
        output=flash_output,
    )

    # --- Torch ref path ---
    # Kernel and ref see the same raw FP4 tensor; ref's chunk(dim=0) gives
    # (w3, w1) which matches flashinfer's own reference convention in
    # _run_w4a16_moe_hopper. No explicit reorder needed on either side.
    w13_bf16 = _dequant_mxfp4(w13_fp4, w13_scale)
    w2_bf16 = _dequant_mxfp4(w2_fp4, w2_scale)

    ref_output = torch.zeros_like(x)
    torch_ref_cutlass_fused_moe(
        input=x,
        token_selected_experts=topk_ids.to(torch.int32).contiguous(),
        token_final_scales=topk_weights.to(torch.float32).contiguous(),
        fc1_expert_weights=w13_bf16,
        fc2_expert_weights=w2_bf16,
        output_dtype=torch.bfloat16,
        swiglu_limit=swiglu_limit_tensor,
        ep_size=1,
        ep_rank=0,
        output=ref_output,
    )

    # Compare
    diff = (ref_output.float() - flash_output.float()).abs()
    tol = 0.1 + 1e-1 * ref_output.float().abs()
    close_pct = (diff <= tol).float().mean().item()
    max_abs = diff.max().item()
    print(
        f"m={m} k={k} e={e} top_k={top_k} n={n} limit={swiglu_limit_val} "
        f"close%={close_pct:.4%} max_abs={max_abs:.4f}"
    )
    assert close_pct >= 0.99, (
        f"torch-ref vs kernel mismatch: only {close_pct:.4%} within tol; "
        f"max_abs={max_abs:.4f}"
    )


def main() -> None:
    for cfg in SHAPES:
        _run_one(*cfg)
    print("ALL PASS")


if __name__ == "__main__":
    main()
