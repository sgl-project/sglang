from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.utils import cache_once, is_arch_support_pdl, load_jit
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_SCORING_FUNC_MAP = {
    "sigmoid": 0,
    "sqrtsoftplus": 1,
    "softmax": 2,
}


@cache_once
def _jit_moe_fused_gate_module() -> Module:
    return load_jit(
        "moe_fused_gate",
        cuda_files=["moe/moe_fused_gate.cuh"],
        cuda_wrappers=[("moe_fused_gate", "MoEFusedGateKernel::run")],
    )


@cache_once
def can_use_moe_fused_gate() -> bool:
    logger = logging.getLogger(__name__)
    try:
        _jit_moe_fused_gate_module()
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT MoE fused gate kernel: {e}")
        return False


def moe_fused_gate_jit(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scoring_func_int = _SCORING_FUNC_MAP.get(scoring_func.lower())
    assert (
        scoring_func_int is not None
    ), f"Unknown scoring_func '{scoring_func}', must be one of {list(_SCORING_FUNC_MAP.keys())}"

    assert input.dtype == torch.float32, "input must be float32"
    assert bias.dtype == torch.float32, "bias must be float32"
    assert input.ndim == 2, "input must be 2D"
    assert bias.ndim == 1, "bias must be 1D"
    assert input.size(1) == bias.size(0), "input and bias must have same num_experts"
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"

    num_rows, _ = input.shape
    device = input.device

    output = torch.empty(num_rows, topk, dtype=torch.float32, device=device)
    indices = torch.empty(num_rows, topk, dtype=torch.int32, device=device)

    module = _jit_moe_fused_gate_module()
    module.moe_fused_gate(
        input,
        bias,
        output,
        indices,
        topk,
        scoring_func_int,
        num_fused_shared_experts,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )

    return output, indices


@triton.jit
def _router_triton_kernel(
    scores_ptr,  # [M, N] fp32, GEMM output (raw logits)
    bias_ptr,  # [N]    fp32
    out_weights_ptr,  # [M, K] fp32
    out_indices_ptr,  # [M, K] int32
    M,
    routed_scaling_factor,
    moe_softcapping,
    N: tl.constexpr,
    K: tl.constexpr,  # total topk (includes fused shared experts)
    K_ROUTED: tl.constexpr,  # K - num_fused_shared_experts
    BLOCK_N: tl.constexpr,  # >= N, power of 2
    BLOCK_K: tl.constexpr,  # >= K, power of 2
    SCORING_FUNC: tl.constexpr,  # 0 = sigmoid, 1 = sqrtsoftplus, 2 = softmax
    HAS_SOFTCAP: tl.constexpr,  # tanh softcapping (softmax only)
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,  # apply_routed_scaling_factor_on_output
    USE_PDL: tl.constexpr,
    stride_sm,
    stride_sn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
) -> None:
    pid = tl.program_id(0)
    if pid >= M:
        return

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    # prefetch bias before PDL wait
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    row_ptr = scores_ptr + pid * stride_sm + offs_n * stride_sn
    scores = tl.load(row_ptr, mask=mask_n, other=0.0).to(tl.float32)

    if SCORING_FUNC == 0:
        # sigmoid(x) = 1 / (1 + exp(-x)); bias is for ranking only, weight is bias-free.
        activated = tl.sigmoid(scores)
        biased = activated + bias
    elif SCORING_FUNC == 1:
        # sqrt(softplus(x)) = sqrt(log1p(exp(x))); guard against overflow when x is large.
        sp = tl.where(
            scores > 20.0,
            scores,  # log1p(exp(big)) = big
            tl.log(1.0 + tl.exp(scores)),
        )
        activated = tl.sqrt(sp)
        biased = activated + bias
    else:
        # softmax over the row: weight is the softmax probability (bias kept), with
        # optional tanh softcapping. Ranking by the (softcapped, biased) logit is
        # monotonic with the softmax prob, so the topk loop below ranks on `biased`.
        logit = scores
        if HAS_SOFTCAP:
            # tanh(z) = 2*sigmoid(2z) - 1 (avoids relying on tl.math.tanh availability).
            z = logit / moe_softcapping
            logit = moe_softcapping * (2.0 * tl.sigmoid(2.0 * z) - 1.0)
        biased = logit + bias
        biased = tl.where(mask_n, biased, -float("inf"))
        row_max = tl.max(biased, axis=0)
        exp_row = tl.where(mask_n, tl.exp(biased - row_max), 0.0)
        row_sum = tl.sum(exp_row, axis=0)
        activated = exp_row / row_sum

    biased = tl.where(mask_n, biased, -float("inf"))
    offs_k = tl.arange(0, BLOCK_K)
    mask_k_total = offs_k < K
    mask_k_routed = offs_k < K_ROUTED
    selected_vals = tl.zeros([BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([BLOCK_K], dtype=tl.int32)

    cur = biased
    for k in tl.static_range(K_ROUTED):
        max_val = tl.max(cur, axis=0)
        is_max = cur == max_val
        lane_id = tl.where(is_max, offs_n, N + 1)
        win_lane = tl.min(lane_id, axis=0).to(tl.int32)
        win_activated = tl.sum(tl.where(offs_n == win_lane, activated, 0.0), axis=0)
        slot = offs_k == k
        selected_vals = tl.where(slot, win_activated, selected_vals)
        selected_idx = tl.where(slot, win_lane, selected_idx)
        cur = tl.where(offs_n == win_lane, -float("inf"), cur)

    routed_sum = tl.sum(tl.where(mask_k_routed, selected_vals, 0.0), axis=0)

    # Fill fused-shared-expert slots: weight = routed_sum / routed_scaling_factor,
    # id = num_experts + (slot - K_ROUTED).
    if K_ROUTED < K:
        is_shared = (offs_k >= K_ROUTED) & mask_k_total
        shared_weight = routed_sum / routed_scaling_factor
        shared_idx = N + (offs_k - K_ROUTED)
        selected_vals = tl.where(is_shared, shared_weight, selected_vals)
        selected_idx = tl.where(is_shared, shared_idx, selected_idx)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    if RENORMALIZE:
        norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)
        selected_vals = selected_vals / norm
    if APPLY_SCALE:
        selected_vals = selected_vals * routed_scaling_factor

    out_w_ptr = out_weights_ptr + pid * stride_wm + offs_k * stride_wk
    out_i_ptr = out_indices_ptr + pid * stride_im + offs_k * stride_ik
    tl.store(out_w_ptr, selected_vals, mask=mask_k_total)
    tl.store(out_i_ptr, selected_idx, mask=mask_k_total)


@debug_kernel_api
def moe_fused_gate(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    moe_softcapping: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton fused router: scoring + bias + topk + (optional) renorm/scale.

    Mirrors the semantics of :func:`moe_fused_gate_jit` (the CUDA JIT kernel)
    for the ungrouped case (``num_expert_group == 1``). The first argument is
    named ``scores`` (raw GEMM logits) to match the existing call sites.
    """
    scoring_func_int = _SCORING_FUNC_MAP.get(scoring_func.lower())
    assert (
        scoring_func_int is not None
    ), f"Unknown scoring_func '{scoring_func}', must be one of {list(_SCORING_FUNC_MAP.keys())}"
    assert scores.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), "scores must be float32/float16/bfloat16"
    assert bias.dtype == torch.float32, "bias must be float32"
    assert scores.ndim == 2, "scores must be 2D"
    assert bias.ndim == 1, "bias must be 1D"
    assert scores.size(1) == bias.size(0), "scores and bias must have same num_experts"
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"

    M, N = scores.shape
    K = topk
    K_routed = topk - num_fused_shared_experts

    weights = torch.empty((M, K), dtype=torch.float32, device=scores.device)
    indices = torch.empty((M, K), dtype=torch.int32, device=scores.device)

    BLOCK_N = triton.next_power_of_2(N)  # 256 -> 256, 384 -> 512
    BLOCK_K = triton.next_power_of_2(K)  # 6 -> 8, 8 -> 8
    grid = (M,)
    use_pdl = is_arch_support_pdl()
    extra = {"launch_pdl": True} if use_pdl else {}
    # A single warp keeps the per-row reductions cheap to synchronize.
    _router_triton_kernel[grid](
        scores,
        bias,
        weights,
        indices,
        M,
        float(routed_scaling_factor),
        float(moe_softcapping),
        N=N,
        K=K,
        K_ROUTED=K_routed,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SCORING_FUNC=scoring_func_int,
        HAS_SOFTCAP=bool(moe_softcapping != 0.0),
        RENORMALIZE=bool(renormalize),
        APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
        USE_PDL=use_pdl,
        stride_sm=scores.stride(0),
        stride_sn=scores.stride(1),
        stride_wm=weights.stride(0),
        stride_wk=weights.stride(1),
        stride_im=indices.stride(0),
        stride_ik=indices.stride(1),
        num_warps=1,
        **extra,
    )
    return weights, indices
