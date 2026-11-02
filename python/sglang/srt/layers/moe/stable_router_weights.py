"""Fused selected-logit sigmoid weights with stable normalization for MiMo."""

import torch
import triton
import triton.language as tl


@triton.jit
def _stable_router_weights_kernel(
    logits,
    ids,
    output,
    logits_row_stride,
    logits_col_stride,
    ids_row_stride,
    ids_col_stride,
    SCALE: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < K
    expert = tl.load(
        ids + row * ids_row_stride + col * ids_col_stride, valid, other=0
    ).to(tl.int64)
    x = tl.load(
        logits + row * logits_row_stride + expert * logits_col_stride, valid, other=0
    ).to(tl.float32)
    # log(sigmoid(x)) = min(x, 0) - log1p(exp(-abs(x))). Normalizing
    # log-probabilities avoids underflow when all selected logits are negative.
    logp = tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x)))
    if RENORMALIZE:
        logp = tl.where(valid, logp, -float("inf"))
        value = tl.exp(logp - tl.max(logp, axis=0))
        value = value / tl.sum(value, axis=0) * SCALE
        # Torch softmax propagates a selected NaN to every selected weight.
        has_nan = tl.sum((valid & (x != x)).to(tl.int32), axis=0) > 0
        value = tl.where(has_nan, float("nan"), value)
    else:
        value = tl.exp(logp)
    tl.store(output + row * K + col, value, valid)


def stable_router_weights(
    router_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    renormalize: bool,
    routed_scaling_factor: float | None = None,
    apply_routed_scaling_factor_on_output: bool = False,
) -> torch.Tensor:
    """Preserve selected experts and return FP32 weights; no routing decisions."""
    rows, k = topk_ids.shape
    output = torch.empty((rows, k), dtype=torch.float32, device=router_logits.device)
    if rows == 0 or k == 0:
        return output
    scale = (
        routed_scaling_factor
        if renormalize and apply_routed_scaling_factor_on_output
        else 1.0
    )
    _stable_router_weights_kernel[(rows,)](
        router_logits,
        topk_ids,
        output,
        *router_logits.stride(),
        *topk_ids.stride(),
        SCALE=scale,
        RENORMALIZE=renormalize,
        K=k,
        BLOCK=triton.next_power_of_2(k),
        num_warps=1,
    )
    return output
