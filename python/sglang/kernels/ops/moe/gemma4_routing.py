import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_routing_post_topk_kernel(
    Logits_ptr,
    Ids_ptr,
    Scale_ptr,
    Out_weights_ptr,
    Out_ids_ptr,
    stride_l,
    stride_ow,
    stride_oi,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused: softmax(topk_logits) * per_expert_scale[topk_ids] → float32 weights, int32 ids.

    One program per token. K is the number of top-k experts (e.g. 8).
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K

    logits = tl.load(
        Logits_ptr + row * stride_l + cols, mask=mask, other=float("-inf")
    ).to(tl.float32)
    ids_i64 = tl.load(Ids_ptr + row * stride_l + cols, mask=mask, other=0)

    # Stable softmax
    max_val = tl.max(logits, axis=0)
    exp_val = tl.exp(logits - max_val)
    sum_exp = tl.sum(exp_val, axis=0)
    weights = exp_val / sum_exp

    # Gather per_expert_scale and multiply
    scale = tl.load(Scale_ptr + ids_i64, mask=mask, other=1.0).to(tl.float32)
    weights = weights * scale

    tl.store(Out_weights_ptr + row * stride_ow + cols, weights, mask=mask)
    tl.store(Out_ids_ptr + row * stride_oi + cols, ids_i64.to(tl.int32), mask=mask)


def gemma_routing_post_topk(
    topk_logits: torch.Tensor,
    topk_ids: torch.Tensor,
    per_expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused softmax + scale-gather + casts for Gemma4 routing.

    Replaces: softmax(topk_logits) * per_expert_scale[topk_ids] → (f32, i32).
    """
    B, K = topk_logits.shape
    BLOCK_K = triton.next_power_of_2(K)
    out_weights = torch.empty((B, K), dtype=torch.float32, device=topk_logits.device)
    out_ids = torch.empty((B, K), dtype=torch.int32, device=topk_logits.device)

    _gemma_routing_post_topk_kernel[(B,)](
        topk_logits,
        topk_ids,
        per_expert_scale,
        out_weights,
        out_ids,
        topk_logits.stride(0),
        out_weights.stride(0),
        out_ids.stride(0),
        K=K,
        BLOCK_K=BLOCK_K,
    )
    return out_weights, out_ids


@triton.jit
def _gemma4_routing_kernel(
    gating_ptr,  # [T, E] router logits, any float dtype
    per_expert_scale_ptr,  # [E] per-expert scale (any float dtype)
    topk_weights_ptr,  # [T, K] fp32 out
    topk_ids_ptr,  # [T, K] int32 out
    stride_g_t,  # stride of gating in the token dim
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    valid = offs_e < E

    logits = tl.load(
        gating_ptr + pid * stride_g_t + offs_e,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)

    # Pack (sort_key, expert_id) into one int64 so a single signed-ascending
    # tl.sort yields logits in descending float order. The key bijection is
    # anti-monotone on the float value, and the <<32 shift moves its high bit
    # into the int64 sign bit. Ties break by expert id ascending. Invalid
    # lanes use a max key so they sort last.
    MIN32 = -2147483648
    logit_bits = logits.to(tl.int32, bitcast=True)
    sign = logit_bits >> 31
    key = tl.where(sign == 0, logit_bits ^ -1, logit_bits ^ MIN32)
    key = tl.where(valid, key, 0x7FFFFFFF)
    sk64 = key.to(tl.int64) & 0x00000000FFFFFFFF
    packed = (sk64 << 32) | offs_e.to(tl.int64)

    sorted_p = tl.sort(packed, descending=False)
    all_keys = ((sorted_p >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    all_ids = (sorted_p & 0x00000000FFFFFFFF).to(tl.int32)

    # Invert the key bijection to recover the original logit value.
    sign_k = all_keys >> 31
    all_bits = tl.where(sign_k < 0, all_keys ^ -1, all_keys ^ MIN32)
    all_logits = all_bits.to(tl.float32, bitcast=True)

    # softmax over the top-K logits; max sits at index 0 (sorted descending).
    top_mask = offs_e < K
    max_l = tl.max(tl.where(top_mask, all_logits, -float("inf")), axis=0)
    raw_exp = tl.where(top_mask, tl.exp(all_logits - max_l), 0.0)

    denom = tl.sum(raw_exp, axis=0)
    denom = tl.where(denom > 0.0, denom, 1.0)
    weights = raw_exp / denom

    scales = tl.load(
        per_expert_scale_ptr + all_ids.to(tl.int64),
        mask=top_mask,
        other=1.0,
    ).to(tl.float32)
    weights = weights * scales

    base_off = pid * K + offs_e
    tl.store(topk_weights_ptr + base_off, weights, mask=top_mask)
    tl.store(topk_ids_ptr + base_off, all_ids, mask=top_mask)


def gemma4_fused_routing(
    gating_output: torch.Tensor,
    per_expert_scale: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-launch Gemma4 router.

    Args:
        gating_output: [T, E] router logits in any floating dtype; will be
            cast to fp32 inside the kernel.
        per_expert_scale: [E] per-expert scale, any floating dtype.
        topk: number of experts to keep per token.

    Returns:
        topk_weights: [T, topk] fp32 (matches SGLang TopK contract).
        topk_ids: [T, topk] int32 (matches SGLang TopK contract).
    """
    assert gating_output.dim() == 2, "expected [T, E] router logits"
    assert per_expert_scale.dim() == 1
    assert per_expert_scale.shape[0] == gating_output.shape[1]
    T, E = gating_output.shape
    assert topk <= E, f"topk ({topk}) must be <= E ({E})"
    assert E <= 1024, f"gemma4_fused_routing only supports E<=1024, got E={E}"

    gating_output = gating_output.contiguous()
    per_expert_scale = per_expert_scale.contiguous()

    BLOCK_E = triton.next_power_of_2(E)
    topk_weights = torch.empty(
        (T, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty((T, topk), dtype=torch.int32, device=gating_output.device)

    if T == 0:
        return topk_weights, topk_ids

    _gemma4_routing_kernel[(T,)](
        gating_output,
        per_expert_scale,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        E,
        topk,
        BLOCK_E,
        num_warps=1,
    )
    return topk_weights, topk_ids
