"""Shared BF16 MoE arithmetic, matching the pinned DeepEP expert program.

Local scatter only changes ownership. Expert GEMMs, activation and slot-order
combine reuse the serving kernels used by the distributed implementation.
"""

import torch


def _true_on_policy_enabled() -> bool:
    """MoE kernels are shared with non-TOP engines; a process that never published server args is not serving TOP."""
    from sglang.srt.true_on_policy import is_true_on_policy_enabled

    try:
        return is_true_on_policy_enabled()
    except ValueError:
        return False


def should_use_slot_order_moe_combine() -> bool:
    """Keep DeepEP expert rows unweighted until the token-owner combine."""
    return _true_on_policy_enabled()


def should_use_precise_masked_silu() -> bool:
    """TOP BF16 experts use the accurate SiLU (the trainer's) for prefill and decode on every architecture.

    The fast builds differ from it in about one element in 3e7 (H200), enough to flip a long rollout's bytes.
    """
    return _true_on_policy_enabled()


def top_clamp_swiglu_input(gate_up, swiglu_limit) -> None:
    """Stands in for the stock silu_and_mul_clamp on contiguous (prefill) expert rows: that kernel is a fast-math JIT
    build, while decode's masked rows run the precise masked SiLU with the clamp inside, so prefill and decode would
    differ.

    TOP BF16 experts with a SwiGLU limit (GLM-5.3-Flash): clamp gate and up in place in BF16 before the accurate
    SiLU, as silu_and_mul_clamp does (gate <= limit, |up| <= limit). The trainer clamps the same rows."""
    if swiglu_limit is None or not _true_on_policy_enabled():
        return
    n = gate_up.shape[-1] // 2
    gate_up[..., :n].clamp_(max=swiglu_limit)
    gate_up[..., n:].clamp_(min=-swiglu_limit, max=swiglu_limit)


def top_swiglu_limit(swiglu_limit):
    """The limit the masked SiLU applies to the valid rows of TOP BF16 experts (same clamp as
    top_clamp_swiglu_input); None when TOP is off or the model has no limit."""
    if swiglu_limit is None or not _true_on_policy_enabled():
        return None
    return swiglu_limit


def refuse_routed_experts_capture(router: str) -> None:
    """Only TopQwenRouter feeds the routed-experts capturer under TOP; another TOP router would return no routes."""
    from sglang.srt.runtime_context import get_exec

    if get_exec().features.enable_return_routed_experts:
        raise NotImplementedError(
            f"[top] the {router} router does not capture routed experts, so rollout routing replay has nothing to "
            "replay; drop --enable-return-routed-experts"
        )


def route_qwen(hidden_states, weight, topk, replayed_ids=None):
    """BF16 gate output, then the fused TOP routing (FP32 softmax, top-k, renormalize, one cast).

    ``replayed_ids`` [T, topk] forces the experts (rollout routing replay); -1 rows route live.
    """
    from sglang.srt.true_on_policy.qwen_route import (
        route_qwen_logits,
        route_qwen_logits_replayed,
    )

    logits = torch.matmul(hidden_states, weight.t().to(hidden_states.dtype))
    if replayed_ids is not None:
        return route_qwen_logits_replayed(
            logits, topk, hidden_states.dtype, replayed_ids
        )
    return route_qwen_logits(logits, topk, hidden_states.dtype)


class _LocalExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w13, w2, weights, ids):
        # Optional GPU dependencies are resolved only when the operation runs.
        from sglang.kernels.ops.activation.activation import silu_and_mul_accurate
        from sglang.kernels.ops.moe.ep_moe_kernels import (
            ep_gather,
            ep_scatter,
            fused_moe_dispatch_index,
        )
        from sglang.kernels.ops.moe.triton_pad_expert_counts import pad_expert_counts
        from sglang.srt.layers.deep_gemm_wrapper import grouped_gemm_nt_bf16_contig

        if x.shape[0] == 0:
            return torch.empty_like(x)
        experts = w13.shape[0]
        # At most one expert per assignment can be nonempty. Match the native
        # compact-layout bound without reading routing counts back to the host.
        assignments = ids.numel()
        nonempty = min(assignments, experts)
        capacity = 128 * (nonempty + (assignments - nonempty) // 128)
        counts, unused = fused_moe_dispatch_index(ids, experts, 1)
        padded_counts = pad_expert_counts(counts, 128, capacity)
        starts = torch.empty_like(counts)
        # Scatter writes every live row; expert indices mask the padding.
        packed = x.new_empty((capacity, x.shape[1]))
        scales = torch.empty((capacity, 1), device=x.device, dtype=torch.float32)
        expert_ids = torch.empty(capacity, device=x.device, dtype=torch.int32)
        input_index = torch.empty_like(ids)
        ep_scatter(
            x,
            None,
            ids,
            padded_counts,
            counts,
            starts,
            packed,
            scales,
            expert_ids,
            input_index,
            expert_alignment=128,
        )
        gate_up = x.new_empty((capacity, w13.shape[1]))
        grouped_gemm_nt_bf16_contig(packed, w13, gate_up, expert_ids)
        activated = x.new_empty((capacity, w13.shape[1] // 2))
        # The accurate build: decode's precise masked SiLU and the trainer use the same bytes.
        silu_and_mul_accurate(gate_up, activated)
        rows = x.new_empty((capacity, w2.shape[1]))
        grouped_gemm_nt_bf16_contig(activated, w2, rows, expert_ids)
        output = x.new_empty((x.shape[0], w2.shape[1]))
        # Match the pinned owner-side combine: weight and sum in FP32 slot order,
        # then round once to BF16. Do not weight the expert rows beforehand.
        ep_gather(rows, ids, weights.float(), input_index, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "[top] MoE expert and slot-combine backward is deferred"
        )


def local_experts(x, w13, w2, weights, ids):
    """EP1 forward over every expert of the layer; all ranks hold identical token rows and routing decisions."""
    if any(t.dtype != torch.bfloat16 for t in (x, w13, w2, weights)):
        raise NotImplementedError(
            "[top] local MoE requires BF16 inputs, weights and routing weights"
        )
    if x.ndim != 2 or w13.ndim != 3 or w2.ndim != 3:
        raise ValueError("[top] expected token rows and stacked expert weights")
    e, n, h = w13.shape
    if w2.shape != (e, h, n // 2) or x.shape[1] != h or n % 2:
        raise ValueError("[top] incompatible expert weight shapes")
    if (
        ids.shape != weights.shape
        or ids.shape[0] != x.shape[0]
        or ids.dtype != torch.int32
    ):
        raise ValueError(
            "[top] expected int32 ordered top-k IDs and matching routing weights"
        )
    if h % 128 or (n // 2) % 128:
        raise NotImplementedError(
            "[top] local experts require K128 alignment of both expert GEMMs"
        )
    return _LocalExperts.apply(
        x.contiguous(), w13.contiguous(), w2.contiguous(), weights, ids
    )
