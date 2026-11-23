"""Qwen softmax routing for TOP, shared by serving and the trainer.

The forward is SGLang's fused router (``moe_fused_gate``, softmax scoring): FP32 softmax of the router logits, the
top-k experts with ties ranked by expert id, and the FP32 renormalization, then one cast. Its launch shape depends
only on the expert count and k, and every reduction is per row, so a token's bytes do not depend on the batch.
"""

import torch

from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate


def _route_forward(logits, topk, out_dtype):
    weights, ids = moe_fused_gate(
        logits, None, topk, scoring_func="softmax", renormalize=True
    )
    return weights.to(out_dtype), ids


def _route_backward(logits, ids, grad_weights):
    """Analytic FP32 gradient of softmax -> gather(ids) -> renormalize, the function the kernel computes."""
    p = torch.softmax(logits.float(), dim=-1)
    selected = torch.gather(p, 1, ids.long())
    total = selected.sum(dim=-1, keepdim=True)
    w = selected / total
    dw = grad_weights.float()
    d_selected = (dw - (dw * w).sum(dim=-1, keepdim=True)) / total
    dp = torch.zeros_like(p).scatter_(1, ids.long(), d_selected)
    d_logits = p * (dp - (dp * p).sum(dim=-1, keepdim=True))
    return d_logits.to(logits.dtype)


class _QwenRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, topk, out_dtype):
        weights, ids = _route_forward(logits, topk, out_dtype)
        ctx.save_for_backward(logits, ids)
        ctx.mark_non_differentiable(ids)
        return weights, ids

    @staticmethod
    def backward(ctx, grad_weights, _grad_ids):
        logits, ids = ctx.saved_tensors
        return _route_backward(logits, ids, grad_weights), None, None


def _replay_forward(logits, topk, out_dtype, replayed_ids):
    live_weights, live_ids = _route_forward(logits, topk, out_dtype)
    replayed_ids = replayed_ids.to(live_ids.dtype)
    # Pad rows (any -1) route live; rows whose ordered experts match keep the kernel's bytes.
    keep = (replayed_ids < 0).any(dim=-1, keepdim=True) | (
        replayed_ids == live_ids
    ).all(dim=-1, keepdim=True)
    ids = torch.where(keep, live_ids, replayed_ids)
    selected = torch.softmax(logits.float(), dim=-1).gather(1, ids.long())
    replayed = (selected / selected.sum(dim=-1, keepdim=True)).to(out_dtype)
    return torch.where(keep, live_weights, replayed), ids


class _QwenRouteReplayed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, topk, out_dtype, replayed_ids):
        weights, ids = _replay_forward(logits, topk, out_dtype, replayed_ids)
        ctx.save_for_backward(logits, ids)
        ctx.mark_non_differentiable(ids)
        return weights, ids

    @staticmethod
    def backward(ctx, grad_weights, _grad_ids):
        logits, ids = ctx.saved_tensors
        return _route_backward(logits, ids, grad_weights), None, None, None


def _checked_logits(logits, topk):
    if logits.dim() != 2 or not logits.is_cuda or topk > logits.shape[1]:
        raise ValueError(
            "[top] Qwen routing expects CUDA logits [T, E] with top-k <= E"
        )
    return logits if logits.stride(1) == 1 else logits.contiguous()


def route_qwen_logits(logits, topk, out_dtype):
    """Routing weights [T, topk] in ``out_dtype`` and expert ids [T, topk] (int32) from router logits [T, E]."""
    logits = _checked_logits(logits, topk)
    if torch.is_grad_enabled() and logits.requires_grad:
        return _QwenRoute.apply(logits, topk, out_dtype)
    return _route_forward(logits, topk, out_dtype)


def route_qwen_logits_replayed(logits, topk, out_dtype, replayed_ids):
    """Routing replay: experts forced to ``replayed_ids`` [T, topk] in their slot order, weights the FP32 softmax at
    those experts renormalized and cast once. Rows equal to the live route, or padded with -1, return the live bytes.
    """
    logits = _checked_logits(logits, topk)
    if replayed_ids.shape != (logits.shape[0], topk):
        raise ValueError(
            f"[top] replayed top-k {tuple(replayed_ids.shape)} != ({logits.shape[0]}, {topk})"
        )
    if torch.is_grad_enabled() and logits.requires_grad:
        return _QwenRouteReplayed.apply(logits, topk, out_dtype, replayed_ids)
    return _replay_forward(logits, topk, out_dtype, replayed_ids)
