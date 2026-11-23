"""Shared GLM expert selection using FlashInfer's deterministic top-k."""

import torch


def _validate_scores(scores, top_k):
    if scores.dim() != 2:
        raise ValueError(f"GLM top-k expects a 2D tensor, got {scores.shape}")
    n_cols = scores.shape[1]
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 0 < top_k <= n_cols:
        raise ValueError(f"top_k must be an integer in [1, {n_cols}], got {top_k}")
    if not scores.is_cuda:
        raise NotImplementedError("TOP GLM top-k requires CUDA")
    if scores.dtype not in (torch.bfloat16, torch.float32):
        raise NotImplementedError("TOP GLM top-k requires BF16 or FP32 scores")
    if torch.is_grad_enabled() and scores.requires_grad:
        raise NotImplementedError("TOP GLM top-k is forward-only")
    if n_cols > 1024:
        raise NotImplementedError("TOP GLM top-k supports at most 1024 experts")


def _prepare_scores(scores):
    n_cols = scores.shape[1]
    finite_max = torch.finfo(scores.dtype).max
    scores_2d = torch.nan_to_num(
        scores, nan=-finite_max, posinf=finite_max, neginf=-finite_max
    ).contiguous()
    scores_f32 = scores_2d.float()
    expert_ids = torch.arange(n_cols, device=scores.device, dtype=torch.float32)
    step = torch.finfo(torch.float32).eps * scores_f32.abs().clamp_min(1.0 / n_cols)
    # Near-ties go to the lower expert id: each score drops by id * eps *
    # max(|s|, 1/n) before selection (about 3e-5 relative for expert 255).
    # HF and native SGLang GLM routing do not do this, so on a near-tie TOP may
    # pick another expert than they would. The trainer and serving both route
    # through route_glm and agree. Values returned to the caller remain
    # unperturbed; the adjustment only selects.
    adjusted = torch.addcmul(scores_f32, expert_ids, step, value=-1)
    adjusted = torch.where(torch.isnan(adjusted), -float("inf"), adjusted)
    return scores_2d, adjusted


# Keep eager fallback available when a process exercises many dtype/stride
# specializations; a compiler cache limit must not reject valid score shapes.
_prepare_scores_fused = torch.compile(
    _prepare_scores,
    dynamic=True,
    options={"emulate_precision_casts": True},
)


def _select_ids(adjusted, top_k, *, sorted=True):
    import flashinfer

    if adjusted.shape[0] == 0:
        return torch.empty((0, top_k), device=adjusted.device, dtype=torch.long)
    # FlashInfer finalizes deterministic selections in ID order, then stably
    # sorts by adjusted value: score descending, expert ID ascending on ties.
    return flashinfer.top_k(
        adjusted,
        top_k,
        sorted=sorted,
        deterministic=True,
        tie_break=1,
        dsa_graph_safe=True,
    )[1]
