from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import torch
import torch.nn.functional as F
from sglang.kernels.ops.sampling import softmax as sampling_softmax
from sglang.srt.layers.sampler import top_p_normalize_probs_torch
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu

if is_cuda():
    from flashinfer.sampling import top_k_renorm_probs as top_k_renorm_prob
    from flashinfer.sampling import top_p_renorm_probs as top_p_renorm_prob
elif is_hip():
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_k_renorm_probs_triton as top_k_renorm_prob,
    )
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_p_renorm_probs_triton as top_p_renorm_prob,
    )
else:
    top_k_renorm_prob = None
    top_p_renorm_prob = None


def _npu_top_k_top_p_renorm_prob(
    probs: torch.Tensor,
    *,
    top_ks: Optional[torch.Tensor] = None,
    top_ps: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if not is_npu() or probs.device.type != "npu":
        return None
    try:
        import torch_npu
    except ImportError:
        return None
    if not hasattr(torch_npu, "npu_top_k_top_p"):
        return None

    logits = probs.log()
    npu_top_ps = (
        top_ps.reshape(-1).to(device=probs.device, dtype=probs.dtype)
        if top_ps is not None
        else None
    )
    npu_top_ks = (
        top_ks.reshape(-1).to(device=probs.device, dtype=torch.int32)
        if top_ks is not None
        else None
    )
    if npu_top_ks is not None and not bool(
        torch.all((npu_top_ks >= 1) & (npu_top_ks <= 1024)).item()
    ):
        return None
    filtered_logits = torch_npu.npu_top_k_top_p(logits, npu_top_ps, npu_top_ks)
    return filtered_logits.softmax(dim=-1)


def _top_k_renorm_prob(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    if top_k_renorm_prob is not None:
        return top_k_renorm_prob(probs, top_ks)
    if is_musa():
        from sgl_kernel import top_k_renorm_prob as renorm

        return renorm(probs, top_ks)

    npu_probs = _npu_top_k_top_p_renorm_prob(probs, top_ks=top_ks)
    if npu_probs is not None:
        return npu_probs

    vocab_size = probs.shape[-1]
    top_ks = top_ks.reshape(-1).to(device=probs.device, dtype=torch.int64)
    top_ks = top_ks.clamp(min=1, max=vocab_size)
    max_top_k = int(top_ks.max().item())
    topk_probs, topk_indices = torch.topk(probs, k=max_top_k, dim=-1)
    ranks = torch.arange(max_top_k, device=probs.device)[None, :]
    topk_probs.masked_fill_(ranks >= top_ks[:, None], 0.0)
    topk_probs.div_(topk_probs.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs).scatter_(1, topk_indices, topk_probs)


def _top_p_renorm_prob(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    if top_p_renorm_prob is not None:
        return top_p_renorm_prob(probs, top_ps)
    if is_musa():
        from sgl_kernel import top_p_renorm_prob as renorm

        return renorm(probs, top_ps)
    npu_probs = _npu_top_k_top_p_renorm_prob(probs, top_ps=top_ps)
    if npu_probs is not None:
        return npu_probs
    return top_p_normalize_probs_torch(probs, top_ps)


def _apply_min_p(probs, sampling_info, width):
    if not getattr(sampling_info, "need_min_p_sampling", False):
        return probs
    min_ps = torch.repeat_interleave(sampling_info.min_ps, width).reshape(-1, 1)
    shape = probs.shape
    rows = probs.reshape(-1, shape[-1])
    filtered = rows.masked_fill(rows < rows.amax(dim=-1, keepdim=True) * min_ps, 0)
    return torch.where(
        min_ps > 0, filtered / filtered.sum(dim=-1, keepdim=True), rows
    ).reshape(shape)


def build_verify_target_probs(
    *,
    next_token_logits: torch.Tensor,
    sampling_info: Any,
    draft_token_num: int,
    bs: int,
    max_top_k: Optional[int] = None,
    uniform_top_k_value: Optional[int] = None,
    use_sparse_topk: bool = True,
    sparse_top_k_mode: Literal["rank", "threshold"] = "rank",
    renorm_top_k: Callable = _top_k_renorm_prob,
    renorm_top_p: Callable = _top_p_renorm_prob,
    probe: Optional[Callable] = None,
) -> torch.Tensor:
    """Build (batch, verify width, vocab) probabilities with top-k before top-p.

    max_top_k is a host-side upper bound, required for sparse graph capture.
    Rank-limited sparse support and cutoff-inclusive support differ on ties;
    callers must retain their eager policy when selecting a captured fast path.
    """
    if sparse_top_k_mode not in ("rank", "threshold"):
        raise ValueError(f"Unknown sparse top-k mode: {sparse_top_k_mode}")
    if next_token_logits.ndim != 2 or draft_token_num <= 0 or bs <= 0:
        raise ValueError("Verify logits must be 2D with positive batch and width")
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError("Verify logit rows must equal batch size times verify width")
    if sampling_info.temperatures.shape != (bs, 1):
        raise ValueError("Verify temperatures must have shape (batch size, 1)")
    device = next_token_logits.device
    need_top_k = bool(getattr(sampling_info, "need_top_k_sampling", True))
    need_top_p = bool(getattr(sampling_info, "need_top_p_sampling", False))
    if (
        use_sparse_topk
        and sparse_top_k_mode == "threshold"
        and need_top_k
        and need_top_p
        and max_top_k is not None
        and 0 < max_top_k <= 64
        and max_top_k < next_token_logits.shape[-1]
        and next_token_logits.is_cuda
        and next_token_logits.dtype == torch.float32
        and next_token_logits.is_contiguous()
    ):
        from sglang.kernels.ops.sampling.verify_probs import sparse_target_probs

        return _apply_min_p(
            sparse_target_probs(next_token_logits, sampling_info, draft_token_num),
            sampling_info,
            draft_token_num,
        )
    if (
        use_sparse_topk
        and sparse_top_k_mode == "rank"
        and need_top_k
        and max_top_k is None
        and next_token_logits.is_cuda
        and torch.cuda.is_current_stream_capturing()
    ):
        raise ValueError("Sparse verify graph capture requires max_top_k")
    expanded_temperature = torch.repeat_interleave(
        sampling_info.temperatures, draft_token_num, dim=0
    )
    scaled_logits = next_token_logits / expanded_temperature
    sparse_topk_applied = False

    if use_sparse_topk and need_top_k and sparse_top_k_mode == "rank":
        repeated_top_ks = torch.repeat_interleave(
            sampling_info.top_ks, draft_token_num, dim=0
        ).to(dtype=torch.int64)
        vocab_size = int(scaled_logits.shape[-1])
        repeated_top_ks.clamp_(min=1, max=vocab_size)
        if max_top_k is None:
            max_top_k = int(repeated_top_ks.max().item())
        else:
            max_top_k = int(max_top_k)
        if max_top_k < 1:
            max_top_k = 1
        elif max_top_k > vocab_size:
            max_top_k = vocab_size

        # Sparse exact path for top-k/top-p (top-k-first semantics), then scatter to dense.
        if 0 < max_top_k < vocab_size:
            topk_logits, topk_indices = torch.topk(scaled_logits, k=max_top_k, dim=-1)
            if uniform_top_k_value is None or int(uniform_top_k_value) != max_top_k:
                ranks = torch.arange(max_top_k, device=device, dtype=torch.int64)[
                    None, :
                ]
                valid = ranks < repeated_top_ks.unsqueeze(1)
                topk_logits = topk_logits.masked_fill(~valid, float("-inf"))

            topk_probs = F.softmax(topk_logits, dim=-1)
            if need_top_p:
                repeated_top_ps = torch.repeat_interleave(
                    sampling_info.top_ps, draft_token_num, dim=0
                )
                topk_probs = renorm_top_p(topk_probs, repeated_top_ps)

            target_probs = torch.zeros_like(scaled_logits, dtype=topk_probs.dtype)
            target_probs.scatter_(1, topk_indices, topk_probs)
            sparse_topk_applied = True

    if not sparse_topk_applied:
        target_probs = (
            sampling_softmax(next_token_logits, temperatures=expanded_temperature)
            if sparse_top_k_mode == "threshold"
            else F.softmax(scaled_logits, dim=-1)
        )
        if probe is not None:
            probe(target_probs, "v2 verify: target_probs after softmax")
        if need_top_k:
            target_probs = renorm_top_k(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0),
            )
            if probe is not None:
                probe(target_probs, "v2 verify: target_probs after top_k_renorm")
        if need_top_p:
            target_probs = renorm_top_p(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0),
            )
            if probe is not None:
                probe(target_probs, "v2 verify: target_probs after top_p_renorm")
    return (
        _apply_min_p(target_probs, sampling_info, draft_token_num)
        .view(bs, draft_token_num, -1)
        .contiguous()
    )
