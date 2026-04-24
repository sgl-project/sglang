from __future__ import annotations

from typing import Optional, Sequence

import torch


def _normalize_accept_lengths(accept_lengths) -> list[int]:
    if accept_lengths is None:
        return []
    if isinstance(accept_lengths, torch.Tensor):
        return [int(x) for x in accept_lengths.detach().cpu().tolist()]
    return [int(x) for x in accept_lengths]


def _format_req_accepts(
    reqs: Optional[Sequence[object]],
    accept_lengths: list[int],
    draft_tokens_per_req: int,
    limit: int = 6,
) -> str:
    if not accept_lengths:
        return "-"

    if reqs is None:
        return ", ".join(str(x) for x in accept_lengths[:limit])

    preview = []
    for req, accepted in zip(reqs[:limit], accept_lengths[:limit]):
        rid = getattr(req, "rid", "?")
        preview.append(f"{rid}:{accepted}/{draft_tokens_per_req}")
    return ", ".join(preview)


def log_verify_summary(
    logger,
    worker_label: str,
    step: int,
    batch_size: int,
    verify_tokens: int,
    draft_tokens_per_req: int,
    can_run_cuda_graph: bool,
    accept_lengths,
    target_forward_ms: float,
    verify_total_ms: float,
    reqs: Optional[Sequence[object]] = None,
) -> None:
    accept_list = _normalize_accept_lengths(accept_lengths)
    accepted_total = sum(accept_list)
    accept_avg = accepted_total / len(accept_list) if accept_list else 0.0
    accept_max = max(accept_list) if accept_list else 0
    proposed_total = len(accept_list) * max(draft_tokens_per_req, 0)
    accept_rate = accepted_total / proposed_total if proposed_total > 0 else 0.0
    req_accepts = _format_req_accepts(reqs, accept_list, draft_tokens_per_req)

    logger.info(
        "[SpecDecodeDiag][%s] verify_step=%s batch=%s verify_tokens=%s "
        "draft_tokens_per_req=%s can_run_cuda_graph=%s "
        "target_forward_ms=%.3f verify_total_ms=%.3f "
        "accepted_total=%s accept_avg=%.3f accept_max=%s accept_rate=%.3f "
        "req_accepts=[%s]",
        worker_label,
        step,
        batch_size,
        verify_tokens,
        draft_tokens_per_req,
        can_run_cuda_graph,
        target_forward_ms,
        verify_total_ms,
        accepted_total,
        accept_avg,
        accept_max,
        accept_rate,
        req_accepts,
    )
