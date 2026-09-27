from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from sglang.kernels.ops.speculative.dflash import selector_walk_triton

# One Triton lane group in the selector walk; the config refuses a wider head.
MAX_FUSED_CANDIDATE_TOPK = 16


def _lattice_scores(log_start: torch.Tensor, log_pair: torch.Tensor) -> torch.Tensor:
    # The walk reads only scores[:, 0, 0, :] at slot 0, so broadcasting is sound.
    topk = int(log_start.shape[-1])
    start = log_start.float()[:, None, None, :].expand(-1, 1, topk, topk)
    return torch.cat([start, log_pair.float()], dim=1)


def _selector_walk_torch(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _, num_slots, topk = candidate_ids.shape
    temps = temperatures.view(-1, 1).to(torch.float32)
    greedy = greedy_mask.view(-1)
    previous = torch.zeros(greedy.shape[0], dtype=torch.int64, device=scores.device)
    tokens, q_rows = [], []
    for slot in range(num_slots):
        node = torch.gather(
            scores[:, slot].float(), 1, previous.view(-1, 1, 1).expand(-1, 1, topk)
        ).squeeze(1)
        probs = torch.softmax(node / temps, dim=-1)
        sampled = (
            uniforms[:, slot : slot + 1]
            .ge(probs.cumsum(dim=-1))
            .sum(dim=-1)
            .clamp_max(topk - 1)
        )
        previous = torch.where(greedy, node.argmax(dim=-1), sampled)
        q_rows.append(
            torch.where(
                greedy.unsqueeze(-1),
                F.one_hot(previous, topk).to(torch.float32),
                probs,
            )
        )
        tokens.append(
            torch.gather(candidate_ids[:, slot], 1, previous.view(-1, 1)).squeeze(1)
        )
    return torch.stack(tokens, dim=-1).to(torch.int64), torch.stack(q_rows, dim=1)


def lilicorr_sample_path(
    log_start: torch.Tensor,
    log_pair: torch.Tensor,
    candidate_tokens: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # No log-prob prior in the proposal: the head was trained without one. Greedy rows
    # report a point mass so min(1, p/q) stays the right acceptance test.
    scores = _lattice_scores(log_start, log_pair)
    walk = selector_walk_triton if scores.is_cuda else _selector_walk_torch
    return walk(
        candidate_ids=candidate_tokens,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )
