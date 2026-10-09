from typing import Callable, Optional, Tuple

import torch
from torch import nn

StepSampler = Callable[[torch.Tensor, int], torch.Tensor]


def run_markov_block(
    head: nn.Module,
    base_logits: torch.Tensor,
    *,
    first_prev_tokens: torch.Tensor,
    hidden_states: Optional[torch.Tensor],
    sampler: StepSampler,
    collect_corrected: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sample a proposal block; head state is local to this invocation/replay."""
    batch_size, proposal_len = base_logits.shape[:2]
    if proposal_len == 0:
        empty = torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
        return empty, base_logits

    sampled_tokens = []
    corrected_logits = []
    prev_tokens = first_prev_tokens.long()
    state = None
    for step_idx in range(proposal_len):
        step_hidden = None if hidden_states is None else hidden_states[:, step_idx, ...]
        step_logits, state = head.apply_step_logits(
            base_logits[:, step_idx, :],
            token_ids=prev_tokens,
            hidden_states=step_hidden,
            state=state,
        )
        next_tokens = sampler(step_logits, step_idx)
        sampled_tokens.append(next_tokens)
        if collect_corrected:
            corrected_logits.append(step_logits.unsqueeze(1))
        prev_tokens = next_tokens
    return (
        torch.stack(sampled_tokens, dim=1),
        torch.cat(corrected_logits, dim=1) if collect_corrected else None,
    )


def sample_markov_block_greedy(head, base_logits, *, first_prev_tokens):
    if not base_logits.is_cuda:
        return None
    batch_size, proposal_len = base_logits.shape[:2]
    if proposal_len == 0:
        return torch.empty(batch_size, 0, dtype=torch.long, device=base_logits.device)
    tokens = []
    prev = first_prev_tokens.long()
    for step in range(proposal_len):
        prev = head.compute_greedy_step(base_logits[:, step], token_ids=prev)
        if prev is None:
            return None
        tokens.append(prev)
    return torch.stack(tokens, dim=1)
