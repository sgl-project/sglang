# SPDX-License-Identifier: Apache-2.0
"""Tensor-only acceptance for Clover's linear candidate chain."""

import torch


def clover_correct_drafts(
    logits, candidates, accept_ratio=0.9, accept_delta=0.6, accept_rule="threshold"
):
    """Return counts excluding bonus; logits[j] verifies candidates[j + 1].

    Exact matches the verify argmax; threshold uses raw target probabilities.
    Both are independent of temperature/top-k/top-p used for the boundary token.
    """
    if accept_rule == "exact":
        matches = candidates[:, 1:] == logits[:, :-1].argmax(dim=-1)
        return matches.to(torch.int32).cumprod(dim=-1).sum(-1).to(torch.int32)
    if accept_rule != "threshold":
        raise ValueError(f"Unknown Clover accept rule: {accept_rule}")
    probabilities = torch.softmax(logits[:, :-1].float(), dim=-1)
    proposal_probs = probabilities.gather(-1, candidates[:, 1:, None]).squeeze(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    thresholds = torch.minimum(
        torch.full_like(entropy, accept_ratio), accept_delta * torch.exp(-entropy)
    )
    # A hard penalty (e.g. min_new_tokens) must still reject an impossible
    # candidate when ratio or delta is zero and therefore the threshold is zero.
    matches = (proposal_probs >= thresholds) & (proposal_probs > 0)
    return matches.to(torch.int32).cumprod(dim=-1).sum(-1).to(torch.int32)


def clover_accept_tokens(candidates, num_correct_drafts, bonus_tokens):
    """Padded output rows: correct drafts followed by the newly sampled bonus."""
    width = candidates.shape[1]
    offsets = torch.arange(width, device=candidates.device)[None, :]
    shifted = torch.cat((candidates[:, 1:], bonus_tokens[:, None]), dim=1)
    return torch.where(
        offsets < num_correct_drafts[:, None], shifted, bonus_tokens[:, None]
    )
