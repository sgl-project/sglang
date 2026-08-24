"""Confidence-driven, cross-request verification planning for DFlash2.

This module deliberately schedules only the *currently proposed* linear prefix.
A suffix that is not selected for target verification is discarded and drafted
again from the next target-verified bonus token.  It is therefore not a source
of generated output or KV state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.speculative.dspark_components.dspark_sps import (
    SpsAdditiveCostTable,
    SpsCostTable,
)


@dataclass(frozen=True)
class DFlashSpsBudgetDecision:
    """SPS-selected total target-verify width and its throughput prediction."""

    budget: int
    predicted_step_seconds: float | None = None
    predicted_theta: float | None = None


@dataclass(frozen=True)
class DFlashConfidenceDecision:
    """A deterministic verify-prefix decision for one target-verify batch."""

    verify_lens: torch.Tensor
    deferred_tokens: int
    low_confidence_tokens: int


def selector_confidence_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """Return a conservative per-position selector certainty proxy in [0, 1].

    ``scores`` is the selector lattice ``[batch, position, predecessor, token]``.
    The first position has one valid predecessor (the verified anchor).  For later
    positions we use the largest conditional next-token probability among possible
    predecessor candidates.  This is intentionally a scheduling signal only; it
    never participates in acceptance.
    """

    if scores.ndim != 4:
        raise ValueError(f"expected selector scores [B, L, K, K], got {scores.shape}")
    first = torch.softmax(scores[:, 0, 0].float(), dim=-1).amax(dim=-1)
    if scores.shape[1] == 1:
        return first.unsqueeze(1)
    transitions = torch.softmax(scores[:, 1:].float(), dim=-1)
    rest = transitions.amax(dim=-1).amax(dim=-1)
    return torch.cat([first.unsqueeze(1), rest], dim=1)


def select_sps_verify_token_budget(
    confidence: torch.Tensor,
    *,
    verify_num_draft_tokens: int,
    min_verify_len: int,
    sps_table: SpsCostTable | SpsAdditiveCostTable,
) -> DFlashSpsBudgetDecision:
    """Choose DFlash's verify width with DSpark's SPS table objective.

    This intentionally depends only on the lightweight SPS table module, rather
    than the DSpark worker planner. It maximizes expected committed tokens per
    measured verify-step second over every admissible number of draft positions.
    """

    if confidence.ndim != 2:
        raise ValueError(f"expected confidence [B, gamma], got {confidence.shape}")
    bs, gamma = confidence.shape
    if verify_num_draft_tokens != gamma + 1:
        raise ValueError(
            "verify_num_draft_tokens must equal confidence width + 1; "
            f"got {verify_num_draft_tokens=} and gamma={gamma}"
        )
    if not 2 <= min_verify_len <= verify_num_draft_tokens:
        raise ValueError(
            f"min_verify_len must be in [2, {verify_num_draft_tokens}], got {min_verify_len}"
        )

    survival = torch.cumprod(confidence.float().clamp(0.0, 1.0), dim=1)
    candidates = torch.sort(survival.reshape(-1).double(), descending=True).values
    expected = torch.cat((
        torch.tensor([float(bs)], dtype=torch.float64),
        float(bs) + torch.cumsum(candidates, dim=0),
    ))
    budgets = range(expected.numel())
    if isinstance(sps_table, SpsAdditiveCostTable):
        step_seconds = torch.tensor(
            [sps_table.step_time(num_reqs=bs, budget=k) for k in budgets],
            dtype=torch.float64,
        )
        theta = expected / step_seconds
        best = int(torch.argmax(theta))
        predicted_step_seconds = float(step_seconds[best])
    else:
        sps = torch.tensor(
            [sps_table.lookup(bs + k) for k in budgets], dtype=torch.float64
        )
        theta = expected * sps
        best = int(torch.argmax(theta))
        predicted_step_seconds = float(1.0 / sps[best]) if sps[best] > 0 else None

    total = min(
        bs * verify_num_draft_tokens,
        max(bs * min_verify_len, bs + best),
    )
    return DFlashSpsBudgetDecision(
        budget=total,
        predicted_step_seconds=predicted_step_seconds,
        predicted_theta=float(theta[best]),
    )


def plan_verify_prefixes(
    confidence: torch.Tensor,
    *,
    verify_num_draft_tokens: int,
    confidence_threshold: float,
    min_verify_len: int,
    target_verify_tokens: int,
) -> DFlashConfidenceDecision:
    """Allocate a target-verify token budget across request-local prefixes.

    Every request verifies an anchor plus at least one proposed token, preserving
    progress and preventing starvation.  Optional positions compete globally by
    uncertainty (``1 - cumulative_confidence``), so lower-confidence paths are
    verified earlier.  Prefix expansion is performed one round at a time to keep
    every selected per-request set contiguous.
    """

    if confidence.ndim != 2:
        raise ValueError(f"expected confidence [B, gamma], got {confidence.shape}")
    bs, gamma = confidence.shape
    if verify_num_draft_tokens != gamma + 1:
        raise ValueError(
            "verify_num_draft_tokens must equal confidence width + 1; "
            f"got {verify_num_draft_tokens=} and gamma={gamma}"
        )
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 2 <= min_verify_len <= verify_num_draft_tokens:
        raise ValueError(
            "min_verify_len must verify an anchor and at least one proposal, got "
            f"{min_verify_len} for verify width {verify_num_draft_tokens}"
        )

    confidence = confidence.float().clamp_(0.0, 1.0)
    survival = torch.cumprod(confidence, dim=1)
    # A lower survival is less trustworthy and should be verified sooner. The
    # threshold forms an explicit urgency class; within each class uncertainty
    # retains a stable, interpretable ordering.
    uncertainty = 1.0 - survival
    urgency = (survival < confidence_threshold).to(uncertainty.dtype)
    priority = urgency * 2.0 + uncertainty
    required_extra = min_verify_len - 1
    max_extra = gamma
    requested_total = max(bs * min_verify_len, int(target_verify_tokens))
    budget_extra = min(max(0, requested_total - bs), bs * max_extra)
    budget_extra = max(budget_extra, bs * required_extra)

    lengths = torch.full(
        (bs,), min_verify_len, dtype=torch.int32, device=confidence.device
    )
    remaining = budget_extra - bs * required_extra
    low_confidence_tokens = int((survival < confidence_threshold).sum().item())

    # Per-request priority is monotonic along its chain because survival only
    # decreases as confidence is multiplied. Consequently, after a request wins
    # its next-position comparison, it remains at least as urgent as every
    # request it beat until its prefix is full. Allocate whole winning runs,
    # rather than launching one device-wide argmax per token. Stable sorting
    # preserves the original request-index tie break of the token-by-token form.
    if remaining:
        capacity_per_request = max_extra - required_extra
        next_priority = priority[:, required_extra]
        request_order = torch.argsort(next_priority, descending=True, stable=True)
        rank = torch.arange(bs, device=confidence.device, dtype=torch.int32)
        rank = rank.unsqueeze(1)
        starts = rank * capacity_per_request
        extras = (remaining - starts).clamp_(min=0, max=capacity_per_request)
        allocations = extras.squeeze(1).to(torch.int32)
        lengths += allocations[request_order.argsort(stable=True)]

    deferred_tokens = int((verify_num_draft_tokens - lengths).sum().item())
    return DFlashConfidenceDecision(
        verify_lens=lengths,
        deferred_tokens=deferred_tokens,
        low_confidence_tokens=low_confidence_tokens,
    )
