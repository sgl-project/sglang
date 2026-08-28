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


def selector_selected_path_confidence(
    scores: torch.Tensor, path_indices: torch.Tensor
) -> torch.Tensor:
    """Return conditional probabilities for the selector path actually proposed.

    ``scores[b, j, p, c]`` describes candidate ``c`` at position ``j`` given
    predecessor candidate ``p``. ``path_indices[b, j]`` identifies the candidate
    selected by the selector walk. The returned factor is
    ``P(path_j | anchor)`` for ``j=0`` and ``P(path_j | path_{j-1})`` afterward.
    Prefix survival is formed downstream with ``cumprod`` (see DSpark's
    ``ScheduleVerifyLensTopk``).
    The score uses the selector's native temperature rather than request sampling
    temperature, so it remains a model-likelihood signal suitable for later STS.
    """

    if scores.ndim != 4:
        raise ValueError(f"expected selector scores [B, L, K, K], got {scores.shape}")
    batch, positions, predecessors, candidates = scores.shape
    if predecessors != candidates:
        raise ValueError(
            "expected square selector lattice [B, L, K, K], got " f"{scores.shape}"
        )
    if path_indices.shape != (batch, positions):
        raise ValueError(
            "expected path_indices [B, L] matching selector scores, got "
            f"{path_indices.shape} for scores {scores.shape}"
        )
    if path_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"path_indices must be int32 or int64, got {path_indices.dtype}"
        )
    # `path_indices` is produced by CandidateSelector.sample_path, which
    # guarantees its values are in [0, candidates). Keep this hot path fully
    # asynchronous; converting a CUDA reduction to bool would synchronize.

    predecessor_indices = torch.cat(
        (torch.zeros_like(path_indices[:, :1]), path_indices[:, :-1]), dim=1
    )
    selected_rows = scores.gather(
        2,
        predecessor_indices.to(torch.long)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(-1, -1, 1, candidates),
    ).squeeze(2)
    return (
        torch.softmax(selected_rows.float(), dim=-1)
        .gather(-1, path_indices.to(torch.long).unsqueeze(-1))
        .squeeze(-1)
    )


def select_sps_verify_token_budget(
    confidence: torch.Tensor,
    *,
    verify_num_draft_tokens: int,
    sps_table: SpsCostTable | SpsAdditiveCostTable,
    survival_eps: float = 1e-6,
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
    if survival_eps < 0:
        raise ValueError(f"survival_eps must be >= 0, got {survival_eps}")

    survival = torch.cumprod(confidence.float().clamp(0.0, 1.0), dim=1)
    candidates = torch.sort(
        survival[survival >= survival_eps].double(), descending=True
    ).values
    expected = torch.cat(
        (
            torch.tensor([float(bs)], dtype=torch.float64),
            float(bs) + torch.cumsum(candidates, dim=0),
        )
    )
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

    return DFlashSpsBudgetDecision(
        budget=best,
        predicted_step_seconds=predicted_step_seconds,
        predicted_theta=float(theta[best]),
    )
