from __future__ import annotations

import logging
import math
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_correct_drafts_and_bonus,
)

logger = logging.getLogger(__name__)

EPS_PROB = 1e-8


def _format_float(value: float, digits: int = 4) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


class PerPositionConfidenceMetrics:

    def __init__(
        self,
        *,
        gamma: int,
        device: torch.device,
        num_coarse_bins: int = 15,
        num_fine_bins: int = 1024,
    ) -> None:
        self.gamma = int(gamma)
        self.num_coarse_bins = int(num_coarse_bins)
        self.num_fine_bins = int(num_fine_bins)
        self.coarse_count = torch.zeros(
            (self.gamma, self.num_coarse_bins), dtype=torch.float64, device=device
        )
        self.coarse_pred = torch.zeros_like(self.coarse_count)
        self.coarse_target = torch.zeros_like(self.coarse_count)
        self.fine_pos = torch.zeros(
            (self.gamma, self.num_fine_bins), dtype=torch.float64, device=device
        )
        self.fine_neg = torch.zeros_like(self.fine_pos)
        self.brier_num = torch.zeros(self.gamma, dtype=torch.float64, device=device)

    def update(self, *, survival: torch.Tensor, prefix_mask: torch.Tensor) -> None:
        assert survival.shape == prefix_mask.shape
        assert survival.dim() == 2 and survival.shape[1] == self.gamma

        probs = survival.to(torch.float64).clamp(EPS_PROB, 1.0 - EPS_PROB)
        targets = prefix_mask.to(torch.float64)
        bs = probs.shape[0]

        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        weights = torch.ones_like(probs_flat)
        pos_idx = (
            torch.arange(self.gamma, device=probs.device)
            .view(1, -1)
            .expand(bs, self.gamma)
            .reshape(-1)
        )

        coarse_idx = (
            (probs_flat * self.num_coarse_bins)
            .long()
            .clamp_(0, self.num_coarse_bins - 1)
        )
        flat_coarse = pos_idx * self.num_coarse_bins + coarse_idx
        self.coarse_count.view(-1).scatter_add_(0, flat_coarse, weights)
        self.coarse_pred.view(-1).scatter_add_(0, flat_coarse, probs_flat)
        self.coarse_target.view(-1).scatter_add_(0, flat_coarse, targets_flat)

        fine_idx = (
            (probs_flat * self.num_fine_bins).long().clamp_(0, self.num_fine_bins - 1)
        )
        flat_fine = pos_idx * self.num_fine_bins + fine_idx
        self.fine_pos.view(-1).scatter_add_(0, flat_fine, targets_flat)
        self.fine_neg.view(-1).scatter_add_(0, flat_fine, 1.0 - targets_flat)

        self.brier_num.add_((probs - targets).pow(2).sum(dim=0))

    @staticmethod
    def _auroc_from_hist(pos_hist: torch.Tensor, neg_hist: torch.Tensor) -> float:
        total_pos = float(pos_hist.sum())
        total_neg = float(neg_hist.sum())
        if total_pos <= 0.0 or total_neg <= 0.0:
            return float("nan")
        cum_neg = torch.cumsum(neg_hist, dim=0)
        cum_neg_before = cum_neg - neg_hist
        pair = (pos_hist * cum_neg_before).sum() + 0.5 * (pos_hist * neg_hist).sum()
        return float(pair) / (total_pos * total_neg)

    def compute(self) -> list[dict]:
        coarse_count = self.coarse_count.cpu()
        coarse_pred = self.coarse_pred.cpu()
        coarse_target = self.coarse_target.cpu()
        fine_pos = self.fine_pos.cpu()
        fine_neg = self.fine_neg.cpu()
        brier_num = self.brier_num.cpu()

        out: list[dict] = []
        for pos in range(self.gamma):
            weights = coarse_count[pos]
            total = float(weights.sum())
            if total <= 1e-12:
                out.append(
                    {
                        "position": pos,
                        "total_weight": 0.0,
                        "ece": float("nan"),
                        "auc": float("nan"),
                        "brier": float("nan"),
                        "pred_mean": float("nan"),
                        "target_mean": float("nan"),
                        "reliability": [],
                    }
                )
                continue

            denom = weights.clamp_min(1e-12)
            avg_pred = coarse_pred[pos] / denom
            avg_target = coarse_target[pos] / denom
            bin_err = (avg_pred - avg_target).abs()
            ece = float((bin_err * weights).sum()) / total
            auc = self._auroc_from_hist(fine_pos[pos], fine_neg[pos])
            brier = float(brier_num[pos]) / total
            reliability = []
            for bin_idx in range(self.num_coarse_bins):
                weight = float(weights[bin_idx])
                if weight <= 0.0:
                    continue
                reliability.append(
                    {
                        "bin": bin_idx,
                        "range": [
                            bin_idx / self.num_coarse_bins,
                            (bin_idx + 1) / self.num_coarse_bins,
                        ],
                        "avg_pred": float(avg_pred[bin_idx]),
                        "avg_target": float(avg_target[bin_idx]),
                        "weight": weight,
                    }
                )
            out.append(
                {
                    "position": pos,
                    "total_weight": total,
                    "ece": ece,
                    "auc": auc,
                    "brier": brier,
                    "pred_mean": float(coarse_pred[pos].sum()) / total,
                    "target_mean": float(coarse_target[pos].sum()) / total,
                    "reliability": reliability,
                }
            )
        return out

    def format_table(self) -> str:
        rows = self.compute()
        header = (
            f"{'pos':>3} {'count':>12} {'pred':>8} {'target':>8} "
            f"{'ece':>8} {'auc':>8} {'brier':>8}"
        )
        lines = [
            "DSpark confidence-head per-position calibration "
            "(cumprod survival vs leading-correct-prefix)",
            header,
        ]
        for row in rows:
            lines.append(
                f"{row['position']:>3} {row['total_weight']:>12.0f} "
                f"{_format_float(row['pred_mean']):>8} "
                f"{_format_float(row['target_mean']):>8} "
                f"{_format_float(row['ece']):>8} "
                f"{_format_float(row['auc']):>8} "
                f"{_format_float(row['brier']):>8}"
            )
        return "\n".join(lines)


class ConfidenceMetricsProbe:

    def __init__(
        self,
        *,
        gamma: int,
        verify_num_draft_tokens: int,
        tp_rank: int,
        print_every: int = 256,
    ) -> None:
        self.gamma = int(gamma)
        self.verify_num_draft_tokens = int(verify_num_draft_tokens)
        self.tp_rank = int(tp_rank)
        self.print_every = int(print_every)
        self._metrics: Optional[PerPositionConfidenceMetrics] = None
        self._step_ct: int = 0
        self._compact_warned: bool = False

    def maybe_observe(
        self,
        *,
        carries_confidence: bool,
        is_compact_mode: bool,
        confidence_raw: Optional[torch.Tensor],
        verify_ids_2d: torch.Tensor,
        target_logits: torch.Tensor,
        bs: int,
    ) -> None:
        if not envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS.get():
            return
        if self.tp_rank != 0:
            return
        if not carries_confidence:
            return
        if is_compact_mode:
            if not self._compact_warned:
                logger.warning(
                    "SGLANG_DSPARK_DEBUG_CONFIDENCE_METRICS is ignored under "
                    "SGLANG_RAGGED_VERIFY_MODE=compact (padded verify rows corrupt the "
                    "per-position prefix label); run cap-accept or static to measure it."
                )
                self._compact_warned = True
            return
        if confidence_raw is None:
            return

        target_predict = torch.argmax(target_logits, dim=-1).view(
            bs, self.verify_num_draft_tokens
        )
        num_correct_drafts, _ = compute_dflash_correct_drafts_and_bonus(
            candidates=verify_ids_2d,
            target_predict=target_predict,
        )
        positions = torch.arange(self.gamma, device=confidence_raw.device).view(1, -1)
        prefix_mask = (positions < num_correct_drafts.view(-1, 1)).to(torch.float32)
        survival = torch.cumprod(torch.sigmoid(confidence_raw.float()), dim=1)

        if self._metrics is None:
            self._metrics = PerPositionConfidenceMetrics(
                gamma=self.gamma, device=confidence_raw.device
            )
        self._metrics.update(survival=survival, prefix_mask=prefix_mask)
        self._step_ct += 1
        if self._step_ct % self.print_every == 0:
            logger.info("%s", self._metrics.format_table())
