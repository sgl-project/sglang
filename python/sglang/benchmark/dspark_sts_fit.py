from __future__ import annotations

import argparse
import glob
import logging
import math
from pathlib import Path
from typing import Optional

import torch

from sglang.srt.speculative.dspark_components.dspark_sts_table import (
    DSparkStsCalibration,
)

logger = logging.getLogger(__name__)

_EPS_PROB = 1e-8


def default_temperature_grid() -> torch.Tensor:
    return torch.logspace(math.log10(0.1), math.log10(10.0), steps=41)


def expected_calibration_error(
    *,
    probs: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int,
) -> float:
    probs = probs.reshape(-1).to(torch.float64).clamp(_EPS_PROB, 1.0 - _EPS_PROB)
    targets = targets.reshape(-1).to(torch.float64)
    total = probs.numel()
    if total == 0:
        return float("nan")
    bin_index = (probs * num_bins).long().clamp_(0, num_bins - 1)
    count = torch.zeros(num_bins, dtype=torch.float64)
    pred_sum = torch.zeros(num_bins, dtype=torch.float64)
    target_sum = torch.zeros(num_bins, dtype=torch.float64)
    count.scatter_add_(0, bin_index, torch.ones_like(probs))
    pred_sum.scatter_add_(0, bin_index, probs)
    target_sum.scatter_add_(0, bin_index, targets)
    denom = count.clamp_min(1.0)
    bin_error = (pred_sum / denom - target_sum / denom).abs()
    return float((bin_error * count).sum().item() / total)


def fit_sts_temperatures(
    *,
    logits: torch.Tensor,
    prefix_mask: torch.Tensor,
    grid: torch.Tensor,
    num_bins: int = 15,
) -> dict[str, list[float]]:
    logits = logits.to(torch.float64)
    prefix_mask = prefix_mask.to(torch.float64)
    num_samples, gamma = logits.shape
    if num_samples == 0:
        raise ValueError("fit_sts_temperatures requires at least one sample.")
    grid_values = grid.to(torch.float64).tolist()

    temperatures: list[float] = []
    ece_before: list[float] = []
    ece_after: list[float] = []

    survival_at_one = torch.ones(num_samples, dtype=torch.float64)
    survival_fitted = torch.ones(num_samples, dtype=torch.float64)
    for position in range(gamma):
        position_logits = logits[:, position]
        position_target = prefix_mask[:, position]

        survival_at_one = survival_at_one * torch.sigmoid(position_logits)
        ece_before.append(
            expected_calibration_error(
                probs=survival_at_one,
                targets=position_target,
                num_bins=num_bins,
            )
        )

        best_temperature = grid_values[0]
        best_survival = survival_fitted * torch.sigmoid(
            position_logits / best_temperature
        )
        best_ece = expected_calibration_error(
            probs=best_survival, targets=position_target, num_bins=num_bins
        )
        for temperature in grid_values[1:]:
            candidate_survival = survival_fitted * torch.sigmoid(
                position_logits / temperature
            )
            candidate_ece = expected_calibration_error(
                probs=candidate_survival,
                targets=position_target,
                num_bins=num_bins,
            )
            if candidate_ece < best_ece:
                best_ece = candidate_ece
                best_temperature = temperature
                best_survival = candidate_survival

        temperatures.append(float(best_temperature))
        ece_after.append(float(best_ece))
        survival_fitted = best_survival

    return {
        "temperatures": temperatures,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }


def load_collected_shards(*, data_glob: str) -> tuple[torch.Tensor, torch.Tensor]:
    shard_paths = sorted(glob.glob(data_glob))
    if not shard_paths:
        raise ValueError(f"No STS data shards matched {data_glob!r}.")

    logits_shards: list[torch.Tensor] = []
    prefix_mask_shards: list[torch.Tensor] = []
    shard_gamma: Optional[int] = None
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu")
        shard_logits = shard["logits"]
        shard_prefix_mask = shard["prefix_mask"]
        if shard_logits.shape != shard_prefix_mask.shape:
            raise ValueError(
                f"Shard {shard_path!r} logits / prefix_mask shape mismatch: "
                f"{tuple(shard_logits.shape)} vs {tuple(shard_prefix_mask.shape)}."
            )
        if shard_gamma is None:
            shard_gamma = int(shard_logits.shape[1])
        elif int(shard_logits.shape[1]) != shard_gamma:
            raise ValueError(
                f"Shard {shard_path!r} gamma {int(shard_logits.shape[1])} disagrees "
                f"with earlier shards' gamma {shard_gamma}."
            )
        logits_shards.append(shard_logits)
        prefix_mask_shards.append(shard_prefix_mask)

    return torch.cat(logits_shards, dim=0), torch.cat(prefix_mask_shards, dim=0)


def fit(
    *,
    data_glob: str,
    out: Path,
    num_bins: int = 15,
    gamma: Optional[int] = None,
) -> None:
    logits, prefix_mask = load_collected_shards(data_glob=data_glob)
    resolved_gamma = int(logits.shape[1])
    if gamma is not None and gamma != resolved_gamma:
        raise ValueError(
            f"Collected shards have gamma={resolved_gamma} but --gamma={gamma}."
        )
    num_samples = int(logits.shape[0])

    result = fit_sts_temperatures(
        logits=logits,
        prefix_mask=prefix_mask,
        grid=default_temperature_grid(),
        num_bins=num_bins,
    )
    calibration = DSparkStsCalibration(
        temperatures=result["temperatures"],
        dataset=data_glob,
        num_samples=num_samples,
        ece_before=result["ece_before"],
        ece_after=result["ece_after"],
    )
    out.write_text(calibration.to_json(), encoding="utf-8")

    print(
        f"Fit STS temperatures over {num_samples} samples (gamma={resolved_gamma}) "
        f"-> {out}"
    )
    print("pos  temperature  ece_before  ece_after")
    for position in range(resolved_gamma):
        print(
            f"{position:>3}  {result['temperatures'][position]:>11.4f}  "
            f"{result['ece_before'][position]:>10.4f}  "
            f"{result['ece_after'][position]:>9.4f}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Fit DSpark Sequential Temperature Scaling (STS) calibration "
        "temperatures from collected confidence shards."
    )
    parser.add_argument(
        "--data-glob",
        required=True,
        help="Glob of collected .pt shards, each a dict with [n, gamma] "
        "'logits' and 'prefix_mask' tensors.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output STS calibration JSON path.",
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=15,
        help="Number of equal-width ECE bins.",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=None,
        help="Optional gamma override to validate the shards against.",
    )
    args = parser.parse_args()

    fit(
        data_glob=args.data_glob,
        out=args.out,
        num_bins=args.num_bins,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
