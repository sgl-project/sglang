# SPDX-License-Identifier: Apache-2.0
"""TeaCache coefficient calibration.

Records, per denoising step, the relative change of the modulated input
(``diff_tm``) and of the transformer output (``diff_x``), averaged across CFG
branches and samples. ``fit`` regresses ``diff_x`` on ``diff_tm`` as a degree-4
polynomial and derives a threshold from the flattest stretch of ``diff_x``.

A process-global calibrator is set while the calibration tool runs; DiT models
forward their per-step tensors via ``TeaCacheMixin.maybe_record_calibration``.
"""

from __future__ import annotations

import json

import numpy as np
import torch

# Expert tag for single-transformer models (no MoE high/low split).
SINGLE_EXPERT = "single"

_active_calibrator: TeaCacheCalibrator | None = None


def get_active_calibrator() -> TeaCacheCalibrator | None:
    return _active_calibrator


def set_active_calibrator(calibrator: TeaCacheCalibrator | None) -> None:
    global _active_calibrator
    _active_calibrator = calibrator


def record_from_env(
    modulated_inp: torch.Tensor,
    output: torch.Tensor,
    *,
    step_index: int,
    expert: str = SINGLE_EXPERT,
) -> None:
    """Record one step for calibration, gated by env (for models without the mixin).

    Lazily creates the process-global calibrator and dumps rows to
    ``SGLANG_TEACACHE_CALIBRATE_OUT`` so the fit step can pick them up.
    """
    from sglang.multimodal_gen import envs

    if not envs.SGLANG_TEACACHE_CALIBRATE:
        return
    # Record on global rank 0 only: avoids workers racing on the rows file, and
    # ties the diffs to rank 0's shard -- a shard-local approximation that is
    # adequate since the relative-diff signal is near-uniform across shards.
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    calibrator = get_active_calibrator()
    if calibrator is None:
        calibrator = TeaCacheCalibrator()
        set_active_calibrator(calibrator)
    calibrator.record(
        modulated_inp,
        output,
        step_index=step_index,
        is_cfg_negative=False,
        expert=expert,
    )
    out_path = envs.SGLANG_TEACACHE_CALIBRATE_OUT
    if out_path:
        calibrator.dump_rows(out_path)


def _relative_diff(t2: torch.Tensor, t1: torch.Tensor, eps: float = 1e-8) -> float:
    return float((torch.abs(t2 - t1).mean() / (torch.abs(t1).mean() + eps)).item())


def calculate_threshold(y_data: np.ndarray, slope_threshold: float = 0.01) -> float:
    """Twice the mean of ``diff_x`` over its longest run of near-flat slope."""
    if len(y_data) <= 1:
        return 0.0

    slopes = np.diff(y_data)
    valid_indices = np.where(np.abs(slopes) < slope_threshold)[0]

    max_group: list[int] = []
    current_group: list[int] = []
    for idx in valid_indices:
        if not current_group or idx == current_group[-1] + 1:
            current_group.append(int(idx))
        else:
            if len(current_group) > len(max_group):
                max_group = current_group
            current_group = [int(idx)]
    if len(current_group) > len(max_group):
        max_group = current_group

    if max_group:
        # slope i spans y[i..i+1], so the flat run ends at index max_group[-1]+1.
        longest_y = y_data[max_group[0] : max_group[-1] + 2]
        return float(np.mean(longest_y) * 2)
    return 0.2


class TeaCacheCalibrator:
    """Accumulates per-step (diff_tm, diff_x) and fits TeaCache coefficients.

    Records are bucketed by ``expert`` (``"single"`` for non-MoE models; e.g.
    ``"high"``/``"low"`` for Wan2.2's two transformers) and, within an expert,
    by denoising step index. Each step's value is the running mean over every
    CFG branch and sample that reached it.
    """

    def __init__(self, degree: int = 4, slope_threshold: float = 0.01) -> None:
        self.degree = degree
        self.slope_threshold = slope_threshold
        # expert -> {"prev": {branch_key: (prev_e, prev_x)},
        #            "rows": {step_index: [count, diff_tm_mean, diff_x_mean]}}
        self._experts: dict[str, dict] = {}
        self._seen_nonzero_step = False

    def _expert(self, expert: str) -> dict:
        return self._experts.setdefault(expert, {"prev": {}, "rows": {}})

    def begin_sample(self) -> None:
        """Drop carried predecessors for every expert at a sample boundary.

        The MoE low-noise expert's first call happens at a later global step, so
        its ``prev`` would otherwise diff against the previous sample's last step.
        """
        for state in self._experts.values():
            state["prev"].clear()
        self._seen_nonzero_step = False

    def record(
        self,
        modulated_inp: torch.Tensor,
        output: torch.Tensor,
        *,
        step_index: int,
        is_cfg_negative: bool,
        expert: str = SINGLE_EXPERT,
    ) -> None:
        """Record one transformer call.

        The first call of each (expert, branch) has no predecessor and yields a
        zero diff, which is harmless since early steps are always computed.
        """
        # A global step 0 after any higher step is a new sample: drop every
        # expert's predecessor (the low-noise expert never records step 0 itself).
        if step_index == 0 and self._seen_nonzero_step:
            self.begin_sample()
        if step_index > 0:
            self._seen_nonzero_step = True

        state = self._expert(expert)
        branch_key = "neg" if is_cfg_negative else "pos"
        prev = state["prev"].get(branch_key)

        e = modulated_inp.detach()
        x = output.detach()
        if prev is None:
            diff_tm, diff_x = 0.0, 0.0
        else:
            prev_e, prev_x = prev
            diff_tm = _relative_diff(e, prev_e)
            diff_x = _relative_diff(x, prev_x)
        state["prev"][branch_key] = (e.clone(), x.clone())

        row = state["rows"].get(step_index)
        if row is None:
            state["rows"][step_index] = [1, diff_tm, diff_x]
        else:
            n = row[0] + 1
            row[1] += (diff_tm - row[1]) / n
            row[2] += (diff_x - row[2]) / n
            row[0] = n

    def _fit_expert(self, state: dict) -> dict:
        steps = sorted(state["rows"])
        diff_tm = np.array([state["rows"][s][1] for s in steps], dtype=np.float64)
        diff_x = np.array([state["rows"][s][2] for s in steps], dtype=np.float64)
        coefficients = np.polyfit(diff_tm, diff_x, self.degree)
        threshold = calculate_threshold(diff_x, self.slope_threshold)
        return {
            "coefficients": coefficients.tolist(),
            "teacache_thresh": float(threshold),
        }

    def fit(self) -> dict:
        """Fit coefficients + threshold per expert.

        Returns ``{"coefficients": [...], "teacache_thresh": float}`` for a
        single-expert (non-MoE) run, otherwise ``{expert: {...}, ...}``.
        """
        result = {expert: self._fit_expert(s) for expert, s in self._experts.items()}
        if set(result) == {SINGLE_EXPERT}:
            return result[SINGLE_EXPERT]
        return result

    def to_rows_dict(self) -> dict:
        """Serialize accumulated rows so another process can fit them.

        The DiT forward runs in a worker subprocess, so the worker records and
        dumps these rows to a file; the calibration tool loads them and fits.
        """
        return {
            expert: {str(step): row for step, row in s["rows"].items()}
            for expert, s in self._experts.items()
        }

    def dump_rows(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_rows_dict(), f)

    @classmethod
    def from_rows_dict(
        cls, rows: dict, degree: int = 4, slope_threshold: float = 0.01
    ) -> TeaCacheCalibrator:
        calib = cls(degree=degree, slope_threshold=slope_threshold)
        for expert, steps in rows.items():
            calib._experts[expert] = {
                "prev": {},
                "rows": {int(step): list(row) for step, row in steps.items()},
            }
        return calib

    def save_json(self, path: str) -> dict:
        result = self.fit()
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        return result


def merge_rows_dicts(rows_dicts: list[dict]) -> dict:
    """Count-weighted merge of several ``to_rows_dict()`` outputs.

    Merging per-shard rows by count-weighted mean per (expert, step) matches
    running a single process over all prompts, without needing NCCL.
    """
    merged: dict[str, dict[str, list]] = {}
    for rows in rows_dicts:
        for expert, steps in rows.items():
            dst = merged.setdefault(expert, {})
            for step, row in steps.items():
                count, diff_tm, diff_x = row
                if step not in dst:
                    dst[step] = [count, diff_tm, diff_x]
                    continue
                c0, tm0, x0 = dst[step]
                total = c0 + count
                dst[step] = [
                    total,
                    (tm0 * c0 + diff_tm * count) / total,
                    (x0 * c0 + diff_x * count) / total,
                ]
    return merged
