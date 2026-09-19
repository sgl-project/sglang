"""Throughput-aware adaptive speculative decoding policy.

Replaces the EMA-hysteresis decision logic with a throughput score:

    score(S) = E[tokens_produced | S steps drafted] / cost_ms(batch_size, S)

where:
  - ``E`` is derived from per-position sliding-window acceptance rates
    (shared across all batch sizes).
  - ``cost_ms`` comes from a startup cost table (real-path decode profiling).

Integration
-----------
``AdaptiveController`` owns CUDA-graph capture, runtime-state switching, and
profiling execution. This policy provides the decision and profiling inputs:

  * ``candidate_steps`` / ``cuda_graph_bs_for_step`` — runtime-state shape inputs.
  * ``get_steps_for_batch`` — sole decision point. Every
    ``update_interval`` batches (and only when all active positions have
    accumulated a full window), scores every candidate step and returns the
    winner.
  * ``on_verify_complete`` — data collection only.  Updates the per-position
    acceptance tracker and advances the batch counter.

Config JSON format
------------------
Set ``"strategy": "throughput_aware"`` in the
``--speculative-adaptive-config`` JSON to select this controller. The same file
contains the candidate steps and throughput-specific tuning knobs.

Integer-string keys are batch-size lower bounds (same as the standard
adaptive config); non-integer keys are throughput-specific settings::

    {
        "strategy": "throughput_aware",
        "window_size": 20,
        "update_interval": 5,
        "profile_run_batch_sizes": null,
        "max_profile_run_batch_size": null,
        "profile_run_n_warmup": 5,
        "profile_run_n_measure": 10,
        "profile_run_seq_len": 128,
        "switch_hysteresis": 0.1,
        "1":   {"candidate_steps": [1, 3, 5, 7]},
        "8":   {"candidate_steps": [1, 3, 5]},
        "32":  {"candidate_steps": [1, 3]},
        "128": {"candidate_steps": [1]}
    }

``profile_run_batch_sizes`` (optional list[int]): explicit list of batch
sizes to profile.  When ``null``, the server's ``cuda_graph_bs`` list is used
(filtered by ``max_profile_run_batch_size`` if set).

``max_profile_run_batch_size`` (optional int): upper bound on which batch
sizes are profiled (to keep startup time reasonable).

``profile_run_seq_len`` (optional int, default ``128``): synthetic context
length used by startup profiling.  Set it to ``null`` to retain automatic
selection based on the model context length.

``switch_hysteresis`` (optional float, default ``0.1``): fractional margin
required before switching steps.  A challenger must satisfy
``score_new > score_current * (1 + switch_hysteresis)``; at the default,
the new score must exceed 110% of the current step's score.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Optional

from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveSpecWorker,
    SpecProfilePlan,
    SpecProfilePoint,
)
from sglang.srt.speculative.adaptive_step_router import AdaptiveStepRouter
from sglang.srt.speculative.throughput_aware_spec_params import (
    BatchSizeCostTable,
    PositionAcceptanceTracker,
    format_position_rates,
    format_score_rows,
    pick_best_step,
    pick_best_step_with_hysteresis,
    score_candidates,
)
from sglang.srt.utils.common import log_debug_on_rank0, log_info_on_rank0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

DEFAULT_THROUGHPUT_AWARE_CONFIG: dict = {
    "window_size": 20,
    "update_interval": 5,
    "max_profile_run_batch_size": None,
    "profile_run_n_warmup": 5,
    "profile_run_n_measure": 10,
    "profile_run_seq_len": 128,
    "switch_hysteresis": 0.1,
    "1": {"candidate_steps": [1, 3, 5, 7]},
}


def load_throughput_aware_config(path: Optional[str]) -> dict:
    """Load and validate the throughput-aware JSON config.

    Uses ``DEFAULT_THROUGHPUT_AWARE_CONFIG`` when *path* is ``None``.
    """
    if path is None:
        return DEFAULT_THROUGHPUT_AWARE_CONFIG
    with open(path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"throughput-aware adaptive config must be a JSON object, "
            f"got {type(cfg).__name__}"
        )
    return cfg


def _parse_bs_candidates(cfg: dict) -> tuple[list[int], dict[int, list[int]]]:
    """Parse integer-string keys from config into (bs_list, bs_candidates).

    Returns:
        bs_list: Sorted list of batch-size lower-bound keys.
        bs_candidates: Mapping from each bs key to its ``candidate_steps``.

    Raises:
        ValueError: If no valid BS entries are found.
    """
    bs_candidates: dict[int, list[int]] = {}
    for key, entry in cfg.items():
        if not key.isdigit():
            continue
        if int(key) < 1:
            raise ValueError(
                f"throughput-aware batch-size key must be positive, got {key!r}"
            )
        if not isinstance(entry, dict):
            raise ValueError(
                f"throughput-aware config key '{key}' must map to a JSON object, "
                f"got {type(entry).__name__}"
            )
        steps = entry.get("candidate_steps")
        if (
            not isinstance(steps, list)
            or not steps
            or not all(
                isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in steps
            )
        ):
            raise ValueError(
                f"throughput-aware config key '{key}': "
                f"candidate_steps must be a non-empty list of positive ints, "
                f"got {steps!r}"
            )
        bs_candidates[int(key)] = sorted(set(steps))

    if not bs_candidates:
        raise ValueError(
            "throughput-aware adaptive config must contain at least one "
            'integer-string BS key, e.g. {"1": {"candidate_steps": [1, 3, 7]}}. '
            f"Got keys: {list(cfg.keys())}"
        )
    return sorted(bs_candidates), bs_candidates


def _config_int(cfg: dict, name: str, default: int, *, minimum: int) -> int:
    value = cfg.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _config_optional_positive_int(
    cfg: dict, name: str, default: Optional[int] = None
) -> Optional[int]:
    value = cfg.get(name, default)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer or null, got {value!r}")
    return value


def _config_profile_batch_sizes(cfg: dict) -> Optional[list[int]]:
    value = cfg.get("profile_run_batch_sizes")
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or not value
        or not all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in value
        )
    ):
        raise ValueError(
            "profile_run_batch_sizes must be a non-empty list of positive "
            f"integers or null, got {value!r}"
        )
    return sorted(set(value))


def resolve_throughput_aware_candidate_steps(
    cfg_path: Optional[str] = None,
) -> list[int]:
    """Return the union of all candidate steps across all BS slots.

    Used by ``server_args.max_speculative_num_draft_tokens`` to pre-size
    KV cache buffers.
    """
    cfg = load_throughput_aware_config(cfg_path)
    _, bs_candidates = _parse_bs_candidates(cfg)
    all_steps: set[int] = set()
    for steps in bs_candidates.values():
        all_steps.update(steps)
    return sorted(all_steps)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class ThroughputAwarePolicy:
    """Choose adaptive steps by scoring expected output tokens per decode cost."""

    def __init__(self, initial_steps: int, config_path: Optional[str] = None):
        cfg = load_throughput_aware_config(config_path)
        _, bs_candidates = _parse_bs_candidates(cfg)
        self._router = AdaptiveStepRouter(bs_candidates)
        all_candidate_steps = self._router.candidate_steps
        self._all_candidate_steps = all_candidate_steps

        window_size = _config_int(cfg, "window_size", 20, minimum=1)
        self._update_interval = _config_int(cfg, "update_interval", 5, minimum=1)
        self._tracker = PositionAcceptanceTracker(
            max_steps=max(all_candidate_steps),
            window_size=window_size,
        )
        self._cost_table = BatchSizeCostTable()

        self._profile_batch_sizes = _config_profile_batch_sizes(cfg)
        self._max_profile_bs = _config_optional_positive_int(
            cfg, "max_profile_run_batch_size"
        )
        self._profile_n_warmup = _config_int(cfg, "profile_run_n_warmup", 5, minimum=0)
        self._profile_n_measure = _config_int(
            cfg, "profile_run_n_measure", 10, minimum=1
        )
        self._profile_run_seq_len = _config_optional_positive_int(
            cfg, "profile_run_seq_len", default=128
        )
        switch_hysteresis = cfg.get("switch_hysteresis", 0.1)
        if (
            not isinstance(switch_hysteresis, (int, float))
            or isinstance(switch_hysteresis, bool)
            or not math.isfinite(switch_hysteresis)
            or switch_hysteresis < 0
        ):
            raise ValueError(
                "switch_hysteresis must be a finite non-negative number, "
                f"got {switch_hysteresis!r}"
            )
        self._switch_hysteresis = float(switch_hysteresis)

        first_candidates = self._router.candidates_for_batch(0)
        self._current_steps = initial_steps
        if self._current_steps not in all_candidate_steps:
            self._current_steps = first_candidates[len(first_candidates) // 2]
        self._batches_since_reevaluation = 0

        log_info_on_rank0(
            logger,
            f"ThroughputAwarePolicy initialized: "
            f"bs_list={self._router.batch_size_keys}, "
            f"all_candidate_steps={self._all_candidate_steps}, "
            f"initial_steps={self._current_steps}, "
            f"window_size={window_size}, "
            f"update_interval={self._update_interval}, "
            f"switch_hysteresis={self._switch_hysteresis}, "
            f"profile_run_seq_len={self._profile_run_seq_len!r}",
        )

    @property
    def candidate_steps(self) -> list[int]:
        return self._router.candidate_steps

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        """cuda_graph_bs values where step is a valid candidate (prunes graph captures)."""
        return self._router.cuda_graph_bs_for_step(step)

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        self._router.set_cuda_graph_bs(cuda_graph_bs)

    def _resolve_profile_seq_len(self, worker: AdaptiveSpecWorker) -> int:
        """Prefill context length for profiling (config or auto, clamped to context_length)."""
        ctx = int(worker.model_config.context_len)
        max_step = max(self._all_candidate_steps) if self._all_candidate_steps else 1
        decode_growth = (self._profile_n_warmup + self._profile_n_measure) * (
            max_step + 1
        )
        headroom = decode_growth + 16
        if ctx <= headroom:
            raise ValueError(
                "throughput-aware profiling needs context_length greater than "
                f"warmup/measurement headroom, got {ctx} <= {headroom}"
            )
        default_len = min(2048, max(256, ctx - headroom))
        seq_len = self._profile_run_seq_len or default_len
        return int(max(1, min(seq_len, ctx - headroom)))

    def profile_plan(
        self, worker: AdaptiveSpecWorker, *, max_running_requests: int
    ) -> SpecProfilePlan:
        points = self._build_profile_points(max_running_requests)
        return SpecProfilePlan(
            points=points,
            seq_len=self._resolve_profile_seq_len(worker) if points else 1,
            n_warmup=self._profile_n_warmup,
            n_measure=self._profile_n_measure,
        )

    def record_profile(self, batch_size: int, steps: int, median_ms: float) -> None:
        self._cost_table.set(batch_size, steps, median_ms)
        log_debug_on_rank0(
            logger,
            f"[ThroughputAware] steps={steps:2d}  bs={batch_size:4d}  "
            f"decode_median={median_ms:.3f}ms",
        )

    def profile_summary(self) -> str:
        return self._cost_table.summary()

    def _build_profile_points(
        self, max_running_requests: int
    ) -> tuple[SpecProfilePoint, ...]:
        """Return the valid (steps, batch-size) measurements to run."""
        cuda_graph_bs = self._router.cuda_graph_bs
        if cuda_graph_bs is None:
            return ()
        pool = (
            sorted(set(self._profile_batch_sizes) & set(cuda_graph_bs))
            if self._profile_batch_sizes is not None
            else list(cuda_graph_bs)
        )
        if self._max_profile_bs is not None:
            pool = [b for b in pool if b <= self._max_profile_bs]
        pool = [b for b in pool if b <= max_running_requests]
        return tuple(
            SpecProfilePoint(steps, batch_size)
            for steps in self._all_candidate_steps
            for batch_size in sorted(
                set(pool) & set(self.cuda_graph_bs_for_step(steps) or [])
            )
        )

    def get_steps_for_batch(self, batch_size: int) -> int:
        """Pick best step every update_interval when positions are warmed."""
        candidates = self._router.candidates_for_batch(batch_size)
        if self._current_steps not in candidates:
            # Batch-size routing constrains which graphs were captured. Honor
            # it even at cold start or between periodic score updates.
            rows = score_candidates(
                self._tracker, self._cost_table, candidates, batch_size
            )
            target = pick_best_step(rows, fallback=candidates[len(candidates) // 2])
            if target < self._current_steps:
                self._tracker.clear_positions_above(target)
            self._current_steps = target

        if self._batches_since_reevaluation >= self._update_interval:
            if self._should_reevaluate():
                self._reevaluate_and_switch(batch_size)
            self._batches_since_reevaluation = 0

        return self._current_steps

    def on_verify_complete(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int = 0,
        num_steps: int | None = None,
    ) -> int | None:
        """Update acceptance tracker only (no step switch here)."""
        if not num_correct_drafts_per_req:
            return None
        observed_steps = self._current_steps if num_steps is None else num_steps
        self._tracker.update(num_correct_drafts_per_req, observed_steps)
        self._batches_since_reevaluation += 1
        return None

    def on_state_activated(self, steps: int) -> None:
        self._current_steps = steps

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _should_reevaluate(self) -> bool:
        """Return True if this is a valid moment to re-score candidates."""
        return (
            self._tracker.all_positions_warmed(self._current_steps)
            and not self._cost_table.is_empty()
        )

    def _reevaluate_and_switch(self, batch_size: int) -> None:
        """Score all candidates for the given batch size and switch if beneficial."""
        candidates = self._router.candidates_for_batch(batch_size)
        rows = score_candidates(self._tracker, self._cost_table, candidates, batch_size)
        raw_best = pick_best_step(rows, fallback=self._current_steps)
        best_steps = pick_best_step_with_hysteresis(
            rows,
            current_steps=self._current_steps,
            hysteresis=self._switch_hysteresis,
        )

        logger.debug(
            "[ThroughputAware] batches_since_reevaluation=%d  bs=%d  "
            "pos_rates=%s  scores=%s",
            self._batches_since_reevaluation,
            batch_size,
            format_position_rates(self._tracker, max(candidates) if candidates else 0),
            format_score_rows(rows, raw_best),
        )
        if raw_best != best_steps:
            score_map = {r["steps"]: r["score"] for r in rows}
            current_score = score_map.get(self._current_steps)
            raw_score = score_map.get(raw_best)
            logger.debug(
                "[ThroughputAware] hysteresis blocked switch: "
                "raw_best=%d (score=%s) vs current=%d (score=%s), "
                "need score > %.4f (margin=%.2f)",
                raw_best,
                f"{raw_score:.4f}" if raw_score is not None else "n/a",
                self._current_steps,
                f"{current_score:.4f}" if current_score is not None else "n/a",
                (current_score or 0) * (1.0 + self._switch_hysteresis),
                self._switch_hysteresis,
            )

        if best_steps != self._current_steps:
            old_steps = self._current_steps
            direction = "expand" if best_steps > old_steps else "shrink"

            # Extract throughput scores for old and new steps (for the INFO summary).
            score_map = {r["steps"]: r["score"] for r in rows}
            old_score = score_map.get(old_steps)
            new_score = score_map.get(best_steps)
            score_summary = (
                f"{old_score:.4f} → {new_score:.4f} tok/ms"
                if old_score is not None and new_score is not None
                else "n/a"
            )

            if best_steps < old_steps:
                self._tracker.clear_positions_above(best_steps)
            self._current_steps = best_steps
            log_info_on_rank0(
                logger,
                f"[ThroughputAware] Step {direction}: {old_steps} → {best_steps}  "
                f"(bs={batch_size}, "
                f"batches_since_reevaluation={self._batches_since_reevaluation}, "
                f"throughput={score_summary})",
            )
