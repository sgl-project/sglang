"""Throughput-aware adaptive speculative decoding controller.

Replaces the EMA-hysteresis decision logic with a throughput score:

    score(S) = E[tokens_produced | S steps drafted] / cost_ms(batch_size, S)

where:
  - ``E`` is derived from per-position sliding-window acceptance rates
    (shared across all batch sizes).
  - ``cost_ms`` comes from a startup cost table (real-path decode profiling).

Integration
-----------
This controller inherits all CUDA-graph / runtime-state switching machinery
from :class:`AdaptiveController` and overrides only the decision logic:

  * ``init_states`` — build SpecRuntimeStates; profiling deferred to ``run_profiling``.
  * ``activate_step_by_batch`` — sole decision point.  Every
    ``update_interval`` batches (and only when all active positions have
    accumulated a full window), scores every candidate step and activates the
    winner.
  * ``on_verify_complete`` — data collection only.  Updates the per-position
    acceptance tracker and advances the batch counter.

Config JSON format
------------------
Integer-string keys are batch-size lower bounds (same as the standard
adaptive config); non-integer keys are throughput-specific settings::

    {
        "window_size": 20,
        "update_interval": 10,
        "profile_run_batch_sizes": null,
        "max_profile_run_batch_size": 128,
        "profile_run_n_warmup": 5,
        "profile_run_n_measure": 10,
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
"""

from __future__ import annotations

import bisect
import json
import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from sglang.srt.speculative.adaptive_runtime_state import _SpecAdaptiveBase
from sglang.srt.speculative.throughput_aware_spec_params import (
    BatchSizeCostTable,
    PositionAcceptanceTracker,
    format_position_rates,
    format_score_rows,
    pick_best_step,
    score_candidates,
)
from sglang.srt.utils.common import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState

logger = logging.getLogger(__name__)


def _broadcast_float_from_rank0(value: float) -> float:
    """Broadcast a float from TP rank 0 to all ranks; no-op when TP=1 or dist not active."""
    if not dist.is_initialized():
        return value
    try:
        from sglang.srt.distributed import (
            get_tensor_model_parallel_world_size,
            get_tp_group,
        )
        if get_tensor_model_parallel_world_size() <= 1:
            return value
        t = torch.tensor([value], dtype=torch.float64,
                         device=get_tp_group().device)
        dist.broadcast(t, src=0, group=get_tp_group().device_group)
        return float(t.item())
    except Exception:
        return value


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_throughput_aware_config(path: Optional[str]) -> dict:
    """Load and validate the throughput-aware JSON config.

    Returns an empty dict when *path* is ``None`` (caller should handle the
    missing-config case).
    """
    if path is None:
        return {}
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
        steps = entry.get("candidate_steps")
        if (
            not isinstance(steps, list)
            or not steps
            or not all(isinstance(s, int) and s > 0 for s in steps)
        ):
            raise ValueError(
                f"throughput-aware config key '{key}': "
                f"candidate_steps must be a non-empty list of positive ints, "
                f"got {steps!r}"
            )
        bs_candidates[int(key)] = sorted(steps)

    if not bs_candidates:
        raise ValueError(
            "throughput-aware adaptive config must contain at least one "
            'integer-string BS key, e.g. {"1": {"candidate_steps": [1, 3, 7]}}. '
            f"Got keys: {list(cfg.keys())}"
        )
    return sorted(bs_candidates), bs_candidates


def resolve_throughput_aware_candidate_steps(cfg_path: str) -> list[int]:
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


class ThroughputAwareAdaptiveController(_SpecAdaptiveBase):
    """Throughput score = E[tokens] / cost_ms; decision in activate_step_by_batch."""

    def __init__(self, worker, config_path: Optional[str] = None):
        super().__init__(worker)
        cfg = load_throughput_aware_config(config_path)
        self._bs_list, self._bs_candidates = _parse_bs_candidates(cfg)

        self._cuda_graph_bs: list[int] | None = None

        all_candidate_steps = sorted(
            {s for steps in self._bs_candidates.values() for s in steps}
        )
        self._all_candidate_steps: list[int] = all_candidate_steps

        window_size: int = int(cfg.get("window_size", 20))
        self._update_interval: int = int(cfg.get("update_interval", 10))
        self._tracker = PositionAcceptanceTracker(
            max_steps=max(all_candidate_steps),
            window_size=window_size,
        )
        self._cost_table = BatchSizeCostTable()

        self._profile_batch_sizes: Optional[list[int]] = cfg.get("profile_run_batch_sizes")
        self._max_profile_bs: Optional[int] = cfg.get("max_profile_run_batch_size")
        self._profile_n_warmup: int = int(cfg.get("profile_run_n_warmup", 5))
        self._profile_n_measure: int = int(cfg.get("profile_run_n_measure", 10))
        self._profile_run_seq_len: Optional[int] = cfg.get("profile_run_seq_len")

        first_candidates = self._bs_candidates[self._bs_list[0]]
        self._current_steps: int = worker.speculative_num_steps
        if self._current_steps not in all_candidate_steps:
            self._current_steps = first_candidates[len(first_candidates) // 2]
        self._batch_count: int = 0

        log_info_on_rank0(
            logger,
            f"ThroughputAwareAdaptiveController initialized: "
            f"bs_list={self._bs_list}, "
            f"all_candidate_steps={self._all_candidate_steps}, "
            f"initial_steps={self._current_steps}, "
            f"window_size={window_size}, "
            f"update_interval={self._update_interval}, "
            f"profile_run_seq_len={self._profile_run_seq_len!r}",
        )

    @property
    def candidate_steps(self) -> list[int]:
        return self._all_candidate_steps

    def _find_closest_bs_key(self, target: int) -> int:
        idx = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, idx)]

    def _candidates_for_batch(self, batch_size: int) -> list[int]:
        if self._cuda_graph_bs is not None:
            idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
            if idx < len(self._cuda_graph_bs):
                batch_size = self._cuda_graph_bs[idx]
        return self._bs_candidates[self._find_closest_bs_key(batch_size)]

    def _cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        """cuda_graph_bs values where step is a valid candidate (prunes graph captures)."""
        if self._cuda_graph_bs is None:
            return None
        return [
            bs for bs in self._cuda_graph_bs
            if step in self._bs_candidates[self._find_closest_bs_key(bs)]
        ]

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        """Build SpecRuntimeStates for all candidate steps."""
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

        for steps in self._all_candidate_steps:
            if steps in self._states:
                continue
            pruned_bs = self._cuda_graph_bs_for_step(steps)
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                cuda_graph_bs=pruned_bs,
            )
            self._states[steps] = state

        # Profiling needs tree_cache; runs in run_profiling() before serving.
        self._activate(self._current_steps)

    def _resolve_profile_seq_len(self) -> int:
        """Prefill context length for profiling (config or auto, clamped to context_length)."""
        ctx = int(getattr(self.worker.server_args, "context_length", None) or 4096)
        max_step = max(self._all_candidate_steps) if self._all_candidate_steps else 1
        decode_growth = (self._profile_n_warmup + self._profile_n_measure) * (
            max_step + 1
        )
        headroom = decode_growth + 16
        default_len = min(2048, max(256, ctx - headroom))
        seq_len = self._profile_run_seq_len or default_len
        return int(max(1, min(seq_len, ctx - headroom)))

    def run_profiling(self, tree_cache) -> None:
        """Fill cost table via SpecProfilingSession for each (steps, batch_size)."""
        from sglang.srt.speculative.spec_profiling_session import SpecProfilingSession

        steps_to_profile_bs = self._build_profile_grid()
        if not any(steps_to_profile_bs.values()):
            log_info_on_rank0(
                logger,
                "[ThroughputAware] No batch sizes to profile; "
                "cost table will be empty.",
            )
            return

        seq_len = self._resolve_profile_seq_len()
        original_steps = self._current_steps

        all_points = [
            (steps, bs)
            for steps, bs_list in sorted(steps_to_profile_bs.items())
            for bs in sorted(bs_list)
        ]

        log_info_on_rank0(
            logger,
            f"[ThroughputAware] Starting cost table profiling: "
            f"{len(all_points)} points, seq_len={seq_len}, "
            f"n_warmup={self._profile_n_warmup}, n_measure={self._profile_n_measure}",
        )

        for steps, bs in all_points:
            self._current_steps = steps  # pin step during profiling
            self._activate(steps)

            avg_ms = SpecProfilingSession(
                worker=self.worker,
                tree_cache=tree_cache,
                batch_size=bs,
                num_steps=steps,
                seq_len=seq_len,
                n_warmup=self._profile_n_warmup,
                n_measure=self._profile_n_measure,
            ).measure()

            # Sync cost across TP ranks so all ranks make identical step decisions.
            avg_ms = _broadcast_float_from_rank0(avg_ms)

            self._cost_table.set(bs, steps, avg_ms)
            log_info_on_rank0(
                logger,
                f"[ThroughputAware] steps={steps:2d}  bs={bs:4d}  "
                f"seq_len={seq_len}  decode_avg={avg_ms:.3f}ms",
            )

        self._current_steps = original_steps
        self._activate(original_steps)

        log_info_on_rank0(
            logger,
            f"[ThroughputAware] Cost table ready: {self._cost_table.summary()}",
        )

    def _build_profile_grid(self) -> dict[int, list[int]]:
        """Map num_steps -> batch sizes to profile."""
        if self._cuda_graph_bs is None:
            return {}
        pool = (
            sorted(set(self._profile_batch_sizes) & set(self._cuda_graph_bs))
            if self._profile_batch_sizes is not None
            else list(self._cuda_graph_bs)
        )
        if self._max_profile_bs is not None:
            pool = [b for b in pool if b <= self._max_profile_bs]
        return {
            steps: profiled
            for steps in self._all_candidate_steps
            if (profiled := sorted(set(pool) & set(self._cuda_graph_bs_for_step(steps) or [])))
        }

    def activate_step_by_batch(self, batch_size: int) -> None:
        """Pick best step every update_interval when positions are warmed."""
        if self._should_reevaluate():
            self._reevaluate_and_switch(batch_size)

        # Apply current step (no-op if already active).
        if self._current_steps != self.worker.speculative_num_steps:
            self._activate(self._current_steps)

    def on_verify_complete(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int = 0,
    ) -> None:
        """Update acceptance tracker only (no step switch here)."""
        if not num_correct_drafts_per_req:
            return
        self._tracker.update(num_correct_drafts_per_req, self._current_steps)
        self._batch_count += 1

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _should_reevaluate(self) -> bool:
        """Return True if this is a valid moment to re-score candidates."""
        return (
            self._batch_count > 0
            and self._batch_count % self._update_interval == 0
            and self._tracker.all_positions_warmed(self._current_steps)
            and not self._cost_table.is_empty()
        )

    def _reevaluate_and_switch(self, batch_size: int) -> None:
        """Score all candidates for the given batch size and switch if beneficial."""
        candidates = self._candidates_for_batch(batch_size)
        rows = score_candidates(self._tracker, self._cost_table, candidates, batch_size)
        best_steps = pick_best_step(rows, fallback=self._current_steps)

        pos_rates_str = format_position_rates(self._tracker, max(candidates) if candidates else 0)
        scores_str = format_score_rows(rows, best_steps)
        logger.debug(
            f"[ThroughputAware] batch_count={self._batch_count}  "
            f"bs={batch_size}  pos_rates={pos_rates_str}  scores={scores_str}"
        )

        if best_steps != self._current_steps:
            old_steps = self._current_steps
            direction = "expand" if best_steps > old_steps else "shrink"
            if best_steps < old_steps:
                self._tracker.clear_positions_above(best_steps)
            self._current_steps = best_steps
            log_info_on_rank0(
                logger,
                f"[ThroughputAware] Step {direction}: {old_steps} → {best_steps}  "
                f"(bs={batch_size}, batch_count={self._batch_count})  "
                f"pos_rates={pos_rates_str}  scores={scores_str}",
            )
