"""Adaptive speculative decoding parameters.

Adjusts speculative_num_steps at runtime based on observed acceptance lengths.
"""

from __future__ import annotations

import bisect
import json
import logging
from typing import TYPE_CHECKING, Dict, Optional

from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default per-BS hysteresis for Route 1 (accept_len + hysteresis)
# Keys are lower bounds of BS ranges: 1 covers [1,64), 64 covers [64,128), etc.
# ---------------------------------------------------------------------------
DEFAULT_BS_HYSTERESIS: Dict[int, Dict[str, float]] = {
    1: {"up_hysteresis": 0.0, "down_hysteresis": -0.25},
    64: {"up_hysteresis": 0.0, "down_hysteresis": 0.0},
    128: {"up_hysteresis": -0.1, "down_hysteresis": 0.25},
}

DEFAULT_BS_STEPS: Dict[int, list] = {
    1: [1, 3, 7],
    64: [1, 2, 5],
    128: [1, 2, 6],
}


def adaptive_unsupported_reason(server_args: ServerArgs) -> str | None:
    """Return why adaptive spec cannot run under the given server args, or None if supported."""
    if server_args.speculative_algorithm not in ("EAGLE", "EAGLE3"):
        return (
            f"speculative_algorithm={server_args.speculative_algorithm} "
            "(only EAGLE/EAGLE3 are supported)"
        )
    if server_args.speculative_eagle_topk != 1:
        return (
            f"speculative_eagle_topk={server_args.speculative_eagle_topk} "
            "(only topk=1 is supported)"
        )
    if server_args.enable_dp_attention:
        return (
            "enable_dp_attention=True is not supported "
            "(adaptive tier decisions are not synchronized across DP ranks)"
        )
    if server_args.enable_multi_layer_eagle:
        return (
            "enable_multi_layer_eagle=True is not supported "
            "(MultiLayerEagleWorker does not implement adaptive)"
        )
    if server_args.enable_two_batch_overlap:
        return (
            "enable_two_batch_overlap=True is not supported "
            "(adaptive state swap would discard the TboAttnBackend wrapper)"
        )
    if server_args.enable_pdmux:
        return (
            "enable_pdmux=True is not supported "
            "(adaptive state swap does not update decode_attn_backend_group)"
        )
    return None


def _bisect_lookup(table: Dict[int, dict], bs: int) -> dict:
    """Find the entry in *table* whose key is the largest <= *bs*."""
    keys = sorted(table.keys())
    idx = bisect.bisect_right(keys, bs) - 1
    idx = max(0, idx)
    return table[keys[idx]]


def get_default_hysteresis(bs: int) -> Dict[str, float]:
    return _bisect_lookup(DEFAULT_BS_HYSTERESIS, bs)


def load_adaptive_config(path: str | None) -> dict:
    """Load adaptive speculative config from a JSON file.

    The file is a flat JSON object. Integer keys (``"1"``, ``"64"``, …) are
    per-BS entries parsed by :func:`load_bs_config`.  All other keys
    (``ema_alpha``, ``update_interval``, ``warmup_batches``, …) are global
    overrides applied to every BS slot.

    Returns an empty dict when *path* is ``None``.
    """
    if path is None:
        return {}
    with open(path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            "speculative_adaptive_config must be a JSON object, "
            f"got {type(cfg).__name__}"
        )
    return cfg


def load_bs_config(config: dict) -> Optional[Dict[int, dict]]:
    """Parse per-BS config from the loaded JSON.

    The config is a flat dict whose keys are BS lower-bound strings::

        {"1": {"steps": [1,3,7], ...}, "64": {"steps": [1,3,5], ...}}

    Non-integer keys (e.g. ``ema_alpha``) are treated as global overrides
    and ignored here — they are passed through in *config* directly.

    Returns ``{bs_int: entry_dict}`` or ``None`` when no BS entries found.
    """
    result: Dict[int, dict] = {}
    for key, entry in config.items():
        try:
            bs = int(key)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            result[bs] = {"steps": entry if isinstance(entry, list) else []}
        else:
            result[bs] = entry

    return result if result else None


class AdaptiveSpeculativeParams:
    """Tracks acceptance rate via EMA and adapts num_steps accordingly.

    The core idea: if drafts are consistently accepted, try more steps;
    if drafts are consistently rejected early, reduce steps to avoid waste.

    Formula: target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)
    - Probes one step beyond observed acceptance
    - EMA smoothing prevents oscillation
    - Only updates every `update_interval` batches for stability
    - num_steps can be selected from different candidate sets on different batch_sizes
    """

    def __init__(
        self,
        initial_steps: int,
        config: dict | None = None,
    ):
        cfg = config or {}
        candidates = sorted(set(cfg.get("candidate_steps", [1, 3, 7])))

        assert len(candidates) >= 1, "candidate_steps must have at least 1 value"
        self.candidate_steps = candidates

        self.min_steps = self.candidate_steps[0]
        self.max_steps = self.candidate_steps[-1]
        self.ema_alpha = cfg.get("ema_alpha", 0.2)
        self.update_interval = cfg.get("update_interval", 5)
        self.warmup_batches = cfg.get("warmup_batches", 10)
        self.down_hysteresis = cfg.get("down_hysteresis", -0.25)
        self.up_hysteresis = cfg.get("up_hysteresis", 0.0)

        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]
            log_info_on_rank0(
                logger,
                f"initial_steps={initial_steps} not in "
                f"candidate_steps={self.candidate_steps}, "
                f"snapping to middle entry {self.current_steps}",
            )

        # Initialize EMA at current steps - 1 (neutral starting point)
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0

        log_info_on_rank0(
            logger,
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={self.current_steps}, candidate_steps={self.candidate_steps}",
        )

    def update(self, num_correct_drafts_per_req: list[int]) -> bool:
        """Update EMA with observed accept lengths. Returns True if params changed.

        Args:
            num_correct_drafts_per_req: Per-request accepted draft token counts from last verify.
        """
        if not num_correct_drafts_per_req:
            return False

        batch_avg = sum(num_correct_drafts_per_req) / len(num_correct_drafts_per_req)
        self.ema_accept_len = (
            1 - self.ema_alpha
        ) * self.ema_accept_len + self.ema_alpha * batch_avg

        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False

        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False

        return self._recompute_params()

    def _recompute_params(self) -> bool:
        """Recompute steps from EMA. Returns True if params changed."""
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)

        # TODO: Consider limiting step changes to avoid overshooting.
        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            drop_threshold = prev_step - 0.5 + self.down_hysteresis
            if self.ema_accept_len <= drop_threshold:
                current_idx -= 1
            else:
                break

        while current_idx < len(self.candidate_steps) - 1:
            current_step = self.candidate_steps[current_idx]
            rise_threshold = current_step - 0.5 + self.up_hysteresis
            if self.ema_accept_len > rise_threshold:
                current_idx += 1
            else:
                break

        target = self.candidate_steps[current_idx]

        if target != old_steps:
            self.current_steps = target
            log_info_on_rank0(
                logger,
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})",
            )
            return True
        return False
