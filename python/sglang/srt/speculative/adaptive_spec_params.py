"""Adaptive speculative decoding parameters.

Adjusts speculative_num_steps at runtime based on observed acceptance lengths.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING

from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default adaptive config (conservative).
# Used when --speculative-adaptive is enabled without --speculative-adaptive-config.
#
# Config format: integer-keyed dict where each key is a BS lower bound.
# BS lookup uses bisect (largest key <= actual padded BS).
# See docs/advanced_features/adaptive_speculative_decoding_per_bs.md for
# recommended configs for different draft model qualities.
# ---------------------------------------------------------------------------
# TODO: add step=0 (nospec fallback) for BS>=64 once supported —
# on hard workloads, even step=1 loses to nospec at high batch sizes.
DEFAULT_ADAPTIVE_CONFIG: dict[str, dict] = {
    "1": {
        "steps": [1, 3, 7],
        "up_hysteresis": 0.0,
        "down_hysteresis": -0.25,
        "ceiling_coeff": 0,
    },
    "8": {
        "steps": [1],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
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


def load_adaptive_config(path: str | None) -> dict:
    """Load adaptive speculative config from a JSON file.

    The file is a JSON object with integer-string keys as BS lower bounds::

        {"1": {"steps": [1,3,7], ...}, "64": {"steps": [1,2,5], ...}}

    Non-integer keys (``ema_alpha``, ``update_interval``, …) are global
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


def _resolve_candidate_steps(config: dict) -> dict[int, dict] | None:
    """Extract per-BS entries from a config dict.

    Integer-string keys (``"1"``, ``"64"``, …) become ``{1: {...}, 64: {...}}``.
    Non-integer keys are ignored (they are global overrides).

    Returns ``None`` when no BS entries found.
    """
    result: dict[int, dict] = {}
    for key, entry in config.items():
        try:
            bs = int(key)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            raise ValueError(
                f"Invalid adaptive config for BS {bs}: "
                f"expected a dict, got {type(entry).__name__}"
            )
        result[bs] = entry
    return result if result else None


def _load_validated_config(
    cfg_path: str | None,
) -> tuple[dict, dict[int, dict]]:
    """Load, parse, and validate adaptive config. Falls back to default on error.

    Returns ``(cfg, bs_config)`` where *cfg* is the full config dict and
    *bs_config* is the validated ``{bs_int: entry_dict}`` mapping.
    Both ``resolve_candidate_steps_from_config`` and ``build_per_bs_params``
    use this so that allocation sizing and runtime controller always agree.
    """
    try:
        cfg = load_adaptive_config(cfg_path) if cfg_path else DEFAULT_ADAPTIVE_CONFIG
        bs_config = _resolve_candidate_steps(cfg)
        if bs_config is None:
            raise ValueError("no per-BS entries found")
        for bs, entry in bs_config.items():
            steps = entry.get("steps")
            if steps is not None and (
                not isinstance(steps, list)
                or not steps
                # TODO: allow step=0 (nospec fallback) once supported
                or not all(isinstance(s, int) and s > 0 for s in steps)
            ):
                raise ValueError(
                    f"BS {bs}: 'steps' must be a non-empty list of positive ints, "
                    f"got {steps!r}"
                )
    except Exception as e:
        log_info_on_rank0(
            logger,
            f"Invalid adaptive config ({e}), falling back to default",
        )
        cfg = DEFAULT_ADAPTIVE_CONFIG
        bs_config = _resolve_candidate_steps(cfg)
    return cfg, bs_config


def resolve_candidate_steps_from_config(
    initial_steps: int,
    cfg_path: str | None = None,
) -> list[int]:
    """Resolve the union of all candidate steps from config.

    Used by ``ServerArgs.effective_max_speculative_num_draft_tokens()``
    to determine the max draft-token count without building the full
    AdaptiveController.
    """
    _, bs_config = _load_validated_config(cfg_path)
    all_steps: set[int] = set()
    for entry in bs_config.values():
        all_steps.update(entry.get("steps", [1, 3, 7]))
    all_steps.add(initial_steps)
    return sorted(all_steps)


def build_per_bs_params(
    cfg_path: str | None = None,
) -> tuple[list[int], dict[int, "AdaptiveSpeculativeParams"]]:
    """Parse config and build one ``AdaptiveSpeculativeParams`` per BS slot.

    Returns ``(bs_list, bs_params)`` where *bs_list* is the sorted list of
    BS lower-bound keys and *bs_params* maps each key to its params instance.
    """
    cfg, bs_config = _load_validated_config(cfg_path)

    bs_list = sorted(bs_config.keys())
    bs_params: dict[int, AdaptiveSpeculativeParams] = {}
    for bs, entry in sorted(bs_config.items()):
        steps = entry.get("steps", [1, 3, 7])
        initial = steps[len(steps) // 2]
        params_cfg = {
            **cfg,
            "candidate_steps": steps,
            "up_hysteresis": entry.get("up_hysteresis", cfg.get("up_hysteresis", 0.0)),
            "down_hysteresis": entry.get(
                "down_hysteresis", cfg.get("down_hysteresis", -0.25)
            ),
        }
        if "ceiling_coeff" in entry:
            params_cfg["ceiling_coeff"] = entry["ceiling_coeff"]
        bs_params[bs] = AdaptiveSpeculativeParams(
            initial_steps=initial,
            bs_cfg=params_cfg,
        )
    return bs_list, bs_params


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
        bs_cfg: str | dict | None = None,
    ):
        if isinstance(bs_cfg, dict):
            cfg = bs_cfg
        else:
            cfg = load_adaptive_config(bs_cfg)
        candidates = sorted(set(cfg.get("candidate_steps", [1, 3, 7])))

        assert len(candidates) >= 1, "candidate_steps must have at least 1 value"
        self.candidate_steps = candidates

        self.ema_alpha = cfg.get("ema_alpha", 0.2)
        self.update_interval = cfg.get("update_interval", 5)
        self.warmup_batches = cfg.get("warmup_batches", 10)
        self.down_hysteresis = cfg.get("down_hysteresis", -0.25)
        self.up_hysteresis = cfg.get("up_hysteresis", 0.0)
        self.ceiling_coeff = cfg.get("ceiling_coeff", 0)

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

        # EMA ceiling: cap num_steps proportionally to observed draft quality.
        # Only applied as a downward cap — never blocks step-ups that hysteresis
        # allows, so the system can explore higher steps and let the EMA catch up.
        target = self.candidate_steps[current_idx]
        if self.ceiling_coeff > 0:
            ceiling = max(1, math.ceil(self.ema_accept_len * self.ceiling_coeff))
            if target > ceiling and target <= old_steps:
                while current_idx > 0 and self.candidate_steps[current_idx] > ceiling:
                    current_idx -= 1
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
