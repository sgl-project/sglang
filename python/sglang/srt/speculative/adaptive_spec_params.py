"""Adaptive speculative decoding parameters.

Adjusts speculative_num_steps at runtime based on observed acceptance lengths.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from functools import cached_property
from typing import TYPE_CHECKING

from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

DEFAULT_ADAPTIVE_CONFIG: dict[str, dict] = {
    "1": {
        "candidate_steps": [1, 3, 7],
        "up_hysteresis": 0.0,
        "down_hysteresis": -0.25,
        "ceiling_coeff": 0,
    },
    "8": {
        "candidate_steps": [0, 1, 3],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "32": {
        "candidate_steps": [0, 1],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "64": {
        "candidate_steps": [0],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
}


def adaptive_unsupported_reason(server_args: ServerArgs) -> str | None:
    """Return why adaptive spec cannot run under the given server args, or None if supported."""
    from sglang.srt.arg_groups.overrides import resolved_view

    if server_args.speculative_algorithm not in ("EAGLE", "EAGLE3"):
        return (
            f"speculative_algorithm={server_args.speculative_algorithm} "
            "(only EAGLE/EAGLE3 are supported)"
        )
    if (
        server_args.speculative_eagle_topk is not None
        and server_args.speculative_eagle_topk != 1
    ):
        return (
            f"speculative_eagle_topk={server_args.speculative_eagle_topk} "
            "(only topk=1 is supported)"
        )
    # enable_dp_attention IS supported: tier decisions are synchronized across
    # DP ranks via a min-reduce in the scheduler's MLP-sync all_gather
    # (dp_attn.MLPSyncBatchInfo.local_adaptive_steps), so all ranks run equal
    # draft-token shapes. See AdaptiveController(sync_across_dp=...).
    if resolved_view(server_args).enable_multi_layer_eagle:
        return (
            "enable_multi_layer_eagle=True is not supported "
            "(MultiLayerEagleWorkerV2 does not implement adaptive)"
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


def _load_adaptive_config(
    cfg_path: str | None,
) -> tuple[dict, dict[int, dict]]:
    """Load and validate adaptive config.

    Uses ``DEFAULT_ADAPTIVE_CONFIG`` when *cfg_path* is ``None``.
    """
    if cfg_path is not None:
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = DEFAULT_ADAPTIVE_CONFIG

    bs_entries: dict[int, dict] = {}
    for key, entry in cfg.items():
        if not key.isdigit():
            continue

        steps = entry.get("candidate_steps")
        if (
            not isinstance(steps, list)
            or not steps
            or not all(isinstance(s, int) and s >= 0 for s in steps)
        ):
            raise ValueError(
                f"BS {key}: candidate_steps must be a list of non-negative ints, "
                f"got {steps!r}"
            )
        bs_entries[int(key)] = entry

    if not bs_entries:
        raise ValueError(
            "speculative_adaptive_config must contain at least one integer-string "
            'BS key, e.g. {"1": {"candidate_steps": [1,3,7]}}. '
            f"Got keys: {list(cfg.keys())}"
        )
    return cfg, bs_entries


def resolve_candidate_steps_from_config(
    cfg_path: str | None = None,
) -> list[int]:
    """Union of every BS slot's candidate steps; sizes the runtime buffers."""
    _, bs_entries = _load_adaptive_config(cfg_path)
    all_steps: set[int] = set()
    for entry in bs_entries.values():
        all_steps.update(entry["candidate_steps"])
    return sorted(all_steps)


class AdaptiveStepSlot:
    """Tracks acceptance rate via EMA and adapts num_steps accordingly.

    The core idea: if drafts are consistently accepted, try more steps;
    if drafts are consistently rejected early, reduce steps to avoid waste.

    Formula: target_steps = clamp(round(ema_accept_len) + 1, min_steps, max_steps)
    - Probes one step beyond observed acceptance
    - EMA smoothing prevents oscillation
    - Only updates every `update_interval` batches for stability
    - num_steps can be selected from different candidate sets on different batch_sizes
    """

    def __init__(self, initial_steps: int, cfg: dict):
        candidates = sorted(set(cfg["candidate_steps"]))
        assert len(candidates) >= 1, "candidate_steps must have at least 1 value"
        self.candidate_steps = candidates

        self.ema_alpha = cfg.get("ema_alpha", 0.2)
        self.update_interval = cfg.get("update_interval", 5)
        self.warmup_batches = cfg.get("warmup_batches", 10)
        self.down_hysteresis = cfg.get("down_hysteresis", -0.25)
        self.up_hysteresis = cfg.get("up_hysteresis", 0.0)
        self.ceiling_coeff = cfg.get("ceiling_coeff", 0)
        # Min-dwell: after a tier change, hold the new tier for this many
        # verify rounds before another change is allowed (0 = off). Damps
        # threshold-straddling oscillation, both TP-local and DP-shared.
        self.min_dwell_batches = cfg.get("min_dwell_batches", 0)

        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]

        # Initialize EMA at current steps - 1 (neutral starting point)
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0
        self._dwell_remaining = 0
        # Cross-rank (mean) EMA folded in from the DP MLP-sync all_gather.
        # The DP vote is a pure function of this + current_steps, so every
        # rank computes an identical vote (None until the first fold).
        self.shared_ema_accept_len: float | None = None

    def update_ema(self, num_correct_drafts_per_req: list[int]) -> None:
        """Fold one verify round into the EMA. No tier decision."""
        if not num_correct_drafts_per_req:
            return
        if self.current_steps > 0:
            batch_avg = sum(num_correct_drafts_per_req) / len(
                num_correct_drafts_per_req
            )
            self.ema_accept_len = (
                1 - self.ema_alpha
            ) * self.ema_accept_len + self.ema_alpha * batch_avg

    def update(self, num_correct_drafts_per_req: list[int]) -> bool:
        """Update EMA with observed accept lengths. Returns True if params changed.

        Args:
            num_correct_drafts_per_req: Per-request accepted draft token counts from last verify.
        """
        if not num_correct_drafts_per_req:
            return False

        self.update_ema(num_correct_drafts_per_req)

        self._batch_count += 1
        if self._batch_count <= self.warmup_batches:
            return False

        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False

        if self._dwell_remaining > 0:
            self._dwell_remaining -= 1
            return False

        return self._recompute_params()

    def fresh_vote(self) -> int:
        """Tier a pure threshold walk over the EMA selects right now.

        Under DP attention the gathered vote must be a deterministic function
        of state every rank already shares — the folded cross-rank EMA and the
        consensus-activated tier — never of when each rank's local
        `update_interval` last fired, or per-rank timing diverges the votes.
        """
        if self._dwell_remaining > 0:
            return self.current_steps
        ema = (
            self.shared_ema_accept_len
            if self.shared_ema_accept_len is not None
            else self.ema_accept_len
        )
        return self.candidate_steps[self._walk_target_idx(ema)]

    def apply_consensus(self, steps: int) -> None:
        """Mirror the DP-consensus activation into the slot; re-arm min-dwell."""
        if steps != self.current_steps:
            self.current_steps = steps
            self._dwell_remaining = self.min_dwell_batches

    def fold_shared_ema(self, ema: float) -> None:
        """Adopt the cross-rank (mean) EMA carried by the MLP-sync all_gather."""
        self.shared_ema_accept_len = ema

    def _walk_target_idx(self, ema: float) -> int:
        """Pure threshold walk: candidate idx *ema* selects (no mutation)."""
        old_idx = self.candidate_steps.index(self.current_steps)
        current_idx = old_idx

        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            # A zero-step candidate disables drafting. Treat zero accepted drafts
            # as low enough to reach it when it is the floor candidate.
            drop_threshold = 0.5 if prev_step == 0 else prev_step - 0.5
            drop_threshold += self.down_hysteresis
            if ema <= drop_threshold:
                current_idx -= 1
            else:
                break

        if current_idx >= old_idx:
            while current_idx < len(self.candidate_steps) - 1:
                current_step = self.candidate_steps[current_idx]
                rise_threshold = current_step - 0.5 + self.up_hysteresis
                if ema > rise_threshold:
                    current_idx += 1
                else:
                    break

        # EMA ceiling: only caps downward — never blocks step-ups, so the
        # system can explore higher steps and let the EMA catch up.
        if self.ceiling_coeff > 0:
            ceiling = max(1, math.ceil(ema * self.ceiling_coeff))
            target = self.candidate_steps[current_idx]
            if target > ceiling and target <= self.candidate_steps[old_idx]:
                while current_idx > 0 and self.candidate_steps[current_idx] > ceiling:
                    current_idx -= 1

        return current_idx

    def _recompute_params(self) -> bool:
        """Recompute steps from EMA. Returns True if params changed."""
        old_steps = self.current_steps

        # Probe the smallest positive step after a zero-step nospec interval.
        if old_steps == 0:
            current_idx = self.candidate_steps.index(old_steps)
            current_idx = min(current_idx + 1, len(self.candidate_steps) - 1)
            target = self.candidate_steps[current_idx]
            if target > 0 and self.ema_accept_len < 0:
                # A slot initialized at steps=0 has no draft acceptance history;
                # start the first positive-step probe from that step's neutral EMA.
                self.ema_accept_len = float(target - 1)
            return self._apply_target_steps(old_steps, target)

        # TODO: Consider limiting step changes to avoid overshooting.
        target = self.candidate_steps[self._walk_target_idx(self.ema_accept_len)]
        return self._apply_target_steps(old_steps, target)

    def _apply_target_steps(self, old_steps: int, target: int) -> bool:
        if target != old_steps:
            self.current_steps = target
            self._dwell_remaining = self.min_dwell_batches
            log_info_on_rank0(
                logger,
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})",
            )
            return True
        return False


class AdaptiveSpeculativeParams:
    """Routes ``batch_size`` to the correct per-BS slot.

    A slot is a per-BS configuration of adaptive step selection.
    """

    def __init__(
        self,
        initial_steps: int,
        cfg_path: str | None = None,
    ):
        cfg, bs_entries = _load_adaptive_config(cfg_path)
        self._bs_list: list[int] = sorted(bs_entries)
        self._slots: dict[int, AdaptiveStepSlot] = {}
        self._cuda_graph_bs: list[int] | None = None

        for bs, entry in sorted(bs_entries.items()):
            self._slots[bs] = AdaptiveStepSlot(
                initial_steps=initial_steps,
                cfg={**cfg, **entry},
            )

        first_slot = self._slots[self._bs_list[0]]
        log_info_on_rank0(
            logger,
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={first_slot.current_steps}, "
            f"candidate_steps={first_slot.candidate_steps}",
        )

    @cached_property
    def candidate_steps(self) -> list[int]:
        """Union of all BS slots' candidate steps."""
        return sorted({s for p in self._slots.values() for s in p.candidate_steps})

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

    def get_steps_for_batch(self, batch_size: int) -> int:
        return self._route(batch_size).current_steps

    def fresh_steps_for_batch(self, batch_size: int) -> int:
        """DP vote: tier the pure shared-EMA threshold walk selects now."""
        return self._route(batch_size).fresh_vote()

    def slot_key_for_batch(self, batch_size: int) -> int:
        """Config key of the slot *batch_size* routes to (DP EMA-fold key)."""
        return self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))

    def get_ema_for_batch(self, batch_size: int) -> float:
        """Live EMA of the routed slot (contributed to the DP gather)."""
        return self._route(batch_size).ema_accept_len

    def update_ema_only(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> None:
        """DP mode: feed verify results to the EMA, no local tier decision."""
        self._route(batch_size).update_ema(num_correct_drafts_per_req)

    def fold_shared_ema(self, per_slot_ema: dict[int, float]) -> None:
        """Adopt per-slot cross-rank mean EMAs gathered over DP."""
        for bs, ema in per_slot_ema.items():
            slot = self._slots.get(bs)
            if slot is not None:
                slot.fold_shared_ema(ema)

    def apply_consensus(self, batch_size: int, steps: int) -> None:
        """Sync the routed slot's tier to the DP-consensus activation."""
        self._route(batch_size).apply_consensus(steps)

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> int | None:
        """Feed verify results to the matching BS slot's EMA.

        Returns the new step if a switch is warranted, else ``None``.
        """
        params = self._route(batch_size)
        if params.update(num_correct_drafts_per_req):
            return params.current_steps
        return None

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        """Return cuda_graph_bs values that can reach *step* at runtime.

        Returns ``None`` when CUDA graphs are disabled (``set_cuda_graph_bs``
        was never called or was called with ``None``).
        """
        if self._cuda_graph_bs is None:
            return None
        return [
            v
            for v in self._cuda_graph_bs
            if step in self._slots[self._find_closest_bs(v)].candidate_steps
        ]

    def _route(self, batch_size: int) -> AdaptiveStepSlot:
        """Map *batch_size* → pad to CUDA-graph BS → closest slot."""
        return self._slots[
            self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))
        ]

    def _pad_to_cuda_graph_bs(self, batch_size: int) -> int:
        if self._cuda_graph_bs is None:
            return batch_size
        idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
        return (
            self._cuda_graph_bs[idx] if idx < len(self._cuda_graph_bs) else batch_size
        )

    def _find_closest_bs(self, target: int) -> int:
        idx = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, idx)]
