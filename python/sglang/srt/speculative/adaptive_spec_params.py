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
    if resolved_view(server_args).enable_dp_attention:
        return (
            "enable_dp_attention=True is not supported "
            "(adaptive tier decisions are not synchronized across DP ranks)"
        )
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


_CTX_BUCKET_MAX = 2**31 - 1


def _parse_ctx_bucket_key(key: str) -> tuple[int, int]:
    if not isinstance(key, str) or "-" not in key:
        raise ValueError(f"ctx_buckets key must be a 'lo-hi' string, got {key!r}")
    lo_str, hi_str = key.split("-", 1)
    try:
        lo, hi = int(lo_str), int(hi_str)
    except ValueError as e:
        raise ValueError(f"ctx_buckets key {key!r} must contain integer 'lo-hi'") from e
    if lo < 1 or hi < lo:
        raise ValueError(f"ctx_buckets key {key!r} must satisfy 1 <= lo <= hi")
    return lo, hi


def _validate_ctx_bucket_coverage(
    bs: int, ordered: list[tuple[int, int, dict]]
) -> None:
    if not ordered:
        raise ValueError(f"BS {bs}: ctx_buckets must not be empty")
    if ordered[0][0] != 1:
        raise ValueError(
            f"BS {bs}: ctx_buckets must start at 1, got lo={ordered[0][0]}"
        )
    for i in range(1, len(ordered)):
        prev_hi = ordered[i - 1][1]
        cur_lo = ordered[i][0]
        if cur_lo != prev_hi + 1:
            raise ValueError(
                f"BS {bs}: ctx_buckets gap between "
                f"{prev_hi} and {cur_lo}; ranges must be contiguous"
            )


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

        ctx_raw = entry.get("ctx_buckets")
        if ctx_raw is not None:
            if not isinstance(ctx_raw, dict) or not ctx_raw:
                raise ValueError(
                    f"BS {key}: ctx_buckets must be a non-empty dict, "
                    f"got {ctx_raw!r}"
                )
            for bucket_key, bucket_entry in ctx_raw.items():
                _parse_ctx_bucket_key(bucket_key)
                steps = (
                    bucket_entry.get("candidate_steps")
                    if isinstance(bucket_entry, dict)
                    else None
                )
                if (
                    not isinstance(steps, list)
                    or not steps
                    or not all(isinstance(s, int) and s >= 0 for s in steps)
                ):
                    raise ValueError(
                        f"BS {key} ctx bucket {bucket_key!r}: candidate_steps "
                        f"must be a list of non-negative ints, got {steps!r}"
                    )
        else:
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
        ctx_raw = entry.get("ctx_buckets")
        if ctx_raw is not None:
            for bucket_entry in ctx_raw.values():
                all_steps.update(bucket_entry["candidate_steps"])
        else:
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

        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]

        # Initialize EMA at current steps - 1 (neutral starting point)
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0

    def update(self, num_correct_drafts_per_req: list[int]) -> bool:
        """Update EMA with observed accept lengths. Returns True if params changed.

        Args:
            num_correct_drafts_per_req: Per-request accepted draft token counts from last verify.
        """
        if not num_correct_drafts_per_req:
            return False

        if self.current_steps > 0:
            batch_avg = sum(num_correct_drafts_per_req) / len(
                num_correct_drafts_per_req
            )
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
        old_idx = current_idx

        # Probe the smallest positive step after a zero-step nospec interval.
        if old_steps == 0:
            current_idx = min(current_idx + 1, len(self.candidate_steps) - 1)
            target = self.candidate_steps[current_idx]
            if target > 0 and self.ema_accept_len < 0:
                # A slot initialized at steps=0 has no draft acceptance history;
                # start the first positive-step probe from that step's neutral EMA.
                self.ema_accept_len = float(target - 1)
            return self._apply_target_steps(old_steps, target)

        # TODO: Consider limiting step changes to avoid overshooting.
        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            # A zero-step candidate disables drafting. Treat zero accepted drafts
            # as low enough to reach it when it is the floor candidate.
            drop_threshold = 0.5 if prev_step == 0 else prev_step - 0.5
            drop_threshold += self.down_hysteresis
            if self.ema_accept_len <= drop_threshold:
                current_idx -= 1
            else:
                break

        moved_down = current_idx < old_idx
        if not moved_down:
            while current_idx < len(self.candidate_steps) - 1:
                current_step = self.candidate_steps[current_idx]
                rise_threshold = current_step - 0.5 + self.up_hysteresis
                if self.ema_accept_len > rise_threshold:
                    current_idx += 1
                else:
                    break

        target = self.candidate_steps[current_idx]
        # EMA ceiling: only caps downward — never blocks step-ups, so the
        # system can explore higher steps and let the EMA catch up.
        if self.ceiling_coeff > 0:
            ceiling = max(1, math.ceil(self.ema_accept_len * self.ceiling_coeff))
            if target > ceiling and target <= old_steps:
                while current_idx > 0 and self.candidate_steps[current_idx] > ceiling:
                    current_idx -= 1
                target = self.candidate_steps[current_idx]

        return self._apply_target_steps(old_steps, target)

    def _apply_target_steps(self, old_steps: int, target: int) -> bool:
        if target != old_steps:
            self.current_steps = target
            log_info_on_rank0(
                logger,
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"(ema_accept_len={self.ema_accept_len:.2f})",
            )
            return True
        return False


class AdaptiveSpeculativeParams:
    """Routes ``(batch_size, ctx_repr)`` to the correct per-(BS, ctx) slot.

    A slot is a per-(BS, ctx-bucket) configuration of adaptive step
    selection. Legacy 1D configs (no ``ctx_buckets`` key) degenerate to a
    single ctx bucket covering ``[0, ∞)`` per BS.
    """

    def __init__(
        self,
        initial_steps: int,
        cfg_path: str | None = None,
    ):
        cfg, bs_entries = _load_adaptive_config(cfg_path)
        self._bs_list: list[int] = sorted(bs_entries)
        self._slots: dict[int, AdaptiveStepSlot] = {}
        self._ctx_slots_by_bs: dict[int, dict[int, AdaptiveStepSlot]] = {}
        self._ctx_lo_by_bs: dict[int, list[int]] = {}
        self._ctx_hi_by_bs: dict[int, list[int]] = {}
        self._cuda_graph_bs: list[int] | None = None

        for bs, entry in sorted(bs_entries.items()):
            ctx_raw = entry.get("ctx_buckets")
            bs_slots: dict[int, AdaptiveStepSlot] = {}
            los: list[int] = []
            his: list[int] = []
            if ctx_raw is None:
                bs_slots[1] = AdaptiveStepSlot(
                    initial_steps=initial_steps,
                    cfg={**cfg, **entry},
                )
                los.append(1)
                his.append(_CTX_BUCKET_MAX)
            else:
                ordered: list[tuple[int, int, dict]] = []
                for bucket_key, bucket_entry in ctx_raw.items():
                    lo, hi = _parse_ctx_bucket_key(bucket_key)
                    ordered.append((lo, hi, bucket_entry))
                ordered.sort(key=lambda t: t[0])
                _validate_ctx_bucket_coverage(bs, ordered)
                for lo, hi, bucket_entry in ordered:
                    merged = {**cfg, **entry, **bucket_entry}
                    merged.pop("ctx_buckets", None)
                    bs_slots[lo] = AdaptiveStepSlot(
                        initial_steps=initial_steps,
                        cfg=merged,
                    )
                    los.append(lo)
                    his.append(hi)
            self._ctx_slots_by_bs[bs] = bs_slots
            self._slots[bs] = bs_slots[los[0]]
            self._ctx_lo_by_bs[bs] = los
            self._ctx_hi_by_bs[bs] = his

        first_slot = self._slots[self._bs_list[0]]
        log_info_on_rank0(
            logger,
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={first_slot.current_steps}, "
            f"candidate_steps={first_slot.candidate_steps}",
        )

    @cached_property
    def candidate_steps(self) -> list[int]:
        """Union of all (BS, ctx) slots' candidate steps."""
        return sorted(
            {
                s
                for buckets in self._ctx_slots_by_bs.values()
                for slot in buckets.values()
                for s in slot.candidate_steps
            }
        )

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

    def get_steps_for_batch(self, batch_size: int, ctx_repr: int = 0) -> int:
        return self._route(batch_size, ctx_repr).current_steps

    def on_verify_complete(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int,
        ctx_repr: int = 0,
    ) -> int | None:
        """Feed verify results to the matching (BS, ctx) slot's EMA.

        Returns the new step if a switch is warranted, else ``None``.
        """
        params = self._route(batch_size, ctx_repr)
        if params.update(num_correct_drafts_per_req):
            return params.current_steps
        return None

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        """Return cuda_graph_bs values that can reach *step* at runtime.

        Returns ``None`` when CUDA graphs are disabled (``set_cuda_graph_bs``
        was never called or was called with ``None``). Under the 2D schema
        the reachable step set at a given BS is the union of every ctx
        bucket's ``candidate_steps`` — the capture set is BS-only, so the
        pruning shape matches the 1D-union config.
        """
        if self._cuda_graph_bs is None:
            return None
        return [
            v
            for v in self._cuda_graph_bs
            if step in self._steps_reachable_at_bs(self._find_closest_bs(v))
        ]

    def _steps_reachable_at_bs(self, bs: int) -> set[int]:
        return {
            s
            for slot in self._ctx_slots_by_bs[bs].values()
            for s in slot.candidate_steps
        }

    def _route(self, batch_size: int, ctx_repr: int = 0) -> AdaptiveStepSlot:
        """Map *(batch_size, ctx_repr)* → pad to CUDA-graph BS → slot."""
        bs = self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))
        return self._slot_for_ctx(bs, ctx_repr)

    def _slot_for_ctx(self, bs: int, ctx_repr: int) -> AdaptiveStepSlot:
        buckets = self._ctx_slots_by_bs[bs]
        if len(buckets) == 1:
            return next(iter(buckets.values()))
        his = self._ctx_hi_by_bs[bs]
        idx = bisect.bisect_left(his, ctx_repr)
        if idx == len(his):
            idx = len(his) - 1
        return buckets[self._ctx_lo_by_bs[bs][idx]]

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
