"""Auto-speculative decoding engine.

Dynamically adjusts speculative decoding parameters (num_steps, num_draft_tokens)
based on runtime acceptance rate feedback. The engine monitors per-batch-size
acceptance rates and increases/decreases the speculation depth accordingly.
"""

import bisect
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Default batch size to step mappings
DEFAULT_BS_STEPS_MAPPING: Dict[int, List[int]] = {
    1: [3, 4, 5, 6],
    2: [3, 4, 5, 6],
    4: [3, 4, 5, 6],
    8: [2, 3, 4],
    16: [2, 3, 4],
    32: [2, 3, 4],
    64: [1, 2, 3, 4],
    128: [1, 2, 3, 4],
}

# Default initial num_steps for each batch size
DEFAULT_INITIAL_STEPS: Dict[int, int] = {
    1: 5,
    2: 5,
    4: 5,
    8: 4,
    16: 3,
    32: 3,
    64: 3,
    128: 1,
}

# Default acceptance rate thresholds (increase_threshold, decrease_threshold)
DEFAULT_THRESHOLDS: Dict[int, Tuple[float, float]] = {
    1: (0.55, 0.50),
    2: (0.55, 0.50),
    4: (0.60, 0.50),
    8: (0.80, 0.50),
    16: (0.95, 0.60),
    32: (0.91, 0.66),
    64: (0.95, 0.65),
    128: (0.95, 0.65),
}


@dataclass
class SpecParams:
    """Speculative decoding parameters for a given configuration."""

    num_steps: int
    topk: int = 1
    num_draft_tokens: int = -1  # auto-computed from num_steps + 1

    def __post_init__(self):
        if self.num_draft_tokens < 0:
            self.num_draft_tokens = self.num_steps + 1


class AutoSpecEngine:
    """Engine for dynamically adjusting speculative decoding parameters.

    Monitors acceptance rates per batch size and adjusts num_steps up or down
    based on configurable thresholds. Uses EMA (exponential moving average)
    for smooth acceptance rate tracking.
    """

    def __init__(self, server_args: "ServerArgs"):
        self.cuda_graph_bs = server_args.cuda_graph_bs
        self.device_id = server_args.device
        self.model_name = server_args.served_model_name or server_args.model_path
        self.speculative_config_file = getattr(
            server_args, "speculative_config_file", None
        )

        # EMA smoothing factor for acceptance rate tracking
        self.ema_alpha = 0.3

        # Load or build bs -> steps mapping
        self._init_bs_steps_mapping()

        # Build sorted batch size list and closest-bs lookup
        self.bs_list = sorted(self.bs_steps_mapping.keys())

        # Build thresholds per batch size
        self._init_thresholds(server_args)

        # Will be finalized in initialize() after GPU memory is known
        self._initialized = False

    def initialize(self, gpu_id: int, is_spec_v2: bool = False):
        """Must be called after model loading to finalize based on available GPU memory.

        Args:
            gpu_id: The GPU device index.
            is_spec_v2: If True, use more conservative memory estimates for SpecV2
                which captures both draft and verify CUDA graphs per step.
        """
        from sglang.srt.utils.common import get_available_gpu_memory

        available_memory = get_available_gpu_memory(self.device_id, gpu_id, empty_cache=False)
        # SpecV2 captures both draft + verify graphs per step (~3.5GB each),
        # while V1 only captures draft graphs (~2GB each).
        # Reserve safety margin for inference runtime allocations (MoE, etc).
        if is_spec_v2:
            safety_margin = 6.0  # GB reserved for runtime
            cost_per_step = 3.5  # GB per step (draft + verify graphs)
        else:
            safety_margin = 4.0
            cost_per_step = 2.0
        max_num_graphs = int((available_memory - safety_margin) // cost_per_step)
        max_num_graphs = max(max_num_graphs, 1)
        logger.info(
            f"AutoSpecEngine: available_memory={available_memory:.1f}GB, "
            f"safety_margin={safety_margin}GB, cost_per_step={cost_per_step}GB, "
            f"max graph sets={max_num_graphs}"
        )

        self._trim_steps_by_memory(max_num_graphs)
        self._build_reverse_mapping()
        self._init_current_params()
        self._initialized = True

        logger.info(
            f"AutoSpecEngine initialized: step_range={self.step_range}, "
            f"bs_steps_mapping={self.bs_steps_mapping}, "
            f"initial params={self._format_params()}"
        )

    def _init_bs_steps_mapping(self):
        """Load batch-size to step mappings from config file or defaults."""
        if self.speculative_config_file and os.path.exists(
            self.speculative_config_file
        ):
            try:
                with open(self.speculative_config_file, "r") as f:
                    config = json.load(f)
                if self.model_name and self.model_name in config:
                    model_config = config[self.model_name]
                    self.bs_steps_mapping = {}
                    for bs_str, steps in model_config.items():
                        try:
                            self.bs_steps_mapping[int(bs_str)] = sorted(steps)
                        except ValueError:
                            pass
                    if self.bs_steps_mapping:
                        logger.info(
                            f"AutoSpecEngine: loaded config for '{self.model_name}'"
                        )
                        return
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"AutoSpecEngine: failed to load config: {e}, using defaults"
                )

        self.bs_steps_mapping = {k: list(v) for k, v in DEFAULT_BS_STEPS_MAPPING.items()}

    def _init_thresholds(self, server_args: "ServerArgs"):
        """Initialize acceptance rate thresholds per batch size."""
        pos_threshold = getattr(server_args, "pos_threshold", None)
        neg_threshold = getattr(server_args, "neg_threshold", None)

        self.increase_thresholds: Dict[int, float] = {}
        self.decrease_thresholds: Dict[int, float] = {}

        sorted_default_bs = sorted(DEFAULT_THRESHOLDS.keys())

        for bs in self.bs_steps_mapping.keys():
            # Find closest default threshold
            idx = bisect.bisect_right(sorted_default_bs, bs) - 1
            idx = max(0, min(idx, len(sorted_default_bs) - 1))
            closest_bs = sorted_default_bs[idx]
            inc_t, dec_t = DEFAULT_THRESHOLDS[closest_bs]
            self.increase_thresholds[bs] = inc_t
            self.decrease_thresholds[bs] = dec_t

        # Override with user-specified thresholds if provided
        if pos_threshold is not None and len(pos_threshold) > 0:
            for i, bs in enumerate(sorted(self.bs_steps_mapping.keys())):
                if i < len(pos_threshold):
                    self.increase_thresholds[bs] = pos_threshold[i]
        if neg_threshold is not None and len(neg_threshold) > 0:
            for i, bs in enumerate(sorted(self.bs_steps_mapping.keys())):
                if i < len(neg_threshold):
                    self.decrease_thresholds[bs] = neg_threshold[i]

    def _trim_steps_by_memory(self, max_num_graphs: int):
        """Reduce number of unique steps if GPU memory is limited."""
        all_steps = sorted(set(s for steps in self.bs_steps_mapping.values() for s in steps))

        if len(all_steps) <= max_num_graphs:
            self.step_range = all_steps
            return

        # Prioritize keeping smaller steps (more frequently used) + largest step
        if max_num_graphs >= 2:
            kept = all_steps[:max_num_graphs - 1] + [all_steps[-1]]
        else:
            kept = [all_steps[len(all_steps) // 2]]
        self.step_range = sorted(set(kept))

        # Update bs_steps_mapping to only include available steps
        for bs in list(self.bs_steps_mapping.keys()):
            available = [s for s in self.bs_steps_mapping[bs] if s in self.step_range]
            if not available:
                available = list(self.step_range)
            self.bs_steps_mapping[bs] = available

        logger.info(
            f"AutoSpecEngine: trimmed to step_range={self.step_range} "
            f"due to memory constraint (max_graphs={max_num_graphs})"
        )

    def _build_reverse_mapping(self):
        """Build step -> batch_sizes reverse mapping."""
        self.steps_bs_mapping: Dict[int, List[int]] = {}
        for bs, steps in self.bs_steps_mapping.items():
            for step in steps:
                if step not in self.steps_bs_mapping:
                    self.steps_bs_mapping[step] = []
                self.steps_bs_mapping[step].append(bs)
        for step in self.steps_bs_mapping:
            self.steps_bs_mapping[step].sort()

    def _init_current_params(self):
        """Initialize current best parameters for each batch size."""
        self.current_params: Dict[int, SpecParams] = {}
        for bs in self.bs_list:
            initial_steps = DEFAULT_INITIAL_STEPS.get(bs, 3)
            # Clamp to available steps
            available = self.bs_steps_mapping[bs]
            if initial_steps not in available:
                # Pick closest available
                initial_steps = min(available, key=lambda s: abs(s - initial_steps))
            self.current_params[bs] = SpecParams(num_steps=initial_steps)

        # EMA acceptance rate tracker per batch size
        self.ema_accept_rate: Dict[int, float] = {bs: 0.5 for bs in self.bs_list}

    def sync_with_graph_capacity(self, step_max_bs: Dict[int, int]):
        """Sync BS->step mappings with actual CUDA graph capacity.

        After CUDA graphs are captured, each step's graph runner has a max_bs.
        This method:
        1. Expands: allows every BS to use any step whose graph supports it
        2. Prunes: removes BS->step mappings where the graph can't handle that BS

        This ensures all batch sizes can use all capable steps (maximizing
        auto-spec flexibility) while preventing eager mode fallback.

        Args:
            step_max_bs: Mapping from num_steps -> max batch size supported by graphs.
        """
        changed = False
        all_steps = sorted(step_max_bs.keys())

        for bs in list(self.bs_steps_mapping.keys()):
            # Expand: include all steps whose graph can handle this BS
            expanded = sorted(s for s in all_steps if bs <= step_max_bs.get(s, 0))
            if not expanded:
                # Fallback: keep the step with the largest max_bs
                best_step = max(all_steps, key=lambda s: step_max_bs.get(s, 0))
                expanded = [best_step]
            if expanded != self.bs_steps_mapping[bs]:
                changed = True
                self.bs_steps_mapping[bs] = expanded

        if changed:
            self._build_reverse_mapping()
            self._init_current_params()
            logger.info(
                f"AutoSpecEngine: synced bs_steps_mapping with graph capacity: "
                f"{self.bs_steps_mapping}"
            )

    def _find_closest_bs(self, target: int) -> int:
        """Find the closest batch size in our configured list."""
        if target <= 0:
            return self.bs_list[0]
        idx = bisect.bisect_left(self.bs_list, target)
        if idx == 0:
            return self.bs_list[0]
        if idx == len(self.bs_list):
            return self.bs_list[-1]
        left, right = self.bs_list[idx - 1], self.bs_list[idx]
        return left if (target - left) <= (right - target) else right

    def get_spec_params(self, batch_size: int) -> SpecParams:
        """Get the current best speculative parameters for a given batch size."""
        bs = self._find_closest_bs(batch_size)
        return self.current_params[bs]

    def update(
        self,
        batch_size: int,
        num_accepted_tokens: int,
        num_draft_tokens_per_req: int,
    ):
        """Update the tuner with acceptance feedback from one decode step.

        Args:
            batch_size: Number of requests in the batch.
            num_accepted_tokens: Total number of accepted tokens across all requests.
            num_draft_tokens_per_req: Number of draft tokens per request.
        """
        if batch_size <= 0:
            return

        bs = self._find_closest_bs(batch_size)
        available_steps = self.bs_steps_mapping[bs]

        # Compute acceptance rate for this step
        avg_accept_length = (num_accepted_tokens + batch_size) / batch_size
        accept_rate = avg_accept_length / num_draft_tokens_per_req if num_draft_tokens_per_req > 0 else 0.0

        # Update EMA
        old_ema = self.ema_accept_rate[bs]
        self.ema_accept_rate[bs] = self.ema_alpha * accept_rate + (1 - self.ema_alpha) * old_ema

        current_rate = self.ema_accept_rate[bs]
        current_steps = self.current_params[bs].num_steps

        inc_threshold = self.increase_thresholds.get(bs, 0.55)
        dec_threshold = self.decrease_thresholds.get(bs, 0.50)

        new_steps = current_steps
        if current_rate >= inc_threshold:
            # Try to increase steps
            higher = [s for s in available_steps if s > current_steps]
            if higher:
                new_steps = higher[0]  # smallest step larger than current
                self.current_params[bs] = SpecParams(num_steps=new_steps)
        elif current_rate < dec_threshold:
            # Try to decrease steps
            lower = [s for s in available_steps if s < current_steps]
            if lower:
                new_steps = lower[-1]  # largest step smaller than current
                self.current_params[bs] = SpecParams(num_steps=new_steps)

        if new_steps != current_steps:
            logger.info(
                f"AutoSpecEngine: bs={batch_size}(mapped={bs}) step change "
                f"{current_steps}->{new_steps}, "
                f"ema_rate={current_rate:.4f}, "
                f"thresholds=({inc_threshold:.2f},{dec_threshold:.2f})"
            )
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"AutoSpecEngine: bs={batch_size}(mapped={bs}) "
                f"accept_rate={accept_rate:.4f}, ema={current_rate:.4f}, "
                f"steps={current_steps}, "
                f"thresholds=({inc_threshold:.2f},{dec_threshold:.2f})"
            )

    def _format_params(self) -> str:
        parts = []
        for bs in sorted(self.current_params.keys()):
            p = self.current_params[bs]
            parts.append(f"bs{bs}:steps={p.num_steps}")
        return ", ".join(parts)
