from __future__ import annotations

import logging
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.perturb.config import PerturbConfig, TargetGroupKind

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class WarmupGate:
    """Per-hook warmup window check + once-per-lifetime disable/enable log emission.

    Shared across the four perturb-point hooks so warmup state is decided in one place
    rather than duplicated per hook.
    """

    def __init__(
        self,
        *,
        config: PerturbConfig,
        outer_step_counter_getter: Callable[[], int],
    ) -> None:
        self._config = config
        self._outer_step_counter_getter = outer_step_counter_getter
        self._warmup_disable_logged: bool = False
        self._warmup_enable_logged: bool = False

    def is_in_warmup(self) -> bool:
        step = self._outer_step_counter_getter()
        warmup_steps = self._config.warmup_steps

        if step < warmup_steps:
            self._log_warmup_disabled_once(warmup_steps)
            return True

        self._log_warmup_enabled_once(step)
        return False

    def _log_warmup_disabled_once(self, warmup_steps: int) -> None:
        if self._warmup_disable_logged:
            return

        logger.info(
            "kv_canary perturb: disabled during warmup window "
            "(first %d forward steps)",
            warmup_steps,
        )
        self._warmup_disable_logged = True

    def _log_warmup_enabled_once(self, step: int) -> None:
        if self._warmup_enable_logged:
            return

        logger.info("kv_canary perturb: enabled after warmup window at step=%d", step)
        self._warmup_enable_logged = True


def should_run_perturbation(
    *,
    perturb_name: str,
    probability: float,
    warmup_gate: WarmupGate,
    maybe_inaccurate_forward_batch: Optional["ForwardBatch"],
    require_forward_batch: bool = True,
) -> bool:
    if probability <= 0.0:
        return False
    if warmup_gate.is_in_warmup():
        return False
    if require_forward_batch and maybe_inaccurate_forward_batch is None:
        logger.info(
            "kv_canary perturb %s: skipped because maybe_inaccurate_forward_batch is unavailable",
            perturb_name,
        )
        return False
    return torch.rand((), device="cpu").item() < probability


def pick_target_group(
    *,
    buffer_groups: tuple[CanaryBufferGroup, ...],
    target_kind: TargetGroupKind,
) -> Optional[CanaryBufferGroup]:
    """Filter buffer_groups by target_kind.

    Returns None if no group matches.
    """
    if target_kind == TargetGroupKind.FULL:
        want = PoolKind.FULL
    elif target_kind == TargetGroupKind.SWA:
        want = PoolKind.SWA
    else:
        raise ValueError(f"Unsupported target_group_kind: {target_kind!r}")
    filtered = [group for group in buffer_groups if group.kind == want]
    if not filtered:
        return None
    pick = random.randrange(len(filtered))
    return filtered[pick]
