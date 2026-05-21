from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_canary.verify import RealKvSource
from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.perturb.config import PerturbConfig, TargetGroupKind

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class WarmupGate:
    """Per-hook warmup window check + once-per-lifetime disable/enable log emission.

    Shared across the three perturb-point hooks so warmup state is decided in one place
    rather than duplicated per hook.
    """

    def __init__(
        self,
        *,
        config: PerturbConfig,
        step_counter_getter: Callable[[], int],
    ) -> None:
        self._config = config
        self._step_counter_getter = step_counter_getter
        self._warmup_disable_logged: bool = False
        self._warmup_enable_logged: bool = False

    def is_in_warmup(self) -> bool:
        step = self._step_counter_getter()
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
    forward_batch: Optional["ForwardBatch"],
    require_forward_batch: bool = True,
) -> bool:
    if probability <= 0.0:
        return False
    if warmup_gate.is_in_warmup():
        return False
    if require_forward_batch and forward_batch is None:
        logger.info(
            "kv_canary perturb %s: skipped because forward_batch is unavailable",
            perturb_name,
        )
        return False
    return torch.rand((), device="cpu").item() < probability


def pick_target_group(
    *,
    buffer_groups: tuple[CanaryBufferGroup, ...],
    target_kind: TargetGroupKind,
) -> Optional[CanaryBufferGroup]:
    """Filter buffer_groups by target_kind restricted to groups with non-empty real_kv_sources_k.

    Returns None if no group matches.
    """
    eligible = [group for group in buffer_groups if group.real_kv_sources_k]
    if not eligible:
        return None
    if target_kind == TargetGroupKind.FULL:
        want = PoolKind.FULL
    elif target_kind == TargetGroupKind.SWA:
        want = PoolKind.SWA
    else:
        raise ValueError(f"Unsupported target_group_kind: {target_kind!r}")
    filtered = [group for group in eligible if group.kind == want]
    if not filtered:
        return None
    pick = int(torch.randint(0, len(filtered), (1,)).item())
    return filtered[pick]


def flip_first_byte_in_source(
    *,
    group: CanaryBufferGroup,
    source: RealKvSource,
    slot_idx: int,
) -> Optional[tuple[int, int, int]]:
    """XOR 0xFF on byte_offset=0 of slot_idx in source.tensor.

    For SWA groups, slot_idx is translated through group.swa_index_lut before computing
    (row, col). Returns (row, col, original_byte) for logging, or None if the slot is
    out-of-range / source is degenerate.
    """
    if source.num_bytes_per_token <= 0 or source.read_bytes <= 0:
        return None

    physical_slot = slot_idx
    if group.kind == PoolKind.SWA and group.swa_index_lut is not None:
        lut = group.swa_index_lut
        if slot_idx < 0 or slot_idx >= int(lut.shape[0]):
            return None
        physical_slot = int(lut[slot_idx].detach().to("cpu").item())
        if physical_slot < 0:
            return None

    page_size = max(1, source.page_size)
    row = physical_slot // page_size
    col = (physical_slot % page_size) * source.num_bytes_per_token
    if row < 0 or row >= int(source.tensor.shape[0]):
        return None
    if col < 0 or col >= int(source.tensor.shape[1]):
        return None

    flat = source.tensor
    original_byte = int(flat[row, col].item())
    flat[row, col] = original_byte ^ 0xFF
    return row, col, original_byte
