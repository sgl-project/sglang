from __future__ import annotations

import logging
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.kv_canary.verify import RealKvSource
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
    maybe_inaccurate_forward_batch: Optional[ForwardBatch],
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
    pick = random.randrange(len(filtered))
    return filtered[pick]


def flip_random_source_byte_and_log(
    *,
    perturb_name: str,
    group: CanaryBufferGroup,
    slot_idx: int,
) -> None:
    """Pick a random K-half real_kv source on group, flip byte 0 of slot_idx's tile
    in it, and log the result. Logs and returns silently when the group has no
    real_kv_sources_k or the slot cannot be mapped into the chosen source."""
    if not group.real_kv_sources_k:
        logger.info(
            "kv_canary perturb %s: skipped because group=%s has no real_kv_sources_k",
            perturb_name,
            group.kind.name,
        )
        return
    source_pick = random.randrange(len(group.real_kv_sources_k))
    source = group.real_kv_sources_k[source_pick]
    flip_result = flip_first_byte_in_source(
        group=group, source=source, slot_idx=slot_idx
    )
    if flip_result is None:
        logger.info(
            "kv_canary perturb %s: skipped because slot=%d could not be mapped "
            "into group=%s source_idx=%d",
            perturb_name,
            slot_idx,
            group.kind.name,
            source_pick,
        )
        return
    row, col, original_byte = flip_result
    logger.info(
        "kv_canary perturb %s: group=%s source_idx=%d slot=%d row=%d col=%d "
        "original_byte=0x%02X new_byte=0x%02X",
        perturb_name,
        group.kind.name,
        source_pick,
        slot_idx,
        row,
        col,
        original_byte,
        original_byte ^ 0xFF,
    )


def flip_first_byte_in_source(
    *,
    group: CanaryBufferGroup,
    source: RealKvSource,
    slot_idx: int,
    slot_is_physical: bool = False,
) -> Optional[tuple[int, int, int]]:
    """XOR 0xFF on byte 0 of slot_idx's tile in source.tensor (column
    `(physical_slot % page_size) * num_bytes_per_token`, not row offset 0).

    For SWA groups, slot_idx is translated through group.swa_index_lut before computing
    (row, col). Returns (row, col, original_byte) for logging, or None if the slot is
    out-of-range / source is degenerate.
    """
    if source.num_bytes_per_token <= 0 or source.read_bytes <= 0:
        return None

    physical_slot = slot_idx
    if (
        not slot_is_physical
        and group.kind == PoolKind.SWA
        and group.swa_index_lut is not None
    ):
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
