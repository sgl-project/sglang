from __future__ import annotations

import logging

from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.runner.pump import PumpAndAllreduce
from sglang.srt.kv_canary.violation_state import CanaryDeviceState

logger = logging.getLogger(__name__)

_ON_MODE_VERBOSE_LIMIT: int = 10


class ViolationReporter:
    def __init__(
        self,
        *,
        config: CanaryConfig,
        device_state: CanaryDeviceState,
        pump: PumpAndAllreduce,
    ) -> None:
        self._config = config
        self._device_state = device_state
        self._pump = pump
        self._on_mode_violations_logged: int = 0
        self._raised: bool = False

    @property
    def is_raised(self) -> bool:
        return self._raised

    def raise_violation(self) -> None:
        violation_log = self._device_state.violation_log
        write_index = int(violation_log.violation_write_index.cpu().item())
        if write_index == 0:
            return
        ring = violation_log.violation_ring.cpu()
        ring_overflow = write_index > int(ring.shape[0])
        message = _format_violation(
            row=ring[0].tolist(),
            total=write_index,
            ring_overflow=ring_overflow,
            step_when_pumped=self._pump.step_counter,
        )
        if self._config.mode == "on":
            self._on_mode_violations_logged += 1
            if self._on_mode_violations_logged <= _ON_MODE_VERBOSE_LIMIT:
                logger.warning(message)
            else:
                logger.debug(message)
            return
        self._raised = True
        raise RuntimeError(message)


def _canary_kind_label(tag: CanaryLaunchTag) -> str:
    name_lower = tag.name.lower()
    if tag in (
        CanaryLaunchTag.SWEEP_K_FULL,
        CanaryLaunchTag.SWEEP_V_FULL,
        CanaryLaunchTag.SWEEP_K_SWA,
        CanaryLaunchTag.SWEEP_V_SWA,
    ):
        return name_lower
    return f"per_forward_{name_lower}"


def _format_violation(
    *,
    row: list[int],
    total: int,
    ring_overflow: bool,
    step_when_pumped: int,
) -> str:
    (
        kernel_kind,
        slot_idx,
        position,
        stored_token,
        expected_token,
        stored_chain_hash,
        expected_aux,
        fail_reason_bits,
    ) = row
    try:
        tag_label = CanaryLaunchTag(int(kernel_kind)).name
        canary_kind = _canary_kind_label(CanaryLaunchTag(int(kernel_kind)))
    except ValueError:
        tag_label = f"unknown({int(kernel_kind)})"
        canary_kind = tag_label
    bits = int(fail_reason_bits)
    reasons: list[str] = []
    if bits & 0x1:
        reasons.append("chain_hash")
    if bits & 0x2:
        reasons.append("position")
    if bits & 0x4:
        reasons.append("real_kv_hash")
    u64_mask = (1 << 64) - 1
    stored_prev_hash = int(stored_chain_hash) & u64_mask
    expected_prev_hash = int(expected_aux) & u64_mask

    return "\n".join(
        [
            (
                f"KV cache canary violation detected (kernel_kind={tag_label}, "
                f"slot_idx={int(slot_idx)}, position={int(position)})"
            ),
            f"canary_kind:       {canary_kind}",
            f"  fail_reasons: {' '.join(reasons) if reasons else 'none'}",
            (
                f"  stored:   token_id={int(stored_token)}   position={int(position)} "
                f"prev_hash={stored_prev_hash:#018x}"
            ),
            (
                f"  expected: token_id={int(expected_token)}   position={int(position)} "
                f"prev_hash={expected_prev_hash:#018x}"
            ),
            (
                f"  total_violations={total} ring_overflow={ring_overflow} "
                f"step_when_pumped={step_when_pumped}"
            ),
        ]
    )
