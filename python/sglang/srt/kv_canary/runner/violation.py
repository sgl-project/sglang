from __future__ import annotations

import logging

from sglang.jit_kernel.kv_canary.consts import FailReason
from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.runner.violation_pump import ViolationPump
from sglang.srt.kv_canary.state import CanaryDeviceState

logger = logging.getLogger(__name__)

_WRITE_BITS = FailReason.WRITE_TOKEN_MISMATCH | FailReason.WRITE_POSITION_MISMATCH
_REASON_LABELS: dict[FailReason, str] = {
    FailReason.CHAIN_HASH: "chain_hash",
    FailReason.POSITION: "position",
    FailReason.REAL_KV_HASH: "real_kv_hash",
    FailReason.WRITE_TOKEN_MISMATCH: "write_token",
    FailReason.WRITE_POSITION_MISMATCH: "write_position",
}


class ViolationReporter:
    def __init__(
        self,
        *,
        config: CanaryConfig,
        device_state: CanaryDeviceState,
        violation_pump: ViolationPump,
    ) -> None:
        self._config = config
        self._device_state = device_state
        self._violation_pump = violation_pump
        self._raised: bool = False

    @property
    def is_raised(self) -> bool:
        return self._raised

    def log_or_raise_violation(self) -> None:
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
            step_when_pumped=self._violation_pump.step_counter,
        )
        # log mode: always surface every violation as WARNING. Never rate-limit or demote to
        # DEBUG: if violation volume is high enough to feel like spam, that's a bug in whatever
        # is producing them, not a reason to hide them.
        if self._config.mode == "log":
            logger.warning(message)
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
    bits_int = int(fail_reason_bits)
    reasons = [label for bit, label in _REASON_LABELS.items() if bits_int & int(bit)]
    is_write = bool(bits_int & int(_WRITE_BITS))
    u64_mask = (1 << 64) - 1

    # Stable single-line key=value summary, parsed by CanaryE2EBase.assert_violation_logged.
    # Format frozen: do not reorder / rename / change separators without updating the test
    # helper in python/sglang/test/kv_canary/e2e_base.py.
    structured_line = (
        f"kv_canary violation: "
        f"launch_tag={tag_label} "
        f"fail_reason={'+'.join(reasons) if reasons else 'none'} "
        f"slot_idx={int(slot_idx)} "
        f"position={int(position)} "
        f"stored_token={int(stored_token)} "
        f"expected_token={int(expected_token)} "
        f"stored_chain_hash={int(stored_chain_hash) & u64_mask:#018x} "
        f"expected_aux={int(expected_aux) & u64_mask:#018x}"
    )

    header = (
        f"KV cache canary violation detected (kernel_kind={tag_label}, "
        f"slot_idx={int(slot_idx)}, position={int(position)})"
    )
    kind_line = f"canary_kind:       {canary_kind}"
    reasons_line = f"  fail_reasons: {' '.join(reasons) if reasons else 'none'}"
    footer = (
        f"  total_violations={total} ring_overflow={ring_overflow} "
        f"step_when_pumped={step_when_pumped}"
    )

    if is_write:
        # Write-path row layout: POSITION holds the actually-written position; EXPECTED_AUX holds
        # the expected position (NOT a chain hash); EXPECTED_TOKEN holds the expected token.
        # STORED_CHAIN_HASH is the running chain hash at write time (kept for debug context).
        running_prev_hash = int(stored_chain_hash) & u64_mask
        body = [
            (
                f"  actual:   token_id={int(stored_token)}   position={int(position)} "
                f"prev_hash={running_prev_hash:#018x}"
            ),
            (
                f"  expected: token_id={int(expected_token)}   position={int(expected_aux)}"
            ),
        ]
    else:
        # Verify-path row layout: POSITION holds the stored position; EXPECTED_AUX holds the
        # expected chain hash; EXPECTED_TOKEN is unused (always 0) so don't print it.
        stored_prev_hash = int(stored_chain_hash) & u64_mask
        expected_prev_hash = int(expected_aux) & u64_mask
        body = [
            (
                f"  stored:   token_id={int(stored_token)}   position={int(position)} "
                f"prev_hash={stored_prev_hash:#018x}"
            ),
            f"  expected: prev_hash={expected_prev_hash:#018x}",
        ]

    return "\n".join([structured_line, header, kind_line, reasons_line, *body, footer])
