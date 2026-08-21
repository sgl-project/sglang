from __future__ import annotations

import logging

from sglang.kernels.ops.kv_canary.consts import FailReason
from sglang.kernels.ops.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.config import CanaryConfig, CanaryMode
from sglang.srt.kv_canary.state import CanaryDeviceState

logger = logging.getLogger(__name__)

_WRITE_BITS = FailReason.WRITE_TOKEN_MISMATCH | FailReason.WRITE_POSITION_MISMATCH
_TOKEN_BITS = FailReason.WRITE_TOKEN_MISMATCH | FailReason.VERIFY_TOKEN_MISMATCH


def _reason_label(bit: FailReason) -> str:
    return bit.name.lower().removesuffix("_mismatch")


class ViolationReporter:
    def __init__(
        self,
        *,
        config: CanaryConfig,
        device_state: CanaryDeviceState,
    ) -> None:
        self._config = config
        self._device_state = device_state
        self._raised: bool = False
        self._last_logged_write_index: int = 0

    @property
    def is_raised(self) -> bool:
        return self._raised

    def log_or_raise_violation(self, *, outer_step_counter: int) -> None:
        violation_log = self._device_state.violation_log
        write_index = int(violation_log.violation_write_index.cpu().item())
        if write_index == 0:
            return
        ring = violation_log.violation_ring.cpu()
        ring_capacity = int(ring.shape[0])
        valid_count = min(write_index, ring_capacity)
        ring_overflow = write_index > ring_capacity

        start = min(self._last_logged_write_index, valid_count)
        if start >= valid_count:
            return

        messages: list[str] = [
            _format_violation(
                row=ring[i].tolist(),
                total=write_index,
                ring_overflow=ring_overflow,
                step_when_pumped=outer_step_counter,
            )
            for i in range(start, valid_count)
        ]
        self._last_logged_write_index = valid_count

        # log mode: always surface every violation as WARNING.
        if self._config.mode is CanaryMode.LOG:
            for message in messages:
                logger.warning(message)
            return
        self._raised = True
        raise RuntimeError("\n".join(messages))


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
    reasons = [_reason_label(bit) for bit in FailReason if bits_int & int(bit)]
    is_write = bool(bits_int & int(_WRITE_BITS))
    u64_mask = (1 << 64) - 1

    # Stable single-line key=value summary, parsed by the regex in
    # python/sglang/test/kv_canary/violation_log_utils.py and asserted by
    # assert_violation_logged_any in python/sglang/test/kv_canary/violation_assert_mixin.py.
    # Format frozen: do not reorder / rename / change separators without updating those helpers.
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

    has_token_check = bool(bits_int & int(_TOKEN_BITS))
    if is_write:
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
        stored_prev_hash = int(stored_chain_hash) & u64_mask
        expected_prev_hash = int(expected_aux) & u64_mask
        stored_body = (
            f"  stored:   token_id={int(stored_token)}   position={int(position)} "
            f"prev_hash={stored_prev_hash:#018x}"
        )
        expected_body = (
            f"  expected: token_id={int(expected_token)} prev_hash={expected_prev_hash:#018x}"
            if has_token_check
            else f"  expected: prev_hash={expected_prev_hash:#018x}"
        )
        body = [stored_body, expected_body]

    return "\n".join([structured_line, header, kind_line, reasons_line, *body, footer])
