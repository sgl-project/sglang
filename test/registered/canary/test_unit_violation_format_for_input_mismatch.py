"""Lock the human-readable rendering of INPUT_TOKEN / INPUT_POSITION mismatch.

The canary kernel reuses the existing 8-field violation row layout for
the two new pseudo-mode fail reasons by overloading
``expected_hash`` / ``actual_hash`` (INPUT_TOKEN_MISMATCH carries
expected_token / actual_token in those slots) and ``expected_position``
(INPUT_POSITION_MISMATCH carries the oracle's expected position there).
The runner's ``_format_violation`` is the only thing that translates
those overloaded fields back into a debuggable error message, so this
test pins that rendering: the formatted lines must surface the expected
vs actual values without exposing the field-reuse trick.
"""

from __future__ import annotations

import unittest

from sglang.jit_kernel.kv_cache_canary import (
    _VIOLATION_FIELD_ACTUAL_HASH,
    _VIOLATION_FIELD_EXPECTED_HASH,
    _VIOLATION_FIELD_EXPECTED_POSITION,
    _VIOLATION_FIELD_FAIL_REASON,
    _VIOLATION_FIELD_KERNEL_KIND,
    _VIOLATION_FIELD_POSITION,
    _VIOLATION_FIELD_SLOT_IDX,
    _VIOLATION_FIELD_TOKEN_ID,
    KERNEL_KIND_HEAD,
    KERNEL_KIND_TAIL,
    VIOLATION_FIELDS,
    FailReason,
)
from sglang.srt.kv_cache_canary.runner import CanaryRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2)


def _make_row(
    *,
    kernel_kind: int,
    fail_reason: int,
    slot_idx: int,
    token_id: int,
    position: int,
    expected_hash: int = 0,
    actual_hash: int = 0,
    expected_position: int = 0,
) -> list[int]:
    # Build by field-offset index so a layout change in
    # _VIOLATION_FIELD_* surfaces here at test time, not in the kernel.
    row = [0] * VIOLATION_FIELDS
    row[_VIOLATION_FIELD_KERNEL_KIND] = kernel_kind
    row[_VIOLATION_FIELD_FAIL_REASON] = fail_reason
    row[_VIOLATION_FIELD_SLOT_IDX] = slot_idx
    row[_VIOLATION_FIELD_TOKEN_ID] = token_id
    row[_VIOLATION_FIELD_POSITION] = position
    row[_VIOLATION_FIELD_EXPECTED_HASH] = expected_hash
    row[_VIOLATION_FIELD_ACTUAL_HASH] = actual_hash
    row[_VIOLATION_FIELD_EXPECTED_POSITION] = expected_position
    return row


class TestFormatViolationInputTokenMismatch(unittest.TestCase):
    """INPUT_TOKEN_MISMATCH renders expected vs actual token without leaking field reuse."""

    def test_lines_surface_expected_and_actual_token(self) -> None:
        row = _make_row(
            kernel_kind=KERNEL_KIND_HEAD,
            fail_reason=int(FailReason.INPUT_TOKEN_MISMATCH),
            slot_idx=42,
            token_id=999,
            position=3,
            expected_hash=123,
            actual_hash=999,
        )
        text = CanaryRunner._format_violation("head_k", row, write_index=1)

        self.assertIn("INPUT_TOKEN_MISMATCH", text)
        self.assertIn("token_id:", text)
        self.assertIn("expected=123", text)
        self.assertIn("actual=999", text)
        self.assertIn("slot_idx:          42", text)
        self.assertIn("HEAD", text)

    def test_lines_do_not_render_token_as_hex_hash(self) -> None:
        """expected_hash field carries the token id; format must NOT print 0x hex for tokens."""
        row = _make_row(
            kernel_kind=KERNEL_KIND_HEAD,
            fail_reason=int(FailReason.INPUT_TOKEN_MISMATCH),
            slot_idx=1,
            token_id=88,
            position=0,
            expected_hash=77,
            actual_hash=88,
        )
        text = CanaryRunner._format_violation("head_k", row, write_index=1)
        self.assertNotIn("expected_hash:", text)
        self.assertNotIn("hash_xor_diff:", text)


class TestFormatViolationInputPositionMismatch(unittest.TestCase):
    """INPUT_POSITION_MISMATCH renders oracle-expected vs actual position."""

    def test_lines_surface_expected_and_actual_position(self) -> None:
        row = _make_row(
            kernel_kind=KERNEL_KIND_TAIL,
            fail_reason=int(FailReason.INPUT_POSITION_MISMATCH),
            slot_idx=5,
            token_id=222,
            position=17,
            expected_position=16,
        )
        text = CanaryRunner._format_violation("tail_v", row, write_index=2)

        self.assertIn("INPUT_POSITION_MISMATCH", text)
        self.assertIn("position:", text)
        self.assertIn("expected=16", text)
        self.assertIn("actual=17", text)
        self.assertIn("TAIL", text)
        self.assertIn("tail_v", text)


class TestFormatViolationOtherReasonsUnchanged(unittest.TestCase):
    """Existing HASH / REAL_KV_HASH formatting is not regressed."""

    def test_hash_reason_still_renders_hex_diff(self) -> None:
        row = _make_row(
            kernel_kind=KERNEL_KIND_HEAD,
            fail_reason=int(FailReason.HASH),
            slot_idx=3,
            token_id=10,
            position=2,
            expected_hash=0xDEADBEEF,
            actual_hash=0xCAFEBABE,
            expected_position=2,
        )
        text = CanaryRunner._format_violation("head_k", row, write_index=1)
        self.assertIn("expected_hash:", text)
        self.assertIn("hash_xor_diff:", text)
        self.assertIn("expected=2 actual=2", text)


if __name__ == "__main__":
    unittest.main(verbosity=3)
