"""CPU-only unit test for the CanaryViolationView decoder.

The decoder is the only thing standing between a raw int64 violation
row and the assertions Block 5 / 6 will make on it, so pin the
positional-to-named-field mapping here. The mapping is fragile because
some fail_reasons overload ``expected_hash`` / ``actual_hash`` to carry
token ids and others carry the chain hash.
"""

from __future__ import annotations

import unittest
from test.registered.pseudo_mode._violation_view import CanaryViolationView

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
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2)


def _make_row(
    *,
    kernel_kind: int,
    fail_reason: int,
    slot_idx: int = 0,
    token_id: int = 0,
    position: int = 0,
    expected_hash: int = 0,
    actual_hash: int = 0,
    expected_position: int = 0,
) -> list[int]:
    """Build a violation row by field-offset constant.

    Constructing by offset (not by positional index) means a layout
    change in ``_VIOLATION_FIELD_*`` surfaces at this test boundary
    rather than silently shifting the decoded values.
    """
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


class TestFromRowFieldDecode(unittest.TestCase):
    """Each named field on the view reads from the right row offset."""

    def test_hash_reason_decodes_full_field_set(self) -> None:
        row = _make_row(
            kernel_kind=KERNEL_KIND_HEAD,
            fail_reason=int(FailReason.HASH),
            slot_idx=3,
            token_id=7,
            position=11,
            expected_hash=0xDEADBEEF,
            actual_hash=0xCAFEBABE,
            expected_position=11,
        )
        view = CanaryViolationView.from_row(
            row=row, canary_kind="head_k", write_index=4
        )
        self.assertEqual(view.canary_kind, "head_k")
        self.assertEqual(view.kernel_kind, "HEAD")
        self.assertEqual(view.fail_reason, "HASH")
        self.assertEqual(view.fail_reason_int, int(FailReason.HASH))
        self.assertEqual(view.slot_idx, 3)
        self.assertEqual(view.token_id, 7)
        self.assertEqual(view.position, 11)
        self.assertEqual(view.expected_hash, 0xDEADBEEF)
        self.assertEqual(view.actual_hash, 0xCAFEBABE)
        self.assertEqual(view.expected_position, 11)
        self.assertEqual(view.write_index, 4)
        self.assertTrue(view.is_real())

    def test_input_token_mismatch_overloads_hash_fields(self) -> None:
        row = _make_row(
            kernel_kind=KERNEL_KIND_TAIL,
            fail_reason=int(FailReason.INPUT_TOKEN_MISMATCH),
            slot_idx=9,
            token_id=999,
            position=4,
            expected_hash=123,
            actual_hash=999,
        )
        view = CanaryViolationView.from_row(
            row=row, canary_kind="tail_k", write_index=1
        )
        self.assertEqual(view.fail_reason, "INPUT_TOKEN_MISMATCH")
        self.assertEqual(view.kernel_kind, "TAIL")
        # Field reuse: expected_hash carries expected token, actual_hash actual.
        self.assertEqual(view.expected_hash, 123)
        self.assertEqual(view.actual_hash, 999)
        self.assertEqual(view.token_id, 999)
        self.assertEqual(view.position, 4)

    def test_input_position_mismatch_carries_expected_position(self) -> None:
        row = _make_row(
            kernel_kind=KERNEL_KIND_HEAD,
            fail_reason=int(FailReason.INPUT_POSITION_MISMATCH),
            slot_idx=1,
            position=17,
            expected_position=16,
        )
        view = CanaryViolationView.from_row(
            row=row, canary_kind="head_v", write_index=2
        )
        self.assertEqual(view.fail_reason, "INPUT_POSITION_MISMATCH")
        self.assertEqual(view.expected_position, 16)
        self.assertEqual(view.position, 17)


class TestFromRowSentinelRow(unittest.TestCase):
    """An empty (all-zero) row decodes as a NONE fail_reason and ``is_real`` is False."""

    def test_none_row_is_not_real(self) -> None:
        row = [0] * VIOLATION_FIELDS
        view = CanaryViolationView.from_row(
            row=row, canary_kind="head_k", write_index=0
        )
        self.assertEqual(view.fail_reason, "NONE")
        self.assertEqual(view.fail_reason_int, int(FailReason.NONE))
        self.assertFalse(view.is_real())


class TestFromRowUnknownReason(unittest.TestCase):
    """Out-of-range fail_reason ids render as ``unknown(N)`` rather than crashing."""

    def test_unknown_reason_falls_back_to_unknown_name(self) -> None:
        row = _make_row(
            kernel_kind=KERNEL_KIND_HEAD,
            fail_reason=99,  # not in FailReason
        )
        view = CanaryViolationView.from_row(
            row=row, canary_kind="head_k", write_index=1
        )
        self.assertEqual(view.fail_reason, "unknown(99)")
        self.assertEqual(view.fail_reason_int, 99)
        # is_real treats any non-NONE as real, including unknown.
        self.assertTrue(view.is_real())


class TestStrRendering(unittest.TestCase):
    """``__str__`` includes the essentials for log inspection."""

    def test_str_carries_canary_kernel_reason_slot_token_pos(self) -> None:
        row = _make_row(
            kernel_kind=KERNEL_KIND_TAIL,
            fail_reason=int(FailReason.TOKEN_ID),
            slot_idx=42,
            token_id=7,
            position=3,
        )
        view = CanaryViolationView.from_row(
            row=row, canary_kind="tail_v", write_index=5
        )
        text = str(view)
        self.assertIn("canary=tail_v", text)
        self.assertIn("kernel=TAIL", text)
        self.assertIn("reason=TOKEN_ID", text)
        self.assertIn("slot=42", text)
        self.assertIn("token=7", text)
        self.assertIn("pos=3", text)
        self.assertIn("write_index=5", text)


if __name__ == "__main__":
    unittest.main(verbosity=3)
