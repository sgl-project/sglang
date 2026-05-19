"""Lock the human-readable rendering of INPUT_TOKEN / INPUT_POSITION mismatch.

The canary kernel reuses the existing 10-field violation row layout for
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
    KERNEL_KIND_HEAD,
    KERNEL_KIND_TAIL,
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
    req_id: int,
    token_id: int,
    position: int,
    expected_hash: int = 0,
    actual_hash: int = 0,
    expected_req_id: int = 0,
    expected_position: int = 0,
) -> list[int]:
    return [
        kernel_kind,
        fail_reason,
        slot_idx,
        req_id,
        token_id,
        position,
        expected_hash,
        actual_hash,
        expected_req_id,
        expected_position,
    ]


class TestFormatViolationInputTokenMismatch(unittest.TestCase):
    """INPUT_TOKEN_MISMATCH renders expected vs actual token without leaking field reuse."""

    def test_lines_surface_expected_and_actual_token(self) -> None:
        row = _make_row(
            kernel_kind=KERNEL_KIND_HEAD,
            fail_reason=int(FailReason.INPUT_TOKEN_MISMATCH),
            slot_idx=42,
            req_id=7,
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
            req_id=2,
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
            req_id=11,
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

    def test_unused_expected_req_id_is_not_misrendered(self) -> None:
        """expected_req_id is forced to 0 by the kernel for this reason; do not show as 'expected=0'."""
        row = _make_row(
            kernel_kind=KERNEL_KIND_HEAD,
            fail_reason=int(FailReason.INPUT_POSITION_MISMATCH),
            slot_idx=0,
            req_id=9,
            token_id=4,
            position=2,
            expected_position=1,
        )
        text = CanaryRunner._format_violation("head_k", row, write_index=1)
        self.assertIn("req_id:            9", text)
        self.assertNotIn("expected=0 actual=9", text)


class TestFormatViolationOtherReasonsUnchanged(unittest.TestCase):
    """Existing HASH / REAL_KV_HASH formatting is not regressed."""

    def test_hash_reason_still_renders_hex_diff(self) -> None:
        row = _make_row(
            kernel_kind=KERNEL_KIND_HEAD,
            fail_reason=int(FailReason.HASH),
            slot_idx=3,
            req_id=1,
            token_id=10,
            position=2,
            expected_hash=0xDEADBEEF,
            actual_hash=0xCAFEBABE,
            expected_req_id=1,
            expected_position=2,
        )
        text = CanaryRunner._format_violation("head_k", row, write_index=1)
        self.assertIn("expected_hash:", text)
        self.assertIn("hash_xor_diff:", text)
        self.assertIn("expected=1 actual=1", text)


if __name__ == "__main__":
    unittest.main(verbosity=3)
