from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.jit_kernel.kv_canary import consts
from sglang.jit_kernel.kv_canary.consts import FailReason
from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.config import CanaryMode
from sglang.srt.kv_canary.runner import violation_reporter as violation_reporter_module
from sglang.srt.kv_canary.runner.violation_reporter import (
    ViolationReporter,
    _format_violation,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="extra-a-test-1-gpu-small-amd")


def _make_row(
    *,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
    slot_idx: int = 17,
    position: int = 42,
    stored_token: int = 111,
    expected_token: int = 0,
    stored_chain_hash: int = 0,
    expected_aux: int = 0,
    fail_reason_bits: int = 0,
) -> list[int]:
    row = [0] * consts.VIOLATION_FIELDS
    row[consts.VIOLATION_FIELD_KERNEL_KIND] = int(kernel_kind)
    row[consts.VIOLATION_FIELD_SLOT_IDX] = slot_idx
    row[consts.VIOLATION_FIELD_POSITION] = position
    row[consts.VIOLATION_FIELD_STORED_TOKEN] = stored_token
    row[consts.VIOLATION_FIELD_EXPECTED_TOKEN] = expected_token
    row[consts.VIOLATION_FIELD_STORED_CHAIN_HASH] = stored_chain_hash
    row[consts.VIOLATION_FIELD_EXPECTED_AUX] = expected_aux
    row[consts.VIOLATION_FIELD_FAIL_REASON_BITS] = fail_reason_bits
    return row


class TestViolationReporter(CustomTestCase):
    def test_format_violation_verify_path_labels_each_bit(self) -> None:
        """Verify verify-path violations render each fail-reason bit."""
        row = _make_row(
            stored_chain_hash=0x1111111111111111,
            expected_aux=0x2222222222222222,
            fail_reason_bits=int(
                FailReason.VERIFY_CHAIN_HASH_MISMATCH
                | FailReason.VERIFY_POSITION_MISMATCH
                | FailReason.VERIFY_REAL_KV_HASH_MISMATCH
            ),
        )
        out = _format_violation(
            row=row, total=1, ring_overflow=False, step_when_pumped=7
        )
        self.assertEqual(
            out,
            "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=verify_chain_hash+verify_position+verify_real_kv_hash "
            "slot_idx=17 position=42 stored_token=111 expected_token=0 stored_chain_hash=0x1111111111111111 "
            "expected_aux=0x2222222222222222\n"
            "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=17, position=42)\n"
            "canary_kind:       per_forward_head_k_full\n"
            "  fail_reasons: verify_chain_hash verify_position verify_real_kv_hash\n"
            "  stored:   token_id=111   position=42 prev_hash=0x1111111111111111\n"
            "  expected: prev_hash=0x2222222222222222\n"
            "  total_violations=1 ring_overflow=False step_when_pumped=7",
        )

    def test_format_violation_write_token_mismatch_labels_and_position(self) -> None:
        """Verify write-token violations render token and position details."""
        row = _make_row(
            position=42,
            stored_token=999,
            expected_token=888,
            stored_chain_hash=0xDEADBEEFCAFEBABE,
            expected_aux=43,
            fail_reason_bits=int(FailReason.WRITE_TOKEN_MISMATCH),
        )
        out = _format_violation(
            row=row, total=1, ring_overflow=False, step_when_pumped=0
        )
        self.assertEqual(
            out,
            "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=write_token slot_idx=17 position=42 "
            "stored_token=999 expected_token=888 stored_chain_hash=0xdeadbeefcafebabe "
            "expected_aux=0x000000000000002b\n"
            "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=17, position=42)\n"
            "canary_kind:       per_forward_head_k_full\n"
            "  fail_reasons: write_token\n"
            "  actual:   token_id=999   position=42 prev_hash=0xdeadbeefcafebabe\n"
            "  expected: token_id=888   position=43\n"
            "  total_violations=1 ring_overflow=False step_when_pumped=0",
        )

    def test_format_violation_write_position_mismatch_uses_expected_aux_as_position(
        self,
    ) -> None:
        """Verify write-position violations render expected_aux as a position."""
        row = _make_row(
            position=42,
            stored_token=111,
            expected_token=111,
            expected_aux=99,
            fail_reason_bits=int(FailReason.WRITE_POSITION_MISMATCH),
        )
        out = _format_violation(
            row=row, total=1, ring_overflow=False, step_when_pumped=0
        )
        self.assertEqual(
            out,
            "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=write_position slot_idx=17 position=42 "
            "stored_token=111 expected_token=111 stored_chain_hash=0x0000000000000000 "
            "expected_aux=0x0000000000000063\n"
            "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=17, position=42)\n"
            "canary_kind:       per_forward_head_k_full\n"
            "  fail_reasons: write_position\n"
            "  actual:   token_id=111   position=42 prev_hash=0x0000000000000000\n"
            "  expected: token_id=111   position=99\n"
            "  total_violations=1 ring_overflow=False step_when_pumped=0",
        )

    def test_format_violation_combined_write_bits_render_both_labels(self) -> None:
        """Verify combined write violation bits render both labels."""
        row = _make_row(
            fail_reason_bits=int(
                FailReason.WRITE_TOKEN_MISMATCH | FailReason.WRITE_POSITION_MISMATCH
            ),
        )
        out = _format_violation(
            row=row, total=1, ring_overflow=False, step_when_pumped=0
        )
        self.assertEqual(
            out,
            "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=write_token+write_position "
            "slot_idx=17 position=42 stored_token=111 expected_token=0 stored_chain_hash=0x0000000000000000 "
            "expected_aux=0x0000000000000000\n"
            "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=17, position=42)\n"
            "canary_kind:       per_forward_head_k_full\n"
            "  fail_reasons: write_token write_position\n"
            "  actual:   token_id=111   position=42 prev_hash=0x0000000000000000\n"
            "  expected: token_id=0   position=0\n"
            "  total_violations=1 ring_overflow=False step_when_pumped=0",
        )

    def test_format_violation_unknown_kernel_kind_renders_unknown_label(self) -> None:
        """Verify unknown kernel kinds render an unknown label."""
        row = _make_row(fail_reason_bits=int(FailReason.VERIFY_CHAIN_HASH_MISMATCH))
        row[consts.VIOLATION_FIELD_KERNEL_KIND] = 9999
        out = _format_violation(
            row=row, total=1, ring_overflow=False, step_when_pumped=0
        )
        self.assertEqual(
            out,
            "kv_canary violation: launch_tag=unknown(9999) fail_reason=verify_chain_hash slot_idx=17 position=42 "
            "stored_token=111 expected_token=0 stored_chain_hash=0x0000000000000000 "
            "expected_aux=0x0000000000000000\n"
            "KV cache canary violation detected (kernel_kind=unknown(9999), slot_idx=17, position=42)\n"
            "canary_kind:       unknown(9999)\n"
            "  fail_reasons: verify_chain_hash\n"
            "  stored:   token_id=111   position=42 prev_hash=0x0000000000000000\n"
            "  expected: prev_hash=0x0000000000000000\n"
            "  total_violations=1 ring_overflow=False step_when_pumped=0",
        )


def _make_reporter(
    *,
    rows: list[list[int]],
    write_index: int,
    ring_capacity: int,
    mode: CanaryMode = CanaryMode.LOG,
) -> ViolationReporter:
    ring = torch.zeros(ring_capacity, consts.VIOLATION_FIELDS, dtype=torch.int64)
    for i, row in enumerate(rows):
        ring[i] = torch.tensor(row, dtype=torch.int64)
    violation_log = SimpleNamespace(
        violation_ring=ring,
        violation_write_index=torch.tensor([write_index], dtype=torch.int32),
    )
    device_state = SimpleNamespace(violation_log=violation_log)
    config = SimpleNamespace(mode=mode)
    return ViolationReporter(config=config, device_state=device_state)


class TestLogOrRaiseViolation(CustomTestCase):
    def test_log_or_raise_violation_empty_ring_is_noop(self) -> None:
        """Empty ring (write_index=0) emits no warning and leaves reporter non-raised."""
        reporter = _make_reporter(
            rows=[], write_index=0, ring_capacity=4, mode=CanaryMode.LOG
        )
        with patch.object(violation_reporter_module.logger, "warning") as mock_warning:
            reporter.log_or_raise_violation(outer_step_counter=0)
        mock_warning.assert_not_called()
        self.assertFalse(reporter.is_raised)

    def test_log_mode_emits_one_warning_per_violation(self) -> None:
        """Log mode with 3 valid rows emits 3 warnings, each a full _format_violation snapshot for that row."""
        rows = [
            _make_row(
                slot_idx=11,
                position=101,
                fail_reason_bits=int(FailReason.VERIFY_CHAIN_HASH_MISMATCH),
            ),
            _make_row(
                slot_idx=22,
                position=202,
                fail_reason_bits=int(FailReason.VERIFY_POSITION_MISMATCH),
            ),
            _make_row(
                slot_idx=33,
                position=303,
                fail_reason_bits=int(FailReason.VERIFY_REAL_KV_HASH_MISMATCH),
            ),
        ]
        reporter = _make_reporter(
            rows=rows, write_index=3, ring_capacity=4, mode=CanaryMode.LOG
        )
        with patch.object(violation_reporter_module.logger, "warning") as mock_warning:
            reporter.log_or_raise_violation(outer_step_counter=7)

        self.assertEqual(mock_warning.call_count, 3)
        messages: list[str] = [call.args[0] for call in mock_warning.call_args_list]
        self.assertEqual(
            messages[0],
            "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=verify_chain_hash slot_idx=11 position=101 "
            "stored_token=111 expected_token=0 stored_chain_hash=0x0000000000000000 "
            "expected_aux=0x0000000000000000\n"
            "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=11, position=101)\n"
            "canary_kind:       per_forward_head_k_full\n"
            "  fail_reasons: verify_chain_hash\n"
            "  stored:   token_id=111   position=101 prev_hash=0x0000000000000000\n"
            "  expected: prev_hash=0x0000000000000000\n"
            "  total_violations=3 ring_overflow=False step_when_pumped=7",
        )
        self.assertEqual(
            messages[1],
            "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=verify_position slot_idx=22 position=202 "
            "stored_token=111 expected_token=0 stored_chain_hash=0x0000000000000000 "
            "expected_aux=0x0000000000000000\n"
            "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=22, position=202)\n"
            "canary_kind:       per_forward_head_k_full\n"
            "  fail_reasons: verify_position\n"
            "  stored:   token_id=111   position=202 prev_hash=0x0000000000000000\n"
            "  expected: prev_hash=0x0000000000000000\n"
            "  total_violations=3 ring_overflow=False step_when_pumped=7",
        )
        self.assertEqual(
            messages[2],
            "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=verify_real_kv_hash slot_idx=33 position=303 "
            "stored_token=111 expected_token=0 stored_chain_hash=0x0000000000000000 "
            "expected_aux=0x0000000000000000\n"
            "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=33, position=303)\n"
            "canary_kind:       per_forward_head_k_full\n"
            "  fail_reasons: verify_real_kv_hash\n"
            "  stored:   token_id=111   position=303 prev_hash=0x0000000000000000\n"
            "  expected: prev_hash=0x0000000000000000\n"
            "  total_violations=3 ring_overflow=False step_when_pumped=7",
        )
        self.assertFalse(reporter.is_raised)

    def test_raise_mode_raises_one_error_containing_all_violations(self) -> None:
        """Raise mode raises a single RuntimeError whose text is the 3 formatted rows joined with single newlines."""
        rows = [
            _make_row(
                slot_idx=11,
                position=101,
                fail_reason_bits=int(FailReason.VERIFY_CHAIN_HASH_MISMATCH),
            ),
            _make_row(
                slot_idx=22,
                position=202,
                fail_reason_bits=int(FailReason.VERIFY_POSITION_MISMATCH),
            ),
            _make_row(
                slot_idx=33,
                position=303,
                fail_reason_bits=int(FailReason.VERIFY_REAL_KV_HASH_MISMATCH),
            ),
        ]
        reporter = _make_reporter(
            rows=rows, write_index=3, ring_capacity=4, mode=CanaryMode.RAISE
        )
        with self.assertRaises(RuntimeError) as ctx:
            reporter.log_or_raise_violation(outer_step_counter=5)

        self.assertEqual(
            str(ctx.exception),
            "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=verify_chain_hash slot_idx=11 position=101 "
            "stored_token=111 expected_token=0 stored_chain_hash=0x0000000000000000 "
            "expected_aux=0x0000000000000000\n"
            "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=11, position=101)\n"
            "canary_kind:       per_forward_head_k_full\n"
            "  fail_reasons: verify_chain_hash\n"
            "  stored:   token_id=111   position=101 prev_hash=0x0000000000000000\n"
            "  expected: prev_hash=0x0000000000000000\n"
            "  total_violations=3 ring_overflow=False step_when_pumped=5\n"
            "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=verify_position slot_idx=22 position=202 "
            "stored_token=111 expected_token=0 stored_chain_hash=0x0000000000000000 "
            "expected_aux=0x0000000000000000\n"
            "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=22, position=202)\n"
            "canary_kind:       per_forward_head_k_full\n"
            "  fail_reasons: verify_position\n"
            "  stored:   token_id=111   position=202 prev_hash=0x0000000000000000\n"
            "  expected: prev_hash=0x0000000000000000\n"
            "  total_violations=3 ring_overflow=False step_when_pumped=5\n"
            "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=verify_real_kv_hash slot_idx=33 position=303 "
            "stored_token=111 expected_token=0 stored_chain_hash=0x0000000000000000 "
            "expected_aux=0x0000000000000000\n"
            "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=33, position=303)\n"
            "canary_kind:       per_forward_head_k_full\n"
            "  fail_reasons: verify_real_kv_hash\n"
            "  stored:   token_id=111   position=303 prev_hash=0x0000000000000000\n"
            "  expected: prev_hash=0x0000000000000000\n"
            "  total_violations=3 ring_overflow=False step_when_pumped=5",
        )
        self.assertTrue(reporter.is_raised)

    def test_log_mode_ring_overflow_marks_overflow_in_each_row(self) -> None:
        """Log mode with write_index=5 but ring_capacity=2 emits 2 warnings, each a full snapshot with overflow footer."""
        rows = [
            _make_row(slot_idx=11, position=101),
            _make_row(slot_idx=22, position=202),
        ]
        reporter = _make_reporter(
            rows=rows, write_index=5, ring_capacity=2, mode=CanaryMode.LOG
        )
        with patch.object(violation_reporter_module.logger, "warning") as mock_warning:
            reporter.log_or_raise_violation(outer_step_counter=0)

        self.assertEqual(mock_warning.call_count, 2)
        messages: list[str] = [call.args[0] for call in mock_warning.call_args_list]
        self.assertEqual(
            messages[0],
            "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=none slot_idx=11 position=101 "
            "stored_token=111 expected_token=0 stored_chain_hash=0x0000000000000000 "
            "expected_aux=0x0000000000000000\n"
            "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=11, position=101)\n"
            "canary_kind:       per_forward_head_k_full\n"
            "  fail_reasons: none\n"
            "  stored:   token_id=111   position=101 prev_hash=0x0000000000000000\n"
            "  expected: prev_hash=0x0000000000000000\n"
            "  total_violations=5 ring_overflow=True step_when_pumped=0",
        )
        self.assertEqual(
            messages[1],
            "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=none slot_idx=22 position=202 "
            "stored_token=111 expected_token=0 stored_chain_hash=0x0000000000000000 "
            "expected_aux=0x0000000000000000\n"
            "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=22, position=202)\n"
            "canary_kind:       per_forward_head_k_full\n"
            "  fail_reasons: none\n"
            "  stored:   token_id=111   position=202 prev_hash=0x0000000000000000\n"
            "  expected: prev_hash=0x0000000000000000\n"
            "  total_violations=5 ring_overflow=True step_when_pumped=0",
        )


if __name__ == "__main__":
    unittest.main()
