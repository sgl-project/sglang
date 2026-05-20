from __future__ import annotations

from sglang.jit_kernel.kv_canary import consts
from sglang.jit_kernel.kv_canary.consts import FailReason
from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.runner.violation import _format_violation
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="extra-a", runner_config="1-gpu-large")


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


def test_format_violation_verify_path_labels_each_bit() -> None:
    row = _make_row(
        stored_chain_hash=0x1111111111111111,
        expected_aux=0x2222222222222222,
        fail_reason_bits=int(
            FailReason.CHAIN_HASH | FailReason.POSITION | FailReason.REAL_KV_HASH
        ),
    )
    out = _format_violation(
        row=row, total=1, ring_overflow=False, step_when_pumped=7
    )
    assert "chain_hash" in out
    assert "position" in out
    assert "real_kv_hash" in out
    assert "write_token" not in out
    assert "write_position" not in out
    assert "stored:" in out
    assert "prev_hash=0x1111111111111111" in out
    assert "prev_hash=0x2222222222222222" in out


def test_format_violation_write_token_mismatch_labels_and_position() -> None:
    """Regression: write violations were silently labeled `fail_reasons: none` and `expected_aux`
    was printed as a hex prev_hash even though it carries `expected_position`."""
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
    assert "write_token" in out
    assert "fail_reasons: none" not in out
    assert "actual:" in out
    assert "token_id=999" in out
    assert "token_id=888" in out
    assert "position=43" in out
    # expected_aux must NOT be rendered as a chain hash on the write path.
    assert "prev_hash=0x000000000000002b" not in out


def test_format_violation_write_position_mismatch_uses_expected_aux_as_position() -> None:
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
    assert "write_position" in out
    assert "position=42" in out
    assert "position=99" in out


def test_format_violation_combined_write_bits_render_both_labels() -> None:
    row = _make_row(
        fail_reason_bits=int(
            FailReason.WRITE_TOKEN_MISMATCH | FailReason.WRITE_POSITION_MISMATCH
        ),
    )
    out = _format_violation(
        row=row, total=1, ring_overflow=False, step_when_pumped=0
    )
    assert "write_token" in out
    assert "write_position" in out


def test_format_violation_unknown_kernel_kind_renders_unknown_label() -> None:
    row = _make_row(fail_reason_bits=int(FailReason.CHAIN_HASH))
    row[consts.VIOLATION_FIELD_KERNEL_KIND] = 9999
    out = _format_violation(
        row=row, total=1, ring_overflow=False, step_when_pumped=0
    )
    assert "unknown(9999)" in out
