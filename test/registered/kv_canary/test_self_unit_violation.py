from __future__ import annotations

from sglang.jit_kernel.kv_canary import consts
from sglang.jit_kernel.kv_canary.consts import FailReason
from sglang.jit_kernel.kv_canary.verify import CanaryLaunchTag
from sglang.srt.kv_canary.runner.violation_reporter import _format_violation
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
    """Verify verify-path violations render each fail-reason bit."""
    row = _make_row(
        stored_chain_hash=0x1111111111111111,
        expected_aux=0x2222222222222222,
        fail_reason_bits=int(
            FailReason.CHAIN_HASH | FailReason.POSITION | FailReason.REAL_KV_HASH
        ),
    )
    out = _format_violation(row=row, total=1, ring_overflow=False, step_when_pumped=7)
    assert out == (
        "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=chain_hash+position+real_kv_hash "
        "slot_idx=17 position=42 stored_token=111 expected_token=0 stored_chain_hash=0x1111111111111111 "
        "expected_aux=0x2222222222222222\n"
        "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=17, position=42)\n"
        "canary_kind:       per_forward_head_k_full\n"
        "  fail_reasons: chain_hash position real_kv_hash\n"
        "  stored:   token_id=111   position=42 prev_hash=0x1111111111111111\n"
        "  expected: prev_hash=0x2222222222222222\n"
        "  total_violations=1 ring_overflow=False step_when_pumped=7"
    )


def test_format_violation_write_token_mismatch_labels_and_position() -> None:
    """Verify write-token violations render token and position details."""
    row = _make_row(
        position=42,
        stored_token=999,
        expected_token=888,
        stored_chain_hash=0xDEADBEEFCAFEBABE,
        expected_aux=43,
        fail_reason_bits=int(FailReason.WRITE_TOKEN_MISMATCH),
    )
    out = _format_violation(row=row, total=1, ring_overflow=False, step_when_pumped=0)
    assert out == (
        "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=write_token slot_idx=17 position=42 "
        "stored_token=999 expected_token=888 stored_chain_hash=0xdeadbeefcafebabe "
        "expected_aux=0x000000000000002b\n"
        "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=17, position=42)\n"
        "canary_kind:       per_forward_head_k_full\n"
        "  fail_reasons: write_token\n"
        "  actual:   token_id=999   position=42 prev_hash=0xdeadbeefcafebabe\n"
        "  expected: token_id=888   position=43\n"
        "  total_violations=1 ring_overflow=False step_when_pumped=0"
    )


def test_format_violation_write_position_mismatch_uses_expected_aux_as_position() -> (
    None
):
    """Verify write-position violations render expected_aux as a position."""
    row = _make_row(
        position=42,
        stored_token=111,
        expected_token=111,
        expected_aux=99,
        fail_reason_bits=int(FailReason.WRITE_POSITION_MISMATCH),
    )
    out = _format_violation(row=row, total=1, ring_overflow=False, step_when_pumped=0)
    assert out == (
        "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=write_position slot_idx=17 position=42 "
        "stored_token=111 expected_token=111 stored_chain_hash=0x0000000000000000 "
        "expected_aux=0x0000000000000063\n"
        "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=17, position=42)\n"
        "canary_kind:       per_forward_head_k_full\n"
        "  fail_reasons: write_position\n"
        "  actual:   token_id=111   position=42 prev_hash=0x0000000000000000\n"
        "  expected: token_id=111   position=99\n"
        "  total_violations=1 ring_overflow=False step_when_pumped=0"
    )


def test_format_violation_combined_write_bits_render_both_labels() -> None:
    """Verify combined write violation bits render both labels."""
    row = _make_row(
        fail_reason_bits=int(
            FailReason.WRITE_TOKEN_MISMATCH | FailReason.WRITE_POSITION_MISMATCH
        ),
    )
    out = _format_violation(row=row, total=1, ring_overflow=False, step_when_pumped=0)
    assert out == (
        "kv_canary violation: launch_tag=HEAD_K_FULL fail_reason=write_token+write_position "
        "slot_idx=17 position=42 stored_token=111 expected_token=0 stored_chain_hash=0x0000000000000000 "
        "expected_aux=0x0000000000000000\n"
        "KV cache canary violation detected (kernel_kind=HEAD_K_FULL, slot_idx=17, position=42)\n"
        "canary_kind:       per_forward_head_k_full\n"
        "  fail_reasons: write_token write_position\n"
        "  actual:   token_id=111   position=42 prev_hash=0x0000000000000000\n"
        "  expected: token_id=0   position=0\n"
        "  total_violations=1 ring_overflow=False step_when_pumped=0"
    )


def test_format_violation_unknown_kernel_kind_renders_unknown_label() -> None:
    """Verify unknown kernel kinds render an unknown label."""
    row = _make_row(fail_reason_bits=int(FailReason.CHAIN_HASH))
    row[consts.VIOLATION_FIELD_KERNEL_KIND] = 9999
    out = _format_violation(row=row, total=1, ring_overflow=False, step_when_pumped=0)
    assert out == (
        "kv_canary violation: launch_tag=unknown(9999) fail_reason=chain_hash slot_idx=17 position=42 "
        "stored_token=111 expected_token=0 stored_chain_hash=0x0000000000000000 "
        "expected_aux=0x0000000000000000\n"
        "KV cache canary violation detected (kernel_kind=unknown(9999), slot_idx=17, position=42)\n"
        "canary_kind:       unknown(9999)\n"
        "  fail_reasons: chain_hash\n"
        "  stored:   token_id=111   position=42 prev_hash=0x0000000000000000\n"
        "  expected: prev_hash=0x0000000000000000\n"
        "  total_violations=1 ring_overflow=False step_when_pumped=0"
    )
