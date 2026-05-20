"""C++/Python constant parity test for the kv_canary jit_kernel."""

from __future__ import annotations

import re
from pathlib import Path

from sglang.jit_kernel.kv_canary_verify import (
    _FAIL_REASON_BIT_CHAIN_HASH,
    _FAIL_REASON_BIT_POSITION,
    _FAIL_REASON_BIT_REAL_KV_HASH,
    _MAX_REAL_KV_SOURCES,
    _VIOLATION_FIELD_EXPECTED_AUX,
    _VIOLATION_FIELD_EXPECTED_TOKEN,
    _VIOLATION_FIELD_FAIL_REASON_BITS,
    _VIOLATION_FIELD_KERNEL_KIND,
    _VIOLATION_FIELD_POSITION,
    _VIOLATION_FIELD_SLOT_IDX,
    _VIOLATION_FIELD_STORED_CHAIN_HASH,
    _VIOLATION_FIELD_STORED_TOKEN,
    VIOLATION_FIELDS,
    RealKvHashMode,
)
from sglang.jit_kernel.kv_canary_verify_ref import (
    _FIELD_POSITION,
    _FIELD_PREV_HASH,
    _FIELD_REAL_KV_HASH,
    _FIELD_TOKEN,
)
from sglang.jit_kernel.kv_canary_write import (
    _FAIL_REASON_BIT_WRITE_POSITION_MISMATCH,
    _FAIL_REASON_BIT_WRITE_TOKEN_MISMATCH,
    CanaryPseudoMode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=30, suite="nightly-kernel-1-gpu", nightly=True)


_CANARY_COMMON_CUH: Path = (
    Path(__file__).resolve().parents[1] / "csrc" / "kv_canary" / "canary_common.cuh"
)


def _parse_int_constants(*, source: str) -> dict[str, int]:
    """Extract every ``constexpr <type> kName = <integer>;`` symbol from a cuh source.

    Handles decimal literals with optional trailing type suffix (``U``, ``L``, ``LL``, ``ULL``) and
    leading ``1LL << N`` shift literals. Hex literals (e.g. the chain anchor) are decoded via int(...,16).
    """
    constants: dict[str, int] = {}
    int_re = re.compile(
        r"constexpr\s+(?:[a-zA-Z_][\w:]*)\s+(k[A-Za-z][A-Za-z0-9_]*)\s*=\s*([^;]+);"
    )
    for name, raw_expr in int_re.findall(source):
        expr = raw_expr.strip()
        # Match patterns like "1LL << 3" or "1 << 4".
        shift_match = re.match(r"^\s*1\s*L*\s*<<\s*(\d+)\s*$", expr)
        if shift_match is not None:
            constants[name] = 1 << int(shift_match.group(1))
            continue
        hex_match = re.match(r"^\s*0[xX]([0-9A-Fa-f]+)U*L*L*\s*$", expr)
        if hex_match is not None:
            constants[name] = int(hex_match.group(1), 16)
            continue
        dec_match = re.match(r"^\s*(-?\d+)U*L*L*\s*$", expr)
        if dec_match is not None:
            constants[name] = int(dec_match.group(1))
            continue
        # Anything else (e.g. references to other constants) is skipped — caller will fail loudly when it
        # asks for a missing key.
    return constants


def _parse_enum_values(*, source: str, enum_name: str) -> dict[str, int]:
    """Extract members of a strongly-typed ``enum class`` from a cuh source.

    Body grammar parsed: ``kFoo = <integer>,``. Hex / decimal / shift literals are handled identically to
    ``_parse_int_constants``.
    """
    pattern = re.compile(
        r"enum\s+class\s+" + re.escape(enum_name) + r"\s*:\s*[^\{]+\{([^}]+)\}"
    )
    match = pattern.search(source)
    if match is None:
        raise AssertionError(
            f"kv-canary const sync: enum class {enum_name} not found in cuh"
        )
    body = match.group(1)
    members: dict[str, int] = {}
    for line in body.splitlines():
        member_match = re.match(
            r"\s*(k[A-Za-z][A-Za-z0-9_]*)\s*=\s*([^,]+),?", line.strip()
        )
        if member_match is None:
            continue
        name, raw_expr = member_match.group(1), member_match.group(2).strip()
        shift_match = re.match(r"^\s*1\s*L*\s*<<\s*(\d+)\s*$", raw_expr)
        if shift_match is not None:
            members[name] = 1 << int(shift_match.group(1))
            continue
        hex_match = re.match(r"^\s*0[xX]([0-9A-Fa-f]+)U*L*L*\s*$", raw_expr)
        if hex_match is not None:
            members[name] = int(hex_match.group(1), 16)
            continue
        dec_match = re.match(r"^\s*(-?\d+)U*L*L*\s*$", raw_expr)
        if dec_match is not None:
            members[name] = int(dec_match.group(1))
            continue
    return members


def _read_cuh() -> str:
    return _CANARY_COMMON_CUH.read_text(encoding="utf-8")


def test_canary_field_offsets() -> None:
    """C++ ``kCanaryFieldToken / Position / PrevHash / RealKvHash`` parity with Python ``_FIELD_*``."""
    constants = _parse_int_constants(source=_read_cuh())
    assert constants["kCanaryFieldToken"] == _FIELD_TOKEN
    assert constants["kCanaryFieldPosition"] == _FIELD_POSITION
    assert constants["kCanaryFieldPrevHash"] == _FIELD_PREV_HASH
    assert constants["kCanaryFieldRealKvHash"] == _FIELD_REAL_KV_HASH


def test_violation_field_offsets() -> None:
    """C++ ``kViolationField{KernelKind, SlotIdx, ...}`` parity with Python ``_VIOLATION_FIELD_*``."""
    constants = _parse_int_constants(source=_read_cuh())
    assert constants["kViolationFieldKernelKind"] == _VIOLATION_FIELD_KERNEL_KIND
    assert constants["kViolationFieldSlotIdx"] == _VIOLATION_FIELD_SLOT_IDX
    assert constants["kViolationFieldPosition"] == _VIOLATION_FIELD_POSITION
    assert constants["kViolationFieldStoredToken"] == _VIOLATION_FIELD_STORED_TOKEN
    assert constants["kViolationFieldExpectedToken"] == _VIOLATION_FIELD_EXPECTED_TOKEN
    assert (
        constants["kViolationFieldStoredChainHash"]
        == _VIOLATION_FIELD_STORED_CHAIN_HASH
    )
    assert constants["kViolationFieldExpectedAux"] == _VIOLATION_FIELD_EXPECTED_AUX
    assert (
        constants["kViolationFieldFailReasonBits"] == _VIOLATION_FIELD_FAIL_REASON_BITS
    )


def test_fail_reason_bits() -> None:
    """``kFailReason{ChainHash, Position, RealKvHash, WriteTokenMismatch, WritePositionMismatch}`` parity."""
    constants = _parse_int_constants(source=_read_cuh())
    assert constants["kFailReasonChainHash"] == _FAIL_REASON_BIT_CHAIN_HASH
    assert constants["kFailReasonPosition"] == _FAIL_REASON_BIT_POSITION
    assert constants["kFailReasonRealKvHash"] == _FAIL_REASON_BIT_REAL_KV_HASH
    assert (
        constants["kFailReasonWriteTokenMismatch"]
        == _FAIL_REASON_BIT_WRITE_TOKEN_MISMATCH
    )
    assert (
        constants["kFailReasonWritePositionMismatch"]
        == _FAIL_REASON_BIT_WRITE_POSITION_MISMATCH
    )


def test_real_kv_hash_mode_enum() -> None:
    """C++ ``RealKvHashMode {kOff, kBit, kAll}`` parity with Python ``RealKvHashMode``."""
    members = _parse_enum_values(source=_read_cuh(), enum_name="RealKvHashMode")
    assert members["kOff"] == int(RealKvHashMode.OFF)
    assert members["kBit"] == int(RealKvHashMode.BIT)
    assert members["kAll"] == int(RealKvHashMode.ALL)


def test_canary_mock_mode_enum() -> None:
    """C++ ``CanaryPseudoMode {kOff, kOn}`` parity with Python ``CanaryPseudoMode``."""
    members = _parse_enum_values(source=_read_cuh(), enum_name="CanaryPseudoMode")
    assert members["kOff"] == int(CanaryPseudoMode.OFF)
    assert members["kOn"] == int(CanaryPseudoMode.ON)


def test_max_real_kv_sources() -> None:
    """C++ ``kMaxRealKvSources == 4`` parity with Python ``_MAX_REAL_KV_SOURCES == 4``."""
    constants = _parse_int_constants(source=_read_cuh())
    assert constants["kMaxRealKvSources"] == _MAX_REAL_KV_SOURCES
    assert constants["kMaxRealKvSources"] == 4


def test_violation_fields_count() -> None:
    """C++ ``kViolationFields`` count parity with Python ``VIOLATION_FIELDS`` and the per-field count."""
    constants = _parse_int_constants(source=_read_cuh())
    assert constants["kViolationFields"] == VIOLATION_FIELDS
    # Sanity: the eight kViolationField* indices [0..7] line up with the count.
    field_indices = {
        name: value
        for name, value in constants.items()
        if name.startswith("kViolationField") and name != "kViolationFields"
    }
    assert sorted(field_indices.values()) == list(range(VIOLATION_FIELDS))
