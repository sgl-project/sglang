"""Python <-> C++ constant sync check for the canary kernel.

Scheme A from the design doc: rather than hand-maintaining two copies
of the same integers (slot field offsets, violation field offsets,
fail reasons, real-KV hash modes), this test parses the C++ ``constexpr
int kXxx = N;`` declarations out of ``canary.cuh`` and asserts each one
matches the Python module's exported value.

Drift between the two sides is caught at test time, before it can
silently corrupt a chain hash or a violation row.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Dict

from sglang.jit_kernel.kv_cache_canary import (
    _CANARY_FIELD_POSITION,
    _CANARY_FIELD_PREV_HASH,
    _CANARY_FIELD_REAL_KV_HASH,
    _CANARY_FIELD_REQ_ID,
    _CANARY_FIELD_TOKEN_ID,
    _VIOLATION_FIELD_ACTUAL_HASH,
    _VIOLATION_FIELD_EXPECTED_HASH,
    _VIOLATION_FIELD_FAIL_REASON,
    _VIOLATION_FIELD_KERNEL_KIND,
    _VIOLATION_FIELD_POSITION,
    _VIOLATION_FIELD_REQ_ID,
    _VIOLATION_FIELD_SLOT_IDX,
    _VIOLATION_FIELD_TOKEN_ID,
    CANARY_FIELDS_PER_SLOT,
    KERNEL_KIND_HEAD,
    KERNEL_KIND_TAIL,
    REAL_KV_HASH_BIT_BYTES,
    REAL_KV_HASH_MODE_ALL,
    REAL_KV_HASH_MODE_BIT,
    REAL_KV_HASH_MODE_OFF,
    VIOLATION_FIELDS,
    FailReason,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="extra-a", runner_config="1-gpu-small")

_CANARY_CUH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sglang"
    / "jit_kernel"
    / "csrc"
    / "kv_cache_canary"
    / "canary.cuh"
)

_CONSTEXPR_INT_PATTERN = re.compile(
    r"^\s*constexpr\s+int\s+(k[A-Za-z0-9_]+)\s*=\s*(-?\d+)\s*;",
    re.MULTILINE,
)


def _parse_cpp_constants() -> Dict[str, int]:
    text = _CANARY_CUH.read_text()
    return {name: int(value) for name, value in _CONSTEXPR_INT_PATTERN.findall(text)}


# Mapping: C++ identifier -> Python value the test asserts against.
_EXPECTED_PAIRS: Dict[str, int] = {
    "kCanaryFieldsPerSlot": CANARY_FIELDS_PER_SLOT,
    "kCanaryFieldReqId": _CANARY_FIELD_REQ_ID,
    "kCanaryFieldTokenId": _CANARY_FIELD_TOKEN_ID,
    "kCanaryFieldPosition": _CANARY_FIELD_POSITION,
    "kCanaryFieldPrevHash": _CANARY_FIELD_PREV_HASH,
    "kCanaryFieldRealKvHash": _CANARY_FIELD_REAL_KV_HASH,
    "kViolationFields": VIOLATION_FIELDS,
    "kViolationFieldKernelKind": _VIOLATION_FIELD_KERNEL_KIND,
    "kViolationFieldFailReason": _VIOLATION_FIELD_FAIL_REASON,
    "kViolationFieldSlotIdx": _VIOLATION_FIELD_SLOT_IDX,
    "kViolationFieldReqId": _VIOLATION_FIELD_REQ_ID,
    "kViolationFieldTokenId": _VIOLATION_FIELD_TOKEN_ID,
    "kViolationFieldPosition": _VIOLATION_FIELD_POSITION,
    "kViolationFieldExpectedHash": _VIOLATION_FIELD_EXPECTED_HASH,
    "kViolationFieldActualHash": _VIOLATION_FIELD_ACTUAL_HASH,
    "kFailReasonReqId": int(FailReason.REQ_ID),
    "kFailReasonTokenId": int(FailReason.TOKEN_ID),
    "kFailReasonPosition": int(FailReason.POSITION),
    "kFailReasonHash": int(FailReason.HASH),
    "kFailReasonPositionMonotonic": int(FailReason.POSITION_MONOTONIC),
    "kFailReasonRealKvHash": int(FailReason.REAL_KV_HASH),
    "kRealKvHashModeOff": REAL_KV_HASH_MODE_OFF,
    "kRealKvHashModeBit": REAL_KV_HASH_MODE_BIT,
    "kRealKvHashModeAll": REAL_KV_HASH_MODE_ALL,
}


class TestPythonCppConstantSync(unittest.TestCase):
    """Every shared integer constant in canary.cuh must match Python."""

    def test_canary_cuh_constexpr_int_values_match_python_module(self) -> None:
        self.assertTrue(_CANARY_CUH.exists(), f"missing source: {_CANARY_CUH}")
        cpp_constants = _parse_cpp_constants()

        self.assertGreater(
            len(cpp_constants),
            0,
            f"no `constexpr int k... = N;` declarations found in {_CANARY_CUH}; "
            "parser regex needs updating",
        )

        mismatches: Dict[str, str] = {}
        missing_in_cpp: list[str] = []
        for cpp_name, py_value in _EXPECTED_PAIRS.items():
            if cpp_name not in cpp_constants:
                missing_in_cpp.append(cpp_name)
                continue
            if cpp_constants[cpp_name] != py_value:
                mismatches[cpp_name] = (
                    f"cpp={cpp_constants[cpp_name]} python={py_value}"
                )

        self.assertFalse(
            missing_in_cpp,
            f"expected constants missing from canary.cuh: {missing_in_cpp}",
        )
        self.assertFalse(
            mismatches,
            "canary.cuh / Python const mismatch:\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in mismatches.items()),
        )

    def test_kernel_kind_constants_have_expected_layout(self) -> None:
        # KERNEL_KIND_HEAD / TAIL aren't ``constexpr int`` in canary.cuh
        # (they're plumbed through as the kernel_kind kwarg), so the
        # regex above wouldn't catch them; assert the documented values
        # here as a separate guard.
        self.assertEqual(KERNEL_KIND_HEAD, 0)
        self.assertEqual(KERNEL_KIND_TAIL, 1)
        self.assertNotEqual(KERNEL_KIND_HEAD, KERNEL_KIND_TAIL)

    def test_real_kv_hash_bit_bytes_within_slot_stride(self) -> None:
        # 16 bytes = 2 int64 fields' worth; we pick a fixed cheap prefix
        # rather than a stride-derived value so the bit budget is the
        # same regardless of the real-KV pool's element size.
        self.assertEqual(REAL_KV_HASH_BIT_BYTES, 16)


if __name__ == "__main__":
    unittest.main(verbosity=3)
