"""Python <-> C++ constant sync check for the canary kernel.

Scheme A: the C++ kernel exposes a host-side ``canary_get_constants``
function (registered as a tvm-ffi wrapper). Calling it from Python
fills a CPU int64 tensor with every shared ``constexpr int k... = N;``
value in canary.cuh, in the exact order declared by
:data:`_CANARY_CONSTANT_LAYOUT`. This test asserts each kernel-reported
value matches the Python mirror integer of the same role.

Drift between the two sides is caught at test time — and because we
pull from the *running* kernel module, not from parsed source text, we
also catch the case where the .cuh declarations move around or get
overridden in a build configuration.
"""

from __future__ import annotations

import unittest
from typing import Dict

from sglang.jit_kernel.kv_cache_canary import (
    _CANARY_FIELD_POSITION,
    _CANARY_FIELD_PREV_HASH,
    _CANARY_FIELD_REAL_KV_HASH,
    _CANARY_FIELD_TOKEN_ID,
    _VIOLATION_FIELD_ACTUAL_HASH,
    _VIOLATION_FIELD_EXPECTED_HASH,
    _VIOLATION_FIELD_EXPECTED_POSITION,
    _VIOLATION_FIELD_FAIL_REASON,
    _VIOLATION_FIELD_KERNEL_KIND,
    _VIOLATION_FIELD_POSITION,
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
    get_cpp_constants,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="extra-a", runner_config="1-gpu-small")


# C++ identifier -> Python value that the test asserts against.
_EXPECTED_PAIRS: Dict[str, int] = {
    "kCanaryFieldsPerSlot": CANARY_FIELDS_PER_SLOT,
    "kCanaryFieldTokenId": _CANARY_FIELD_TOKEN_ID,
    "kCanaryFieldPosition": _CANARY_FIELD_POSITION,
    "kCanaryFieldPrevHash": _CANARY_FIELD_PREV_HASH,
    "kCanaryFieldRealKvHash": _CANARY_FIELD_REAL_KV_HASH,
    "kViolationFields": VIOLATION_FIELDS,
    "kViolationFieldKernelKind": _VIOLATION_FIELD_KERNEL_KIND,
    "kViolationFieldFailReason": _VIOLATION_FIELD_FAIL_REASON,
    "kViolationFieldSlotIdx": _VIOLATION_FIELD_SLOT_IDX,
    "kViolationFieldTokenId": _VIOLATION_FIELD_TOKEN_ID,
    "kViolationFieldPosition": _VIOLATION_FIELD_POSITION,
    "kViolationFieldExpectedHash": _VIOLATION_FIELD_EXPECTED_HASH,
    "kViolationFieldActualHash": _VIOLATION_FIELD_ACTUAL_HASH,
    "kViolationFieldExpectedPosition": _VIOLATION_FIELD_EXPECTED_POSITION,
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

    def test_kernel_module_constants_match_python_mirrors(self) -> None:
        cpp_constants = get_cpp_constants()
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
            f"expected constants missing from kernel module: {missing_in_cpp}",
        )
        self.assertFalse(
            mismatches,
            "canary.cuh / Python const mismatch:\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in mismatches.items()),
        )

    def test_kernel_kind_constants_have_expected_layout(self) -> None:
        # KERNEL_KIND_HEAD / TAIL are kernel-launch kwargs (not constexpr
        # int in canary.cuh), so the kernel-reported layout cannot
        # cover them; assert the documented values here as a separate
        # guard against accidental flips.
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
