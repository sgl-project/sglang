from __future__ import annotations

import importlib
import re
from enum import IntEnum
from pathlib import Path

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=30, suite="nightly-kernel-1-gpu", nightly=True)


_CSRC_DIR: Path = Path(__file__).resolve().parents[1] / "csrc" / "kv_canary"
_CUH_FILES: tuple[str, ...] = (
    "canary_common.cuh",
    "canary_verify.cuh",
    "canary_write.cuh",
)

_VERIFY_MOD: str = "sglang.jit_kernel.kv_canary.verify"
_WRITE_MOD: str = "sglang.jit_kernel.kv_canary.write"


# Registry: every `constexpr <type> kXxx = <int>;` symbol parsed from the canary .cuh files MUST appear
# here. Value is (module, attr) for Python-mirrored constants, or None for kernel-only launch params with
# no Python counterpart (block sizes etc.).
_CPP_TO_PY: dict[str, tuple[str, str] | None] = {
    "kCanaryChainAnchor": (_VERIFY_MOD, "CANARY_CHAIN_ANCHOR"),
    "kCanaryFieldsPerSlot": (_VERIFY_MOD, "_CANARY_FIELDS_PER_SLOT"),
    "kCanaryFieldToken": (_VERIFY_MOD, "_FIELD_TOKEN"),
    "kCanaryFieldPosition": (_VERIFY_MOD, "_FIELD_POSITION"),
    "kCanaryFieldPrevHash": (_VERIFY_MOD, "_FIELD_PREV_HASH"),
    "kCanaryFieldRealKvHash": (_VERIFY_MOD, "_FIELD_REAL_KV_HASH"),
    "kViolationFields": (_VERIFY_MOD, "VIOLATION_FIELDS"),
    "kViolationFieldKernelKind": (_VERIFY_MOD, "_VIOLATION_FIELD_KERNEL_KIND"),
    "kViolationFieldSlotIdx": (_VERIFY_MOD, "_VIOLATION_FIELD_SLOT_IDX"),
    "kViolationFieldPosition": (_VERIFY_MOD, "_VIOLATION_FIELD_POSITION"),
    "kViolationFieldStoredToken": (_VERIFY_MOD, "_VIOLATION_FIELD_STORED_TOKEN"),
    "kViolationFieldExpectedToken": (_VERIFY_MOD, "_VIOLATION_FIELD_EXPECTED_TOKEN"),
    "kViolationFieldStoredChainHash": (
        _VERIFY_MOD,
        "_VIOLATION_FIELD_STORED_CHAIN_HASH",
    ),
    "kViolationFieldExpectedAux": (_VERIFY_MOD, "_VIOLATION_FIELD_EXPECTED_AUX"),
    "kViolationFieldFailReasonBits": (_VERIFY_MOD, "_VIOLATION_FIELD_FAIL_REASON_BITS"),
    "kFailReasonChainHash": (_VERIFY_MOD, "_FAIL_REASON_BIT_CHAIN_HASH"),
    "kFailReasonPosition": (_VERIFY_MOD, "_FAIL_REASON_BIT_POSITION"),
    "kFailReasonRealKvHash": (_VERIFY_MOD, "_FAIL_REASON_BIT_REAL_KV_HASH"),
    "kFailReasonWriteTokenMismatch": (
        _WRITE_MOD,
        "_FAIL_REASON_BIT_WRITE_TOKEN_MISMATCH",
    ),
    "kFailReasonWritePositionMismatch": (
        _WRITE_MOD,
        "_FAIL_REASON_BIT_WRITE_POSITION_MISMATCH",
    ),
    "kMaxRealKvSources": (_VERIFY_MOD, "_MAX_REAL_KV_SOURCES"),
    "kRealKvSourceFieldsPerEntry": (_VERIFY_MOD, "_REAL_KV_SOURCE_FIELDS_PER_ENTRY"),
    "kRealKvSourceFieldPageSize": (_VERIFY_MOD, "_REAL_KV_SOURCE_FIELD_PAGE_SIZE"),
    "kRealKvSourceFieldNumBytesPerToken": (
        _VERIFY_MOD,
        "_REAL_KV_SOURCE_FIELD_NUM_BYTES_PER_TOKEN",
    ),
    "kRealKvSourceFieldReadBytes": (_VERIFY_MOD, "_REAL_KV_SOURCE_FIELD_READ_BYTES"),
    "kVerifyBlockSize": None,
    "kWriteBlockSize": None,
}

# Strongly-typed C++ enums and their Python IntEnum counterparts. Members are matched by uppercased name
# (kFooBar on C++ side ↔ FOO_BAR is too lax — actual current convention is kFoo ↔ FOO, kPartial ↔ PARTIAL,
# kAll ↔ ALL, kOn ↔ ON, kOff ↔ OFF — i.e. drop "k" prefix and upper-snake the camelCase tail).
_CPP_ENUMS: dict[str, tuple[str, str]] = {
    "RealKvHashMode": (_VERIFY_MOD, "RealKvHashMode"),
    "CanaryPseudoMode": (_WRITE_MOD, "CanaryPseudoMode"),
}

# Module-level Python constants whose names start with one of these prefixes MUST appear as a value in
# _CPP_TO_PY. Catches "added a Python constant but forgot the C++ side". Per-module so we don't drag in
# unrelated symbols.
_PINNED_PY_PREFIXES: dict[str, tuple[str, ...]] = {
    _VERIFY_MOD: (
        "_FIELD_",
        "_VIOLATION_FIELD_",
        "_FAIL_REASON_BIT_",
        "_CANARY_FIELDS_PER_SLOT",
        "_REAL_KV_SOURCE_FIELD",
        "_MAX_REAL_KV_SOURCES",
        "CANARY_CHAIN_ANCHOR",
        "VIOLATION_FIELDS",
    ),
    _WRITE_MOD: ("_FAIL_REASON_BIT_WRITE_",),
}


def _camel_to_upper_snake(camel: str) -> str:
    """``kPartial`` -> ``PARTIAL``, ``kNumBytesPerToken`` -> ``NUM_BYTES_PER_TOKEN``."""
    if not camel.startswith("k"):
        raise AssertionError(
            f"kv-canary const sync: enum member {camel!r} must start with 'k'"
        )
    body = camel[1:]
    snake = re.sub(r"([A-Z])", r"_\1", body).lstrip("_")
    return snake.upper()


def _decode_int_literal(expr: str) -> int | None:
    """Decode an integer literal from a C++ constexpr RHS. Returns None when the form is unrecognized."""
    expr = expr.strip()
    shift_match = re.match(r"^\s*1\s*L*\s*<<\s*(\d+)\s*$", expr)
    if shift_match is not None:
        return 1 << int(shift_match.group(1))
    hex_match = re.match(r"^\s*0[xX]([0-9A-Fa-f]+)U*L*L*\s*$", expr)
    if hex_match is not None:
        return int(hex_match.group(1), 16)
    dec_match = re.match(r"^\s*(-?\d+)U*L*L*\s*$", expr)
    if dec_match is not None:
        return int(dec_match.group(1))
    return None


def _parse_cpp_constants(*, source: str) -> tuple[set[str], dict[str, int]]:
    """Return (all_kXxx_names, decoded_values). Names are collected regardless of whether the RHS could
    be decoded, so the surjective set-check catches "new constexpr added with exotic literal form" too.
    """
    names: set[str] = set()
    values: dict[str, int] = {}
    int_re = re.compile(
        r"constexpr\s+(?:[a-zA-Z_][\w:]*)\s+(k[A-Za-z][A-Za-z0-9_]*)\s*=\s*([^;]+);"
    )
    for name, raw_expr in int_re.findall(source):
        names.add(name)
        value = _decode_int_literal(raw_expr)
        if value is not None:
            values[name] = value
    return names, values


def _parse_cpp_enum_members(*, source: str, enum_name: str) -> dict[str, int]:
    pattern = re.compile(
        r"enum\s+class\s+" + re.escape(enum_name) + r"\s*:\s*[^\{]+\{([^}]+)\}"
    )
    match = pattern.search(source)
    if match is None:
        raise AssertionError(
            f"kv-canary const sync: enum class {enum_name} not found in any cuh"
        )
    members: dict[str, int] = {}
    for line in match.group(1).splitlines():
        member_match = re.match(
            r"\s*(k[A-Za-z][A-Za-z0-9_]*)\s*=\s*([^,]+),?", line.strip()
        )
        if member_match is None:
            continue
        name = member_match.group(1)
        value = _decode_int_literal(member_match.group(2))
        if value is None:
            raise AssertionError(
                f"kv-canary const sync: enum {enum_name}.{name} has un-decodable RHS"
            )
        members[name] = value
    return members


def _read_all_cuh() -> str:
    return "\n".join(
        (_CSRC_DIR / fname).read_text(encoding="utf-8") for fname in _CUH_FILES
    )


def _resolve_py(target: tuple[str, str]) -> int:
    module_path, attr = target
    module = importlib.import_module(module_path)
    return int(getattr(module, attr))


def test_every_cpp_constant_is_registered() -> None:
    """Every constexpr kXxx in the cuh files must be declared in _CPP_TO_PY (catches drift one way)."""
    parsed_names, _ = _parse_cpp_constants(source=_read_all_cuh())
    registered = set(_CPP_TO_PY.keys())
    extra_in_cpp = parsed_names - registered
    extra_in_registry = registered - parsed_names
    assert not extra_in_cpp, (
        f"kv-canary const sync: cuh has constexpr(s) not in _CPP_TO_PY: {sorted(extra_in_cpp)}. "
        "Add them to the registry (with a Python target or None for kernel-only)."
    )
    assert not extra_in_registry, (
        f"kv-canary const sync: _CPP_TO_PY references constexpr(s) not present in any cuh: "
        f"{sorted(extra_in_registry)}. Remove the stale entries."
    )


def test_registered_constants_value_match() -> None:
    _, decoded = _parse_cpp_constants(source=_read_all_cuh())
    for cpp_name, target in _CPP_TO_PY.items():
        if target is None:
            continue
        assert (
            cpp_name in decoded
        ), f"kv-canary const sync: {cpp_name} could not be decoded from cuh (unrecognized literal form?)"
        py_value = _resolve_py(target)
        assert decoded[cpp_name] == py_value, (
            f"kv-canary const sync: {cpp_name} = {decoded[cpp_name]} in cuh but "
            f"{target[0]}.{target[1]} = {py_value} in Python"
        )


def test_every_pinned_python_constant_is_registered() -> None:
    """Catches drift the other way: someone adds a Python _FIELD_FOO without a matching kCanaryFieldFoo."""
    registered_py: set[tuple[str, str]] = {
        target for target in _CPP_TO_PY.values() if target is not None
    }
    for module_path, prefixes in _PINNED_PY_PREFIXES.items():
        module = importlib.import_module(module_path)
        for attr_name in dir(module):
            if not any(attr_name.startswith(p) for p in prefixes):
                continue
            value = getattr(module, attr_name)
            if not isinstance(value, int) or isinstance(value, bool):
                continue
            assert (module_path, attr_name) in registered_py, (
                f"kv-canary const sync: Python constant {module_path}.{attr_name} matches a pinned "
                f"prefix but is not in _CPP_TO_PY. Add a C++ counterpart, or relax the prefix list."
            )


def test_cpp_enum_members_match_python() -> None:
    cuh_source = _read_all_cuh()
    for cpp_enum_name, (module_path, py_enum_name) in _CPP_ENUMS.items():
        cpp_members = _parse_cpp_enum_members(
            source=cuh_source, enum_name=cpp_enum_name
        )
        py_enum = getattr(importlib.import_module(module_path), py_enum_name)
        assert issubclass(
            py_enum, IntEnum
        ), f"kv-canary const sync: {module_path}.{py_enum_name} must be IntEnum to be const-sync-checked"
        cpp_normalized = {
            _camel_to_upper_snake(name): value for name, value in cpp_members.items()
        }
        py_normalized = {member.name: int(member.value) for member in py_enum}
        assert cpp_normalized == py_normalized, (
            f"kv-canary const sync: enum {cpp_enum_name} mismatch.\n"
            f"  C++ (normalized): {sorted(cpp_normalized.items())}\n"
            f"  Python:           {sorted(py_normalized.items())}"
        )


def test_canary_slot_bytes_derivation() -> None:
    """``CANARY_SLOT_BYTES`` is derived (= _CANARY_FIELDS_PER_SLOT * 8). Pin the relation."""
    from sglang.jit_kernel.kv_canary.verify import (
        _CANARY_FIELDS_PER_SLOT,
        CANARY_SLOT_BYTES,
    )

    assert CANARY_SLOT_BYTES == _CANARY_FIELDS_PER_SLOT * 8
