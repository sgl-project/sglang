from __future__ import annotations

import re
from pathlib import Path

import sglang.jit_kernel
from sglang.kernels.ops.kv_canary import consts
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=5, stage="jit-kernel-unit", runner_config="amd")


# Resolve the kernel source against the installed jit_kernel package rather
# than this file's location, so the test stays correct wherever it lives.
_CONSTS_CUH: Path = (
    Path(sglang.jit_kernel.__file__).resolve().parent
    / "csrc"
    / "kv_canary"
    / "consts.cuh"
)


def _camel_to_upper_snake(name: str) -> str:
    return re.sub(r"([A-Z])", r"_\1", name).lstrip("_").upper()


def _decode(expr: str) -> int:
    expr = expr.strip().rstrip("UuLl")
    if "<<" in expr:
        return 1 << int(expr.split("<<")[1].strip())
    return int(expr, 0)


def _parse_constexpr_ints(source: str) -> dict[str, int]:
    pattern = re.compile(r"constexpr\s+(?:[\w:]+)\s+(k[A-Za-z]\w*)\s*=\s*([^;]+);")
    return {name: _decode(rhs) for name, rhs in pattern.findall(source)}


def _parse_enum_class(source: str, enum_name: str) -> dict[str, int]:
    pattern = re.compile(
        r"enum\s+class\s+" + re.escape(enum_name) + r"\s*:\s*[^\{]+\{([^}]+)\}"
    )
    body = pattern.search(source).group(1)
    member_re = re.compile(r"(k[A-Za-z]\w*)\s*=\s*([^,]+)")
    return {name: _decode(rhs) for name, rhs in member_re.findall(body)}


def test_int_consts_sync() -> None:
    cpp = _parse_constexpr_ints(_CONSTS_CUH.read_text(encoding="utf-8"))
    cpp_normalized = {_camel_to_upper_snake(n[1:]): v for n, v in cpp.items()}
    py = {
        n: v
        for n, v in vars(consts).items()
        if isinstance(v, int) and not isinstance(v, bool) and not n.startswith("_")
    }
    assert cpp_normalized == py


def test_enums_sync() -> None:
    cuh = _CONSTS_CUH.read_text(encoding="utf-8")
    for enum_name in ("RealKvHashMode", "FailReason"):
        cpp_members = _parse_enum_class(cuh, enum_name)
        py_enum = getattr(consts, enum_name)
        cpp_normalized = {
            _camel_to_upper_snake(n[1:]): v for n, v in cpp_members.items()
        }
        py_normalized = {m.name: int(m.value) for m in py_enum}
        assert cpp_normalized == py_normalized


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
