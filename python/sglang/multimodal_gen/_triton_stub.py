"""Install a lightweight Triton stub when Triton is unavailable."""

from __future__ import annotations

import importlib.util
import sys
import types


def _identity_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def _decorator(fn):
        return fn

    return _decorator


def _unsupported(*args, **kwargs):
    raise RuntimeError(
        "Triton runtime is unavailable on this platform. "
        "Use native fallbacks or choose a non-Triton backend."
    )


def _next_power_of_2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


class _Config:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def install_triton_stub_if_needed() -> None:
    if importlib.util.find_spec("triton") is not None:
        return
    if "triton" in sys.modules:
        return

    triton_mod = types.ModuleType("triton")
    triton_mod.jit = _identity_decorator
    triton_mod.autotune = _identity_decorator
    triton_mod.heuristics = _identity_decorator
    triton_mod.cdiv = lambda a, b: (a + b - 1) // b
    triton_mod.next_power_of_2 = _next_power_of_2
    triton_mod.Config = _Config

    triton_language_mod = types.ModuleType("triton.language")
    triton_language_mod.constexpr = object()
    triton_language_mod.math = types.SimpleNamespace(rsqrt=_unsupported)

    # Common symbols referenced by Triton kernels. They should never execute when
    # native fallbacks are selected, but defining them keeps imports valid.
    for name in [
        "program_id",
        "arange",
        "zeros",
        "full",
        "where",
        "sum",
        "sqrt",
        "maximum",
        "minimum",
        "exp",
        "log",
        "load",
        "store",
        "make_block_ptr",
        "advance",
        "dot",
        "static_range",
        "trans",
        "broadcast_to",
        "num_programs",
        "atomic_add",
        "atomic_cas",
        "atomic_xchg",
    ]:
        setattr(triton_language_mod, name, _unsupported)

    for dtype_name in ["float16", "bfloat16", "float32", "int32", "int64", "uint8"]:
        setattr(triton_language_mod, dtype_name, object())

    triton_testing_mod = types.ModuleType("triton.testing")
    triton_testing_mod.do_bench = _unsupported

    triton_tools_mod = types.ModuleType("triton.tools")
    triton_tools_disasm_mod = types.ModuleType("triton.tools.disasm")
    triton_tools_disasm_mod.extract = _unsupported

    triton_mod.language = triton_language_mod
    triton_mod.testing = triton_testing_mod
    triton_mod.tools = triton_tools_mod

    sys.modules["triton"] = triton_mod
    sys.modules["triton.language"] = triton_language_mod
    sys.modules["triton.testing"] = triton_testing_mod
    sys.modules["triton.tools"] = triton_tools_mod
    sys.modules["triton.tools.disasm"] = triton_tools_disasm_mod
