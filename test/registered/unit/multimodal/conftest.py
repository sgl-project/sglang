"""Test-wide shims for running unit tests in minimal environments.

These shims are only intended for *unit test collection* on platforms where
optional runtime dependencies are unavailable (e.g. Windows lacking `resource`,
or environments without `triton`). CI for SGLang typically runs on Linux with
full deps; these shims help local development without changing production code.
"""

from __future__ import annotations

import os
import sys
import types


def _ensure_resource_stub():
    if "resource" in sys.modules:
        return
    resource_stub = types.ModuleType("resource")
    resource_stub.RLIMIT_NOFILE = 7

    def _getrlimit(_):
        return (0, 0)

    def _setrlimit(_, __):
        return None

    resource_stub.getrlimit = _getrlimit
    resource_stub.setrlimit = _setrlimit
    sys.modules["resource"] = resource_stub


def _ensure_triton_stub():
    if "triton" in sys.modules:
        return

    triton_stub = types.ModuleType("triton")
    triton_stub.__dict__["__version__"] = "0.0.0-test-stub"

    # Minimal `triton.language` surface used by torchdynamo checks.
    tl_stub = types.ModuleType("triton.language")

    class _StubDType:
        pass

    tl_stub.dtype = _StubDType
    triton_stub.language = tl_stub

    # Minimal `triton.backends.compiler` tree expected by torch internals.
    backends_stub = types.ModuleType("triton.backends")
    compiler_stub = types.ModuleType("triton.backends.compiler")
    backends_stub.compiler = compiler_stub
    triton_stub.backends = backends_stub

    # Some torch paths import `triton.compiler.compiler`.
    compiler_pkg_stub = types.ModuleType("triton.compiler")
    compiler_leaf_stub = types.ModuleType("triton.compiler.compiler")
    compiler_pkg_stub.compiler = compiler_leaf_stub
    triton_stub.compiler = compiler_pkg_stub

    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = tl_stub
    sys.modules["triton.backends"] = backends_stub
    sys.modules["triton.backends.compiler"] = compiler_stub
    sys.modules["triton.compiler"] = compiler_pkg_stub
    sys.modules["triton.compiler.compiler"] = compiler_leaf_stub


if os.name == "nt":
    _ensure_resource_stub()
    _ensure_triton_stub()

