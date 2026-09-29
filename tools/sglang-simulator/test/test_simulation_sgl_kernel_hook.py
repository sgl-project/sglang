"""Regression tests for the sgl_kernel CPU-op import stand-ins.

Pre-fix behavior (sglang issue #41653): with SGLANG_USE_CPU_ENGINE=1, sglang
model modules (qwen3_5.py, muse_glimmer.py, and every module importing them)
resolve torch.ops.sgl_kernel.*_cpu at import time; the CUDA sgl_kernel wheel
does not register those ops, the import raises AttributeError, and
ModelRegistry silently drops the module from get_supported_archs().
"""

import pytest
from sglang_simulator.simulation.sglang.sgl_kernel_hook import (
    _CPU_OP_STANDIN_NAMES,
    install_cpu_op_standins,
)


def _remove_installed_standins():
    """Delattr only the stand-ins this module installed, never a real op."""
    import torch

    namespace = torch.ops.sgl_kernel
    for op_name in _CPU_OP_STANDIN_NAMES:
        standin = getattr(namespace, op_name, None)
        tagged = getattr(standin, "_sglang_simulator_standin", False)
        if standin is not None and tagged:
            delattr(namespace, op_name)


def test_install_is_a_noop_without_cpu_engine(monkeypatch):
    """Without SGLANG_USE_CPU_ENGINE=1 the hook must leave torch.ops.sgl_kernel
    untouched, so real CPU-engine serving never gets raising stand-ins."""
    import torch

    _remove_installed_standins()
    pre = {
        op_name: hasattr(torch.ops.sgl_kernel, op_name)
        for op_name in _CPU_OP_STANDIN_NAMES
    }
    monkeypatch.delenv("SGLANG_USE_CPU_ENGINE", raising=False)

    install_cpu_op_standins()

    for op_name, existed in pre.items():
        if existed:
            # A host-provided op registration must be left exactly as-is.
            assert hasattr(torch.ops.sgl_kernel, op_name)
            continue
        with pytest.raises(AttributeError):
            getattr(torch.ops.sgl_kernel, op_name)


def test_standins_resolve_import_time_names_and_raise_when_called(monkeypatch):
    """Under SGLANG_USE_CPU_ENGINE=1 the simulator must let torch.ops.sgl_kernel
    resolve the four CPU ops that qwen3_5.py and muse_glimmer.py bind at import
    time even though the CUDA wheel lacks them; calling one must still fail
    loudly instead of computing anything."""
    import torch

    namespace = torch.ops.sgl_kernel
    _remove_installed_standins()
    pre = {op_name: hasattr(namespace, op_name) for op_name in _CPU_OP_STANDIN_NAMES}
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")

    try:
        install_cpu_op_standins()
        for op_name, existed in pre.items():
            op = getattr(namespace, op_name)
            if existed:
                # A pre-existing real op must not have been replaced.
                continue
            with pytest.raises(RuntimeError, match="import-time stand-in"):
                op()
    finally:
        _remove_installed_standins()


def test_install_never_shadows_an_existing_attribute(monkeypatch):
    """install_cpu_op_standins must skip names that already resolve, so a real
    op or cached packet is never replaced, and repeated installs stay
    idempotent."""
    import torch

    namespace = torch.ops.sgl_kernel
    _remove_installed_standins()
    sentinel = object()
    monkeypatch.setenv("SGLANG_USE_CPU_ENGINE", "1")
    setattr(namespace, "fused_sigmoid_mul_cpu", sentinel)

    try:
        install_cpu_op_standins()
        install_cpu_op_standins()
        assert getattr(namespace, "fused_sigmoid_mul_cpu") is sentinel
    finally:
        delattr(namespace, "fused_sigmoid_mul_cpu")
        _remove_installed_standins()
