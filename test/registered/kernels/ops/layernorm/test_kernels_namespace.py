"""GPU-free import / registry / selector tests for ``sglang.kernels`` (RFC #29630)."""

import importlib
import subprocess
import sys

import pytest

import sglang.kernels as K
import sglang.kernels.fused_op as fo
import sglang.kernels.selector as sel
from sglang.kernels import KernelBackend, PlatformInfo
from sglang.kernels.spec import CapabilityRequirement as Cap
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_CPU = PlatformInfo(device_type="cpu")
_SM90 = PlatformInfo(device_type="cuda", cuda_arch_major=9, cuda_arch_minor=0)
_SM100 = PlatformInfo(device_type="cuda", cuda_arch_major=10, cuda_arch_minor=0)
_HIP = PlatformInfo(device_type="hip")


def test_single_backend_resolves_without_backend():
    assert (
        K.select_kernel("kvcache.reshape_and_cache_flash").backend
        is KernelBackend.TRITON
    )


def test_fp8_scaled_mm_requires_explicit_registry_backend(monkeypatch):
    monkeypatch.setattr(sel, "_platform", lambda: _SM90)
    with pytest.raises(ValueError, match="multiple backends"):
        K.select_kernel("gemm.fp8_scaled_mm")
    assert (
        K.select_kernel("gemm.fp8_scaled_mm", backend=KernelBackend.AOT).backend
        is KernelBackend.AOT
    )


def test_unknown_op_or_backend_raises():
    with pytest.raises(KeyError):
        K.select_kernel("does_not.exist")
    with pytest.raises(KeyError):
        K.select_kernel("gemm.fp8_scaled_mm", backend=KernelBackend.TRITON)


def test_multi_backend_requires_explicit_backend(monkeypatch):
    # Device is a hard eligibility filter, not a ranking: >1 usable backend on
    # the current device means selection must name one.
    monkeypatch.setattr(sel, "_platform", lambda: _SM90)
    with pytest.raises(ValueError):
        K.select_kernel("layernorm.rmsnorm")
    spec = K.select_kernel("layernorm.rmsnorm", backend=KernelBackend.JIT)
    assert spec.backend is KernelBackend.JIT
    assert spec.target == "sglang.kernels.ops.layernorm:_RMSNORM.forward_jit"


@pytest.mark.parametrize("device, expect", [("cuda", "jit"), ("hip", "aot")])
def test_activation_default_backend(monkeypatch, device, expect):
    # silu_and_mul default matches production: jit on CUDA, aot (sgl_kernel) on HIP.
    from sglang.kernels.ops.activation import _SILU_AND_MUL

    monkeypatch.setattr(fo, "_platform", lambda: PlatformInfo(device_type=device))
    assert _SILU_AND_MUL.auto_selected_backend().value == expect


@pytest.mark.parametrize(
    "op_attr, device, expect",
    [
        ("_RMSNORM", "cuda", "aot"),
        ("_RMSNORM", "hip", "aiter"),
        ("_RMSNORM", "npu", "torch_npu"),
        ("_GEMMA_RMSNORM", "cuda", "aot"),
        ("_GEMMA_RMSNORM", "hip", "jit"),  # rocm-triton JIT pinned to HIP
        ("_GEMMA_RMSNORM", "npu", "torch_npu"),
    ],
)
def test_layernorm_default_backend(monkeypatch, op_attr, device, expect):
    # Same AOT provenance, different device coverage per op: rmsnorm's AOT is
    # CUDA-only, so HIP falls to aiter and NPU to torch_npu.
    ln = importlib.import_module("sglang.kernels.ops.layernorm")
    monkeypatch.setattr(fo, "_platform", lambda: PlatformInfo(device_type=device))
    assert getattr(ln, op_attr).auto_selected_backend().value == expect


@pytest.mark.parametrize(
    "req, plat, ok",
    [
        (Cap.CUDA, _CPU, False),
        (Cap.CUDA, _SM90, True),
        (Cap.CUDA, _HIP, False),
        (Cap.HIP, _HIP, True),
        (Cap.cuda(min_sm=(10, 0)), _SM90, False),
        (Cap.cuda(min_sm=(10, 0)), _SM100, True),
        (Cap.cuda(max_sm=(9, 0)), _SM100, False),
    ],
)
def test_capability_is_satisfied_by(req, plat, ok):
    assert req.is_satisfied_by(plat) is ok


def test_capabilities_or_semantics():
    both = {Cap.CUDA, Cap.HIP}
    assert K.capabilities_satisfied(both, _SM90)
    assert K.capabilities_satisfied(both, _HIP)
    assert not K.capabilities_satisfied(both, _CPU)
    assert K.capabilities_satisfied((), _CPU)  # empty = unrestricted
    assert K.capabilities_satisfied(Cap.CUDA, _SM90)  # single tolerated


@pytest.mark.parametrize(
    "relative_path",
    (
        "srt/utils/common.py",
        "multimodal_gen/runtime/utils/common.py",
    ),
)
def test_amx_backend_probe_is_lazy(relative_path):
    loader = (
        "package = importlib.util.find_spec('sglang'); "
        "path = pathlib.Path(next(iter(package.submodule_search_locations))) / "
        f"{relative_path!r}; "
        "spec = importlib.util.spec_from_file_location('_common_under_test', path); "
        "module = importlib.util.module_from_spec(spec); "
        "sys.modules[spec.name] = module; "
        "spec.loader.exec_module(module)"
    )
    code = "; ".join(
        (
            "import builtins, importlib.util, pathlib, sys",
            "from unittest import mock",
            "real_import = builtins.__import__",
            "import_mock = mock.Mock(wraps=real_import)",
            "builtins.__import__ = import_mock",
            loader,
            "builtins.__import__ = real_import",
            "attempted = any(call.args and call.args[0] == 'sgl_kernel' "
            "for call in import_mock.call_args_list)",
            "print('DIRTY' if attempted else 'CLEAN')",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_import_stays_metadata_only():
    # Importing the namespace must not pull in the AOT backend (sgl_kernel) or
    # the JIT compilation infra (sglang.kernels.jit), which import torch / nvcc.
    code = (
        "import sys, sglang.kernels.ops; "
        "print('DIRTY' if 'sgl_kernel' in sys.modules or any("
        "m.startswith('sglang.kernels.jit') for m in sys.modules) else 'CLEAN')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "CLEAN" in r.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
