"""GPU-free import / registry / selector tests for ``sglang.kernels`` (RFC #29630)."""

import importlib
import subprocess
import sys

import pytest

import sglang.kernels as K
import sglang.kernels.fused_op as fo
import sglang.kernels.ops  # noqa: F401  -- populate the registry
import sglang.kernels.selector as sel
from sglang.kernels import DeviceType, KernelBackend, PlatformInfo
from sglang.kernels.spec import CapabilityRequirement as Cap
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GROUPS = [
    "activation",
    "attention",
    "communication",
    "diffusion",
    "elementwise",
    "embeddings",
    "gemm",
    "grammar",
    "kvcache",
    "layernorm",
    "mamba",
    "memory",
    "moe",
    "quantization",
    "sampling",
    "speculative",
]

# Representative ops checked as a subset (the registry holds many more).
EXPECTED = {
    "activation.silu_and_mul": {"aot", "jit", "aiter", "torch", "torch_compile"},
    "activation.relu2": {"jit", "torch", "torch_compile"},
    "layernorm.rmsnorm": {"aot", "jit", "aiter", "torch_npu", "torch", "torch_compile"},
    "layernorm.gemma_rmsnorm": {"aot", "jit", "torch_npu", "torch", "torch_compile"},
    "gemm.fp8_scaled_mm": {"aot"},
    "moe.moe_align_block_size": {"aot", "jit"},
    "quantization.nvfp4_gemm_swiglu_nvfp4_quant": {"cute_dsl"},
    "kvcache.reshape_and_cache_flash": {"triton"},
}

_CPU = PlatformInfo(device_type="cpu")
_SM90 = PlatformInfo(device_type="cuda", cuda_arch_major=9, cuda_arch_minor=0)
_SM100 = PlatformInfo(device_type="cuda", cuda_arch_major=10, cuda_arch_minor=0)
_HIP = PlatformInfo(device_type="hip")


def test_top_level_exports():
    for name in (
        "KernelSpec",
        "KernelBackend",
        "FormatSignature",
        "CapabilityRequirement",
        "PlatformInfo",
        "registry",
        "get_kernel",
        "select_kernel",
    ):
        assert hasattr(K, name), name


@pytest.mark.parametrize("group", GROUPS)
def test_group_importable(group):
    assert hasattr(importlib.import_module(f"sglang.kernels.ops.{group}"), "__all__")


@pytest.mark.parametrize("op, backends", list(EXPECTED.items()))
def test_registry_backends(op, backends):
    assert {s.backend.value for s in K.registry.get(op)} == backends


def test_specs_well_formed():
    for spec in K.registry.all_specs():
        assert spec.op == f"{spec.group}.{spec.name}"
        mod, sep, attr = spec.target.partition(":")
        assert sep == ":" and mod and attr, spec.target


def test_single_backend_resolves_without_backend():
    assert K.select_kernel("gemm.fp8_scaled_mm").backend is KernelBackend.AOT


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
    assert _SILU_AND_MUL._resolve_backend().value == expect


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
    assert getattr(ln, op_attr)._resolve_backend().value == expect


def test_per_op_backend_subset():
    # silu_and_mul ships an aiter (HIP) kernel; the gelu siblings deliberately
    # do not -- ROCm coverage is a per-(op, backend) subset.
    from sglang.kernels.ops.activation import _GELU_AND_MUL, _SILU_AND_MUL

    assert KernelBackend.AITER in _SILU_AND_MUL.available_backends()
    assert KernelBackend.AITER not in _GELU_AND_MUL.available_backends()


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


def test_capability_shortcuts():
    assert Cap.CUDA == Cap(device=DeviceType.CUDA)
    assert Cap.HIP == Cap(device=DeviceType.HIP)
    assert Cap.NPU == Cap(device=DeviceType.NPU)
    assert {Cap.CUDA, Cap.HIP} == {Cap.HIP, Cap.CUDA}
    assert Cap.cuda(min_sm=(10, 0)) == Cap(
        device=DeviceType.CUDA, min_cuda_arch=(10, 0)
    )


def test_platform_detect_does_not_raise():
    assert PlatformInfo.detect().device_type in ("cpu", "cuda", "hip", "npu")


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
