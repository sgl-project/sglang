"""Architecture gates of kernel entry points, on faked architectures (no GPU)."""

import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch

import sglang.kernels.arch as arch_module
from sglang.kernels.arch._cuda import CudaArch
from sglang.kernels.arch._hip import HipArch
from sglang.kernels.arch.base import DeviceArch
from sglang.kernels.ops.gemm import bf16_fp32, hopper_bf16_gemv
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def _cuda_arch(major: int, minor: int) -> CudaArch:
    arch = CudaArch()
    arch.props = SimpleNamespace(major=major, minor=minor)
    return arch


def _hip_arch(gcn_arch: str) -> HipArch:
    arch = HipArch()
    arch.props = SimpleNamespace(gcnArchName=gcn_arch)
    return arch


# What torch.cuda.get_device_capability() reports next to each architecture;
# ROCm answers too, and gfx950 answers (9, 5).
_ARCHS = {
    "sm90": (_cuda_arch(9, 0), (9, 0)),
    "sm100": (_cuda_arch(10, 0), (10, 0)),
    "sm120": (_cuda_arch(12, 0), (12, 0)),
    "sm121": (_cuda_arch(12, 1), (12, 1)),
    "sm130": (_cuda_arch(13, 0), (13, 0)),
    "gfx950": (_hip_arch("gfx950"), (9, 5)),
    "cpu": (DeviceArch(), None),
}


@pytest.fixture
def fake_arch(monkeypatch):
    def install(name: str) -> None:
        arch, capability = _ARCHS[name]
        monkeypatch.setattr(arch_module, "_DEVICE_ARCH", arch)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: capability is not None)
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda device=None: capability
        )

    return install


@pytest.mark.parametrize(
    "arch,expected",
    [
        ("sm90", True),
        ("sm100", False),
        ("sm120", False),
        ("gfx950", False),
        ("cpu", False),
    ],
)
def test_hpc_bf16xfp32_gemm_needs_nvidia_sm9x(fake_arch, monkeypatch, arch, expected):
    """HPC-Ops ships sm90a kernels only: an installed ``hpc`` package must not
    enable them on gfx950, whose torch capability also has major 9."""
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: object() if name == "hpc" else real_find_spec(name, *args),
    )
    fake_arch(arch)
    bf16_fp32._hpc_gemm_bf16xfp32_available.cache_clear()
    try:
        assert bf16_fp32._hpc_gemm_bf16xfp32_available() is expected
    finally:
        bf16_fp32._hpc_gemm_bf16xfp32_available.cache_clear()


@pytest.mark.parametrize(
    "arch,expected",
    [
        ("sm90", False),
        ("sm120", True),
        ("sm121", True),
        ("sm130", False),
        ("gfx950", False),
    ],
)
def test_hopper_bf16_gemv_sm12x_tuning_is_family_12_only(fake_arch, arch, expected):
    """The SM12x tile and thresholds were tuned on SM120; a later family must
    not inherit them through a ``major >= 12`` test."""
    fake_arch(arch)
    assert hopper_bf16_gemv._is_sm12x() is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
