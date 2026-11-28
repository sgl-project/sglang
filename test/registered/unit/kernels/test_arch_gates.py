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
from sglang.kernels.ops.attention import inkling2_decode_attention_sm90
from sglang.kernels.ops.gemm import bf16_fp32, hopper_bf16_gemv
from sglang.kernels.ops.moe import inkling2_moe_lowm_sm120
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


def _swa_launch_config(batch: int, qlen: int, sms: int):
    # Inkling-2-Mini's SWA layers: 24 query heads over 8 KV heads, 128-token pages,
    # and a 1023-key window ahead of the qlen queries.
    heads_k, group = 8, 3
    return inkling2_decode_attention_sm90._sm12x_swa_launch_config(
        batch=batch,
        heads_k=heads_k,
        qlen=qlen,
        rows=max(16, 1 << (qlen * group - 1).bit_length()),
        head_dim=128,
        page_size=128,
        span=1023 + qlen,
        sms=sms,
        max_splits=max(1, min(2 * sms // (batch * heads_k), 64)),
    )


@pytest.mark.parametrize("arch", ["sm90", "sm100", "sm130", "gfx950", "cpu"])
def test_inkling2_swa_single_wave_tiles_are_family_12_only(fake_arch, arch):
    """The single-wave SWA tiles were sized for SM12x SM counts: SM90, which
    shares the kernel, keeps its own schedule."""
    fake_arch(arch)
    assert _swa_launch_config(batch=1, qlen=1, sms=132) is None


@pytest.mark.parametrize("arch", ["sm120", "sm121"])
@pytest.mark.parametrize("sms", [170, 188])
@pytest.mark.parametrize("qlen", [1, 8])
def test_inkling2_swa_tiles_fit_one_wave_or_fall_back(fake_arch, arch, sms, qlen):
    """A chosen schedule runs every CTA in one wave with at most one tile per
    split; a batch that cannot fit keeps the general schedule."""
    fake_arch(arch)
    chosen = 0
    for batch in range(1, 9):
        config = _swa_launch_config(batch=batch, qlen=qlen, sms=sms)
        if config is None:
            continue
        chosen += 1
        block_n, splits, _ = config
        span = 1023 + qlen
        assert batch * 8 * splits <= sms
        assert 128 % block_n == 0
        # A window may start anywhere in a tile, so it can touch one extra tile.
        assert splits * block_n >= span + block_n - 1
    assert chosen >= 1
    assert _swa_launch_config(batch=8, qlen=qlen, sms=sms) is None


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


@pytest.mark.parametrize(
    "arch,expected",
    [
        ("sm90", False),
        ("sm100", False),
        ("sm120", True),
        ("sm121", True),
        ("sm130", False),
        ("gfx950", False),
        ("cpu", False),
    ],
)
def test_inkling2_low_m_moe_is_family_12_only(fake_arch, arch, expected):
    """The low-M MoE kernels are SM12x-only JIT sources: every other
    architecture has to stay on its general expert path."""
    fake_arch(arch)
    usable = inkling2_moe_lowm_sm120.can_use_lowm_moe(
        tokens=8,
        experts=128,
        top_k=8,
        hidden_size=2304,
        intermediate_size=1152,
        device=torch.device("cuda"),
    )
    assert usable is expected


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
