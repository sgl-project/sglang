"""Architecture gates of kernel entry points, on faked architectures (no GPU)."""

import contextlib
import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch

import sglang.kernels.arch as arch_module
from sglang.kernels.arch._cuda import CudaArch
from sglang.kernels.arch._hip import HipArch
from sglang.kernels.arch.base import DeviceArch
from sglang.kernels.ops.attention.fla import layernorm_gated
from sglang.kernels.ops.attention.linear import fused_gdn_gating as gdn_gating
from sglang.kernels.ops.gemm import bf16_fp32, hopper_bf16_gemv
from sglang.kernels.ops.mamba import gdn_extend_state_copies
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


class _LaunchSpy:
    """Stands in for a Triton kernel and records each launch's grid and arguments."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return launch


@pytest.fixture
def gated_norm_launches(monkeypatch):
    spy = _LaunchSpy()
    monkeypatch.setattr(layernorm_gated, "_layer_norm_fwd_1pass_kernel", spy)
    monkeypatch.setattr(layernorm_gated, "_get_sm_count", lambda device: 148)
    monkeypatch.setattr(layernorm_gated, "is_arch_support_pdl", lambda: False)
    monkeypatch.setattr(
        layernorm_gated, "device_context", lambda device: contextlib.nullcontext()
    )
    return spy.launches


_TILE_ROWS = layernorm_gated.PREFILL_TILE_MIN_ROWS


@pytest.mark.parametrize(
    "arch,rows,dim,quant_heads,expected",
    [
        ("sm100", _TILE_ROWS, 128, 0, (16, 4, True)),
        ("sm120", _TILE_ROWS, 128, 0, (16, 4, True)),
        ("sm100", _TILE_ROWS - 64, 128, 0, (4, 1, False)),
        ("sm100", _TILE_ROWS, 128, 64, (64, 16, False)),
        ("sm100", _TILE_ROWS, 512, 0, (4, 2, False)),
        ("sm90", _TILE_ROWS, 128, 0, (4, 1, False)),
        ("gfx950", _TILE_ROWS, 128, 0, (4, 1, False)),
        ("cpu", _TILE_ROWS, 128, 0, (4, 1, False)),
    ],
)
def test_gated_norm_prefill_tiles_need_sm100(
    fake_arch, gated_norm_launches, arch, rows, dim, quant_heads, expected
):
    """Only SM100+ launches of at least PREFILL_TILE_MIN_ROWS one-warp 4-row tiles
    take several tiles a program; HIP, SM90, small and other launches keep stock."""
    fake_arch(arch)
    x = torch.empty((rows, dim), dtype=torch.bfloat16)
    w = torch.ones(dim, dtype=torch.bfloat16)
    layernorm_gated._layer_norm_fwd(
        x,
        w,
        None,
        1e-6,
        z=torch.empty_like(x),
        is_rms_norm=True,
        quant_heads=quant_heads,
    )
    ((grid, _, kwargs),) = gated_norm_launches
    rows_per_block, num_warps, early_z = expected
    assert kwargs["ROWS_PER_BLOCK"] == rows_per_block
    assert kwargs["num_warps"] == num_warps
    assert kwargs.get("EARLY_Z", False) is early_z
    assert grid == (-(-rows // rows_per_block), 1)


@pytest.mark.parametrize(
    "arch,tokens,heads,expected",
    [
        ("sm100", 1024, 64, 64),
        ("sm100", 4096, 48, 64),
        ("sm100", 1023, 64, 8),
        ("sm100", 4096, 8, 8),
        ("sm100", 4096, 4, 8),
        ("sm90", 4096, 64, 8),
        ("gfx950", 4096, 64, 8),
        ("cpu", 4096, 64, 8),
    ],
)
def test_gdn_gating_one_program_needs_sm100_prefill_sizes(
    fake_arch, monkeypatch, arch, tokens, heads, expected
):
    """All heads of a token go to one program only on SM100+ at prefill token counts;
    decode-sized batches, HIP and SM90 keep 8-head programs."""
    fake_arch(arch)
    spy = _LaunchSpy()
    monkeypatch.setattr(gdn_gating, "fused_gdn_gating_kernel", spy)
    a = torch.empty((tokens, heads), dtype=torch.bfloat16)
    params = torch.empty(heads)
    gdn_gating.fused_gdn_gating(params, a, torch.empty_like(a), params)
    ((grid, args, _),) = spy.launches
    assert args[-1] == expected
    assert grid == (tokens, 1, -(-heads // expected))


def _state_copy_args(dtype=torch.bfloat16):
    pool = torch.empty((8, 4, 16, 16), dtype=dtype)
    indices = torch.tensor([3, 0, 5])
    h = torch.empty((1, 6, 4, 16, 16), dtype=dtype)
    conv = torch.empty((8, 32, 3), dtype=dtype)
    x = torch.empty((40, 32), dtype=dtype)
    tokens = torch.zeros((2, 3), dtype=torch.int64)
    return pool, indices, h, conv, x, tokens


@pytest.fixture
def cuda_tensors(monkeypatch):
    """Lets the CPU tensors of these tests stand for CUDA tensors."""
    monkeypatch.setattr(gdn_extend_state_copies, "_on_cuda", lambda *tensors: True)


def _track_ok(*, pool, h, h_rows=6, snapshot=False, recompute_rows=0):
    return gdn_extend_state_copies.track_state_rows_ok(
        states=pool,
        h=h,
        h_rows=h_rows,
        snapshot=snapshot,
        recompute_rows=recompute_rows,
    )


@pytest.mark.parametrize(
    "arch,on_cuda,expected",
    [
        ("sm100", True, True),
        ("sm90", True, False),
        ("gfx950", True, False),
        ("cpu", True, False),
        ("sm100", False, False),
    ],
)
def test_gdn_extend_state_copies_need_sm100_and_cuda_tensors(
    fake_arch, monkeypatch, arch, on_cuda, expected
):
    """The eager GDN extend takes the one-launch state copies only on SM100+, and
    only for CUDA tensors: the copy kernels are Triton."""
    fake_arch(arch)
    if on_cuda:
        monkeypatch.setattr(gdn_extend_state_copies, "_on_cuda", lambda *t: True)
    sc = gdn_extend_state_copies
    pool, indices, h, conv, x, tokens = _state_copy_args()
    updates = torch.empty((3, 4, 16, 16), dtype=pool.dtype)
    assert sc.gather_pool_rows_ok(states=pool, indices=indices, dtype=pool.dtype) is (
        expected
    )
    assert sc.scatter_pool_rows_ok(states=pool, indices=indices, updates=updates) is (
        expected
    )
    assert _track_ok(pool=pool, h=h) is expected
    assert _track_ok(pool=pool, h=None, h_rows=0) is expected
    assert sc.track_conv_rows_ok(conv=conv, x=x, tokens=tokens) is expected


def test_gdn_extend_state_copies_refuse_what_the_index_ops_do_differently(
    fake_arch, cuda_tensors
):
    """A dtype change, a layout the kernel does not address, snapshot or recomputed
    track rows keep the index ops."""
    fake_arch("sm100")
    sc = gdn_extend_state_copies
    pool, indices, h, conv, x, tokens = _state_copy_args()
    fp32 = torch.float32
    assert not sc.gather_pool_rows_ok(states=pool, indices=indices, dtype=fp32)
    assert not sc.gather_pool_rows_ok(
        states=pool.transpose(2, 3), indices=indices, dtype=pool.dtype
    )
    assert not sc.scatter_pool_rows_ok(
        states=pool, indices=indices, updates=torch.empty((3, 4, 16, 16))
    )
    assert not sc.scatter_pool_rows_ok(
        states=pool, indices=indices, updates=torch.empty((2, 4, 16, 16)).to(pool)
    )
    assert not _track_ok(pool=pool, h=h.to(fp32))
    assert not _track_ok(pool=pool, h=h[..., :8])
    # Same element count per row, other shape: the index ops would raise.
    assert not _track_ok(pool=pool, h=h.reshape(1, 6, 8, 8, 16))
    assert not _track_ok(pool=pool, h=None)
    assert not _track_ok(pool=pool, h=h, snapshot=True)
    assert not _track_ok(pool=pool, h=h, recompute_rows=2)
    assert not sc.track_conv_rows_ok(conv=conv, x=x.to(fp32), tokens=tokens)
    assert not sc.track_conv_rows_ok(conv=conv, x=x, tokens=tokens[:, :2])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
