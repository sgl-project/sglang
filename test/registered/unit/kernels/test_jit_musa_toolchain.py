from pathlib import Path

from sglang.kernels.jit.utils import arch
from sglang.kernels.jit.utils.compile import toolchain
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_musa_arch_flags(monkeypatch):
    monkeypatch.setattr(arch, "is_musa_runtime", lambda: True)
    assert arch.get_default_target_flags() == ["-DUSE_MUSA", "-std=c++20", "-O3"]


def test_musa_home_uses_compiler_override(monkeypatch):
    monkeypatch.delenv("MUSA_HOME", raising=False)
    monkeypatch.delenv("MUSA_PATH", raising=False)
    monkeypatch.setenv("SGLANG_MUSA_MCC", "/opt/musa/bin/mcc")
    assert toolchain.musa_home.__wrapped__() == "/opt/musa"


def test_musa_runtime_alias_owner_is_torchada():
    utils = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/kernels/jit/include/sgl_kernel/utils.cuh"
    ).read_text()
    musa_block = utils.split("#if defined(__MUSACC__)", 1)[1].split(
        "#elif !defined(USE_ROCM)", 1
    )[0]
    assert "#include <cuda.h>" not in musa_block
    for header in ("cuda_bf16.h", "cuda_fp16.h", "cuda_fp8.h", "cuda_runtime.h"):
        assert f"#include <{header}>" in musa_block
    for symbol in (
        "cudaFuncAttributeMaxDynamicSharedMemorySize",
        "cudaLaunchKernelEx",
        "cudaRuntimeGetVersion",
        "cuGetErrorString",
    ):
        assert f"#define {symbol}" not in musa_block


def test_musa_runtime_aliases_stay_on_torchada():
    runtime = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/kernels/jit/include/sgl_kernel/runtime.cuh"
    ).read_text()
    for symbol in (
        "musaDevAttrComputeCapabilityMajor",
        "musaDevAttrComputeCapabilityMinor",
        "musaDevAttrMaxSharedMemoryPerBlockOptin",
        "musaFuncAttributePreferredSharedMemoryCarveout",
        "musaRuntimeGetVersion",
        "musaSharedmemCarveoutDefault",
        "musaSharedmemCarveoutMaxL1",
        "musaSharedmemCarveoutMaxShared",
    ):
        assert symbol not in runtime


def test_musa_toolchain_paths_and_flags(monkeypatch):
    monkeypatch.setattr(toolchain, "is_musa_runtime", lambda: True)
    monkeypatch.setattr(toolchain, "is_sail_runtime", lambda: False)
    monkeypatch.setattr(
        toolchain, "get_compile_arch", lambda: arch.CompileArch(3, 1, "")
    )
    monkeypatch.setattr(toolchain, "musa_home", lambda: "/opt/musa")
    monkeypatch.setattr(
        toolchain,
        "torchada_cuda_compat_include_dir",
        lambda: "/torchada/cuda_compat",
    )
    monkeypatch.setattr(
        toolchain, "tvm_ffi_paths", lambda: (("/tvm/include",), "/tvm/lib", "tvm")
    )

    assert toolchain.device_compiler_path.__wrapped__() == "/opt/musa/bin/mcc"
    assert toolchain.gpu_arch_name() == "mp_31"
    assert toolchain.target_flags() == ["--offload-arch=mp_31"]
    assert toolchain.base_cuda_flags() == ["-fPIC", "-x", "musa"]
    assert toolchain.base_include_paths() == [
        "/torchada/cuda_compat",
        "/tvm/include",
        "/opt/musa/include",
    ]
    assert toolchain.base_link_flags(with_device=True)[-2:] == [
        "-L/opt/musa/lib",
        "-lmusart",
    ]
