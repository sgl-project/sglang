from pathlib import Path

from sglang.kernels.jit.utils import arch
from sglang.kernels.jit.utils.compile import cache, toolchain
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_musa_arch_flags(monkeypatch):
    monkeypatch.setattr(arch, "is_musa_runtime", lambda: True)
    assert arch.get_default_target_flags() == ["-DUSE_MUSA", "-std=c++20", "-O3"]


def test_musa_runtime_alias_owner_is_torchada():
    include_dir = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/kernels/jit/include/sgl_kernel"
    )
    headers = {
        path.name: path.read_text()
        for path in (
            include_dir / "runtime.cuh",
            include_dir / "tensor.h",
            include_dir / "utils.cuh",
            include_dir / "utils.h",
        )
    }
    assert (
        "#if defined(__MUSACC__)\n#include <cuda_runtime.h>"
        not in headers["runtime.cuh"]
    )
    assert "__MUSACC__" not in headers["runtime.cuh"]
    assert "__MUSACC__" in headers["tensor.h"]
    assert "__MUSACC__" in headers["utils.h"]
    assert "USE_MUSA" not in headers["tensor.h"]
    assert "USE_MUSA" not in headers["utils.h"]
    pdl = (
        headers["utils.cuh"]
        .split("auto enable_pdl", 1)[1]
        .split("auto enable_cluster", 1)[0]
    )
    assert "m_config.numAttrs = 0" not in pdl
    for header in ("cuda_bf16.h", "cuda_fp16.h", "cuda_fp8.h", "cuda_runtime.h"):
        assert f"#include <{header}>" in headers["utils.cuh"]
    for source in headers.values():
        for symbol in (
            "musaDeviceGetAttribute",
            "musaLaunchKernelEx",
            "musaFuncSetAttribute",
            "musaRuntimeGetVersion",
            "__mt_bfloat16",
            "__mt_fp8",
        ):
            assert symbol not in source


def test_musa_toolchain_paths_and_flags(monkeypatch):
    monkeypatch.setattr(toolchain, "is_musa_runtime", lambda: True)
    monkeypatch.setattr(toolchain, "is_sail_runtime", lambda: False)
    monkeypatch.setattr(
        toolchain, "get_compile_arch", lambda: arch.CompileArch(3, 1, "")
    )
    monkeypatch.delenv("MTGPU_TARGET", raising=False)
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


def test_musa_explicit_arch_wins_over_environment_and_cache_tag(monkeypatch):
    explicit = arch.CompileArch(4, 0, "a")
    monkeypatch.setenv("MTGPU_TARGET", "mp_31")
    monkeypatch.setattr(toolchain, "is_musa_runtime", lambda: True)
    monkeypatch.setattr(cache, "is_musa_runtime", lambda: True)
    monkeypatch.setattr(cache, "is_hip_runtime", lambda: False)
    monkeypatch.setattr(cache, "is_sail_runtime", lambda: False)

    assert toolchain.gpu_arch_name.__wrapped__(explicit) == "mp_40a"
    assert toolchain.gpu_arch_name.__wrapped__() == "mp_31"
    assert toolchain.target_flags(explicit) == ["--offload-arch=mp_40a"]
    assert toolchain.target_flags() == ["--offload-arch=mp_31"]
    assert cache._target_tag.__wrapped__(explicit) == "mp_40a"
    assert cache._target_tag.__wrapped__() == "mp_31"


def test_musa_checks_preserve_existing_platform_precedence(monkeypatch):
    monkeypatch.setattr(toolchain, "is_sail_runtime", lambda: False)
    monkeypatch.setattr(toolchain, "is_hip_runtime", lambda: True)
    monkeypatch.setattr(toolchain, "is_musa_runtime", lambda: True)
    monkeypatch.setattr(toolchain, "rocm_home", lambda: "/opt/rocm")

    assert toolchain.device_compiler_path.__wrapped__() == "/opt/rocm/bin/hipcc"
    assert toolchain.base_cuda_flags() == [
        "-fPIC",
        "-D__HIP_PLATFORM_AMD__=1",
        "-fno-gpu-rdc",
    ]
