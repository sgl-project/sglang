from sglang.kernels.jit.utils import arch
from sglang.kernels.jit.utils.compile import toolchain
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_musa_arch_flags(monkeypatch):
    monkeypatch.setattr(arch, "is_musa_runtime", lambda: True)
    compile_arch = arch.CompileArch(3, 1, "")
    assert arch.get_default_target_flags(compile_arch) == [
        "--offload-arch=mp_31",
        "-DSGL_CUDA_ARCH=310",
        "-std=c++20",
        "-O3",
    ]


def test_musa_toolchain_paths_and_flags(monkeypatch):
    monkeypatch.setattr(toolchain, "is_musa_runtime", lambda: True)
    monkeypatch.setattr(toolchain, "is_sail_runtime", lambda: False)
    monkeypatch.setattr(toolchain, "musa_home", lambda: "/opt/musa")
    monkeypatch.setattr(
        toolchain, "tvm_ffi_paths", lambda: (("/tvm/include",), "/tvm/lib", "tvm")
    )

    assert toolchain.device_compiler_path() == "/opt/musa/bin/mcc"
    assert toolchain.base_cuda_flags() == ["-x", "musa", "-fPIC"]
    assert toolchain.base_include_paths() == [
        "/tvm/include",
        "/opt/musa/include",
    ]
    assert toolchain.base_link_flags(with_device=True)[-2:] == [
        "-L/opt/musa/lib",
        "-lmusart",
    ]
