"""CPU-only tests for how the JIT toolchain finds libcudart under a CUDA root."""

import pathlib

from sglang.kernels.jit.utils.compile import toolchain
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _touch(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_system_toolkit_uses_lib64(tmp_path):
    _touch(tmp_path / "lib64" / "libcudart.so")
    assert toolchain.cuda_lib_dir(str(tmp_path)) == str(tmp_path / "lib64")


def test_pip_wheel_uses_lib(tmp_path):
    _touch(tmp_path / "lib" / "libcudart.so.13")
    assert toolchain.cuda_lib_dir(str(tmp_path)) == str(tmp_path / "lib")


def test_lib64_preferred_when_both_have_cudart(tmp_path):
    _touch(tmp_path / "lib64" / "libcudart.so")
    _touch(tmp_path / "lib" / "libcudart.so.13")
    assert toolchain.cuda_lib_dir(str(tmp_path)) == str(tmp_path / "lib64")


def test_empty_root_falls_back_to_lib64(tmp_path):
    assert toolchain.cuda_lib_dir(str(tmp_path)) == str(tmp_path / "lib64")


def test_system_toolkit_flags_unchanged(tmp_path):
    _touch(tmp_path / "lib64" / "libcudart.so")
    assert toolchain.cudart_link_flags(str(tmp_path)) == [
        f"-L{tmp_path / 'lib64'}",
        "-lcudart",
    ]


def test_pip_wheel_links_versioned_file(tmp_path):
    _touch(tmp_path / "lib" / "libcudart.so.13")
    assert toolchain.cudart_link_flags(str(tmp_path)) == [
        f"-L{tmp_path / 'lib'}",
        "-l:libcudart.so.13",
    ]


def test_missing_cudart_keeps_old_flags(tmp_path):
    assert toolchain.cudart_link_flags(str(tmp_path)) == [
        f"-L{tmp_path / 'lib64'}",
        "-lcudart",
    ]
