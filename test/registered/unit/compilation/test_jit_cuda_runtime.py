import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels.jit.utils import compile as jit_compile
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestJitCudaRuntime(CustomTestCase):
    def test_versioned_cuda_runtime_is_exposed_to_the_linker(self):
        """A versioned-only pip CUDA runtime must not fall through to a system CUDA."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            cuda_home = root / "cuda"
            runtime_dir = cuda_home / "lib"
            runtime_dir.mkdir(parents=True)
            runtime = runtime_dir / "libcudart.so.13"
            runtime.write_bytes(b"test runtime")
            cache_dir = root / "cache"

            with (
                patch(
                    "tvm_ffi.cpp.extension._find_cuda_home",
                    side_effect=lambda: os.environ["CUDA_HOME"],
                ),
                patch.object(
                    jit_compile,
                    "get_jit_cuda_arch",
                    return_value=SimpleNamespace(target_name="sm_80"),
                ),
                patch.dict(
                    os.environ,
                    {
                        "CUDA_HOME": str(cuda_home),
                        "LIBRARY_PATH": "/existing/link/path",
                        "TVM_FFI_CACHE_DIR": str(cache_dir),
                    },
                ),
            ):
                with jit_compile._jit_compile_context():
                    compat_home = pathlib.Path(os.environ["CUDA_HOME"])
                    runtime_link = compat_home / "lib64" / "libcudart.so"
                    self.assertNotEqual(compat_home, cuda_home)
                    self.assertEqual(runtime_link.resolve(), runtime.resolve())
                    self.assertEqual(os.environ["LIBRARY_PATH"], "/existing/link/path")

                self.assertEqual(os.environ["CUDA_HOME"], str(cuda_home))
                self.assertIn(
                    "__cudart_", jit_compile._jit_build_dir_name("test_module")
                )

    def test_standard_cuda_runtime_layout_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            cuda_home = root / "cuda"
            runtime_dir = cuda_home / "lib64"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "libcudart.so").write_bytes(b"test runtime")

            with (
                patch(
                    "tvm_ffi.cpp.extension._find_cuda_home",
                    side_effect=lambda: os.environ["CUDA_HOME"],
                ),
                patch.object(
                    jit_compile,
                    "get_jit_cuda_arch",
                    return_value=SimpleNamespace(target_name="sm_80"),
                ),
                patch.dict(
                    os.environ,
                    {
                        "CUDA_HOME": str(cuda_home),
                        "LIBRARY_PATH": "/existing/link/path",
                        "TVM_FFI_CACHE_DIR": str(root / "cache"),
                    },
                ),
            ):
                with jit_compile._jit_compile_context():
                    self.assertEqual(os.environ["LIBRARY_PATH"], "/existing/link/path")
                    self.assertEqual(os.environ["CUDA_HOME"], str(cuda_home))

                self.assertNotIn(
                    "__cudart_", jit_compile._jit_build_dir_name("test_module")
                )


if __name__ == "__main__":
    unittest.main()
