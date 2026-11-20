"""Exercise the real wrapper's imports without a CUDA or DeepGEMM installation."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[4]
SGLANG_ROOT = REPO_ROOT / "python" / "sglang"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module(
    "ci_register", SGLANG_ROOT / "test" / "ci" / "ci_register.py"
).register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepGemmHcPrenormImport(unittest.TestCase):
    def setUp(self):
        wrapper = "sglang.srt.layers.deep_gemm_wrapper"
        configurer = types.ModuleType(f"{wrapper}.configurer")
        for name in (
            "ENABLE_JIT_DEEPGEMM",
            "DEEPGEMM_BLACKWELL",
            "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
            "DEEPGEMM_SCALE_UE8M0",
        ):
            setattr(configurer, name, False)
        package = types.ModuleType(wrapper)
        package.compile_utils = types.ModuleType(f"{wrapper}.compile_utils")
        environ = types.ModuleType("sglang.srt.environ")
        environ.envs = mock.Mock()
        environ.envs.SGLANG_DEEPGEMM_SANITY_CHECK.get.return_value = False
        torch = types.ModuleType("torch")
        torch.Tensor = object
        self.modules = {
            wrapper: package,
            f"{wrapper}.configurer": configurer,
            "sglang.srt.environ": environ,
            "torch": torch,
            "deep_gemm": None,
        }
        with mock.patch.dict(sys.modules, self.modules):
            self.entrypoint = _load_module(
                "hc_prenorm_entrypoint",
                SGLANG_ROOT / "srt/layers/deep_gemm_wrapper/entrypoint.py",
            )
        self.assertFalse(self.entrypoint.ENABLE_JIT_DEEPGEMM)
        self.assertNotIn("deep_gemm", vars(self.entrypoint))

    def test_nonempty_input_imports_deep_gemm_when_jit_disabled(self):
        kernel = mock.Mock()
        deep_gemm = types.ModuleType("deep_gemm")
        deep_gemm.tf32_hc_prenorm_gemm = kernel
        x = types.SimpleNamespace(shape=(1, 4))
        fn, out, sqrsum = object(), object(), object()
        with mock.patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            self.entrypoint.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, 2)
        kernel.assert_called_once_with(x, fn, out, sqrsum, num_splits=2)

    def test_empty_input_does_not_require_deep_gemm(self):
        with mock.patch.dict(sys.modules, {"deep_gemm": None}):
            self.entrypoint.tf32_hc_prenorm_gemm(
                types.SimpleNamespace(shape=(0, 4)), None, None, None, 1
            )

    def test_missing_dependency_reports_import_error(self):
        with mock.patch.dict(sys.modules, {"deep_gemm": None}):
            with self.assertRaises(ImportError):
                self.entrypoint.tf32_hc_prenorm_gemm(
                    types.SimpleNamespace(shape=(1, 4)), None, None, None, 1
                )


if __name__ == "__main__":
    unittest.main()
