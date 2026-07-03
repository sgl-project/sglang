import builtins
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _EnvFlag:
    def __init__(self, value: bool):
        self._value = value

    def get(self) -> bool:
        return self._value


def _load_configurer(
    *,
    sm_version: int = 0,
    is_cuda: bool = False,
    is_musa: bool = False,
    env_enabled: bool = True,
    deep_gemm_imports: list[str] | None = None,
):
    repo_root = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "python" / "sglang").is_dir()
    )
    module_path = (
        repo_root
        / "python"
        / "sglang"
        / "srt"
        / "layers"
        / "deep_gemm_wrapper"
        / "configurer.py"
    )

    environ_stub = types.ModuleType("sglang.srt.environ")
    environ_stub.envs = types.SimpleNamespace(
        SGLANG_ENABLE_JIT_DEEPGEMM=_EnvFlag(env_enabled)
    )
    utils_stub = types.ModuleType("sglang.srt.utils")
    utils_stub.get_device_sm = lambda: sm_version
    utils_stub.is_cuda = lambda: is_cuda
    utils_stub.is_musa = lambda: is_musa
    utils_stub.is_sm100_supported = lambda: False

    deep_gemm_stub = types.ModuleType("deep_gemm")
    real_import = builtins.__import__

    def tracked_import(name, *args, **kwargs):
        if name == "deep_gemm" and deep_gemm_imports is not None:
            deep_gemm_imports.append(name)
        return real_import(name, *args, **kwargs)

    spec = importlib.util.spec_from_file_location(
        "_test_deep_gemm_configurer", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with (
        patch.dict(
            sys.modules,
            {
                "sglang.srt.environ": environ_stub,
                "sglang.srt.utils": utils_stub,
                "deep_gemm": deep_gemm_stub,
            },
        ),
        patch("builtins.__import__", side_effect=tracked_import),
    ):
        spec.loader.exec_module(module)
    return module


class TestDeepGemmConfigurer(unittest.TestCase):
    def test_cuda_architecture_allowlist(self):
        configurer = _load_configurer()
        self.assertTrue(
            hasattr(configurer, "_is_deep_gemm_supported_cuda_arch"),
            "configurer must expose a pure CUDA architecture predicate",
        )
        is_supported = configurer._is_deep_gemm_supported_cuda_arch

        cases = {
            80: False,
            89: False,
            90: True,
            91: True,
            100: True,
            101: True,
            103: True,
            110: False,
            120: False,
            121: False,
        }
        for sm_version, expected in cases.items():
            with self.subTest(sm_version=sm_version):
                self.assertIs(is_supported(sm_version), expected)

    def test_unsupported_cuda_arch_returns_before_importing_deep_gemm(self):
        deep_gemm_imports = []

        configurer = _load_configurer(
            sm_version=121,
            is_cuda=True,
            env_enabled=True,
            deep_gemm_imports=deep_gemm_imports,
        )

        self.assertFalse(configurer.ENABLE_JIT_DEEPGEMM)
        self.assertEqual(deep_gemm_imports, [])

    def test_musa_architecture_gate_is_unchanged(self):
        for sm_version, expected in ((30, False), (31, True)):
            with self.subTest(sm_version=sm_version):
                configurer = _load_configurer(
                    sm_version=sm_version,
                    is_musa=True,
                    env_enabled=True,
                )
                self.assertIs(configurer.ENABLE_JIT_DEEPGEMM, expected)


if __name__ == "__main__":
    unittest.main()
