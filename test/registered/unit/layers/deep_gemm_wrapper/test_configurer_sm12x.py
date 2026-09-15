"""Architecture and scale-layout checks without importing GPU extensions."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def load_configurer(sm, *, has_sm12x_entry=True, enabled=True, cuda=True, musa=False):
    root = next(
        p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir()
    )
    source = root / "python/sglang/srt/layers/deep_gemm_wrapper/configurer.py"
    environ = ModuleType("sglang.srt.environ")
    environ.envs = SimpleNamespace(
        SGLANG_ENABLE_JIT_DEEPGEMM=SimpleNamespace(get=lambda: enabled)
    )
    utils = ModuleType("sglang.srt.utils")
    utils.get_device_sm = lambda: sm
    utils.is_cuda = lambda: cuda
    utils.is_musa = lambda: musa
    context = ModuleType("sglang.srt.runtime_context")
    context.get_platform = lambda: SimpleNamespace(is_sm100=cuda and sm // 10 == 10)
    deep_gemm = ModuleType("deep_gemm")
    if has_sm12x_entry:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous = lambda: None
    with patch.dict(
        sys.modules,
        {
            "sglang.srt.environ": environ,
            "sglang.srt.utils": utils,
            "sglang.srt.runtime_context": context,
            "deep_gemm": deep_gemm,
        },
    ):
        spec = importlib.util.spec_from_file_location("configurer_under_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class TestDeepGemmSM12xConfig(unittest.TestCase):
    def test_sm12x_uses_packed_activation_scales(self):
        for sm in (120, 121):
            with self.subTest(sm=sm):
                module = load_configurer(sm)
                self.assertTrue(module.ENABLE_JIT_DEEPGEMM)
                self.assertTrue(module.DEEPGEMM_SCALE_UE8M0)
                self.assertFalse(module.DEEPGEMM_NEED_TMA_ALIGNED_SCALES)
                # SM12x does not acquire datacenter-only kernel dispatch.
                self.assertFalse(module.DEEPGEMM_BLACKWELL)

    def test_sm12x_requires_available_entry_point(self):
        for sm in (120, 121):
            with self.subTest(sm=sm):
                module = load_configurer(sm, has_sm12x_entry=False)
                self.assertFalse(module.ENABLE_JIT_DEEPGEMM)
                self.assertFalse(module.DEEPGEMM_SCALE_UE8M0)

    def test_environment_can_disable_sm12x(self):
        for sm in (120, 121):
            with self.subTest(sm=sm):
                module = load_configurer(sm, enabled=False)
                self.assertFalse(module.ENABLE_JIT_DEEPGEMM)
                self.assertFalse(module.DEEPGEMM_SCALE_UE8M0)

    def test_hopper_and_datacenter_blackwell_unchanged(self):
        for sm, packed in ((90, False), (100, True), (103, True)):
            with self.subTest(sm=sm):
                module = load_configurer(sm, has_sm12x_entry=False)
                self.assertTrue(module.ENABLE_JIT_DEEPGEMM)
                self.assertEqual(module.DEEPGEMM_SCALE_UE8M0, packed)
                self.assertEqual(module.DEEPGEMM_BLACKWELL, packed)
                self.assertEqual(module.DEEPGEMM_NEED_TMA_ALIGNED_SCALES, not packed)

    def test_unsupported_cuda_architecture_stays_disabled(self):
        self.assertFalse(load_configurer(80).ENABLE_JIT_DEEPGEMM)

    def test_cpu_stays_disabled(self):
        module = load_configurer(0, cuda=False)
        self.assertFalse(module.ENABLE_JIT_DEEPGEMM)
        self.assertFalse(module.DEEPGEMM_SCALE_UE8M0)

    def test_musa_keeps_its_scale_layout(self):
        module = load_configurer(31, cuda=False, musa=True)
        self.assertTrue(module.ENABLE_JIT_DEEPGEMM)
        self.assertFalse(module.DEEPGEMM_SCALE_UE8M0)
        self.assertFalse(module.DEEPGEMM_NEED_TMA_ALIGNED_SCALES)


if __name__ == "__main__":
    unittest.main()
