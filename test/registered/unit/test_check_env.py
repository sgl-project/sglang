"""Regression tests for environment checker backend selection."""

import unittest
from unittest.mock import Mock, patch

import sglang.check_env as check_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCheckEnvBackendSelection(CustomTestCase):
    def _get_env(self, *, xpu_available: bool):
        predicates = {
            "is_cuda_v2": False,
            "is_hip": False,
            "is_npu": False,
            "is_musa": False,
            "is_xpu": xpu_available,
            "is_mps": False,
        }
        with patch.multiple(
            check_env,
            **{name: Mock(return_value=value) for name, value in predicates.items()},
        ):
            return check_env._get_env()

    def test_cpu_fallback_without_accelerator(self):
        """CPU-only environments must still select a checker instead of crashing."""
        self.assertIsInstance(
            self._get_env(xpu_available=False),
            check_env.CPUEnv,
        )

    def test_xpu_backend_selection(self):
        """An available Intel XPU must select its checker instead of falling through."""
        xpu = Mock()
        xpu.is_available.return_value = True
        xpu.device_count.return_value = 2
        xpu.get_device_name.side_effect = ["Intel GPU", "Intel GPU"]

        with patch.object(check_env.torch, "xpu", xpu, create=True):
            env = self._get_env(xpu_available=True)
            self.assertIsInstance(env, check_env.XPUEnv)
            self.assertEqual(
                env.get_info(),
                {
                    "XPU available": True,
                    "XPU count": 2,
                    "XPU 0,1": "Intel GPU",
                },
            )


if __name__ == "__main__":
    unittest.main()
