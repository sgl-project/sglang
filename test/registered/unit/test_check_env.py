"""Regression tests for environment checker backend selection."""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import sglang.check_env as check_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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


class TestCheckEnvMissingVendorTool(CustomTestCase):
    def test_missing_vendor_tool_is_reported_not_raised(self):
        """The report must still be produced where the vendor's SMI tool or compiler
        is not installed, for example a CUDA torch wheel on a machine with no GPU."""
        not_available = "Not Available"
        probes = [
            (check_env.GPUEnv, "get_topology", (), {}),
            (check_env.HIPEnv, "get_topology", (), {}),
            (
                check_env.HIPEnv,
                "_get_rocm_driver_version",
                (),
                {"ROCM Driver Version": not_available},
            ),
            (check_env.NPUEnv, "get_topology", (), {}),
            (
                check_env.NPUEnv,
                "_get_ascend_driver_version",
                (),
                {"Ascend Driver Version": not_available},
            ),
            (
                check_env.NPUEnv,
                "_get_cann_info",
                ("/nonexistent-cann-home",),
                {"CANN": not_available, "BiSheng": not_available},
            ),
            (check_env.MUSAEnv, "get_topology", (), {}),
            (
                check_env.MUSAEnv,
                "_get_musa_driver_version",
                (),
                {"MUSA Driver Version": not_available},
            ),
        ]
        with (
            tempfile.TemporaryDirectory() as empty_dir,
            patch.dict(os.environ, {"PATH": empty_dir}),
            # NPUEnv() extends the module-level list in place.
            patch.object(check_env, "PACKAGE_LIST", list(check_env.PACKAGE_LIST)),
        ):
            for env_class, probe, args, expected in probes:
                with self.subTest(env=env_class.__name__, probe=probe):
                    self.assertEqual(getattr(env_class(), probe)(*args), expected)


if __name__ == "__main__":
    unittest.main()
