import unittest
from unittest.mock import patch

import sglang.check_env as check_env
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCheckEnvSelection(unittest.TestCase):
    @patch.object(check_env, "is_cuda_v2", return_value=False)
    @patch.object(check_env, "is_hip", return_value=False)
    @patch.object(check_env, "is_npu", return_value=False)
    @patch.object(check_env, "is_musa", return_value=False)
    @patch.object(check_env, "is_mps", return_value=False)
    def test_cpu_fallback_when_no_accelerator_is_available(self, *_):
        self.assertIsInstance(check_env.get_env_checker(), check_env.CPUEnv)

    def test_cpu_env_reports_cpu_available(self):
        env = check_env.CPUEnv()
        self.assertTrue(env.get_info()["CPU available"])
        self.assertEqual(env.get_topology(), {})

    @patch.object(check_env.subprocess, "check_output", side_effect=FileNotFoundError)
    def test_cpu_env_tolerates_missing_lscpu(self, _):
        env = check_env.CPUEnv()

        self.assertEqual(env.get_info(), {"CPU available": True})


if __name__ == "__main__":
    unittest.main()
