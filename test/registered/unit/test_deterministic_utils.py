"""Unit tests for the deterministic inference test harness."""

import unittest
from unittest.mock import patch

from sglang.test import test_deterministic_utils as deterministic_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestDeterministicUtils(CustomTestCase):
    def test_common_server_args_are_immutable_and_copied(self):
        first = deterministic_utils.TestDeterministicBase.get_server_args()
        second = deterministic_utils.TestDeterministicBase.get_server_args()

        self.assertIsInstance(deterministic_utils.COMMON_SERVER_ARGS, tuple)
        self.assertIsNot(first, second)

        first.extend(["--attention-backend", "fa3"])
        self.assertNotIn("--attention-backend", second)
        self.assertNotIn("--attention-backend", deterministic_utils.COMMON_SERVER_ARGS)

    def test_single_configures_sampling_before_running(self):
        def capture_args(args):
            self.assertEqual(args.test_mode, "single")
            self.assertEqual(args.temperature, 0.5)
            self.assertEqual(args.sampling_seed, 42)
            return [1]

        case = deterministic_utils.TestDeterministicBase(methodName="test_single")

        with patch.object(
            deterministic_utils, "test_deterministic", side_effect=capture_args
        ) as mock_test_deterministic:
            case.test_single()

        mock_test_deterministic.assert_called_once()

    @patch.object(deterministic_utils, "popen_launch_server")
    def test_setup_uses_one_server_args_snapshot(self, mock_launch_server):
        class BackendTest(deterministic_utils.TestDeterministicBase):
            server_args_calls = 0

            @classmethod
            def get_model(cls):
                return "test/model"

            @classmethod
            def get_server_args(cls):
                cls.server_args_calls += 1
                return [
                    "--attention-backend",
                    "fa3",
                    "--snapshot",
                    str(cls.server_args_calls),
                ]

        mock_launch_server.return_value.pid = 123

        BackendTest.setUpClass()

        self.assertEqual(BackendTest.server_args_calls, 1)
        mock_launch_server.assert_called_once_with(
            "test/model",
            deterministic_utils.DEFAULT_URL_FOR_TEST,
            timeout=deterministic_utils.DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "fa3", "--snapshot", "1"],
        )

        with patch.object(deterministic_utils, "kill_process_tree") as mock_kill:
            BackendTest.tearDownClass()
            mock_kill.assert_called_once_with(123)

    def test_teardown_is_safe_after_server_launch_failure(self):
        class BackendTest(deterministic_utils.TestDeterministicBase):
            @classmethod
            def get_server_args(cls):
                return ["--attention-backend", "fa3"]

        with patch.object(
            deterministic_utils,
            "popen_launch_server",
            side_effect=RuntimeError("launch failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                BackendTest.setUpClass()

        with patch.object(deterministic_utils, "kill_process_tree") as mock_kill:
            BackendTest.tearDownClass()
            mock_kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
