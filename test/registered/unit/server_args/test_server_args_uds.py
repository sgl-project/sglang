"""Unit tests for ServerArgs UDS validation.

Usage:
    python3 -m pytest test/registered/unit/server_args/test_server_args_uds.py -v
"""

import sys
import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Mock get_device() so all tests run on CPU-only CI runners.
_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestServerArgsUDS(unittest.TestCase):
    def test_uds_with_default_host_port_accepted(self):
        args = ServerArgs(model_path="dummy", uds="/tmp/sglang-test.sock")
        self.assertEqual(args.uds, "/tmp/sglang-test.sock")
        # Defaults for host/port preserved so internal services keep working.
        self.assertEqual(args.host, ServerArgs.host)
        self.assertEqual(args.port, ServerArgs.port)

    def test_uds_with_explicit_host_rejected(self):
        with self.assertRaises(ValueError) as cm:
            ServerArgs(
                model_path="dummy",
                uds="/tmp/sglang-test.sock",
                host="0.0.0.0",
            )
        self.assertIn("--uds", str(cm.exception))
        self.assertIn("--host", str(cm.exception))

    def test_uds_with_explicit_port_rejected(self):
        with self.assertRaises(ValueError) as cm:
            ServerArgs(
                model_path="dummy",
                uds="/tmp/sglang-test.sock",
                port=31000,
            )
        self.assertIn("--uds", str(cm.exception))
        self.assertIn("--port", str(cm.exception))

    def test_uds_on_windows_rejected(self):
        with patch.object(sys, "platform", "win32"):
            with self.assertRaises(ValueError) as cm:
                ServerArgs(model_path="dummy", uds="/tmp/sglang-test.sock")
            self.assertIn("--uds", str(cm.exception))
            self.assertIn("Linux", str(cm.exception))

    def test_no_uds_unaffected(self):
        # Backward compat: ServerArgs without --uds behaves exactly as before.
        args = ServerArgs(model_path="dummy")
        self.assertIsNone(args.uds)

    def test_uds_cli_flag_parsed(self):
        from sglang.srt.server_args import prepare_server_args
        from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

        args = prepare_server_args(
            [
                "--model-path",
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                "--uds",
                "/tmp/sglang-cli.sock",
            ]
        )
        self.assertEqual(args.uds, "/tmp/sglang-cli.sock")


if __name__ == "__main__":
    unittest.main()
