"""Unit tests for network utility helpers."""

import unittest
from unittest.mock import call, patch

from sglang.srt.utils.network import wait_port_available
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestWaitPortAvailable(CustomTestCase):
    @patch("sglang.srt.utils.network.time.sleep")
    @patch("sglang.srt.utils.network.is_port_available", return_value=False)
    def test_polls_for_the_full_timeout(self, mock_is_port_available, mock_sleep):
        self.assertFalse(
            wait_port_available(
                port=12345,
                port_name="test port",
                timeout_s=3,
                raise_exception=False,
            )
        )

        self.assertEqual(mock_is_port_available.call_count, 30)
        self.assertEqual(mock_sleep.call_args_list, [call(0.1)] * 30)


if __name__ == "__main__":
    unittest.main()
