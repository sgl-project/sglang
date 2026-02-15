import unittest
from unittest.mock import patch

from sglang.srt.utils import common


class TestWaitPortAvailable(unittest.TestCase):
    def test_timeout_with_missing_process_details(self):
        with patch.object(common, "is_port_available", return_value=False), patch.object(
            common, "find_process_using_port", return_value=None
        ), patch.object(common.time, "sleep", return_value=None):
            with self.assertRaisesRegex(ValueError, "process details are not available"):
                common.wait_port_available(
                    12345, "rpc_port", timeout_s=16, raise_exception=True
                )

    def test_timeout_without_process_probe_has_clean_error_message(self):
        with patch.object(common, "is_port_available", return_value=False), patch.object(
            common.time, "sleep", return_value=None
        ):
            with self.assertRaises(ValueError) as ctx:
                common.wait_port_available(
                    12345, "rpc_port", timeout_s=1, raise_exception=True
                )

        self.assertEqual(
            str(ctx.exception),
            "rpc_port at 12345 is not available in 1 seconds.",
        )

    def test_raise_exception_false(self):
        with patch.object(common, "is_port_available", return_value=False), patch.object(
            common.time, "sleep", return_value=None
        ):
            ret = common.wait_port_available(
                12345, "rpc_port", timeout_s=1, raise_exception=False
            )
        self.assertFalse(ret)


if __name__ == "__main__":
    unittest.main()
