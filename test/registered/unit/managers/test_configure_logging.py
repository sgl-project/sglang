import unittest

from sglang.srt.managers.configure_logging import _build_payload, _parse_args
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestConfigureLoggingPayload(unittest.TestCase):
    def test_sends_only_the_options_passed(self):
        """The server applies every key it receives, so a key the user did not
        pass must not be sent: changing the log level once also turned off
        --log-requests and started dumping request contents to /tmp."""
        self.assertEqual(
            _build_payload(_parse_args(["--log-level", "debug"])),
            {"log_level": "debug"},
        )
        self.assertEqual(
            _build_payload(_parse_args(["--no-log-requests"])),
            {"log_requests": False},
        )


if __name__ == "__main__":
    unittest.main()
