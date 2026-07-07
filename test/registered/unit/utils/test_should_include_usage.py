"""Unit tests for should_include_usage() default-resolution logic.

Usage:
    python3 -m pytest test/registered/unit/utils/test_should_include_usage.py -v
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import StreamOptions
from sglang.srt.entrypoints.openai.utils import should_include_usage
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestShouldIncludeUsage(unittest.TestCase):
    # Server-side defaults must OR with client stream_options: either side
    # can turn a flag on; neither can force the other's off. This mirrors
    # how include_usage already worked before continuous_usage_stats got a
    # server default in this PR.

    def test_no_stream_options_no_defaults(self):
        include, continuous = should_include_usage(None, False, False)
        self.assertFalse(include)
        self.assertFalse(continuous)

    def test_no_stream_options_server_defaults_true(self):
        # Operator flipped both server defaults on; client sent nothing.
        include, continuous = should_include_usage(None, True, True)
        self.assertTrue(include)
        self.assertTrue(continuous)

    def test_client_only_no_server_defaults(self):
        # Backward compat: existing clients that already opt in via
        # stream_options still work when server defaults are off.
        opts = StreamOptions(include_usage=True, continuous_usage_stats=True)
        include, continuous = should_include_usage(opts, False, False)
        self.assertTrue(include)
        self.assertTrue(continuous)

    def test_server_default_turns_continuous_on_even_when_client_false(self):
        # This is the whole point of the new flag: operator can turn on
        # per-event token counts for every request regardless of client.
        opts = StreamOptions(include_usage=False, continuous_usage_stats=False)
        include, continuous = should_include_usage(opts, False, True)
        self.assertFalse(include)
        self.assertTrue(continuous)

    def test_server_default_off_client_on(self):
        # Client can still opt in per-request when server default is off.
        opts = StreamOptions(include_usage=False, continuous_usage_stats=True)
        include, continuous = should_include_usage(opts, False, False)
        self.assertFalse(include)
        self.assertTrue(continuous)

    def test_include_usage_and_continuous_independent(self):
        # continuous_usage_stats implies per-event injection but does NOT
        # imply the final aggregated usage chunk. Operators can turn on
        # per-event without the final chunk (though the reverse is more
        # common). Verify both flags are resolved independently.
        opts = StreamOptions(include_usage=False, continuous_usage_stats=True)
        include, continuous = should_include_usage(opts, False, False)
        self.assertFalse(include)
        self.assertTrue(continuous)

        opts = StreamOptions(include_usage=True, continuous_usage_stats=False)
        include, continuous = should_include_usage(opts, False, False)
        self.assertTrue(include)
        self.assertFalse(continuous)


if __name__ == "__main__":
    unittest.main()
