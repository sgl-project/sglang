"""The PD ZMQ server loops must survive a message they cannot handle.

Each loop is the only reader of its socket, so an escaping exception kills the
thread and every later bootstrap / transfer / abort message goes unprocessed --
the process stays alive and health checks keep passing while PD is dead.
"""

import threading
import unittest
from collections import defaultdict

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Shapes a handler can legitimately be handed: empty, too short to index, and
# well-formed framing carrying a payload that will not parse.
MALFORMED = [
    [],
    [b""],
    [b"garbage"],
    [b"not-a-room", b"1", b"2"],
    [b"None", b"ep", b"0"],
    [b"\xff\xfe", b"\x00", b"", b"\x80abc"],
]


def _mooncake_mgr():
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.session_lock = threading.Lock()
    mgr.session_failures = defaultdict(int)
    mgr.failed_sessions = set()
    return mgr


class TestBootstrapMessageGuard(unittest.TestCase):
    def test_mooncake_fail_handler_never_raises(self):
        for msg in MALFORMED:
            with self.subTest(msg=msg):
                _mooncake_mgr()._fail_bootstrap_message(msg)

    def test_mooncake_fail_handler_marks_session(self):
        mgr = _mooncake_mgr()
        mgr._fail_bootstrap_message([b"7", b"ep", b"0", b"session-a"])
        self.assertIn("session-a", mgr.failed_sessions)
        self.assertEqual(mgr.session_failures["session-a"], 1)

    def test_mooncake_fail_handler_tolerates_non_utf8_session(self):
        mgr = _mooncake_mgr()
        mgr._fail_bootstrap_message([b"7", b"ep", b"0", b"\xff\xfe"])
        self.assertEqual(len(mgr.failed_sessions), 1)

    def test_mooncake_fail_handler_attributes_watermark_session(self):
        # [b"WATERMARK", round, tail, session] -- frame 3 really is the session.
        mgr = _mooncake_mgr()
        mgr._fail_bootstrap_message([b"WATERMARK", b"1", b"2", b"session-a"])
        self.assertIn("session-a", mgr.failed_sessions)

    def test_mooncake_fail_handler_skips_frames_without_a_session_at_3(self):
        # STAGING_RSP holds the staging offset at 3 (session at 6) and ABORT holds
        # the decode port, so neither may be recorded as a failed session.
        for msg in (
            [b"STAGING_RSP", b"7", b"0", b"4096", b"1", b"1", b"session-a"],
            [b"ABORT", b"7", b"10.0.0.1", b"31337"],
        ):
            with self.subTest(tag=msg[0]):
                mgr = _mooncake_mgr()
                mgr._fail_bootstrap_message(msg)
                self.assertEqual(mgr.failed_sessions, set())
                self.assertEqual(dict(mgr.session_failures), {})

    def test_mooncake_decode_fail_handler_never_raises(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        for msg in MALFORMED:
            with self.subTest(msg=msg):
                mgr._fail_decode_message(msg)

    def test_nixl_fail_handler_never_raises(self):
        mgr = NixlKVManager.__new__(NixlKVManager)
        for msg in MALFORMED:
            with self.subTest(msg=msg):
                mgr._fail_bootstrap_message(msg)


class TestHandlersAreRaiseProne(unittest.TestCase):
    """The guard is load-bearing: without it these inputs kill the loop."""

    def test_mooncake_bootstrap_handler_raises_on_malformed(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        with self.assertRaises(Exception):
            mgr._handle_bootstrap_message([b"not-a-room", b"1", b"2", b"session-a"])

    def test_mooncake_decode_handler_raises_on_malformed(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        with self.assertRaises(Exception):
            # Unknown header with a part count the 3-tuple unpack cannot take.
            mgr._handle_decode_message([b"UNKNOWN_HEADER", b"1"])

    def test_nixl_bootstrap_handler_raises_on_foreign_traffic(self):
        mgr = NixlKVManager.__new__(NixlKVManager)
        with self.assertRaises(Exception):
            # Missing the GUARD frame -- what the assert there calls "foreign traffic".
            mgr._handle_bootstrap_message([b"foreign", b"", b"", b"agent"])


if __name__ == "__main__":
    unittest.main()
