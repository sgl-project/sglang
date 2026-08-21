import threading
import time
import unittest

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_manager(ttl: float) -> MooncakeKVManager:
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.session_lock = threading.Lock()
    mgr.failed_sessions = set()
    mgr.failed_session_expiry = {}
    mgr.failed_session_blacklist_ttl = ttl
    return mgr


class TestMooncakeFailedSessionTTL(unittest.TestCase):
    def test_blacklisted_session_is_reported_before_ttl_expires(self):
        mgr = _make_manager(ttl=60.0)
        with mgr.session_lock:
            mgr._blacklist_session("session-a")

        with mgr.session_lock:
            self.assertTrue(mgr._session_blacklisted("session-a"))
        self.assertIn("session-a", mgr.failed_sessions)

    def test_blacklist_self_clears_after_ttl_expires(self):
        mgr = _make_manager(ttl=60.0)
        with mgr.session_lock:
            mgr._blacklist_session("session-a")
            # Simulate TTL elapsing without waiting in real time.
            mgr.failed_session_expiry["session-a"] = time.time() - 1

        with mgr.session_lock:
            self.assertFalse(mgr._session_blacklisted("session-a"))
        self.assertNotIn("session-a", mgr.failed_sessions)
        self.assertNotIn("session-a", mgr.failed_session_expiry)

    def test_unrelated_session_is_never_blacklisted(self):
        mgr = _make_manager(ttl=60.0)
        with mgr.session_lock:
            mgr._blacklist_session("session-a")
            self.assertFalse(mgr._session_blacklisted("session-b"))

    def test_repeated_failure_after_ttl_expiry_re_blacklists(self):
        # A genuinely dead session keeps failing on retry and must be
        # blacklisted again -- the TTL only bounds a single transient
        # failure, it does not permanently whitelist a session.
        mgr = _make_manager(ttl=60.0)
        with mgr.session_lock:
            mgr._blacklist_session("session-a")
            mgr.failed_session_expiry["session-a"] = time.time() - 1
            self.assertFalse(mgr._session_blacklisted("session-a"))
            mgr._blacklist_session("session-a")
            self.assertTrue(mgr._session_blacklisted("session-a"))


if __name__ == "__main__":
    unittest.main()
