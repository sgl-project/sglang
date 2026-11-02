"""Unit tests for Mooncake session failure blacklisting — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

import threading
import unittest
from collections import defaultdict

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase


class TestMooncakeSessionFailureThreshold(CustomTestCase):
    def _make_mgr(self, threshold):
        # Bypass __init__: it starts transfer threads and a transfer engine.
        mgr = object.__new__(MooncakeKVManager)
        mgr.session_failures = defaultdict(int)
        mgr.failed_sessions = set()
        mgr.session_failure_threshold = threshold
        mgr.session_lock = threading.Lock()
        return mgr

    def test_default_threshold_blacklists_on_first_failure(self):
        mgr = self._make_mgr(threshold=1)
        mgr._record_session_failure("10.0.0.1:1234")
        self.assertIn("10.0.0.1:1234", mgr.failed_sessions)

    def test_higher_threshold_tolerates_transient_failures(self):
        mgr = self._make_mgr(threshold=3)
        mgr._record_session_failure("10.0.0.1:1234")
        mgr._record_session_failure("10.0.0.1:1234")
        self.assertNotIn("10.0.0.1:1234", mgr.failed_sessions)
        mgr._record_session_failure("10.0.0.1:1234")
        self.assertIn("10.0.0.1:1234", mgr.failed_sessions)

    def test_success_resets_failure_streak(self):
        mgr = self._make_mgr(threshold=3)
        mgr._record_session_failure("10.0.0.1:1234")
        mgr._record_session_failure("10.0.0.1:1234")
        mgr._clear_session_failures("10.0.0.1:1234")
        # Two more failures after a success: still a streak of two, not four.
        mgr._record_session_failure("10.0.0.1:1234")
        mgr._record_session_failure("10.0.0.1:1234")
        self.assertNotIn("10.0.0.1:1234", mgr.failed_sessions)

    def test_failure_streaks_are_independent_per_session(self):
        mgr = self._make_mgr(threshold=2)
        mgr._record_session_failure("10.0.0.1:1234")
        mgr._record_session_failure("10.0.0.2:5678")
        self.assertNotIn("10.0.0.1:1234", mgr.failed_sessions)
        self.assertNotIn("10.0.0.2:5678", mgr.failed_sessions)
        mgr._record_session_failure("10.0.0.1:1234")
        self.assertIn("10.0.0.1:1234", mgr.failed_sessions)
        self.assertNotIn("10.0.0.2:5678", mgr.failed_sessions)

    def test_env_default_is_one(self):
        self.assertEqual(
            envs.SGLANG_MOONCAKE_SESSION_FAILURE_THRESHOLD.get(),
            1,
        )


if __name__ == "__main__":
    unittest.main()
