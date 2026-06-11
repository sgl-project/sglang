"""Unit tests for srt/session/session_controller — no server, no model loading.

Regression tests for https://github.com/sgl-project/sglang/issues/23579:
the idle-timeout reaper must never tear down a session while one of its
requests is still decoding.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import time
import unittest

from sglang.srt.managers.io_struct import CloseSessionReqInput, OpenSessionReqInput
from sglang.srt.session.session_controller import SessionController, SessionReqNode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TIMEOUT = 0.05


class _FakeTreeCache:
    def __init__(self):
        self.released = []

    def release_session(self, session_id):
        self.released.append(session_id)


class _FakeReq:
    """Mimics the bits of Req that SessionController touches."""

    def __init__(self, rid, is_finished=False):
        self.rid = rid
        self.finished_reason = None
        self.multimodal_inputs = None
        self.is_finished = is_finished

    def finished(self):
        return self.is_finished


class TestSessionControllerReaper(CustomTestCase):
    def setUp(self):
        self.tree_cache = _FakeTreeCache()
        self.controller = SessionController(self.tree_cache)

    def _open(self, streaming=False):
        out = self.controller.open(
            OpenSessionReqInput(
                capacity_of_str_len=32768,
                session_id="s1",
                streaming=streaming,
                timeout=_TIMEOUT,
            )
        )
        self.assertTrue(out.success)
        return self.controller.get("s1")

    def _expire_timeout(self, session):
        """Backdate the session past its idle timeout and force a reap tick."""
        session.last_active_time = time.monotonic() - 10 * _TIMEOUT
        self.controller._last_reap_time = 0.0

    def test_reaper_skips_non_streaming_session_with_unfinished_request(self):
        session = self._open(streaming=False)
        req = _FakeReq("r1", is_finished=False)
        session.req_nodes["r1"] = SessionReqNode(req)

        # Simulate a decode that runs longer than the idle timeout.
        self._expire_timeout(session)
        now = time.monotonic()
        self.controller.maybe_reap(now)

        self.assertIn("s1", self.controller)
        self.assertEqual(self.tree_cache.released, [])
        # The reaper must not even schedule a deferred close: the session is
        # busy, not idle, so follow-up requests should still be accepted.
        self.assertFalse(session.close_on_finish)
        # The timeout clock is refreshed so it restarts after the decode.
        self.assertEqual(session.last_active_time, now)

    def test_reaper_skips_streaming_session_with_inflight_request(self):
        session = self._open(streaming=True)
        session._inflight = True

        self._expire_timeout(session)
        self.controller.maybe_reap(time.monotonic())

        self.assertIn("s1", self.controller)
        self.assertEqual(self.tree_cache.released, [])
        self.assertFalse(session.close_on_finish)

    def test_reaper_reaps_idle_session_after_requests_finish(self):
        session = self._open(streaming=False)
        req = _FakeReq("r1", is_finished=False)
        session.req_nodes["r1"] = SessionReqNode(req)

        self._expire_timeout(session)
        self.controller.maybe_reap(time.monotonic())
        self.assertIn("s1", self.controller)

        req.is_finished = True
        self._expire_timeout(session)
        self.controller.maybe_reap(time.monotonic())

        self.assertNotIn("s1", self.controller)
        self.assertEqual(self.tree_cache.released, ["s1"])

    def test_reaper_reaps_idle_session_without_requests(self):
        session = self._open(streaming=False)

        self._expire_timeout(session)
        self.controller.maybe_reap(time.monotonic())

        self.assertNotIn("s1", self.controller)
        self.assertEqual(self.tree_cache.released, ["s1"])

    def test_explicit_close_defers_until_request_finishes(self):
        session = self._open(streaming=False)
        req = _FakeReq("r1", is_finished=False)
        session.req_nodes["r1"] = SessionReqNode(req)

        self.controller.close(CloseSessionReqInput(session_id="s1"))

        # The close is deferred while the request is still decoding.
        self.assertIn("s1", self.controller)
        self.assertEqual(self.tree_cache.released, [])
        self.assertTrue(session.close_on_finish)

        req.is_finished = True
        self.controller._last_reap_time = 0.0
        self.controller.maybe_reap(time.monotonic())

        self.assertNotIn("s1", self.controller)
        self.assertEqual(self.tree_cache.released, ["s1"])


if __name__ == "__main__":
    unittest.main()
