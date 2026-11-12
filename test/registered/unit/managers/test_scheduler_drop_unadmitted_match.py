"""A refused prefix match must return a freshly taken Mamba slot.

A non-streaming session turn has req.session set, but its slot was allocated
by this pass's prefix match, not lent by a streaming session record.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(*, holds_mamba, session=None):
    return SimpleNamespace(
        session=session,
        kv=SimpleNamespace(
            holds_mamba=holds_mamba,
            mamba_cow_src_index=3,
            mamba_needs_clear=True,
        ),
    )


def _scheduler(*, borrowed=None, has_session_mgr=True):
    released = []
    s = Scheduler.__new__(Scheduler)
    s.req_to_token_pool = SimpleNamespace()
    s.token_to_kv_pool_allocator = SimpleNamespace(
        storage=SimpleNamespace(release_state_slot=released.append)
    )
    if has_session_mgr:
        s.tree_cache = SimpleNamespace(
            session=SimpleNamespace(borrowed_slot=lambda req: borrowed)
        )
    else:
        s.tree_cache = SimpleNamespace()
    return s, released


class TestDropUnadmittedMatch(CustomTestCase):
    def _drop(self, req, *, borrowed=None, has_session_mgr=True):
        s, released = _scheduler(borrowed=borrowed, has_session_mgr=has_session_mgr)
        s._drop_unadmitted_match(req)
        return req, released

    def test_plain_request_releases_match_taken_slot(self):
        req, released = self._drop(_req(holds_mamba=True, session=None))
        self.assertEqual(released, [req])
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)

    def test_nonstreaming_session_turn_releases_match_taken_slot(self):
        # req.session is set for every session turn. A non-streaming turn
        # still took its slot from this prefix match, so the slot must be
        # returned when admission refuses the request.
        session = SimpleNamespace(session_id="s1")
        req, released = self._drop(
            _req(holds_mamba=True, session=session), borrowed=None
        )
        self.assertEqual(released, [req])
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)

    def test_streaming_lent_slot_is_kept(self):
        session = SimpleNamespace(session_id="s1")
        lent = SimpleNamespace()
        req, released = self._drop(
            _req(holds_mamba=True, session=session), borrowed=lent
        )
        self.assertEqual(released, [])
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)

    def test_no_mamba_slot_skips_release(self):
        req, released = self._drop(_req(holds_mamba=False, session=None))
        self.assertEqual(released, [])
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)

    def test_cache_without_session_manager_releases_slot(self):
        session = SimpleNamespace(session_id="s1")
        req, released = self._drop(
            _req(holds_mamba=True, session=session), has_session_mgr=False
        )
        self.assertEqual(released, [req])


if __name__ == "__main__":
    unittest.main()
