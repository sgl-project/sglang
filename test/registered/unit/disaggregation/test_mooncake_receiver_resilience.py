"""Unit tests for MooncakeKVManager receiver-thread resilience.

The mooncake ``bootstrap_thread`` / ``decode_thread`` and the shared
``heartbeat_checker`` in ``common/conn.py`` must never die from a bad
message. These tests exercise the dispatch helpers with malformed
inputs (empty, short, stray) and assert the handler drops the message
rather than raising.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    _summarize_zmq_msg,
)
from sglang.test.test_utils import CustomTestCase


class TestSummarizeZmqMsg(CustomTestCase):
    """Log-formatter must never itself raise, and must produce bounded output."""

    def test_bounded_length(self):
        # 20 frames of 200 bytes each = 4000 raw bytes; repr must be capped.
        summary = _summarize_zmq_msg([b"x" * 200] * 20)
        self.assertLessEqual(len(summary), 512)

    def test_readable_ascii_prefix(self):
        summary = _summarize_zmq_msg([b"ABORT", b"1234", b"10.0.0.1", b"9001"])
        self.assertIn("ABORT", summary)

    def test_never_raises_on_broken_input(self):
        # Passing something that isn't iterable should still return a
        # placeholder rather than crashing the caller.
        summary = _summarize_zmq_msg(object())
        self.assertEqual(summary, "<unrepresentable>")


class TestDispatchBootstrapMsg(CustomTestCase):
    """``_dispatch_bootstrap_msg`` must drop malformed messages instead of
    raising (which used to kill the receiver thread)."""

    def _make_manager(self):
        mgr = MagicMock()
        mgr._dispatch_bootstrap_msg = MooncakeKVManager._dispatch_bootstrap_msg.__get__(
            mgr, MooncakeKVManager
        )
        mgr._handle_bootstrap_abort_msg = MagicMock()
        return mgr

    def test_empty_msg_is_dropped(self):
        mgr = self._make_manager()
        # Must not raise.
        mgr._dispatch_bootstrap_msg([])
        mgr._handle_bootstrap_abort_msg.assert_not_called()

    def test_short_abort_msg_is_dropped(self):
        # ABORT frame layout is [b"ABORT", room, ip, port]; only 2 frames
        # here so the original unconditional access to msg[3] would raise
        # IndexError.
        mgr = self._make_manager()
        mgr._dispatch_bootstrap_msg([b"ABORT", b"12345"])
        mgr._handle_bootstrap_abort_msg.assert_not_called()

    def test_short_session_register_msg_is_dropped(self):
        # room != known-prefix and len(msg) < 4 => cannot read
        # mooncake_session_id at index 3.
        mgr = self._make_manager()
        mgr._dispatch_bootstrap_msg([b"None", b"one", b"two"])

    def test_short_transfer_info_msg_is_dropped(self):
        # Transfer-info needs at least 8 frames; only 4 here => cannot
        # read required_dst_info_num at index 7.
        mgr = self._make_manager()
        mgr._dispatch_bootstrap_msg([b"123", b"one", b"two", b"session-id"])

    def test_well_formed_abort_dispatches_to_helper(self):
        # Sanity: on a valid ABORT frame the dispatcher delegates to
        # _handle_bootstrap_abort_msg exactly once.
        mgr = self._make_manager()
        msg = [b"ABORT", b"42", b"10.0.0.1", b"9001"]
        mgr._dispatch_bootstrap_msg(msg)
        mgr._handle_bootstrap_abort_msg.assert_called_once_with(msg)


class TestDispatchDecodeMsg(CustomTestCase):
    """``_dispatch_decode_msg`` must drop malformed messages instead of
    raising ``ValueError`` on the 3-tuple unpack (the observed prod
    failure mode)."""

    def _make_manager(self):
        mgr = MagicMock()
        mgr._dispatch_decode_msg = MooncakeKVManager._dispatch_decode_msg.__get__(
            mgr, MooncakeKVManager
        )
        mgr._handle_prefill_response = MagicMock()
        return mgr

    def test_empty_msg_is_dropped(self):
        mgr = self._make_manager()
        mgr._dispatch_decode_msg([])
        mgr._handle_prefill_response.assert_not_called()

    def test_stray_1_frame_msg_is_dropped(self):
        # No recognized header, only 1 frame => vanilla code would raise
        # ValueError on the unpack.
        mgr = self._make_manager()
        mgr._dispatch_decode_msg([b"stray-frame"])
        mgr._handle_prefill_response.assert_not_called()

    def test_stray_5_frame_msg_is_dropped(self):
        # 5 frames, no recognized header => also mismatches the 3-tuple
        # unpack that ends the dispatch.
        mgr = self._make_manager()
        mgr._dispatch_decode_msg([b"a", b"b", b"c", b"d", b"e"])
        mgr._handle_prefill_response.assert_not_called()

    def test_well_formed_3tuple_dispatches_to_helper(self):
        mgr = self._make_manager()
        # (room, status, prefill_rank) — all ASCII int-encoded bytes.
        mgr._dispatch_decode_msg([b"7", b"3", b"0"])
        mgr._handle_prefill_response.assert_called_once_with(7, 3, 0)


if __name__ == "__main__":
    unittest.main()
