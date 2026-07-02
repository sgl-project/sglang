"""Unit tests for the staging control-plane reliability fixes.

The staging transfer control plane (STAGING_REQ / STAGING_RSP / WATERMARK)
is fire-and-forget ZMQ. These tests cover the recovery paths that keep a
lost message from permanently wedging a transfer:

- handle_staging_req replays the cached allocation and re-broadcasts the
  current watermark (so a prefill-side re-request heals both loss modes),
  and logs (instead of silently swallowing) send failures.
- DecodeStagingHandler._free_and_send_watermark keeps notifying the
  remaining subscribers when one send fails, and logs the failure.
- PrefillStagingStrategy.maybe_resend_staging_req only re-sends after a
  chunk has been stalled for STAGING_STALL_RESEND_INTERVAL_SECS, and
  check_ready clears the stall tracking once the chunk becomes ready.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.common import staging_handler as sh
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingSocket:
    def __init__(self, fail=False):
        self.fail = fail
        self.sent = []

    def send_multipart(self, parts):
        if self.fail:
            raise OSError("send failed")
        self.sent.append(parts)


class _FakeReceiver:
    """Receiver double exposing bootstrap connection plumbing."""

    def __init__(self, bootstrap_infos, fail=False):
        self.bootstrap_infos = bootstrap_infos
        self.sock = _RecordingSocket(fail=fail)
        self.lock = threading.Lock()
        self.chunk_staging_infos = []

    def _connect_to_bootstrap_server(self, bootstrap_info):
        return self.sock, self.lock


def _staging_req_msg(room, chunk_idx, chunk_pages, session_id):
    return [
        b"STAGING_REQ",
        str(room).encode("ascii"),
        str(chunk_idx).encode("ascii"),
        str(chunk_pages).encode("ascii"),
        session_id.encode("ascii"),
    ]


class TestHandleStagingReqReplay(CustomTestCase):
    """Duplicate STAGING_REQ must replay the cached allocation and the
    current watermark — this is what makes prefill-side re-requests heal
    lost STAGING_RSP / WATERMARK messages."""

    def _run(self, receiver):
        allocator = MagicMock()
        allocator.get_watermark.return_value = (3, 4096)
        # Cached allocation for chunk 0: (alloc_id, offset, round, end, pages)
        receiver.chunk_staging_infos = [(7, 128, 2, 640, 4)]
        sh.handle_staging_req(
            _staging_req_msg(11, 0, 4, "sess-a"),
            allocator,
            MagicMock(),  # kv_args: unused on the cached-allocation path
            2,
            4,
            {},
            {11: receiver},
            {11: ["bi0"]},
        )
        # The cached path must not allocate again.
        allocator.assign.assert_not_called()

    def test_duplicate_req_replays_rsp_and_watermark(self):
        receiver = _FakeReceiver(["bi0"])
        self._run(receiver)
        self.assertEqual(len(receiver.sock.sent), 2)
        self.assertEqual(
            receiver.sock.sent[0],
            [b"STAGING_RSP", b"11", b"0", b"128", b"2", b"640", b"sess-a"],
        )
        self.assertEqual(
            receiver.sock.sent[1],
            [b"WATERMARK", b"3", b"4096", b"sess-a"],
        )

    def test_send_failure_is_logged_not_raised(self):
        receiver = _FakeReceiver(["bi0"], fail=True)
        with self.assertLogs(sh.logger, level="WARNING") as cm:
            self._run(receiver)
        self.assertTrue(any("STAGING_RSP" in line for line in cm.output))

    def test_replayed_watermark_is_applied_monotonically(self):
        """The watermark piggybacked on a replayed response goes through
        handle_watermark_msg, which must apply monotonic-max semantics so a
        stale replay can never move the watermark backwards."""
        receiver = _FakeReceiver(["bi0"])
        self._run(receiver)
        ctx = sh.PrefillStagingContext()
        ctx.remote_watermarks["sess-a"] = (5, 100)  # newer than the replay
        sh.handle_watermark_msg(ctx, receiver.sock.sent[1])
        self.assertEqual(ctx.remote_watermarks["sess-a"], (5, 100))
        # And a genuinely newer watermark still advances it.
        sh.handle_watermark_msg(ctx, [b"WATERMARK", b"6", b"1", b"sess-a"])
        self.assertEqual(ctx.remote_watermarks["sess-a"], (6, 1))


class TestFreeAndSendWatermark(CustomTestCase):
    def _make_handler(self):
        return sh.DecodeStagingHandler(
            kv_manager=MagicMock(),
            staging_allocator=MagicMock(),
            kv_buffer_info={},
            decode_tp=1,
            total_kv_heads=8,
            tp_rank=0,
            scheduler=MagicMock(),
        )

    def test_one_failing_subscriber_does_not_block_others(self):
        handler = self._make_handler()
        handler.staging_allocator.get_watermark.return_value = (1, 256)
        bad = _FakeReceiver(["bi-bad"], fail=True)
        good = _FakeReceiver(["bi-good"])
        handler._wm_subscribers = {
            ("bi-bad",): (bad, "sess-bad"),
            ("bi-good",): (good, "sess-good"),
        }
        decode_req = SimpleNamespace(req=SimpleNamespace(bootstrap_room=42))
        with self.assertLogs(sh.logger, level="WARNING") as cm:
            handler._free_and_send_watermark(0, decode_req)
        handler.staging_allocator.free.assert_called_once_with(0)
        # The failure is visible in the logs...
        self.assertTrue(any("WATERMARK" in line for line in cm.output))
        # ...and the healthy subscriber still received the broadcast.
        self.assertEqual(good.sock.sent, [[b"WATERMARK", b"1", b"256", b"sess-good"]])


class TestMaybeResendStagingReq(CustomTestCase):
    def _make_strategy(self):
        kv_manager = MagicMock()
        kv_manager.kv_buffer_tensors = {"page_size": 64}
        kv_manager.server_args.chunked_prefill_size = 8192
        return sh.PrefillStagingStrategy(kv_manager, staging_buffer=MagicMock())

    def test_resend_only_after_stall_interval(self):
        strategy = self._make_strategy()
        t = [1000.0]
        with patch.object(sh.time, "monotonic", side_effect=lambda: t[0]):
            with patch.object(sh, "send_staging_req") as send:
                # First deferral only arms the stall timer.
                self.assertFalse(
                    strategy.maybe_resend_staging_req(1, 0, 4, "s", "host", 9000)
                )
                send.assert_not_called()

                # Still within the interval: no resend.
                t[0] += sh.STAGING_STALL_RESEND_INTERVAL_SECS / 2
                self.assertFalse(
                    strategy.maybe_resend_staging_req(1, 0, 4, "s", "host", 9000)
                )
                send.assert_not_called()

                # Past the interval: resend fires with the STAGING_REQ args.
                t[0] += sh.STAGING_STALL_RESEND_INTERVAL_SECS
                self.assertTrue(
                    strategy.maybe_resend_staging_req(1, 0, 4, "s", "host", 9000)
                )
                send.assert_called_once_with(
                    strategy._resend_sockets, "host", 9000, 1, 0, 4, "s"
                )

                # The timer re-arms after a resend.
                self.assertFalse(
                    strategy.maybe_resend_staging_req(1, 0, 4, "s", "host", 9000)
                )
                send.assert_called_once()

    def test_send_failure_is_logged_and_retried_next_interval(self):
        strategy = self._make_strategy()
        t = [1000.0]
        with patch.object(sh.time, "monotonic", side_effect=lambda: t[0]):
            with patch.object(
                sh, "send_staging_req", side_effect=OSError("connect failed")
            ):
                strategy.maybe_resend_staging_req(1, 0, 4, "s", "host", 9000)
                t[0] += sh.STAGING_STALL_RESEND_INTERVAL_SECS + 1
                with self.assertLogs(sh.logger, level="WARNING") as cm:
                    self.assertFalse(
                        strategy.maybe_resend_staging_req(1, 0, 4, "s", "host", 9000)
                    )
                self.assertTrue(any("STAGING_REQ" in line for line in cm.output))
            # After another interval, a now-working send goes through.
            with patch.object(sh, "send_staging_req") as send:
                t[0] += sh.STAGING_STALL_RESEND_INTERVAL_SECS + 1
                self.assertTrue(
                    strategy.maybe_resend_staging_req(1, 0, 4, "s", "host", 9000)
                )
                send.assert_called_once()

    def test_check_ready_clears_stall_tracking(self):
        strategy = self._make_strategy()
        strategy.kv_manager._is_watermark_ready.return_value = True
        staging = sh.StagingTransferInfo()
        staging.set_chunk(0, 128, 2, 640)
        req = SimpleNamespace(room=1, staging=staging)

        t = [1000.0]
        with patch.object(sh.time, "monotonic", side_effect=lambda: t[0]):
            with patch.object(sh, "send_staging_req"):
                strategy.maybe_resend_staging_req(1, 0, 4, "s", "host", 9000)
                self.assertIn((1, 0, "s"), strategy._stall_since)

                ready, chunk_idx, offset, rnd, end = strategy.check_ready(
                    req, 0, 4, session_id="s"
                )
                self.assertTrue(ready)
                self.assertEqual((chunk_idx, offset, rnd, end), (0, 128, 2, 640))
                self.assertNotIn((1, 0, "s"), strategy._stall_since)


if __name__ == "__main__":
    unittest.main()
