"""Unit tests for Mooncake decode-side status thread handling."""

import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _StopDecodeThread(BaseException):
    pass


class TestMooncakeDecodeThread(unittest.TestCase):
    def _make_decode_manager(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.server_socket = MagicMock()
        mgr.request_status = {7: KVPoll.WaitingForInput}
        mgr.prefill_response_tracker = defaultdict(set)
        mgr.required_prefill_response_num_table = {7: 1}
        mgr.enable_staging = False
        mgr._staging_handler = None
        mgr._chunk_writer_counts = {}
        mgr._handle_aux_data = MagicMock()
        mgr._handle_staging_req = MagicMock()
        mgr.record_failure = MagicMock()

        def update_status(room, status):
            mgr.request_status[room] = status

        mgr.update_status = MagicMock(side_effect=update_status)
        return mgr

    def test_decode_thread_continues_after_malformed_status_message(self):
        mgr = self._make_decode_manager()
        mgr.server_socket.recv_multipart.side_effect = [
            [b"malformed"],
            [b"7", str(KVPoll.Success).encode("ascii"), b"0"],
            _StopDecodeThread(),
        ]
        captured_targets = []

        def capture_thread(*args, **kwargs):
            captured_targets.append(kwargs["target"])
            thread = MagicMock()
            thread.start.return_value = None
            return thread

        with (
            patch(
                "sglang.srt.disaggregation.mooncake.conn.threading.Thread",
                side_effect=capture_thread,
            ),
            patch.object(
                MooncakeKVManager,
                "_start_heartbeat_checker_thread",
                return_value=None,
            ),
        ):
            mgr.start_decode_thread()

        self.assertEqual(len(captured_targets), 1)

        with (
            patch("sglang.srt.disaggregation.mooncake.conn.logger.exception") as log,
            self.assertRaises(_StopDecodeThread),
        ):
            captured_targets[0]()

        log.assert_called_once_with("Mooncake decode status thread iteration failed")
        mgr.update_status.assert_called_once_with(7, KVPoll.Success)
        self.assertEqual(mgr.request_status[7], KVPoll.Success)


if __name__ == "__main__":
    unittest.main()
