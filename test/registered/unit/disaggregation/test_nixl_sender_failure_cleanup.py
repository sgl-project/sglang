import threading
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nixl.conn import NixlKVSender
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestNixlSenderFailureCleanup(unittest.TestCase):
    def test_failure_exception_cleans_room_state_before_raising(self):
        room = 7
        expected_exc = RuntimeError("transfer failed")
        sender = NixlKVSender.__new__(NixlKVSender)
        sender.bootstrap_room = room
        sender.conclude_state = None
        sender._send_failed = False
        sender._send_error = None
        staging_ctx = SimpleNamespace(
            prefetched_rooms={room, 8},
            prefetch_requested={(room, 0, "session-a"), (8, 0, "session-b")},
        )
        sender.kv_mgr = SimpleNamespace(
            enable_staging=True,
            _staging_ctx=staging_ctx,
            request_status={room: object()},
            req_to_decode_prefix_len={room: 3},
            transfer_infos={room: object()},
            exceptions={room: expected_exc},
            failure_records={room: "transfer failed"},
            failure_lock=threading.Lock(),
        )

        with self.assertRaises(RuntimeError) as cm:
            sender.failure_exception()

        self.assertIs(cm.exception, expected_exc)
        self.assertTrue(sender._send_failed)
        self.assertEqual(sender.conclude_state, KVPoll.Failed)
        self.assertNotIn(room, sender.kv_mgr.request_status)
        self.assertNotIn(room, sender.kv_mgr.req_to_decode_prefix_len)
        self.assertNotIn(room, sender.kv_mgr.transfer_infos)
        self.assertNotIn(room, sender.kv_mgr.exceptions)
        self.assertNotIn(room, sender.kv_mgr.failure_records)
        self.assertNotIn(room, staging_ctx.prefetched_rooms)
        self.assertNotIn((room, 0, "session-a"), staging_ctx.prefetch_requested)
        self.assertIn(8, staging_ctx.prefetched_rooms)
        self.assertIn((8, 0, "session-b"), staging_ctx.prefetch_requested)


if __name__ == "__main__":
    unittest.main()
