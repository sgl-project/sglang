import threading
import time
import unittest
from collections import defaultdict

import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOM = 42


class _ExitingSocket:
    """Stops the decode thread at teardown: threading swallows SystemExit."""

    def recv_multipart(self):
        raise SystemExit


class TestMooncakeDecodeThread(CustomTestCase):
    """The decode KV manager binds a plain zmq.PULL socket, so any peer that can
    reach it may send a message that does not match the status wire format. Such
    a message used to raise out of the decode status thread and kill it; the
    process stayed up and healthy while every later KV-transfer completion went
    unprocessed and requests hung until they timed out.
    """

    def test_decode_thread_survives_malformed_message(self):
        ctx = zmq.Context()
        server_socket = ctx.socket(zmq.PULL)
        port = server_socket.bind_to_random_port("tcp://127.0.0.1")

        kv_mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        kv_mgr.server_socket = server_socket
        kv_mgr.request_status = {ROOM: KVPoll.WaitingForInput}
        kv_mgr.prefill_response_tracker = defaultdict(set)
        kv_mgr.required_prefill_response_num_table = {ROOM: 1}
        kv_mgr.enable_staging = False
        kv_mgr._staging_handler = None
        kv_mgr.update_status = kv_mgr.request_status.__setitem__
        kv_mgr._start_heartbeat_checker_thread = lambda: None
        threads_before = set(threading.enumerate())
        MooncakeKVManager.start_decode_thread(kv_mgr)
        decode_threads = set(threading.enumerate()) - threads_before

        sender = ctx.socket(zmq.PUSH)
        sender.connect(f"tcp://127.0.0.1:{port}")
        # A single frame that is neither a known header nor a status triple.
        sender.send_multipart([b"\x00"])
        # A legitimate transfer-done notification from prefill.
        sender.send_multipart([str(ROOM).encode(), str(KVPoll.Success).encode(), b"0"])

        deadline = time.time() + 10
        while kv_mgr.request_status[ROOM] != KVPoll.Success and time.time() < deadline:
            time.sleep(0.05)

        try:
            self.assertEqual(kv_mgr.request_status[ROOM], KVPoll.Success)
        finally:
            # SystemExit on the next recv ends the thread without a traceback.
            kv_mgr.server_socket = _ExitingSocket()
            sender.send_multipart([b"ABORT_ACK", b"0"])
            for thread in decode_threads:
                thread.join(timeout=5)
            sender.close()
            ctx.destroy(linger=0)


if __name__ == "__main__":
    unittest.main()
