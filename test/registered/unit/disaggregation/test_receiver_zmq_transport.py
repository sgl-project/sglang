import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _ConcreteReceiver(CommonKVReceiver):
    def poll(self):
        return KVPoll.WaitingForInput


class TestReceiverZmqTransport(unittest.TestCase):
    def setUp(self):
        self.original_ctx = CommonKVReceiver._ctx
        self.original_socket_cache = CommonKVReceiver._socket_cache
        self.original_socket_locks = CommonKVReceiver._socket_locks
        CommonKVReceiver._ctx = MagicMock()
        CommonKVReceiver._socket_cache = {}
        CommonKVReceiver._socket_locks = {}

    def tearDown(self):
        CommonKVReceiver._ctx = self.original_ctx
        CommonKVReceiver._socket_cache = self.original_socket_cache
        CommonKVReceiver._socket_locks = self.original_socket_locks

    @staticmethod
    def _make_receiver():
        receiver = _ConcreteReceiver.__new__(_ConcreteReceiver)
        receiver.bootstrap_addr = "prefill-a.test:8000"
        receiver.bootstrap_room = 17
        receiver.conclude_state = None
        receiver.kv_mgr = SimpleNamespace(
            connection_lock=threading.Lock(),
            connection_pool={
                "prefill-a.test:8000_0_0_0": [
                    {"rank_ip": "prefill-rank-a.test", "rank_port": 12345}
                ],
                "prefill-b.test:8000_0_0_0": [
                    {"rank_ip": "prefill-rank-b.test", "rank_port": 12346}
                ],
            },
            record_failure=MagicMock(),
            update_status=MagicMock(),
        )
        return receiver

    @patch(
        "sglang.srt.disaggregation.common.conn.envs."
        "SGLANG_DISAGGREGATION_ZMQ_SEND_TIMEOUT_MS.get",
        return_value=4321,
    )
    def test_connect_configures_bounded_reconnectable_socket(self, _mock_timeout):
        sock = MagicMock()
        CommonKVReceiver._ctx.socket.return_value = sock

        returned, lock = CommonKVReceiver._connect("tcp://127.0.0.1:9000")

        self.assertIs(returned, sock)
        self.assertIsInstance(lock, type(threading.Lock()))
        self.assertIn(call(zmq.RECONNECT_IVL, 100), sock.setsockopt.call_args_list)
        self.assertIn(call(zmq.IMMEDIATE, 1), sock.setsockopt.call_args_list)
        self.assertIn(call(zmq.SNDTIMEO, 4321), sock.setsockopt.call_args_list)
        self.assertIn(call(zmq.LINGER, 0), sock.setsockopt.call_args_list)
        self.assertIn(call(zmq.TCP_KEEPALIVE, 1), sock.setsockopt.call_args_list)
        sock.connect.assert_called_once_with("tcp://127.0.0.1:9000")

    @patch(
        "sglang.srt.disaggregation.common.conn.envs."
        "SGLANG_DISAGGREGATION_ZMQ_SEND_TIMEOUT_MS.get",
        return_value=0,
    )
    def test_connect_rejects_unbounded_timeout(self, _mock_timeout):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            CommonKVReceiver._connect("tcp://127.0.0.1:9000")

        CommonKVReceiver._ctx.socket.assert_not_called()

    @patch(
        "sglang.srt.disaggregation.common.conn.envs."
        "SGLANG_DISAGGREGATION_ZMQ_SEND_TIMEOUT_MS.get",
        return_value=5,
    )
    def test_send_timeout_evicts_socket_and_next_send_reconnects(self, _mock_timeout):
        endpoint = "tcp://127.0.0.1:9000"
        bootstrap_info = {"rank_ip": "127.0.0.1", "rank_port": 9000}
        frames = [b"guard", b"room", b"payload"]
        receiver = self._make_receiver()
        stale_sock = MagicMock()
        stale_sock.send_multipart.side_effect = zmq.Again()
        fresh_sock = MagicMock()
        CommonKVReceiver._ctx.socket.side_effect = [stale_sock, fresh_sock]

        with self.assertRaises(zmq.Again):
            receiver._send_multipart_to_bootstrap(bootstrap_info, frames)

        self.assertNotIn(endpoint, CommonKVReceiver._socket_cache)
        self.assertNotIn(endpoint, CommonKVReceiver._socket_locks)
        stale_sock.close.assert_called_once_with(linger=0)
        self.assertNotIn("prefill-a.test:8000_0_0_0", receiver.kv_mgr.connection_pool)
        self.assertIn("prefill-b.test:8000_0_0_0", receiver.kv_mgr.connection_pool)

        receiver._send_multipart_to_bootstrap(bootstrap_info, frames)

        self.assertIs(CommonKVReceiver._socket_cache[endpoint], fresh_sock)
        fresh_sock.send_multipart.assert_called_once_with(frames)

    @patch(
        "sglang.srt.disaggregation.common.conn.envs."
        "SGLANG_DISAGGREGATION_ZMQ_SEND_TIMEOUT_MS.get",
        return_value=5,
    )
    def test_disconnect_does_not_block_fresh_socket_generation(self, _mock_timeout):
        endpoint = "tcp://127.0.0.1:9000"
        stale_sock = MagicMock()
        stale_lock = threading.Lock()
        stale_lock.acquire()
        CommonKVReceiver._socket_cache[endpoint] = stale_sock
        CommonKVReceiver._socket_locks[endpoint] = stale_lock
        fresh_sock = MagicMock()
        CommonKVReceiver._ctx.socket.return_value = fresh_sock
        evicted = threading.Event()
        disconnected = threading.Event()
        original_evict = CommonKVReceiver._evict_cached_socket

        def evict_and_signal(*args, **kwargs):
            result = original_evict(*args, **kwargs)
            evicted.set()
            return result

        def disconnect():
            CommonKVReceiver.disconnect_endpoint(endpoint)
            disconnected.set()

        try:
            with patch.object(
                CommonKVReceiver,
                "_evict_cached_socket",
                side_effect=evict_and_signal,
            ):
                thread = threading.Thread(target=disconnect)
                thread.start()
                self.assertTrue(evicted.wait(timeout=1))
                self.assertFalse(disconnected.is_set())

                returned, _ = CommonKVReceiver._connect(endpoint)
                self.assertIs(returned, fresh_sock)

                stale_lock.release()
                thread.join(timeout=1)
                self.assertFalse(thread.is_alive())
                self.assertTrue(disconnected.is_set())
        finally:
            if stale_lock.locked():
                stale_lock.release()

        stale_sock.close.assert_called_once_with(linger=0)
        self.assertIs(CommonKVReceiver._socket_cache[endpoint], fresh_sock)

    def test_old_failure_cannot_evict_new_socket_generation(self):
        endpoint = "tcp://127.0.0.1:9000"
        stale_sock = MagicMock()
        fresh_sock = MagicMock()
        fresh_lock = threading.Lock()
        CommonKVReceiver._socket_cache[endpoint] = fresh_sock
        CommonKVReceiver._socket_locks[endpoint] = fresh_lock

        evicted_sock, evicted_lock = CommonKVReceiver._evict_cached_socket(
            endpoint, expected_socket=stale_sock
        )

        self.assertIsNone(evicted_sock)
        self.assertIsNone(evicted_lock)
        self.assertIs(CommonKVReceiver._socket_cache[endpoint], fresh_sock)
        self.assertIs(CommonKVReceiver._socket_locks[endpoint], fresh_lock)

    def test_request_send_failure_marks_room_failed(self):
        receiver = self._make_receiver()
        bootstrap_info = {"rank_ip": "prefill-rank-a.test", "rank_port": 12345}
        frames = [b"guard", b"room", b"payload"]

        with patch.object(
            CommonKVReceiver,
            "_send_multipart_to_bootstrap",
            side_effect=zmq.Again(),
        ) as mock_send:
            sent = receiver._send_request_multipart_to_bootstrap(bootstrap_info, frames)

        self.assertFalse(sent)
        mock_send.assert_called_once_with(bootstrap_info, frames)
        receiver.kv_mgr.record_failure.assert_called_once()
        failure_reason = receiver.kv_mgr.record_failure.call_args.args[1]
        self.assertIn("prefill-rank-a.test:12345", failure_reason)
        self.assertNotIn("prefill-a.test:8000_0_0_0", receiver.kv_mgr.connection_pool)
        self.assertIn("prefill-b.test:8000_0_0_0", receiver.kv_mgr.connection_pool)
        receiver.kv_mgr.update_status.assert_called_once_with(17, KVPoll.Failed)
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)

    def test_request_send_success_preserves_multipart_payload(self):
        receiver = self._make_receiver()
        bootstrap_info = {"rank_ip": "prefill-rank-a.test", "rank_port": 12345}
        frames = [b"guard", b"room", b"payload"]

        with patch.object(
            CommonKVReceiver, "_send_multipart_to_bootstrap"
        ) as mock_send:
            sent = receiver._send_request_multipart_to_bootstrap(bootstrap_info, frames)

        self.assertTrue(sent)
        mock_send.assert_called_once_with(bootstrap_info, frames)
        receiver.kv_mgr.record_failure.assert_not_called()
        receiver.kv_mgr.update_status.assert_not_called()
        self.assertIsNone(receiver.conclude_state)


if __name__ == "__main__":
    unittest.main()
