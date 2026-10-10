import threading
import unittest
from unittest.mock import Mock

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAscendTransferLifecycle(unittest.TestCase):
    def setUp(self):
        # Match MemFabric's API: no Mooncake batch_unregister_memory method.
        self.native = Mock(spec=["batch_register_memory", "destroy"])
        self.native.batch_register_memory.return_value = 0
        self.engine = AscendTransferEngine.__new__(AscendTransferEngine)
        self.engine.engine = self.native

    def _manager(self):
        manager = AscendKVManager.__new__(AscendKVManager)
        manager.engine = self.engine
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager._worker_threads = []
        manager._socket_lock = threading.Lock()
        manager._socket_cache = {}
        manager._monitor_cache = {}
        manager.server_socket = Mock(spec=["close"])
        manager._zmq_ctx = Mock(spec=["destroy"])
        return manager

    def test_close_releases_native_session_once(self):
        self.engine.close()
        self.engine.close()
        self.native.destroy.assert_called_once_with()
        self.assertIsNone(self.engine.engine)

    def test_destroy_failure_reaches_role_switch_and_preserves_handle(self):
        self.native.destroy.side_effect = RuntimeError("destroy failed")
        manager = self._manager()
        with self.assertRaisesRegex(RuntimeError, "destroy failed"):
            manager.teardown()
        self.assertIs(self.engine.engine, self.native)

    def test_teardown_uses_destroy_and_releases_cached_connections(self):
        manager = self._manager()
        manager.connection_pool = {"old-prefill": object()}
        manager.connection_lock = threading.Lock()
        manager.teardown()
        self.native.destroy.assert_called_once_with()
        self.assertEqual(manager.connection_pool, {})
        manager.server_socket.close.assert_called_once_with(linger=0)
        manager._zmq_ctx.destroy.assert_called_once_with(linger=0)

    def test_live_worker_prevents_socket_and_engine_destruction(self):
        manager = AscendKVManager.__new__(AscendKVManager)
        manager.engine = self.engine
        worker = Mock(spec=threading.Thread)
        worker.name = "stuck-transfer-worker"
        worker.is_alive.return_value = True
        manager._worker_threads = [worker]
        with self.assertRaisesRegex(RuntimeError, "did not stop"):
            manager.teardown()
        worker.join.assert_called_once_with(timeout=3.0)
        self.native.destroy.assert_not_called()

    def test_registration_uses_memfabric_batch_api(self):
        self.assertEqual(self.engine.batch_register([123, 456], [16, 32]), 0)
        self.native.batch_register_memory.assert_called_once_with([123, 456], [16, 32])

    def test_registration_failure_is_not_silently_ignored(self):
        self.native.batch_register_memory.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            self.engine.batch_register([123], [16])

    def test_registration_exception_reaches_caller(self):
        self.native.batch_register_memory.side_effect = RuntimeError(
            "registration error"
        )
        with self.assertRaisesRegex(RuntimeError, "registration error"):
            self.engine.batch_register([123], [16])


if __name__ == "__main__":
    unittest.main()
