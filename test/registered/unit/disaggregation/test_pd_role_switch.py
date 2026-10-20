import concurrent.futures
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt import runtime_context as rc  # noqa: E402
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# --- teardown: transfer-worker thread-leak fix + prefix-cache release (radix ON) ---
import threading  # noqa: E402
import time  # noqa: E402

import zmq  # noqa: E402

try:
    from sglang.srt.disaggregation.common.utils import (  # noqa: E402,F811
        FastQueue as _FQ,
    )
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager  # noqa: E402

    _HAS_MOONCAKE = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_MOONCAKE = False

try:
    from sglang.srt.disaggregation.role_switch import (  # noqa: E402
        _release_prefix_cache_for_role_switch,
        teardown_disaggregation,
    )

    _HAS_ROLE_SWITCH = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_ROLE_SWITCH = False


@unittest.skipUnless(_HAS_MOONCAKE, "mooncake not importable in this environment")
class TestMooncakeTeardownNoThreadLeak(unittest.TestCase):
    """teardown() must stop+join the transfer workers it started, so a P->D->P
    flip loop does not leak transfer threads per cycle."""

    def test_teardown_joins_transfer_workers(self):
        m = MooncakeKVManager.__new__(MooncakeKVManager)
        m.disaggregation_mode = DisaggregationMode.PREFILL
        m._stopped = False
        m.enable_trace = False
        m._worker_threads = []
        m.transfer_queues = [_FQ() for _ in range(3)]
        m.executors = [concurrent.futures.ThreadPoolExecutor(1) for _ in range(3)]
        m.server_socket = MagicMock()
        m._zmq_ctx = MagicMock()
        m._socket_lock = threading.Lock()
        m._socket_cache = {}
        m._monitor_cache = {}
        m.engine = MagicMock()
        # Built from KVArgs' own annotations: teardown walks several ptr/len
        # pairs, and hardcoding them here goes stale every time one is added.
        from sglang.srt.disaggregation.base.conn import KVArgs

        m.kv_args = SimpleNamespace(**{name: [] for name in KVArgs.__annotations__})
        for i, (q, ex) in enumerate(zip(m.transfer_queues, m.executors)):
            t = threading.Thread(
                target=m.transfer_worker, args=(q, ex, None, i), daemon=True
            )
            t.start()
            m._worker_threads.append(t)
        started = list(m._worker_threads)
        time.sleep(0.05)  # let workers park in FastQueue.get()
        for t in started:
            self.assertTrue(t.is_alive())

        MooncakeKVManager.teardown(m)

        for t in started:
            self.assertFalse(t.is_alive(), "transfer worker survived teardown (leak)")
        self.assertEqual(m._worker_threads, [])
        self.assertEqual(m.transfer_queues, [])
        self.assertEqual(m.executors, [])


@unittest.skipUnless(_HAS_MOONCAKE, "mooncake not importable in this environment")
class TestMooncakeBootstrapThreadRobustness(unittest.TestCase):
    """The prefill bootstrap loop moved from a blocking recv_multipart() to a
    500ms poll + _stopped check (so teardown, i.e. a runtime role switch, can
    stop it). That loop runs on every mooncake PD instance, so pin the
    contract with real ZMQ traffic driven through the ABORT -> ABORT_ACK
    path: no message loss while idle or bursting, and prompt exit once
    _stopped is set. Unlike mori, the loop has no try/except around recv: a
    recv error terminates the thread (see test_recv_error_kills_thread).
    """

    class _FlakySocket(zmq.Socket):
        """Real PULL socket whose next recv can be forced to fail once,
        emulating a transient ZMQ error between poll() and recv()."""

        fail_next_recv = False

        def recv_multipart(self, *args, **kwargs):
            if type(self).fail_next_recv:
                type(self).fail_next_recv = False
                raise RuntimeError("transient recv failure")
            return super().recv_multipart(*args, **kwargs)

    def setUp(self):
        self._FlakySocket.fail_next_recv = False
        self._ctx = zmq.Context()
        sock = self._FlakySocket(self._ctx, zmq.PULL)
        port = sock.bind_to_random_port("tcp://127.0.0.1")
        m = MooncakeKVManager.__new__(MooncakeKVManager)
        m._stopped = False
        m._worker_threads = []
        m.server_socket = sock
        # The receive path is gated on this flag: role switch must be on for
        # the poll-with-timeout loop these tests exercise.
        m.server_args = SimpleNamespace(enable_pd_role_switch=True)
        # Read by the receive loop; off keeps these tests on the plain ACK path.
        m.enable_deferred_decode_kv_release = False
        # ABORT for an unknown room takes the "ignoring" branch and still
        # ACKs, giving a side-effect-free probe of the receive loop.
        m.request_status = {}
        m._socket_send_locks = {}

        def _connect(endpoint, is_ipv6=False):
            m._socket_send_locks.setdefault(endpoint, threading.Lock())
            return m._connect.return_value

        m._connect = MagicMock(side_effect=_connect)
        self.m = m
        self._push = self._ctx.socket(zmq.PUSH)
        self._push.connect(f"tcp://127.0.0.1:{port}")

    def tearDown(self):
        self.m._stopped = True
        for t in self.m._worker_threads:
            t.join(timeout=3.0)
        self._push.close(linger=0)
        self.m.server_socket.close(linger=0)
        self._ctx.destroy(linger=0)

    def _start(self):
        MooncakeKVManager.start_prefill_thread(self.m)
        (thread,) = self.m._worker_threads
        return thread

    def _send_abort(self, room):
        self._push.send_multipart(
            [b"ABORT", str(room).encode("ascii"), b"127.0.0.1", b"9999"]
        )

    def _wait_acks(self, n, timeout=10.0):
        send = self.m._connect.return_value.send_multipart
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if send.call_count >= n:
                return
            time.sleep(0.02)
        self.fail(f"expected {n} ABORT_ACKs, got {send.call_count}")

    def test_messages_processed_across_idle_poll_timeouts(self):
        self._start()
        self._send_abort(1)
        self._wait_acks(1)
        # Idle past a full poll timeout, then traffic must still flow: the
        # empty-poll -> continue path must not disturb the socket.
        time.sleep(0.8)
        self._send_abort(2)
        self._wait_acks(2)

    def test_no_message_loss_under_burst(self):
        self._start()
        n = 200
        for i in range(n):
            self._send_abort(i)
        # Two-step poll+recv must consume every queued message exactly once.
        self._wait_acks(n)

    def test_recv_error_kills_thread(self):
        # No try/except guards recv() in the mooncake loop (unlike mori): a
        # recv error terminates the thread and the loop stops processing.
        # Pin that contract so adding error handling stays a deliberate,
        # reviewed change rather than a silent behavior shift.
        thread = self._start()
        self._FlakySocket.fail_next_recv = True
        self._send_abort(3)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive(), "bootstrap thread survived recv error")
        self.assertFalse(self._FlakySocket.fail_next_recv)  # fault consumed

    def test_exits_promptly_when_stopped_while_idle(self):
        thread = self._start()
        self.m._stopped = True
        # Poll timeout is 500ms, so the flag must be observed within ~1 cycle
        # (this is what keeps teardown / role switch from hanging).
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive(), "bootstrap thread leaked past stop")


def _radix_scheduler(disable_radix_cache):
    s = MagicMock()
    s.disable_radix_cache = disable_radix_cache
    tree = MagicMock()
    del tree.clear_storage_backend  # plain RadixCache has none
    s.tree_cache = tree
    s.req_to_token_pool = MagicMock()
    s.token_to_kv_pool_allocator = MagicMock()
    return s


@unittest.skipUnless(_HAS_ROLE_SWITCH, "role_switch not importable in this env")
class TestReleasePrefixCacheOnRoleSwitch(unittest.TestCase):
    """The flip may run with radix cache ENABLED: teardown resets the tree cache
    + KV pools when radix is on, and is a no-op on the radix-disabled path."""

    def test_noop_when_radix_disabled(self):
        s = _radix_scheduler(disable_radix_cache=True)
        _release_prefix_cache_for_role_switch(s)
        s.tree_cache.reset.assert_not_called()
        s.token_to_kv_pool_allocator.clear.assert_not_called()

    def test_releases_when_radix_enabled(self):
        s = _radix_scheduler(disable_radix_cache=False)
        _release_prefix_cache_for_role_switch(s)
        s.tree_cache.reset.assert_called_once_with()
        s.req_to_token_pool.clear.assert_called_once_with()
        s.token_to_kv_pool_allocator.clear.assert_called_once_with()

    def test_teardown_invokes_release(self):
        s = _radix_scheduler(disable_radix_cache=False)
        s.disaggregation_mode = DisaggregationMode.PREFILL
        s.disagg_prefill_bootstrap_queue = None  # no queue -> skip km.teardown()
        teardown_disaggregation(s)
        self.assertIsNone(s.disagg_metadata_buffers)
        s.tree_cache.reset.assert_called_once_with()


class TestStartupDecodeGraphReuse(unittest.TestCase):
    def test_switch_reuses_startup_graph_or_captures_when_missing(self):
        from sglang.srt.managers.tp_worker import TpModelWorker
        from sglang.srt.model_executor import model_runner as mr

        for graph in (SimpleNamespace(capture_bs=[1, 2, 4]), None):
            with self.subTest(captured_at_startup=graph is not None):
                rc.reset_context()
                try:
                    rc.get_context().set_server_args(ServerArgs(model_path="dummy"))
                    runner = mr.ModelRunner.__new__(mr.ModelRunner)
                    runner.model = MagicMock()
                    runner.decode_cuda_graph_captured = False
                    runner.decode_cuda_graph_capture_bs = []
                    runner.init_decode_cuda_graph = MagicMock()
                    capture = SimpleNamespace(
                        eager_runner=MagicMock(),
                        prefill=SimpleNamespace(runner=None),
                        decode=SimpleNamespace(runner=graph),
                        memory_usage=None,
                        time_usage=None,
                    )
                    with (
                        patch.object(mr, "capture_cuda_graphs", return_value=capture),
                        patch.object(mr, "prebuild_deepep_v2_buffer"),
                    ):
                        runner.init_cuda_graphs()
                    worker = SimpleNamespace(model_runner=runner)
                    self.assertEqual(
                        TpModelWorker.get_decode_cuda_graph_bs(worker),
                        [1, 2, 4] if graph is not None else [],
                    )
                    runner.ensure_decode_cuda_graphs()
                    if graph is not None:
                        runner.init_decode_cuda_graph.assert_not_called()
                        self.assertIs(runner.decode_cuda_graph_runner, graph)
                    else:
                        runner.init_decode_cuda_graph.assert_called_once_with()
                finally:
                    rc.reset_context()


if __name__ == "__main__":
    unittest.main()
