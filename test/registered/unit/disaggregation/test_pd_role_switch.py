import argparse
import concurrent.futures
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt import runtime_context as rc  # noqa: E402
from sglang.srt.disaggregation import role_switch  # noqa: E402
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    PdRoleSwitchReqInput,
    PdRoleSwitchReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestPdRoleSwitchServerArg(unittest.TestCase):
    def test_cli_flag_parses(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        off = parser.parse_args(["--model-path", "dummy"])
        self.assertFalse(off.enable_pd_role_switch)

        on = parser.parse_args(["--model-path", "dummy", "--enable-pd-role-switch"])
        self.assertTrue(on.enable_pd_role_switch)


class TestHandlePdRoleSwitch(unittest.TestCase):
    """Cover the control-plane contract of Scheduler.handle_pd_role_switch.

    Only the role-flip *decision* logic is exercised here (no GPU): the heavy
    teardown/rebuild is mocked, so this asserts the guard branches and the
    orchestration order without standing up a model.
    """

    def setUp(self):
        rc.reset_context()

    def tearDown(self):
        rc.reset_context()

    def _scheduler(self, mode, *, enable=True, idle=True):
        s = Scheduler.__new__(Scheduler)
        s.disaggregation_mode = mode
        sa = ServerArgs(
            model_path="dummy",
            disaggregation_mode=mode.value,
            enable_pd_role_switch=enable,
        )
        rc.get_context().set_server_args(sa)
        s.server_args = sa
        s.is_fully_idle = MagicMock(return_value=idle)
        teardown_patcher = patch.object(role_switch, "teardown_disaggregation")
        s.teardown_disaggregation = teardown_patcher.start()
        self.addCleanup(teardown_patcher.stop)
        s.init_disaggregation = MagicMock()
        s._event_loop_should_restart = False
        s._pd_role_switch_in_progress = False
        s._pd_role_switch_unhealthy = False
        s.tp_worker = MagicMock()
        return s

    def test_rejected_when_flag_disabled(self):
        s = self._scheduler(DisaggregationMode.PREFILL, enable=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertIsInstance(out, PdRoleSwitchReqOutput)
        self.assertFalse(out.success)
        self.assertIn("enable-pd-role-switch", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_on_invalid_role(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(s, PdRoleSwitchReqInput(new_role="both"))
        self.assertFalse(out.success)
        self.assertIn("invalid new_role", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_when_not_in_pd_mode(self):
        s = self._scheduler(DisaggregationMode.NULL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("not running in PD", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_same_role_is_noop(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="prefill")
        )
        self.assertTrue(out.success)
        self.assertEqual(out.message, "already in target role")
        s.teardown_disaggregation.assert_not_called()
        s.init_disaggregation.assert_not_called()
        self.assertFalse(s._event_loop_should_restart)

    def test_rejected_when_not_idle(self):
        s = self._scheduler(DisaggregationMode.PREFILL, idle=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("not idle", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_successful_flip_orchestration(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        self.assertTrue(out.success)
        self.assertEqual(out.old_role, "prefill")
        self.assertEqual(out.new_role, "decode")
        # Orchestration: drain -> teardown -> flip config bag -> rebuild -> signal.
        s.teardown_disaggregation.assert_called_once_with(s)
        self.assertEqual(rc.get_disagg().disaggregation_mode, "decode")
        # The pristine startup record is never mutated.
        self.assertEqual(s.server_args.disaggregation_mode, "prefill")
        s.init_disaggregation.assert_called_once()
        self.assertTrue(s._event_loop_should_restart)
        # Flip to decode ensures decode CUDA graphs exist (idempotent capture).
        s.tp_worker.ensure_decode_cuda_graphs.assert_called_once()
        # The in-progress guard is released after a successful flip.
        self.assertFalse(s._pd_role_switch_in_progress)

    def test_flip_to_prefill_skips_decode_graph_capture(self):
        s = self._scheduler(DisaggregationMode.DECODE)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="prefill")
        )
        self.assertTrue(out.success)
        self.assertEqual(out.new_role, "prefill")
        s.init_disaggregation.assert_called_once()
        # Flipping to prefill must not capture decode graphs.
        s.tp_worker.ensure_decode_cuda_graphs.assert_not_called()
        self.assertTrue(s._event_loop_should_restart)

    def test_rejected_when_switch_in_progress(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s._pd_role_switch_in_progress = True
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("in progress", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_when_unhealthy(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s._pd_role_switch_unhealthy = True
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("unhealthy", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rebuild_failure_marks_unhealthy_and_notifies(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        # Rebuild of the new role fails after the old role was torn down.
        s.init_disaggregation = MagicMock(side_effect=RuntimeError("boom"))

        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        # Fail loud (notify), mark unhealthy, no in-place rollback attempt.
        self.assertFalse(out.success)
        self.assertIn("unhealthy", out.message)
        self.assertIn("restart", out.message)
        self.assertTrue(s._pd_role_switch_unhealthy)
        self.assertFalse(s._event_loop_should_restart)
        self.assertFalse(s._pd_role_switch_in_progress)
        # Teardown + rebuild attempted exactly once (no rollback).
        self.assertEqual(s.teardown_disaggregation.call_count, 1)
        self.assertEqual(s.init_disaggregation.call_count, 1)
        # A subsequent switch is rejected because the instance is unhealthy.
        out2 = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="prefill")
        )
        self.assertFalse(out2.success)
        self.assertIn("unhealthy", out2.message)

    def test_teardown_failure_marks_unhealthy(self):
        """Teardown, the role flip and rebuild are one atomic step: a failure
        during teardown (not only rebuild) must also mark the instance unhealthy
        and must not proceed to rebuild."""
        s = self._scheduler(DisaggregationMode.PREFILL)
        s.teardown_disaggregation.side_effect = RuntimeError("boom")

        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        self.assertFalse(out.success)
        self.assertIn("unhealthy", out.message)
        self.assertIn("restart", out.message)
        self.assertTrue(s._pd_role_switch_unhealthy)
        self.assertFalse(s._event_loop_should_restart)
        self.assertFalse(s._pd_role_switch_in_progress)
        # Teardown raised, so rebuild is never attempted.
        self.assertEqual(s.teardown_disaggregation.call_count, 1)
        s.init_disaggregation.assert_not_called()


class TestPdRoleSwitchReqSerialization(unittest.TestCase):
    """Guard the wire contract of the /pd_role_switch req/resp structs.

    These caught real breakages when upstream moved BaseReq to msgspec: the
    request must accept an optional decode_cuda_graph_bs body field, and the
    response must be encodable for the HTTP layer (msgspec_to_builtins).
    """

    def test_req_accepts_optional_decode_cuda_graph_bs(self):
        req = PdRoleSwitchReqInput(new_role="decode", decode_cuda_graph_bs=[1, 2, 4])
        self.assertEqual(req.new_role, "decode")
        self.assertEqual(req.decode_cuda_graph_bs, [1, 2, 4])
        # Field is optional and defaults to None.
        self.assertIsNone(PdRoleSwitchReqInput(new_role="prefill").decode_cuda_graph_bs)

    def test_resp_is_json_encodable(self):
        from sglang.srt.utils.msgspec_utils import msgspec_to_builtins

        out = PdRoleSwitchReqOutput(
            success=True, message="ok", old_role="prefill", new_role="decode"
        )
        d = msgspec_to_builtins(out)
        self.assertEqual(d["success"], True)
        self.assertEqual(d["old_role"], "prefill")
        self.assertEqual(d["new_role"], "decode")
        self.assertEqual(d["message"], "ok")


class TestPdRoleSwitchStartupValidation(unittest.TestCase):
    """--enable-pd-role-switch only rebuilds the small role-specific disagg
    structures on a flip; the per-role buffers of DP attention / EP / MoE
    all-to-all / pipeline parallelism are sized at startup and not rebuilt, so
    a flip with those on would silently deadlock. The PD arg hook must reject
    the combination up-front instead of failing at flip time."""

    def _sa(self, **kw):
        base = dict(
            disaggregation_transfer_backend="mori",
            disaggregation_mode="prefill",
            enable_pd_role_switch=True,
            enable_dp_attention=False,
            ep_size=1,
            moe_a2a_backend="none",
            pp_size=1,
            dp_size=1,
            dcp_size=1,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def _run(self, sa):
        from sglang.srt.arg_groups.pd_disaggregation_hook import (
            handle_pd_disaggregation,
        )

        handle_pd_disaggregation(sa)

    def test_pure_tp_role_switch_accepted(self):
        # No raise for the validated pure-TP configuration.
        self._run(self._sa())

    def test_reject_dp_attention(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(enable_dp_attention=True))
        self.assertIn("DP attention", str(ctx.exception))

    def test_reject_expert_parallelism(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(ep_size=8))
        self.assertIn("expert parallelism", str(ctx.exception))

    def test_reject_moe_a2a(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(moe_a2a_backend="mori"))
        self.assertIn("MoE all-to-all", str(ctx.exception))

    def test_reject_pipeline_parallelism(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(pp_size=2))
        self.assertIn("pipeline parallelism", str(ctx.exception))

    def test_reject_data_parallelism(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(dp_size=2))
        self.assertIn("data parallelism", str(ctx.exception))

    def test_no_role_switch_is_unaffected(self):
        # The same unsupported feature is fine when role switch is off.
        self._run(self._sa(enable_pd_role_switch=False, moe_a2a_backend="mori"))


# --- teardown: transfer-worker thread-leak fix + prefix-cache release (radix ON) ---
import threading  # noqa: E402
import time  # noqa: E402

import zmq  # noqa: E402

try:
    from sglang.srt.disaggregation.common.utils import FastQueue  # noqa: E402
    from sglang.srt.disaggregation.mori.conn import MoriKVManager  # noqa: E402

    _HAS_MORI = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_MORI = False

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


@unittest.skipUnless(_HAS_MORI, "mori not importable in this environment")
class TestMoriTeardownNoThreadLeak(unittest.TestCase):
    """teardown() must stop+join the transfer workers it started, so a P->D->P
    flip loop does not leak _num_shards transfer threads per cycle."""

    def test_teardown_joins_transfer_workers(self):
        m = MoriKVManager.__new__(MoriKVManager)
        m.disaggregation_mode = DisaggregationMode.PREFILL
        m._stopped = False
        m._worker_threads = []
        m._transfer_queues = [FastQueue() for _ in range(3)]
        m.server_socket = MagicMock()
        m._zmq_ctx = MagicMock()
        m.engine = MagicMock()
        m.kv_mem_descs = m.aux_mem_descs = m.state_mem_descs = []
        for q in m._transfer_queues:
            t = threading.Thread(target=m._transfer_worker, args=(q,), daemon=True)
            t.start()
            m._worker_threads.append(t)
        started = list(m._worker_threads)
        time.sleep(0.05)  # let workers park in FastQueue.get()
        for t in started:
            self.assertTrue(t.is_alive())

        MoriKVManager.teardown(m)

        for t in started:
            self.assertFalse(t.is_alive(), "transfer worker survived teardown (leak)")
        self.assertEqual(m._worker_threads, [])
        self.assertEqual(m._transfer_queues, [])


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
        m.kv_args = SimpleNamespace(
            kv_data_ptrs=[], aux_data_ptrs=[], state_data_ptrs=[]
        )
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
    + KV pools when radix is on, and is a no-op on the historical chunk-cache path."""

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


if __name__ == "__main__":
    unittest.main()
