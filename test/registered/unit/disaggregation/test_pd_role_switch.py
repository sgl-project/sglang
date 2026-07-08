import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

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

    def _scheduler(self, mode, *, enable=True, idle=True):
        s = Scheduler.__new__(Scheduler)
        s.disaggregation_mode = mode
        s.server_args = SimpleNamespace(
            enable_pd_role_switch=enable,
            disaggregation_mode=mode.value,
        )
        s.is_fully_idle = MagicMock(return_value=idle)
        s._teardown_disaggregation = MagicMock()
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
        s._teardown_disaggregation.assert_not_called()

    def test_rejected_on_invalid_role(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(s, PdRoleSwitchReqInput(new_role="both"))
        self.assertFalse(out.success)
        self.assertIn("invalid new_role", out.message)
        s._teardown_disaggregation.assert_not_called()

    def test_rejected_when_not_in_pd_mode(self):
        s = self._scheduler(DisaggregationMode.NULL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("not running in PD", out.message)
        s._teardown_disaggregation.assert_not_called()

    def test_same_role_is_noop(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="prefill")
        )
        self.assertTrue(out.success)
        self.assertEqual(out.message, "already in target role")
        s._teardown_disaggregation.assert_not_called()
        s.init_disaggregation.assert_not_called()
        self.assertFalse(s._event_loop_should_restart)

    def test_rejected_when_not_idle(self):
        s = self._scheduler(DisaggregationMode.PREFILL, idle=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("not idle", out.message)
        s._teardown_disaggregation.assert_not_called()

    def test_successful_flip_orchestration(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        self.assertTrue(out.success)
        self.assertEqual(out.old_role, "prefill")
        self.assertEqual(out.new_role, "decode")
        # Orchestration: drain -> teardown -> flip server arg -> rebuild -> signal.
        s._teardown_disaggregation.assert_called_once()
        self.assertEqual(s.server_args.disaggregation_mode, "decode")
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
        s._teardown_disaggregation.assert_not_called()

    def test_rejected_when_unhealthy(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s._pd_role_switch_unhealthy = True
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("unhealthy", out.message)
        s._teardown_disaggregation.assert_not_called()

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
        self.assertEqual(s._teardown_disaggregation.call_count, 1)
        self.assertEqual(s.init_disaggregation.call_count, 1)
        # A subsequent switch is rejected because the instance is unhealthy.
        out2 = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="prefill")
        )
        self.assertFalse(out2.success)
        self.assertIn("unhealthy", out2.message)


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


# --- teardown: transfer-worker thread-leak fix + prefix-cache release (radix ON) ---
import threading  # noqa: E402
import time  # noqa: E402

try:
    from sglang.srt.disaggregation.common.utils import FastQueue  # noqa: E402
    from sglang.srt.disaggregation.mori.conn import MoriKVManager  # noqa: E402

    _HAS_MORI = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_MORI = False

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
