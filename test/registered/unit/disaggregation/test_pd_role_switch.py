import concurrent.futures
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import torch

from sglang.srt import runtime_context as rc  # noqa: E402
from sglang.srt.disaggregation import role_switch  # noqa: E402
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.disaggregation.common.staging_handler import (
    DecodeStagingHandler,
    PrefillStagingContext,
    handle_watermark_msg,
    is_watermark_ready,
    prefetch_staging_reqs,
)
from sglang.srt.disaggregation.common.utils import FastQueue
from sglang.srt.disaggregation.decode import DecodeTransferQueue
from sglang.srt.disaggregation.mooncake.staging_resource import MooncakeStagingResource
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    PdRoleSwitchReqInput,
    PdRoleSwitchReqOutput,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


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
        s._sync_disaggregation_mode_to_subcomponents = MagicMock()
        s._event_loop_should_restart = False
        s._pd_role_switch_in_progress = False
        s._pd_role_switch_unhealthy = False
        s.tp_worker = MagicMock()
        return s

    def test_preflight_checks_graph_headroom_without_releasing_resources(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s.device = "cuda"
        cases = (
            ("missing", None, [], 2.0, False, "is required"),
            ("negative", -1.0, [], 2.0, False, "non-negative"),
            ("insufficient", 1.0, [], 0.5, False, "insufficient"),
            (
                "probe failure",
                1.0,
                [],
                RuntimeError("probe failed"),
                False,
                "failed to check",
            ),
            ("sufficient", 1.0, [], 1.0, True, "ready"),
            ("already captured", None, [1], 0.0, True, "ready"),
        )
        for name, budget, captured_bs, memory, success, message in cases:
            with self.subTest(name=name):
                s.tp_worker.get_decode_cuda_graph_bs.return_value = captured_bs
                with patch.object(
                    role_switch, "get_available_gpu_memory"
                ) as memory_probe:
                    if isinstance(memory, Exception):
                        memory_probe.side_effect = memory
                    else:
                        memory_probe.return_value = memory
                    out = Scheduler.handle_pd_role_switch(
                        s,
                        PdRoleSwitchReqInput(
                            new_role="decode",
                            prepare_only=True,
                            decode_cuda_graph_memory_gb=budget,
                        ),
                    )
                    self.assertEqual(out.success, success)
                    self.assertTrue(out.safe_to_restore)
                    self.assertIn(message, out.message)
                    if not captured_bs and budget is not None and budget >= 0:
                        memory_probe.assert_called_once()
                    else:
                        memory_probe.assert_not_called()
                s.tp_worker.ensure_decode_cuda_graphs.assert_not_called()
                s.teardown_disaggregation.assert_not_called()
                s.init_disaggregation.assert_not_called()
                self.assertEqual(rc.get_disagg().disaggregation_mode, "prefill")
                self.assertFalse(s._event_loop_should_restart)

    def test_rejected_when_flag_disabled(self):
        s = self._scheduler(DisaggregationMode.PREFILL, enable=False)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertIsInstance(out, PdRoleSwitchReqOutput)
        self.assertFalse(out.success)
        self.assertTrue(out.safe_to_restore)
        self.assertIn("enable-pd-role-switch", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_on_invalid_role(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        out = Scheduler.handle_pd_role_switch(s, PdRoleSwitchReqInput(new_role="both"))
        self.assertFalse(out.success)
        self.assertTrue(out.safe_to_restore)
        self.assertIn("invalid new_role", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_when_not_in_pd_mode(self):
        s = self._scheduler(DisaggregationMode.NULL)
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertTrue(out.safe_to_restore)
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
        self.assertTrue(out.safe_to_restore)
        self.assertIn("not idle", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_when_decode_graph_headroom_is_missing(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s.tp_worker.get_decode_cuda_graph_bs.return_value = []
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )

        self.assertFalse(out.success)
        self.assertTrue(out.safe_to_restore)
        self.assertIn("decode_cuda_graph_memory_gb is required", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_when_decode_graph_headroom_is_insufficient(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s.device = "cuda"
        s.tp_worker.get_decode_cuda_graph_bs.return_value = []
        with patch.object(role_switch, "get_available_gpu_memory", return_value=0.5):
            out = Scheduler.handle_pd_role_switch(
                s,
                PdRoleSwitchReqInput(
                    new_role="decode",
                    decode_cuda_graph_memory_gb=1.0,
                ),
            )

        self.assertFalse(out.success)
        self.assertTrue(out.safe_to_restore)
        self.assertIn("insufficient decode CUDA graph headroom", out.message)
        s.teardown_disaggregation.assert_not_called()
        self.assertEqual(rc.get_disagg().disaggregation_mode, "prefill")

    def test_decode_graph_headroom_allows_flip(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s.device = "cuda"
        s.tp_worker.get_decode_cuda_graph_bs.return_value = []
        with patch.object(role_switch, "get_available_gpu_memory", return_value=1.0):
            out = Scheduler.handle_pd_role_switch(
                s,
                PdRoleSwitchReqInput(
                    new_role="decode",
                    decode_cuda_graph_memory_gb=1.0,
                ),
            )

        self.assertTrue(out.success)
        s.teardown_disaggregation.assert_called_once_with(s)

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
        s._sync_disaggregation_mode_to_subcomponents.assert_called_once()
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
        self.assertFalse(out.safe_to_restore)
        self.assertIn("in progress", out.message)
        s.teardown_disaggregation.assert_not_called()

    def test_rejected_when_unhealthy(self):
        s = self._scheduler(DisaggregationMode.PREFILL)
        s._pd_role_switch_unhealthy = True
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertFalse(out.safe_to_restore)
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
        self.assertFalse(out.safe_to_restore)
        self.assertIn("unhealthy", out.message)
        self.assertIn("restart", out.message)
        self.assertTrue(s._pd_role_switch_unhealthy)
        self.assertFalse(s._event_loop_should_restart)
        self.assertFalse(s._pd_role_switch_in_progress)
        # Teardown + rebuild attempted exactly once (no rollback).
        self.assertEqual(s.teardown_disaggregation.call_count, 1)
        self.assertEqual(s.init_disaggregation.call_count, 1)
        s._sync_disaggregation_mode_to_subcomponents.assert_not_called()
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
        self.assertFalse(out.safe_to_restore)
        self.assertIn("unhealthy", out.message)
        self.assertIn("restart", out.message)
        self.assertTrue(s._pd_role_switch_unhealthy)
        self.assertFalse(s._event_loop_should_restart)
        self.assertFalse(s._pd_role_switch_in_progress)
        # Teardown raised, so rebuild is never attempted.
        self.assertEqual(s.teardown_disaggregation.call_count, 1)
        s.init_disaggregation.assert_not_called()
        s._sync_disaggregation_mode_to_subcomponents.assert_not_called()

    def _shared_scheduler(self, mode=DisaggregationMode.PREFILL):
        s = self._scheduler(mode)
        rc.get_context().override(
            "test.shared", disaggregation_transfer_backend="mooncake"
        )
        resource = SimpleNamespace(unsafe_reason=None, poison=MagicMock())
        manager = SimpleNamespace(
            supports_role_switch=True,
            enable_staging=True,
            staging_resource=resource,
            check_staging_role_switch_ready=MagicMock(return_value=None),
        )
        s._pd_staging_resource = resource
        name = (
            "disagg_prefill_bootstrap_queue"
            if mode == DisaggregationMode.PREFILL
            else "disagg_decode_prealloc_queue"
        )
        setattr(s, name, SimpleNamespace(kv_manager=manager))
        return s, manager

    def test_shared_staging_flips_in_both_directions(self):
        for old, new in (
            (DisaggregationMode.PREFILL, "decode"),
            (DisaggregationMode.DECODE, "prefill"),
        ):
            for failure in (None, "teardown", "rebuild"):
                with self.subTest(old=old, failure=failure):
                    s, manager = self._shared_scheduler(old)
                    parallel = SimpleNamespace(
                        pp_size=1,
                        attn_tp_size=2,
                        attn_tp_group=SimpleNamespace(cpu_group=object()),
                    )
                    with (
                        patch.object(
                            role_switch, "get_parallel", return_value=parallel
                        ),
                        patch("torch.distributed.all_gather_object") as gather,
                    ):
                        ready = Scheduler.handle_pd_role_switch(
                            s, PdRoleSwitchReqInput(new_role=new, prepare_only=True)
                        )
                        self.assertTrue(ready.success, ready.message)
                        s.teardown_disaggregation.assert_not_called()
                        s.init_disaggregation.assert_not_called()
                        if failure == "teardown":
                            s.teardown_disaggregation.side_effect = RuntimeError("boom")
                        elif failure == "rebuild":
                            s.init_disaggregation.side_effect = RuntimeError("boom")
                        out = Scheduler.handle_pd_role_switch(
                            s, PdRoleSwitchReqInput(new_role=new)
                        )
                        gather.assert_not_called()
                    self.assertEqual(out.success, failure is None, out.message)
                    self.assertEqual(
                        manager.check_staging_role_switch_ready.call_count, 2
                    )
                    self.assertEqual(s._pd_role_switch_unhealthy, failure is not None)
                    self.assertFalse(s._pd_role_switch_in_progress)
                    self.assertEqual(s._event_loop_should_restart, failure is None)
                    s.teardown_disaggregation.assert_called_once()
                    self.assertEqual(
                        s.init_disaggregation.call_count, int(failure != "teardown")
                    )
                    if failure is not None:
                        self.assertFalse(out.safe_to_restore)
                        s._pd_staging_resource.poison.assert_called_once()

    def test_shared_staging_busy_rejected_before_teardown(self):
        s, manager = self._shared_scheduler()
        manager.check_staging_role_switch_ready.return_value = (
            "staging allocations are still active"
        )
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertTrue(out.safe_to_restore)
        s.teardown_disaggregation.assert_not_called()

    def test_shared_staging_uncertain_drain_requires_restart(self):
        s, manager = self._shared_scheduler()
        s._pd_staging_resource.unsafe_reason = "missing ACK"
        manager.check_staging_role_switch_ready.return_value = (
            "restart required: missing ACK"
        )
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertFalse(out.safe_to_restore)
        s.teardown_disaggregation.assert_not_called()

    def test_other_backend_staging_remains_rejected(self):
        s, _ = self._shared_scheduler()
        rc.get_context().override("test.other", disaggregation_transfer_backend="nixl")
        out = Scheduler.handle_pd_role_switch(
            s, PdRoleSwitchReqInput(new_role="decode")
        )
        self.assertFalse(out.success)
        self.assertIn("staging", out.message)


class TestPdRoleSwitchReqSerialization(unittest.TestCase):
    """Guard the wire contract of the /pd_role_switch req/resp structs.

    These caught real breakages when upstream moved BaseReq to msgspec: the
    request must accept an optional decode_cuda_graph_bs body field, and the
    response must be encodable for the HTTP layer (msgspec_to_builtins).
    """

    def test_req_accepts_optional_decode_cuda_graph_bs(self):
        req = PdRoleSwitchReqInput(
            new_role="decode",
            decode_cuda_graph_bs=[1, 2, 4],
            decode_cuda_graph_memory_gb=1.25,
        )
        self.assertEqual(req.new_role, "decode")
        self.assertEqual(req.decode_cuda_graph_bs, [1, 2, 4])
        self.assertEqual(req.decode_cuda_graph_memory_gb, 1.25)
        # Field is optional and defaults to None.
        default_req = PdRoleSwitchReqInput(new_role="prefill")
        self.assertIsNone(default_req.decode_cuda_graph_bs)
        self.assertIsNone(default_req.decode_cuda_graph_memory_gb)

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
        self.assertEqual(d["safe_to_restore"], False)


class TestPdRoleSwitchStartupValidation(unittest.TestCase):
    """--enable-pd-role-switch only rebuilds the small role-specific disagg
    structures on a flip; the per-role buffers of EP / MoE
    all-to-all / pipeline parallelism are sized at startup and not rebuilt, so
    a flip with those on would silently deadlock. The PD arg hook must reject
    the combination up-front instead of failing at flip time."""

    def _sa(self, **kw):
        base = dict(
            disaggregation_transfer_backend="mori",
            disaggregation_mode="prefill",
            disaggregation_decode_host_receive_threshold=1.0,
            enable_pd_role_switch=True,
            attn_dp_size=1,
            ep_join_mode=None,
            ep_size=1,
            moe_a2a_backend="none",
            pp_size=1,
            dp_size=1,
            dcp_size=1,
            speculative_algorithm=None,
        )
        base.update(kw)
        # Keep the production defaults for fields read by the shared PD hook.
        return ServerArgs(model_path="dummy", **base)

    def _run(self, sa):
        from sglang.srt.arg_groups.pd_disaggregation_hook import (
            handle_pd_disaggregation,
        )

        handle_pd_disaggregation(sa)

    def test_pure_tp_role_switch_accepted(self):
        # No raise for the validated pure-TP configuration.
        self._run(self._sa())

    def test_dp_attention_role_switch_accepted(self):
        self._run(self._sa(tp_size=2, attn_dp_size=2))

    def test_reject_dp_attention_local_control_broadcast(self):
        with self.assertRaisesRegex(ValueError, "local control broadcast"):
            self._run(
                self._sa(
                    tp_size=2,
                    attn_dp_size=2,
                    enable_dp_attention_local_control_broadcast=True,
                )
            )

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

    def test_data_parallel_role_switch_accepted(self):
        self._run(self._sa(dp_size=2))

    def test_reject_context_parallelism(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(dcp_size=2))
        self.assertIn("decode context parallelism", str(ctx.exception))

        for mode in ("prefill", "decode"):
            for staging in ("0", "1"):
                with (
                    self.subTest(mode=mode, staging=staging),
                    patch.dict("os.environ", {"SGLANG_DISAGG_STAGING_BUFFER": staging}),
                    self.assertRaisesRegex(ValueError, "attention context parallelism"),
                ):
                    self._run(
                        self._sa(
                            disaggregation_mode=mode,
                            disaggregation_transfer_backend="mooncake",
                            tp_size=2,
                            attn_cp_size=2,
                        )
                    )

    def test_reject_speculative_decoding(self):
        with self.assertRaises(ValueError) as ctx:
            self._run(self._sa(speculative_algorithm="EAGLE"))
        self.assertIn("speculative decoding", str(ctx.exception))

    def test_no_role_switch_is_unaffected(self):
        # The same unsupported feature is fine when role switch is off.
        self._run(self._sa(enable_pd_role_switch=False, moe_a2a_backend="mori"))

        for mode in ("prefill", "decode"):
            for staging in ("0", "1"):
                with (
                    self.subTest(mode=mode, staging=staging),
                    patch.dict("os.environ", {"SGLANG_DISAGG_STAGING_BUFFER": staging}),
                ):
                    self._run(
                        self._sa(
                            disaggregation_mode=mode,
                            disaggregation_transfer_backend="mooncake",
                            enable_pd_role_switch=False,
                            tp_size=2,
                            attn_cp_size=2,
                        )
                    )


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


class TestDPRoleSwitch(unittest.IsolatedAsyncioTestCase):
    async def test_all_workers_must_be_ready_before_switching(self):
        for dp_size, attn_dp_size in ((2, 1), (1, 2)):
            with self.subTest(dp_size=dp_size, attn_dp_size=attn_dp_size):
                await self._check_preflight(dp_size, attn_dp_size)

    async def test_insufficient_headroom_prevents_switching_all_workers(self):
        for dp_size, attn_dp_size in ((2, 1), (1, 2)):
            with self.subTest(dp_size=dp_size, attn_dp_size=attn_dp_size):
                await self._check_preflight(
                    dp_size, attn_dp_size, insufficient_memory=True
                )

    async def _check_preflight(self, dp_size, attn_dp_size, insufficient_memory=False):
        from sglang.srt.managers import tokenizer_control_mixin as control

        fixture = TestHandlePdRoleSwitch()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        self.addCleanup(fixture.doCleanups)
        schedulers = [fixture._scheduler(DisaggregationMode.PREFILL) for _ in range(2)]
        available_memory = [2.0, 2.0]
        for scheduler in schedulers:
            scheduler.device = "cuda"
            scheduler.tp_worker.get_decode_cuda_graph_bs.return_value = (
                [] if insufficient_memory else [1]
            )
            scheduler.init_disaggregation.side_effect = lambda s=scheduler: setattr(
                s,
                "disaggregation_mode",
                DisaggregationMode(rc.get_disagg().disaggregation_mode),
            )

        async def fanout(req):
            results = []
            for index, scheduler in enumerate(schedulers):
                rc.get_context().set_server_args(scheduler.server_args)
                rc.get_context().override(
                    "test.role", disaggregation_mode=scheduler.disaggregation_mode.value
                )
                with patch.object(
                    role_switch,
                    "get_available_gpu_memory",
                    return_value=available_memory[index],
                ):
                    results.append(role_switch.handle_pd_role_switch(scheduler, req))
            if req.prepare_only:
                for scheduler in schedulers:
                    scheduler.init_disaggregation.assert_not_called()
            return results

        manager = SimpleNamespace(
            server_args=SimpleNamespace(enable_pd_role_switch=True),
            auto_create_handle_loop=MagicMock(),
            pd_role_switch_communicator=AsyncMock(side_effect=fanout),
            record_config_updates=MagicMock(),
            disaggregation_mode=DisaggregationMode.PREFILL,
        )
        req = PdRoleSwitchReqInput(
            new_role="decode",
            decode_cuda_graph_memory_gb=1.0 if insufficient_memory else None,
        )
        with patch.object(
            control,
            "get_parallel",
            return_value=SimpleNamespace(
                dp_size=dp_size,
                attn_dp_size=attn_dp_size,
                num_dp_ranks=dp_size * attn_dp_size,
            ),
        ):
            if insufficient_memory:
                available_memory[1] = 0.5
            else:
                schedulers[1].is_fully_idle.return_value = False
            out = await control.TokenizerControlMixin.pd_role_switch(manager, req)
            self.assertFalse(out.success)
            self.assertTrue(out.safe_to_restore)
            self.assertIn(
                "insufficient" if insufficient_memory else "not idle", out.message
            )
            manager.pd_role_switch_communicator.assert_awaited_once()
            self.assertTrue(
                manager.pd_role_switch_communicator.call_args.args[0].prepare_only
            )
            manager.record_config_updates.assert_not_called()
            for scheduler in schedulers:
                scheduler.init_disaggregation.assert_not_called()
                scheduler.teardown_disaggregation.assert_not_called()
                self.assertEqual(
                    scheduler.disaggregation_mode, DisaggregationMode.PREFILL
                )

            available_memory[1] = 2.0
            schedulers[1].is_fully_idle.return_value = True
            out = await control.TokenizerControlMixin.pd_role_switch(manager, req)
            self.assertTrue(out.success)
            self.assertEqual(manager.disaggregation_mode, DisaggregationMode.DECODE)
            for scheduler in schedulers:
                scheduler.init_disaggregation.assert_called_once()
                self.assertEqual(
                    scheduler.disaggregation_mode, DisaggregationMode.DECODE
                )


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


def cpu_buffer_init(buf, size_bytes, device, gpu_id, custom_mem_pool=None):
    buf.buffer = torch.empty(size_bytes, dtype=torch.uint8)
    buf.size_bytes = size_bytes
    buf.device = torch.device("cpu")
    buf.device_module = torch.get_device_module(buf.device)
    buf.gpu_id = gpu_id
    buf.data_ptr = buf.buffer.data_ptr()
    buf._gather_stream = None


class TestMooncakeStagingResource(unittest.TestCase):
    def setUp(self):
        self.pool = object()
        p = patch(
            "sglang.srt.disaggregation.mooncake.staging_resource._get_custom_mem_pool",
            return_value=(self.pool, "test"),
        )
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(
            StagingBuffer, "__init__", autospec=True, side_effect=cpu_buffer_init
        )
        self.alloc = p.start()
        self.addCleanup(p.stop)

    def resource(self, decode_bytes=4096, device_type="cuda"):
        engine = SimpleNamespace(
            batch_register=Mock(return_value=0), batch_deregister=Mock(return_value=0)
        )
        args = SimpleNamespace(gpu_id=0, page_size=1, kv_item_lens=[128, 128])
        r = MooncakeStagingResource(
            engine, args, 4, 2, decode_bytes, device_type=device_type
        )
        r.initialize()
        return r

    def test_views_capacity_and_registration_are_shared(self):
        for decode_bytes in (1024, 4096):
            with self.subTest(decode_bytes=decode_bytes):
                r = self.resource(decode_bytes)
                base = r.backing.get_ptr()
                self.assertEqual(r.capacity, max(2048, decode_bytes))
                self.assertEqual(r.decode_allocator.get_total_size(), decode_bytes)
                self.assertEqual(r.decode_allocator.get_base_ptr(), base)
                for i, view in enumerate(r.prefill_buffers):
                    self.assertEqual(view.get_ptr(), base + i * 1024)
                    self.assertEqual(view.get_size(), 1024)
                    self.assertEqual(view.device, view.buffer.device)
                    self.assertIs(
                        view.device_module, torch.get_device_module(view.buffer.device)
                    )
                r.prefill_buffers[0].buffer.fill_(7)
                self.assertTrue(torch.all(r.decode_allocator.buffer.buffer[:1024] == 7))
                self.assertIs(self.alloc.call_args.args[-1], self.pool)
                for role in ("prefill", "decode") * 10:
                    r.acquire(role)
                    r.release(role)
                r.engine.batch_register.assert_called_once_with([base], [r.capacity])
                r.engine.batch_deregister.assert_not_called()
                r.close()
                r.close()
                r.engine.batch_deregister.assert_called_once_with([base])

    def test_allocator_survives_flip_and_old_watermark_cannot_unlock_wrap(self):
        r = self.resource()
        r.acquire("decode")
        a = r.decode_allocator
        for _ in range(5):
            alloc_id, _, _ = a.assign(128)
            a.free(alloc_id)
        old_wm = a.get_watermark()
        next_id = a.next_alloc_id
        peer = PrefillStagingContext(remote_watermarks={"session": old_wm})
        r.release("decode")
        r.acquire("prefill")
        r.prefill_buffers[0].buffer.fill_(42)
        r.release("prefill")
        r.acquire("decode")
        self.assertIs(r.decode_allocator, a)
        self.assertEqual(a.get_watermark(), old_wm)
        self.assertEqual(a.next_alloc_id, next_id)
        first, _, _ = a.assign(a.total_size)
        second, _, second_round = a.assign(a.total_size)
        self.assertFalse(
            is_watermark_ready(peer, "session", second_round, a.total_size)
        )
        a.free(first)
        wm = a.get_watermark()
        handle_watermark_msg(
            peer, [b"WATERMARK", str(wm[0]).encode(), str(wm[1]).encode(), b"session"]
        )
        handle_watermark_msg(
            peer,
            [
                b"WATERMARK",
                str(old_wm[0]).encode(),
                str(old_wm[1]).encode(),
                b"session",
            ],
        )
        self.assertTrue(is_watermark_ready(peer, "session", second_round, a.total_size))
        self.assertEqual(peer.remote_watermarks["session"], wm)
        a.free(second)
        r.release("decode")
        r.close()

    def test_busy_or_poisoned_storage_cannot_change_owner(self):
        r = self.resource()
        r.acquire("decode")
        with self.assertRaises(RuntimeError):
            r.acquire("prefill")
        alloc_id, _, _ = r.decode_allocator.assign(128)
        with self.assertRaisesRegex(RuntimeError, "allocations"):
            r.release("decode")
        r.decode_allocator.free(alloc_id)
        r.poison("missing drain ACK")
        with self.assertRaisesRegex(RuntimeError, "restart required"):
            r.release("decode")
        with self.assertRaises(RuntimeError):
            r.close()
        self.assertIsNotNone(r.backing)
        r.engine.batch_deregister.assert_not_called()

    def test_deregistration_errors_keep_storage_owned(self):
        r = self.resource()
        r.engine.batch_deregister.return_value = -1
        with self.assertRaisesRegex(RuntimeError, "deregistration failed"):
            r.close()
        self.assertIsNotNone(r.backing)
        r.engine.batch_deregister.return_value = 0
        r.close()

    def test_layout_cannot_change_between_roles(self):
        args = SimpleNamespace(gpu_id=0, page_size=1, kv_item_lens=[128, 128])
        for device_type in ("cuda", "cuda:7", "xpu", "xpu:7"):
            with self.subTest(device_type=device_type):
                r = self.resource(device_type=device_type)
                bare_type = torch.device(device_type).type
                self.assertEqual(r.device, f"{bare_type}:0")
                self.assertEqual(self.alloc.call_args.args[2], r.device)
                r.validate_layout(args, 4, 2, 4096, device_type=bare_type)
                for chunk, size, device in (
                    (8, 4096, bare_type),
                    (4, 8192, bare_type),
                    (4, 4096, "cpu"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "layout changed"):
                        r.validate_layout(args, chunk, 2, size, device_type=device)
                r.close()

    def test_registration_errors_keep_storage_owned(self):
        r = MooncakeStagingResource(
            SimpleNamespace(
                batch_register=Mock(return_value=-1),
                batch_deregister=Mock(return_value=0),
            ),
            SimpleNamespace(gpu_id=0, page_size=1, kv_item_lens=[128]),
            4,
            2,
            1024,
            device_type="cuda",
        )
        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            r.initialize()
        self.assertIsNotNone(r.backing)
        with self.assertRaises(RuntimeError):
            r.acquire("decode")
        r.close()

    def test_stream_finishes_before_role_is_released(self):
        r = self.resource()
        observed_roles = []
        r.prefill_buffers[0]._gather_stream = SimpleNamespace(
            synchronize=lambda: observed_roles.append(r.active_role)
        )
        r.acquire("prefill")
        r.release("prefill")
        self.assertEqual(observed_roles, ["prefill"])
        r.acquire("decode")
        r.release("decode")
        r.close()
        self.assertEqual(observed_roles, ["prefill", "decode", None])

    def manager(self, r):
        m = MooncakeKVManager.__new__(MooncakeKVManager)
        m.staging_resource = r
        m.disaggregation_mode = DisaggregationMode.PREFILL
        m._stopped = False
        m._staging_outstanding = {}
        m._staging_ctx = PrefillStagingContext(buffers=r.prefill_buffers)
        m._worker_threads = []
        m.transfer_queues = []
        m.executors = []
        m._socket_lock = threading.Lock()
        m._socket_cache = {}
        m._monitor_cache = {}
        m.server_socket = Mock()
        m._zmq_ctx = Mock()
        m.deregister_buffer_to_engine = Mock()
        r.acquire("prefill")
        return m

    def test_executor_shutdown_failure_keeps_resource_borrowed(self):
        r = self.resource()
        m = self.manager(r)
        executor = Mock()
        executor.shutdown.side_effect = RuntimeError("executor failed")
        m.executors = [executor]
        with self.assertRaisesRegex(RuntimeError, "executor failed"):
            m.teardown()
        executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
        self.assertEqual(r.active_role, "prefill")
        self.assertIn("executor shutdown failed", r.unsafe_reason)
        self.assertIsNotNone(r.backing)
        m.deregister_buffer_to_engine.assert_not_called()

    def test_manager_teardown_releases_resource_for_next_role(self):
        r = self.resource()
        m = self.manager(r)
        m.teardown()
        r.engine.batch_deregister.assert_not_called()
        r.acquire("decode")
        self.assertEqual(r.active_role, "decode")
        r.release("decode")
        r.close()

    def test_unstopped_worker_keeps_resource_borrowed(self):
        r = self.resource()
        m = self.manager(r)
        m._worker_threads = [Mock(is_alive=Mock(return_value=True))]
        with self.assertRaisesRegex(RuntimeError, "did not stop"):
            m.teardown()
        self.assertEqual(r.active_role, "prefill")
        m.deregister_buffer_to_engine.assert_not_called()

    def test_pending_queue_is_not_idle(self):
        r = self.resource()
        m = self.manager(r)
        q = FastQueue()
        q.put(object())
        m.transfer_queues = [q]
        self.assertIn("queues", m.check_staging_role_switch_ready())
        q.get()
        m.transfer_queues = []
        m.teardown()
        r.close()

    def test_prefetch_socket_and_its_private_context_are_closed(self):
        r = self.resource()
        m = self.manager(r)
        ctx = zmq.Context()
        sock = ctx.socket(zmq.PUSH)
        sock.bind("inproc://shared-staging-prefetch")
        m._staging_ctx.prefetch_sockets["peer"] = sock
        endpoints = m._staging_ctx.prefetch_endpoints
        endpoints["peer"] = "inproc://shared-staging-prefetch"
        m.teardown()
        self.assertEqual(endpoints, {})
        self.assertTrue(sock.closed)
        self.assertTrue(ctx.closed)
        self.assertIsNone(r.active_role)
        r.close()

    def test_decode_endpoint_change_replaces_prefetch_socket(self):
        ctx = zmq.Context()
        listeners = [ctx.socket(zmq.PULL), ctx.socket(zmq.PULL)]
        ports = [s.bind_to_random_port("tcp://127.0.0.1") for s in listeners]
        sockets, requested, endpoints = {}, set(), {}
        try:
            for room, listener_index in enumerate((0, 0, 1)):
                port = ports[listener_index]
                info = SimpleNamespace(
                    is_dummy=False,
                    dst_kv_indices=[0],
                    endpoint="localhost",
                    dst_port=port,
                )
                old = sockets.get("same-session")
                old_ctx = old.context if old is not None else None
                with patch.object(
                    zmq.Socket,
                    "getsockopt_string",
                    side_effect=AssertionError("must not query LAST_ENDPOINT"),
                ):
                    prefetch_staging_reqs(
                        room,
                        {room: {"same-session": info}},
                        {"page_size": 1},
                        1,
                        requested,
                        sockets,
                        session_endpoints=endpoints,
                    )
                self.assertTrue(listeners[listener_index].poll(2000))
                self.assertEqual(
                    listeners[listener_index].recv_multipart()[0], b"STAGING_REQ"
                )
                self.assertEqual(len(sockets), 1)
                self.assertEqual(endpoints["same-session"], f"tcp://localhost:{port}")
                if room == 1:
                    self.assertIs(sockets["same-session"], old)
                    self.assertFalse(old.closed)
                elif room == 2:
                    self.assertTrue(old.closed)
                    self.assertTrue(old_ctx.closed)
        finally:
            for sock in sockets.values():
                owner = sock.context
                sock.close(linger=0)
                owner.term()
            for sock in listeners:
                sock.close(linger=0)
            ctx.term()

    def test_handler_close_detaches_without_resetting_allocator(self):
        r = self.resource()
        handler = DecodeStagingHandler.__new__(DecodeStagingHandler)
        handler._room_to_decode_req = {7: object()}
        handler._room_to_receiver = {7: object()}
        handler._writer_counts = {7: {0: [(0, 1, "writer")]}}
        self.assertIn("rooms", handler.check_idle())
        handler._room_to_decode_req.clear()
        self.assertIn("rooms", handler.check_idle())
        handler._room_to_receiver.clear()
        # An arrival after unregister can recreate a nonempty orphan entry.
        self.assertIsNone(handler.check_idle())
        handler._writer_counts[7].clear()
        self.assertIsNone(handler.check_idle())
        handler._writer_counts[7][0] = [(0, 1, "writer")]
        handler._wm_subscribers = {"peer": object()}
        handler.staging_allocator = r.decode_allocator
        handler.kv_manager = handler.scheduler = handler.kv_buffer_info = object()
        handler.close()
        self.assertIsNone(handler.kv_manager)
        self.assertIsNone(handler.scheduler)
        self.assertIsNone(handler.staging_allocator)
        self.assertEqual(handler._wm_subscribers, {})
        self.assertIsNotNone(r.decode_allocator)
        r.close()

    def test_deferred_release_requires_confirmed_drain_for_future_switches(self):
        for drained in (False, True):
            with self.subTest(drained=drained):
                r = self.resource()
                receiver = SimpleNamespace(
                    kv_mgr=SimpleNamespace(
                        staging_resource=r,
                        is_abort_release_safe=Mock(return_value=drained),
                    )
                )
                req = SimpleNamespace(
                    req=SimpleNamespace(bootstrap_room=7), kv_receiver=receiver
                )
                q = DecodeTransferQueue.__new__(DecodeTransferQueue)
                q._deferred_releases = [(req, 0, 0, 0, 1)]
                q.enable_host_receive = False
                q.deferred_kv_release_timeout = 1
                q._do_release = Mock()
                q._observe_deferred_kv_release = Mock()
                q.gloo_group = object()
                with patch("torch.distributed.get_world_size", return_value=1):
                    q.resolve_deferred_releases()
                q._do_release.assert_called_once_with(req, 0)
                self.assertEqual(r.unsafe_reason is None, drained)
                if drained:
                    r.acquire("prefill")
                    r.release("prefill")
                    r.close()

    def test_deferred_release_resource_lookup_failure_is_isolated(self):
        resource = SimpleNamespace(poison=Mock())
        manager = SimpleNamespace(
            staging_resource=resource, is_abort_release_safe=lambda *args: True
        )
        receiver = Mock()
        type(receiver).kv_mgr = PropertyMock(
            side_effect=[manager, RuntimeError("lookup failed")]
        )
        good = SimpleNamespace(
            kv_receiver=SimpleNamespace(kv_mgr=manager),
            req=SimpleNamespace(bootstrap_room=1),
        )
        bad = SimpleNamespace(
            kv_receiver=receiver, req=SimpleNamespace(bootstrap_room=2)
        )
        q = DecodeTransferQueue.__new__(DecodeTransferQueue)
        q._deferred_releases = [
            (good, 0, 0, 0, 1),
            (bad, 0, 0, 1, 1),
            (good, 0, 0, 2, 1),
        ]
        q.enable_host_receive = False
        q._do_release = Mock()
        q._observe_deferred_kv_release = Mock()
        q.gloo_group = object()
        with patch("torch.distributed.get_world_size", return_value=1):
            q.resolve_deferred_releases()
        self.assertEqual(q._do_release.call_count, 2)
        self.assertEqual([c.args[1] for c in q._do_release.call_args_list], [0, 2])
        self.assertEqual(
            [
                c.kwargs["outcome"]
                for c in q._observe_deferred_kv_release.call_args_list
            ],
            ["drained", "error", "drained"],
        )
        resource.poison.assert_not_called()
        self.assertEqual(q._deferred_releases, [])


if __name__ == "__main__":
    unittest.main()
