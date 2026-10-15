"""HiCache progress must not depend on whether a prefill batch is admitted."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.bundles import Streams
from sglang.srt.managers.scheduler_components.engine_state import LaunchCounters
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.scheduler_components.step_planner import StepPlanner
from sglang.srt.runtime_context import get_context
from sglang.test.disagg_owner_shells import prefill_owner_shell

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSchedulerHiCacheEvents(unittest.TestCase):
    def setUp(self):
        self.calls = Mock()
        self.scheduler = s = Scheduler.__new__(Scheduler)
        self.scheduler.mode_owners = ModeOwners()
        s.lists = CoreLists()
        s.counters = LaunchCounters()
        s.scheduler_stage_metrics = None
        s.scripted_scheduler_hook = None
        enter_scope(self, published_topology())
        self._override_flags()
        self.storage_enabled = enter_scope(
            self,
            patch(
                "sglang.srt.managers.scheduler.hicache_storage_enabled",
                return_value=True,
            ),
        )
        s.tree_cache = SimpleNamespace(check_hicache_events=self.calls.drain)
        s.request_intake = SimpleNamespace(
            process_storage_prefetch_retries=self.calls.retry
        )
        s.process_pending_chunked_abort = Mock()
        s.lists.chunked_req = None
        s.pending = []
        s.mode_owners.disagg_prefill = prefill_owner_shell(
            resolve_waiting_bootstrap=lambda waiting: waiting,
            process_prefill_chunk=lambda chunked_req, waiting, pending: (
                chunked_req,
                waiting,
            ),
        )
        s.mode_owners.disagg_decode = None
        s.mode_owners.dllm = None
        s.mode_owners.hisparse_coordinator = None
        s.dp_attn_adapter = Mock()
        s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.return_value = None
        s.step_planner = Mock(spec=StepPlanner)
        s.enable_fpm = False
        self.running_batch = SimpleNamespace(
            batch_is_full=False, is_prefill_only=False, is_empty=lambda: True
        )

    def _override_flags(self):
        get_context().override(
            "test", enable_hierarchical_cache=True, disable_overlap_schedule=True
        )

    def test_feature_gates(self):
        for hierarchical, flexkv, linker, storage in (
            (False, False, False, False),
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
            (True, False, False, True),
        ):
            with (
                self.subTest(
                    hierarchical=hierarchical,
                    flexkv=flexkv,
                    linker=linker,
                    storage=storage,
                ),
                patch(
                    "sglang.srt.managers.scheduler.get_memory",
                    return_value=SimpleNamespace(
                        enable_flexkv=flexkv,
                        enable_hierarchical_cache=hierarchical,
                        enable_unified_cache_external_linker=linker,
                        enable_lmcache=False,
                    ),
                ),
            ):
                self.calls.reset_mock()
                s = self.scheduler
                self.storage_enabled.return_value = storage
                s._process_hicache_events()
                expected = [call.drain()] if hierarchical or flexkv or linker else []
                if storage:
                    expected.append(call.retry())
                self.assertEqual(self.calls.mock_calls, expected)

    def test_pd_prefill_drains_before_admission_even_with_empty_queue(self):
        s = self.scheduler
        s.step_planner.get_new_batch_prefill = self.calls.admit
        s.step_planner.get_new_batch_prefill.return_value = None
        s.ngram_embedding_manager = Mock()
        s.ngram_embedding_manager.prepare_for_forward.return_value = None
        slot = ScheduleSlot()
        for waiting_queue in ([], [SimpleNamespace(rid="pending_l3")]):
            with self.subTest(waiting=bool(waiting_queue)):
                s.lists.waiting_queue = waiting_queue
                self.calls.reset_mock()
                for _ in range(2):
                    self.assertIsNone(s.get_next_disagg_prefill_batch_to_run(slot))
                self.assertEqual(
                    self.calls.mock_calls,
                    [call.drain(), call.retry(), call.admit(slot)] * 2,
                )

    def test_unified_drains_when_prefill_is_deferred(self):
        s = self.scheduler
        s.enable_fpm = False
        s._abort_on_waiting_timeout = Mock()
        s._abort_on_running_timeout = Mock()
        s.mode_owners.disagg_prefill = None
        s.lists.chunked_req = None
        s.require_mlp_sync = False
        s.step_planner.should_defer_prefill = self.calls.defer
        s.step_planner.should_defer_prefill.return_value = True
        s.step_planner.get_new_batch_prefill = Mock()
        s.step_planner.get_decode_step = Mock(return_value=None)
        s.step_planner.arm_prefill_decode_interval = Mock()
        s.ngram_embedding_manager = Mock()
        s.ngram_embedding_manager.prepare_for_forward.return_value = None

        s.init_mode_stages()
        step = s.get_next_batch_to_run(ScheduleSlot())

        self.assertIsNone(step)
        s.step_planner.get_new_batch_prefill.assert_not_called()
        self.assertEqual(
            self.calls.mock_calls, [call.drain(), call.retry(), call.defer()]
        )

    def test_pp_prefill_drains_before_admission(self):
        s = self.scheduler
        enter_scope(self, published_topology(pp_size=2))
        self._override_flags()
        s.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        s.world_group = s.attn_tp_group = s.device = None
        s.schedule_stream = s.forward_stream = s.copy_stream = None
        s.spec_algorithm = None
        s.req_to_token_pool = s.token_to_kv_pool_allocator = None
        s.tp_worker = s.model_worker = s.future_map = None
        s.streams = Streams(
            device="cpu", schedule_stream=None, forward_stream=None, copy_stream=None
        )
        s.device_module = SimpleNamespace(stream=lambda stream: None)
        s.kv = SimpleNamespace(token_to_kv_pool_allocator=None)
        s.pp_spec_forward = None
        s.step_launcher = SimpleNamespace(run_batch=Mock())
        s.recv_skipper = None
        s.ingest_requests = Mock(return_value=[])
        s.lists.waiting_queue = []
        s.disagg_prefill.bootstrap_queue = SimpleNamespace(pop_bootstrapped=lambda: [])
        s.disagg_prefill.poll_transfer_rids = Mock(return_value=[])
        enter_scope(
            self,
            patch(
                "sglang.srt.managers.scheduler.PPChannel",
                return_value=Mock(is_first_rank=True, is_last_rank=True),
            ),
        )
        s.step_planner.get_new_batch_prefill = self.calls.admit
        # Stop the infinite loop at admission; no PP transport or GPU is needed.
        s.step_planner.get_new_batch_prefill.side_effect = StopIteration

        s.init_mode_stages()
        with self.assertRaises(StopIteration):
            s.event_loop_pp()

        self.assertEqual(
            self.calls.mock_calls,
            [call.drain(), call.retry(), call.admit(s.pp_ring.slots[0].schedule_slot)],
        )


if __name__ == "__main__":
    unittest.main()
