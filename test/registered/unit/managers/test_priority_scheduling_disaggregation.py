import json
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base import KVPoll  # noqa: E402
from sglang.srt.disaggregation.decode import DecodePreallocQueue  # noqa: E402
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue  # noqa: E402
from sglang.srt.disaggregation.req_state import DisaggReqState
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.lora.lora_manager import LoRAManager  # noqa: E402
from sglang.srt.managers.io_struct import AbortReq  # noqa: E402
from sglang.srt.managers.request_lists import CoreLists  # noqa: E402
from sglang.srt.managers.schedule_batch import (  # noqa: E402
    FINISH_ABORT,
    Req,
    ReqKvInfo,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.engine_state import (  # noqa: E402
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.lora_admission import LoraAdmission
from sglang.srt.managers.scheduler_components.mode_owners import (  # noqa: E402
    ModeOwners,
)
from sglang.srt.managers.scheduler_components.mode_stages import (  # noqa: E402
    ModeStages,
    ScheduleSlot,
)
from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
from sglang.srt.managers.scheduler_components.request_intake import RequestIntake
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.managers.scheduler_components.step_planner import StepPlanner
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.observability import req_time_stats
from sglang.srt.observability.req_time_stats import SchedulerReqTimeStats
from sglang.srt.runtime_context import get_context, publish, reset_context  # noqa: E402
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import decode_owner_shell, prefill_owner_shell
from sglang.test.scheduler_request_intake import make_request_intake
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402
from sglang.test.scheduler_request_lists import (
    make_request_registry,
    make_request_retirer,
)
from sglang.test.separate_buffer_allocator_double import (
    bind_separate_buffer_capacity,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDisaggregationPriorityQueueing(unittest.TestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _new_intake(self, disaggregation_mode: DisaggregationMode) -> RequestIntake:
        get_context().override("test", enable_priority_scheduling=True)
        get_context().override("test", schedule_low_priority_values_first=False)
        get_context().override("test", abort_on_priority_when_disabled=False)
        get_context().override("test", enable_hierarchical_cache=False)
        get_context().override("test", enable_unified_cache_external_linker=False)
        tree_cache = MagicMock()
        intake = make_request_intake(
            kv=SimpleNamespace(tree_cache=tree_cache, token_to_kv_pool_allocator=None),
            request_retirer=RequestRetirer(
                kv=SimpleNamespace(tree_cache=tree_cache),
                beam_coordinator=MagicMock(),
                hisparse_coordinator=None,
                decode_offload_manager=None,
                pending=PendingSteps(),
                output_streamer=MagicMock(),
            ),
            mode_owners=ModeOwners(
                disagg_prefill=(
                    prefill_owner_shell(bootstrap_queue=MagicMock(), num_kv_heads=8)
                    if disaggregation_mode == DisaggregationMode.PREFILL
                    else None
                ),
                disagg_decode=(
                    decode_owner_shell(prealloc_queue=MagicMock())
                    if disaggregation_mode == DisaggregationMode.DECODE
                    else None
                ),
            ),
        )
        intake._prefetch_kvcache = MagicMock()
        intake._abort_on_queued_limit = MagicMock(return_value=False)
        return intake

    def _new_req(self, priority=None):
        req = MagicMock()
        req.priority = priority
        req.rid = "req"
        req.output_ids = []
        req.weight_version_events = []
        req.time_stats = MagicMock()
        req.time_stats.trace_ctx = MagicMock()
        # At the door a request holds no KV or mamba state.
        req.kv.holds_kv = req.kv.holds_mamba = False
        return req

    def test_prefill_mode_assigns_default_priority_before_bootstrap_queue(self):
        intake = self._new_intake(DisaggregationMode.PREFILL)
        req = self._new_req(priority=None)

        intake.add_request_to_queue(req)

        self.assertEqual(req.priority, -sys.maxsize - 1)
        intake.mode_owners.disagg_prefill.bootstrap_queue.add.assert_called_once_with(
            req, 8
        )
        req.time_stats.set_prefill_bootstrap_queue_entry_time.assert_called_once()

    def test_decode_mode_assigns_default_priority_before_prealloc_queue(self):
        intake = self._new_intake(DisaggregationMode.DECODE)
        req = self._new_req(priority=None)

        intake.add_request_to_queue(req)

        self.assertEqual(req.priority, -sys.maxsize - 1)
        intake.mode_owners.disagg_decode.prealloc_queue.add.assert_called_once_with(
            req, is_retracted=False
        )
        req.time_stats.set_decode_prealloc_queue_entry_time.assert_called_once()

    def test_decode_retraction_queue_time_excludes_decoding_before_it(self):
        """After a PD-decode retraction, queue_time counts waits, not earlier decoding."""
        intake = self._new_intake(DisaggregationMode.DECODE)
        req = self._new_req(priority=0)
        stats = SchedulerReqTimeStats(disagg_mode=DisaggregationMode.DECODE)
        req.time_stats = stats
        t0 = 1000.0
        stats.set_wait_queue_entry_time(t0)
        stats.set_forward_entry_time(t0 + 0.1)

        with patch.object(req_time_stats.time, "perf_counter", return_value=t0 + 10.0):
            intake.add_request_to_queue(req, is_retracted=True)
        stats.set_forward_entry_time(t0 + 12.0)

        self.assertAlmostEqual(stats.get_queueing_time(), 0.1 + 2.0)

    def test_priority_disabled_abort_validation_applies_to_decode_mode(self):
        intake = self._new_intake(DisaggregationMode.DECODE)
        get_context().override("test", enable_priority_scheduling=False)
        get_context().override("test", abort_on_priority_when_disabled=True)
        req = self._new_req(priority=10)

        with patch(
            "sglang.srt.managers.scheduler_components.request_retirer.get_serving",
            return_value=SimpleNamespace(weight_version="v0"),
        ):
            intake.add_request_to_queue(req)

        intake.mode_owners.disagg_decode.prealloc_queue.add.assert_not_called()
        intake.request_retirer.output_streamer.send_abort_output.assert_called_once()
        req.time_stats.trace_ctx.abort.assert_called_once()

    def test_priority_rejection_releases_early_prefetch(self):
        for mode in DisaggregationMode:
            with self.subTest(mode=mode):
                intake = self._new_intake(mode)
                get_context().override("test", enable_priority_scheduling=False)
                get_context().override("test", abort_on_priority_when_disabled=True)
                intake.request_retirer.tree_cache = MagicMock(spec=["finish"])
                req = self._new_req(priority=10)

                with patch(
                    "sglang.srt.managers.scheduler_components.request_retirer.get_serving",
                    return_value=SimpleNamespace(weight_version="v0"),
                ):
                    intake.add_request_to_queue(req)

                intake.request_retirer.tree_cache.finish.assert_called_once_with(
                    req.cache_request_handle, CacheRequestOutcome.ABORT
                )
                intake._prefetch_kvcache.assert_not_called()
                self.assertEqual(intake.lists.waiting_queue, [])
                if intake.mode_owners.disagg_prefill is not None:
                    intake.mode_owners.disagg_prefill.bootstrap_queue.add.assert_not_called()
                if intake.mode_owners.disagg_decode is not None:
                    intake.mode_owners.disagg_decode.prealloc_queue.add.assert_not_called()

    def test_full_queue_releases_only_the_rejected_request(self):
        for priority in (0, 2):
            with self.subTest(incoming_priority=priority):
                intake = self._new_intake(DisaggregationMode.NULL)
                del intake._abort_on_queued_limit  # Exercise the real queue check.
                intake.max_queued_requests = 1
                intake.request_retirer.tree_cache = MagicMock(spec=["finish"])
                intake.beam_coordinator = MagicMock()
                queued = self._new_req(priority=1)
                queued.rid = "queued"
                intake.lists.waiting_queue = [queued]
                incoming = self._new_req(priority=priority)

                with patch(
                    "sglang.srt.managers.scheduler_components.request_retirer.get_serving",
                    return_value=SimpleNamespace(weight_version="v0"),
                ):
                    intake.add_request_to_queue(incoming)

                rejected = incoming if priority == 0 else queued
                intake.request_retirer.tree_cache.finish.assert_called_once_with(
                    rejected.cache_request_handle, CacheRequestOutcome.ABORT
                )
                if priority == 0:
                    self.assertEqual(intake.lists.waiting_queue, [queued])
                    intake._prefetch_kvcache.assert_not_called()
                else:
                    self.assertEqual(intake.lists.waiting_queue, [incoming])
                    intake._prefetch_kvcache.assert_called_once_with(incoming)

    def test_full_queue_skips_the_requests_a_step_admits(self):
        """A request preempted while a prefill step is built is requeued before
        that step's admissions leave the queue at its launch: they neither
        count toward the limit nor are aborted."""
        for limit, expected_abort in ((2, None), (1, "queued")):
            with self.subTest(max_queued_requests=limit):
                intake = self._new_intake(DisaggregationMode.NULL)
                del intake._abort_on_queued_limit  # Exercise the real queue check.
                intake.max_queued_requests = limit
                intake.request_retirer.tree_cache = MagicMock(spec=["finish"])
                intake.beam_coordinator = MagicMock()
                admitted = self._new_req(priority=0)
                admitted.rid = "admitted"
                queued = self._new_req(priority=1)
                queued.rid = "queued"
                intake.lists.waiting_queue = [admitted, queued]
                preempted = self._new_req(priority=2)
                preempted.rid = "preempted"

                with patch(
                    "sglang.srt.managers.scheduler_components.request_retirer.get_serving",
                    return_value=SimpleNamespace(weight_version="v0"),
                ):
                    intake.add_request_to_queue(preempted, admitted={admitted})

                send_abort = intake.request_retirer.output_streamer.send_abort_output
                if expected_abort is None:
                    send_abort.assert_not_called()
                    self.assertEqual(
                        intake.lists.waiting_queue, [admitted, queued, preempted]
                    )
                else:
                    send_abort.assert_called_once()
                    self.assertIs(send_abort.call_args.args[1], queued)
                    self.assertEqual(intake.lists.waiting_queue, [admitted, preempted])


class TestOptimisticPrefillCacheOwnership(unittest.TestCase):
    def _new_unpublished_swa_state(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.is_mamba_enabled = False
        cache.tree_core = SimpleNamespace(is_eagle=False)
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(16, dtype=torch.int64).reshape(2, 8)
        )
        component = MagicMock()
        component.prepare_for_caching_req.return_value = None
        cache._components_tuple = (component,)
        cache.insert = MagicMock()
        req = SimpleNamespace(
            rid="req",
            get_fill_ids=lambda: [1, 2, 3, 4],
            kv=SimpleNamespace(
                req_pool_idx=1,
                cache_protected_len=2,
            ),
            prefix_len=2,
            priority=3,
            extra_key=None,
            cache_salt=None,
            kv_rotation_base=None,
            last_node=object(),
        )
        return cache, component, req

    def _pop_bootstrapping_swa_branch(
        self, write_policy, *, queue_cls=PrefillBootstrapQueue, is_retracted=False
    ):
        queue = queue_cls.__new__(queue_cls)
        queue.scheduler_stage_metrics = None
        queue.pp_size = 1
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = SimpleNamespace(write_policy=write_policy)
        queue.attn_cp_cpu_group = None
        queue.attn_tp_cpu_group = None
        queue.tree_cache = cache
        queue.counters = LaunchCounters()
        queue.ensure_metadata_buffer = MagicMock(return_value=True)
        queue.kv_senders = {"req": object()}
        req = SimpleNamespace(
            rid="req",
            disagg=DisaggReqState(prefill_attempt_count=0),
            is_retracted=is_retracted,
            swa_branching_seqlen=4,
            kv=SimpleNamespace(cache_protected_len=2),
            time_stats=MagicMock(),
        )
        queue.queue = [req]

        with (
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Bootstrapping],
            ),
            patch(
                "sglang.srt.disaggregation.prefill.get_disagg",
                return_value=SimpleNamespace(optimistic_prefill_attempts=2),
            ),
        ):
            result = queue.pop_bootstrapped()

        return queue, req, result

    def test_write_through_pending_swa_branch_waits_for_bootstrap(self):
        queue, req, result = self._pop_bootstrapping_swa_branch("write_through")

        self.assertEqual(result, [])
        self.assertEqual(queue.queue, [req])
        self.assertEqual(req.disagg.prefill_attempt_count, 0)
        queue.ensure_metadata_buffer.assert_not_called()

    def test_other_write_policies_stay_optimistic(self):
        for write_policy in ("write_back", "write_through_selective"):
            with self.subTest(write_policy=write_policy):
                queue, req, result = self._pop_bootstrapping_swa_branch(write_policy)

                self.assertEqual(result, [req])
                self.assertEqual(queue.queue, [])
                self.assertEqual(req.disagg.prefill_attempt_count, 1)
                queue.ensure_metadata_buffer.assert_called_once_with(req)

    def test_subclass_eligibility_admits_prefill_but_never_a_paused_request(self):
        """`can_prefill_before_bootstrap` is the override point; the pause check
        stays outside it, so a permissive subclass cannot resume a retracted req."""

        class _Eager(PrefillBootstrapQueue):
            def can_prefill_before_bootstrap(self, req):
                return True

        queue, req, result = self._pop_bootstrapping_swa_branch(
            "write_through", queue_cls=_Eager
        )
        self.assertEqual(result, [req])
        self.assertEqual(req.disagg.prefill_attempt_count, 1)

        queue, req, result = self._pop_bootstrapping_swa_branch(
            "write_through", queue_cls=_Eager, is_retracted=True
        )
        self.assertEqual(result, [])
        self.assertEqual(queue.queue, [req])
        queue.ensure_metadata_buffer.assert_not_called()

    def test_unpublished_advance_preserves_swa_frontier_and_ownership(self):
        cache, component, req = self._new_unpublished_swa_state()
        last_node = req.last_node

        with envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(True):
            cache.advance_unpublished_req(req)

        self.assertEqual(req.kv.cache_protected_len, 2)
        self.assertIs(req.last_node, last_node)
        prepare_params = component.prepare_for_caching_req.call_args.kwargs[
            "insert_params"
        ]
        self.assertEqual(prepare_params.prev_prefix_len, 2)
        component.free_out_of_window_slots.assert_called_once_with(
            req, 3, prepare_params
        )
        component.cleanup_after_caching_req.assert_called_once_with(
            req, is_finished=False, insert_params=prepare_params
        )
        cache.insert.assert_not_called()

    def test_bootstrap_failure_after_swa_eviction_releases_once(self):
        cache, component, req = self._new_unpublished_swa_state()
        req.bootstrap_room = 7
        req.time_stats = SimpleNamespace(trace_ctx=MagicMock())
        req.return_logprob = False
        req.disagg = DisaggReqState(pending_bootstrap=True, metadata_buffer_index=-1)
        req.cache_request_handle = object()
        req.kv.holds_kv = True
        req.kv.holds_mamba = False
        req.session = None
        req.release_mm_inputs_on_abort = MagicMock()
        cache.finish = MagicMock()
        owner = prefill_owner_shell(
            tree_cache=cache,
            request_retirer=make_request_retirer(kv=SimpleNamespace(tree_cache=cache)),
            clear_pending_chunk_send=MagicMock(),
            req_to_metadata_buffer_idx_allocator=MagicMock(),
            output_streamer=MagicMock(),
            metrics_sink=SimpleNamespace(enable_metrics=False),
        )
        owner.kv_senders[req.rid] = MagicMock()

        with (
            envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS.override(True),
            patch(
                "sglang.srt.managers.scheduler_components.request_retirer."
                "release_kv_cache"
            ) as release_kv_cache,
            patch("sglang.srt.disaggregation.prefill.prepare_abort"),
            patch(
                "sglang.srt.disaggregation.prefill.get_parallel",
                return_value=SimpleNamespace(tp_rank=0),
            ),
        ):
            cache.advance_unpublished_req(req)
            owner.handle_bootstrap_failure(req)

        component.free_out_of_window_slots.assert_called_once()
        cache.insert.assert_not_called()
        release_kv_cache.assert_called_once_with(req, cache, checkpoint=False)
        cache.finish.assert_called_once_with(
            req.cache_request_handle, CacheRequestOutcome.ABORT
        )
        self.assertFalse(req.disagg.pending_bootstrap)
        self.assertNotIn(req.rid, owner.kv_senders)

    def test_write_through_pending_chunk_does_not_publish(self):
        owner = prefill_owner_shell()
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = SimpleNamespace(write_policy="write_through")
        cache.advance_unpublished_req = MagicMock()
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(8, dtype=torch.int32).reshape(1, 8)
        )
        owner.tree_cache = cache
        req = SimpleNamespace(
            disagg=DisaggReqState(pending_bootstrap=True),
            kv=SimpleNamespace(req_pool_idx=0),
            extend_end=5,
        )

        owner.checkpoint(req)

        cache.advance_unpublished_req.assert_called_once_with(req)
        # The next chunk still resumes after this one.
        self.assertEqual(req.prefix_len, 5)

    def test_bootstrap_success_publishes_once(self):
        owner = prefill_owner_shell(bootstrap_queue=MagicMock(), checkpoint=MagicMock())
        owner.bootstrap_queue.finalize_bootstrap.return_value = True
        req = SimpleNamespace(rid="req")

        result = owner.handle_pending_bootstrap(req, KVPoll.WaitingForInput)

        self.assertTrue(result)
        owner.bootstrap_queue.finalize_bootstrap.assert_called_once_with(req)
        owner.checkpoint.assert_called_once_with(req)

    def test_waiting_abort_tears_down_the_transfer_only(self):
        owner = prefill_owner_shell(
            tree_cache=MagicMock(),
            req_to_metadata_buffer_idx_allocator=MagicMock(),
            clear_pending_chunk_send=MagicMock(),
        )
        sender = MagicMock()
        req = SimpleNamespace(
            rid="req",
            disagg=DisaggReqState(metadata_buffer_index=3, pending_bootstrap=True),
            kv=SimpleNamespace(holds_kv=True, holds_mamba=False),
        )
        owner.kv_senders[req.rid] = sender

        owner.release_aborted_waiting_req(req)

        # The KV is the retirer's to release (without an insert).
        owner.tree_cache.assert_not_called()
        sender.abort.assert_called_once()
        self.assertFalse(req.disagg.pending_bootstrap)
        self.assertNotIn(req.rid, owner.kv_senders)

    def test_scheduler_waiting_abort_delegates_release_once(self):
        core_lists = CoreLists()
        core_lists.waiting_queue = [MagicMock(rid="req")]
        kv = SimpleNamespace(tree_cache=MagicMock())
        disagg_prefill = prefill_owner_shell(release_aborted_waiting_req=MagicMock())
        registry = make_request_registry(
            core_lists=core_lists,
            grammar_manager=MagicMock(),
            request_retirer=RequestRetirer(
                kv=kv,
                beam_coordinator=MagicMock(),
                hisparse_coordinator=None,
                decode_offload_manager=None,
                pending=PendingSteps(),
                output_streamer=MagicMock(),
            ),
            kv=kv,
            mode_owners=ModeOwners(disagg_prefill=disagg_prefill),
        )

        with (
            patch(
                "sglang.srt.managers.scheduler_components.request_retirer.get_serving"
            ) as get_serving,
            patch(
                "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache"
            ) as release,
        ):
            get_serving.return_value.weight_version = None
            registry.abort_request(AbortReq(rid="req"))

        disagg_prefill.release_aborted_waiting_req.assert_called_once()
        kv.tree_cache.finish.assert_called_once()
        (req,) = release.call_args.args[:1]
        release.assert_called_once_with(req, kv.tree_cache, checkpoint=False)


class TestDecodePreallocQueuePriority(CustomTestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _new_decode_req(self, rid: str, priority: int, *, failed: bool = False):
        req = SimpleNamespace(
            rid=rid,
            priority=priority,
            origin_input_ids=[1, 2, 3],
            output_ids=[],
            finished_reason=FINISH_ABORT("failed") if failed else None,
            return_logprob=False,
            sampling_params=SimpleNamespace(max_new_tokens=8),
            kv=ReqKvInfo(req_pool_idx=int(priority) % 8, cache_protected_len=0),
            time_stats=MagicMock(),
            disagg=DisaggReqState(),
        )
        return SimpleNamespace(
            req=req,
            waiting_for_input=True,
            kv_receiver=MagicMock(),
            metadata_buffer_index=-1,
            is_rebootstrap=False,
            host_staged=False,
        )

    def _new_queue(self, decode_reqs, *, low_priority_values_first: bool = False):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.request_retirer = MagicMock()
        queue.pp_size = 1
        queue.queue = list(decode_reqs)
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        # `pop_preallocated` credits this counter; `__new__` skips the __init__
        # that seeds it.
        queue._num_published_destinations = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._allocatable_tokens = MagicMock(return_value=1000)

        def pre_alloc_mock(req, prefix_indices=None, prefix_len=0, total_prefix_len=0):
            return torch.arange(
                len(req.origin_input_ids) - prefix_len, dtype=torch.int64
            )

        queue._pre_alloc = MagicMock(side_effect=pre_alloc_mock)

        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 100
        # Non-hybrid pools have no mamba allocator; MagicMock would otherwise
        # auto-create one and break the `available_size() <= 0` comparison in
        # pop_preallocated.
        queue.req_to_token_pool.mamba_allocator = None
        queue.req_to_token_pool.req_to_token = torch.arange(
            8 * 16, dtype=torch.int64
        ).reshape(8, 16)

        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 100
        queue.req_to_metadata_buffer_idx_allocator.alloc.side_effect = iter(range(100))

        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.token_to_kv_pool_allocator.page_size = 1
        queue.token_to_kv_pool_allocator.available_size.return_value = 1000
        queue.token_to_kv_pool = MagicMock()
        queue.transfer_queue = SimpleNamespace(queue=[], enable_staging=False)
        queue.kv_manager = SimpleNamespace(
            kv_args=SimpleNamespace(state_types=[]),
        )
        queue.tree_cache = MagicMock()

        queue.enable_priority_scheduling = True
        queue.schedule_low_priority_values_first = low_priority_values_first
        queue.get_ready = lambda: []
        queue.enable_hisparse = False
        queue.enable_lora = False
        queue.enable_decode_hicache = False
        queue.output_streamer = MagicMock()
        queue.pool_stats_observer = MagicMock()
        return queue

    def test_prealloc_lora_slots_cover_inflight_steps_and_queues(self):
        """In-flight requests retain their adapter slots: the registry's
        adapters in use cover decoding, the steps in flight, and PD decode's
        transfer and ready lists."""
        running, finished, transferring, waiting = [
            self._new_decode_req(rid, 0)
            for rid in ("running", "finished", "transferring", "waiting")
        ]
        reqs = [
            self._new_decode_req("new", 3),
            self._new_decode_req("blocked", 2),
            self._new_decode_req("same-adapter", 1),
        ]
        for entry in [running, finished, transferring, waiting, *reqs]:
            entry.req.lora_id = entry.req.rid
        reqs[2].req.lora_id = "running"

        queue = self._new_queue(reqs)
        queue.transfer_queue.queue = [transferring]
        core_lists = CoreLists()
        core_lists.decoding = [running.req]
        # A step in flight still runs a request its predecessor's commit finished.
        pending = PendingSteps()
        pending.push(SimpleNamespace(step=SimpleNamespace(reqs=[finished.req])))
        queue.held_rebootstrap_reqs = []
        owner = decode_owner_shell(
            prealloc_queue=queue,
            transfer_queue=queue.transfer_queue,
            ready=[waiting.req],
        )
        registry = make_request_registry(
            core_lists=core_lists,
            pending=pending,
            mode_owners=ModeOwners(disagg_decode=owner),
        )
        queue.enable_decode_hicache = False
        queue.enable_lora = True
        get_context().override("test", enable_lora_overlap_loading=False)
        manager = LoRAManager.__new__(LoRAManager)
        manager.max_loras_per_batch = 5
        manager.num_pinned_loras = 0
        lora_admission = LoraAdmission.__new__(LoraAdmission)
        lora_admission.lora_drainer = None
        lora_admission.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(lora_manager=manager)
        )
        queue.can_schedule_lora_req = lora_admission.can_schedule_lora_req

        preallocated, failed = queue.pop_preallocated(
            [], lora_ids_in_use=registry.lora_ids_in_use
        )

        self.assertEqual(preallocated, [reqs[0], reqs[2]])
        self.assertEqual(failed, [])
        self.assertEqual(queue.queue, [reqs[1]])
        self.assertEqual(
            [call.args[0] for call in queue._pre_alloc.call_args_list],
            [reqs[0].req, reqs[2].req],
        )

    def test_hisparse_prealloc_reserves_scratch_for_pending_transfers(self):
        """Pending PD requests consume a full buffer and scratch, not just a buffer."""
        for available, pending, admitted in (
            (4160, 0, 0),
            (8320, 1, 0),
            (10368, 1, 1),
            (10368, 0, 2),
        ):
            with self.subTest(available=available, pending=pending):
                reqs = [self._new_decode_req(f"hisparse-{i}", i) for i in range(3)]
                queue = self._new_queue(reqs)
                queue.enable_priority_scheduling = False
                queue.enable_hisparse = True
                queue.hisparse_coordinator = SimpleNamespace(
                    padded_buffer_size=4160,
                    spec_swap=SimpleNamespace(scratch_capacity=1024),
                )
                queue.token_to_kv_pool_allocator.logical_attn_allocator = (
                    SimpleNamespace(available_size=lambda: 32768)
                )
                queue.token_to_kv_pool_allocator.hisparse_attn_allocator = (
                    SimpleNamespace(available_size=lambda: available)
                )
                queue.token_to_kv_pool_allocator.page_size = 64
                queue.token_to_kv_pool.page_size = 64
                queue.draft_token_to_kv_pool = None
                queue.transfer_queue.queue = [
                    SimpleNamespace(host_staged=False) for _ in range(pending)
                ]
                queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
                queue._uses_swa_reservation = MagicMock(return_value=False)
                queue._send_kv_metadata = MagicMock()

                preallocated, failed = queue.pop_preallocated([], lora_ids_in_use=set)

                self.assertEqual(failed, [])
                self.assertEqual(preallocated, reqs[:admitted])
                self.assertEqual(queue.queue, reqs[admitted:])

    def test_prealloc_queue_schedules_higher_priority_values_first_by_default(self):
        reqs = [
            self._new_decode_req("low", 1),
            self._new_decode_req("high", 10),
            self._new_decode_req("mid", 5),
        ]
        queue = self._new_queue(reqs)

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            preallocated, failed = queue.pop_preallocated([], lora_ids_in_use=set)

        self.assertEqual(
            [decode_req.req.rid for decode_req in preallocated],
            [
                "high",
                "mid",
                "low",
            ],
        )
        self.assertEqual(failed, [])

    def test_prealloc_queue_can_schedule_lower_priority_values_first(self):
        reqs = [
            self._new_decode_req("mid", 5),
            self._new_decode_req("high", 10),
            self._new_decode_req("low", 1),
        ]
        queue = self._new_queue(reqs, low_priority_values_first=True)

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            preallocated, failed = queue.pop_preallocated([], lora_ids_in_use=set)

        self.assertEqual(
            [decode_req.req.rid for decode_req in preallocated],
            [
                "low",
                "mid",
                "high",
            ],
        )
        self.assertEqual(failed, [])

    def test_failed_request_indices_stay_valid_after_priority_sort(self):
        failed_low = self._new_decode_req("failed-low", 1, failed=True)
        healthy_high = self._new_decode_req("healthy-high", 10)
        queue = self._new_queue([failed_low, healthy_high])

        with patch("sglang.srt.disaggregation.decode.CLIP_MAX_NEW_TOKEN", 4096):
            preallocated, failed = queue.pop_preallocated([], lora_ids_in_use=set)

        self.assertEqual(
            [decode_req.req.rid for decode_req in preallocated], ["healthy-high"]
        )
        self.assertEqual([decode_req.req.rid for decode_req in failed], ["failed-low"])
        self.assertEqual(queue.queue, [])
        queue.output_streamer.stream_output.assert_called_once_with(
            [failed_low.req], failed_low.req.return_logprob
        )

    def test_admission_without_radix_cache_charges_the_alloc_hook(self):
        """Bug regression: with the decode radix cache disabled, admission
        charged the raw fill length while ``_pre_alloc`` allocates the
        page-rounded ``_required_alloc_tokens``. On a paged pool a request
        could pass admission and then fail allocation; the radix-cache path
        already charged the rounded amount. Admission must charge what the
        pool will allocate, so the mocked charge decides whether the request
        fits."""
        for charge, fits in ((50, True), (200, False)):
            with self.subTest(charge=charge):
                decode_req = self._new_decode_req("req", 1)
                queue = self._new_queue([decode_req])
                queue.token_to_kv_pool_allocator.available_size.return_value = 100
                queue._required_alloc_tokens = MagicMock(return_value=charge)

                preallocated, failed = queue.pop_preallocated([], lora_ids_in_use=set)

                queue._required_alloc_tokens.assert_called_once_with(
                    fill_len=3, prefix_len=0
                )
                self.assertEqual(preallocated, [decode_req] if fits else [])
                self.assertEqual(queue.queue, [] if fits else [decode_req])
                self.assertEqual(failed, [])


class TestDecodePreallocQueueRebootstrapPayload(unittest.TestCase):
    """The decode scheduler builds the rebootstrap ``/generate`` payload; the
    dispatch itself now lives on the kv manager (see
    ``TestCommonKVManagerPrefillRecompute``)."""

    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _sampling_params(self):
        return SimpleNamespace(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            ignore_eos=False,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
            no_stop_trim=False,
        )

    def _new_req(self):
        return SimpleNamespace(
            rid="rid-0",
            origin_input_ids=np.array([1, 2], dtype=np.int32),
            output_ids=[np.int32(3), np.int32(4)],
            sampling_params=self._sampling_params(),
            bootstrap_host="127.0.0.1",
            bootstrap_port=30000,
            bootstrap_room=7,
            priority=10,
            extra_key=None,
            cache_salt=None,
            routing_key=None,
            disagg_prefill_dp_rank=None,
        )

    def test_build_rebootstrap_payload_converts_numpy_ids_to_json_lists(self):
        req = self._new_req()

        # build_rebootstrap_payload lives on Req; exercise it unbound with a
        # namespace that carries the attributes it reads.
        payload = Req.build_rebootstrap_payload(req)

        # origin_input_ids + output_ids, coerced to plain python ints.
        self.assertEqual(payload["input_ids"], [1, 2, 3, 4])
        self.assertTrue(all(type(x) is int for x in payload["input_ids"]))
        self.assertEqual(payload["sampling_params"]["max_new_tokens"], 1)
        self.assertEqual(payload["bootstrap_room"], 7)
        self.assertIsNone(payload["cache_salt"])
        # The prefill /generate URL is derived from bootstrap info on the decode
        # side, not sent in the payload; and the boundary token is replayed via
        # the decode-side override, so neither belongs in the payload.
        self.assertNotIn("pd_rebootstrap_prefill_url", payload)
        self.assertNotIn("pd_rebootstrap_forced_output_id", payload)
        # Must be JSON-serializable (numpy scalars would raise here).
        json.dumps(payload)


class TestCommonKVManagerPrefillRecompute(unittest.TestCase):
    """The kv manager owns the shared executor + HTTP session and routes any
    rebootstrap ``/generate`` failure through ``kv_receiver.abort()`` ->
    ``KVPoll.Failed`` so the scheduler's normal transfer-failure streaming runs.
    """

    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _new_manager(self):
        from sglang.srt.disaggregation.common.conn import CommonKVManager

        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr._prefill_recompute_executor = None
        mgr._prefill_recompute_executor_lock = threading.Lock()
        mgr._prefill_recompute_sessions = threading.local()
        mgr.waiting_timeout = 300
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        # Only the attn-tp/attn-cp group leader on the first PP stage issues the
        # single rebootstrap /generate; default the mock manager to that leader.
        mgr.attn_tp_rank = 0
        mgr.attn_cp_rank = 0
        mgr.pp_rank = 0
        # Decode-side prefill info cache; the rebootstrap /generate URL is derived
        # from here (bootstrap_addr host + self-registered prefill_http_port)
        # instead of a router-injected pd_rebootstrap_prefill_url.
        mgr.prefill_info_table = {}
        return mgr

    def _register_prefill_info(self, mgr, bootstrap_addr, http_port):
        from sglang.srt.disaggregation.common.conn import PrefillServerInfo

        mgr.prefill_info_table[bootstrap_addr] = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=1,
            kv_cache_dtype=None,
            follow_bootstrap_room=True,
            prefill_http_port=http_port,
        )

    def _payload(self):
        return {
            "input_ids": [1, 2, 3, 4],
            "rid": "rid-0",
        }

    def test_submit_dispatches_run_to_shared_executor(self):
        mgr = self._new_manager()
        mgr._prefill_recompute_executor = MagicMock()
        receiver = MagicMock(bootstrap_room=7, bootstrap_addr="127.0.0.1:8998")
        self._register_prefill_info(mgr, "127.0.0.1:8998", 30000)

        mgr.submit_prefill_recompute(receiver, self._payload())

        mgr._prefill_recompute_executor.submit.assert_called_once()
        args = mgr._prefill_recompute_executor.submit.call_args[0]
        self.assertEqual(args[0], mgr._run_prefill_recompute)
        self.assertIs(args[1], receiver)
        # URL derived from bootstrap_addr host + registered prefill_http_port.
        self.assertEqual(args[2], "http://127.0.0.1:30000")
        receiver.abort.assert_not_called()

    def test_submit_is_noop_on_non_leader_ranks(self):
        # A retracted request is replicated across every rank in its attention
        # TP/CP group and every PP stage; only the group/first-stage leader must
        # POST the single /generate, or the prefill recomputes it once per rank.
        for attn_tp_rank, attn_cp_rank, pp_rank in (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ):
            with self.subTest(
                attn_tp_rank=attn_tp_rank,
                attn_cp_rank=attn_cp_rank,
                pp_rank=pp_rank,
            ):
                mgr = self._new_manager()
                mgr.attn_tp_rank = attn_tp_rank
                mgr.attn_cp_rank = attn_cp_rank
                mgr.pp_rank = pp_rank
                mgr._prefill_recompute_executor = MagicMock()
                receiver = MagicMock(bootstrap_room=7, bootstrap_addr="127.0.0.1:8998")
                self._register_prefill_info(mgr, "127.0.0.1:8998", 30000)

                mgr.submit_prefill_recompute(receiver, self._payload())

                mgr._prefill_recompute_executor.submit.assert_not_called()
                receiver.abort.assert_not_called()
                self.assertEqual(mgr.failure_records, {})

    def test_submit_unresolved_url_fails_via_abort(self):
        mgr = self._new_manager()
        mgr._prefill_recompute_executor = MagicMock()
        # No prefill_info registered for this bootstrap_addr -> URL unresolved.
        receiver = MagicMock(bootstrap_room=7, bootstrap_addr="127.0.0.1:8998")

        mgr.submit_prefill_recompute(receiver, self._payload())

        receiver.abort.assert_called_once()
        mgr._prefill_recompute_executor.submit.assert_not_called()
        self.assertIn(7, mgr.failure_records)

    def test_run_aborts_on_http_error(self):
        mgr = self._new_manager()
        session = MagicMock()
        session.post.return_value = SimpleNamespace(status_code=500, text="boom")
        mgr._prefill_recompute_sessions.session = session
        receiver = MagicMock(bootstrap_room=7)

        mgr._run_prefill_recompute(receiver, "http://prefill", self._payload())

        session.post.assert_called_once()
        receiver.abort.assert_called_once()
        self.assertIn(7, mgr.failure_records)

    def test_run_aborts_on_exception(self):
        mgr = self._new_manager()
        session = MagicMock()
        session.post.side_effect = RuntimeError("network down")
        mgr._prefill_recompute_sessions.session = session
        receiver = MagicMock(bootstrap_room=7)

        mgr._run_prefill_recompute(receiver, "http://prefill", self._payload())

        receiver.abort.assert_called_once()
        self.assertIn(7, mgr.failure_records)

    def test_run_success_does_not_abort(self):
        mgr = self._new_manager()
        session = MagicMock()
        session.post.return_value = SimpleNamespace(status_code=200, text="")
        mgr._prefill_recompute_sessions.session = session
        receiver = MagicMock(bootstrap_room=7)

        mgr._run_prefill_recompute(receiver, "http://prefill", self._payload())

        session.post.assert_called_once()
        receiver.abort.assert_not_called()
        self.assertEqual(mgr.failure_records, {})


class TestDecodeAdmission(unittest.TestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _new_scheduler(self, *, enable_overlap: bool) -> Scheduler:
        scheduler = partial_scheduler()
        scheduler.parts.mode_owners = ModeOwners()
        scheduler.parts.counters = LaunchCounters()
        scheduler.parts.flags = EngineFlags()
        scheduler.parts.lists = CoreLists()
        scheduler.parts.grammar_manager = MagicMock()
        scheduler.parts.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.parts.lists.waiting_queue = []
        scheduler.parts.lists.decoding = []
        scheduler.parts.mode_owners.disagg_decode = decode_owner_shell(
            kv_checksum_computer=None,
            enable_priority_scheduling=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            req_to_token_pool=MagicMock(size=1),
            penalty_store=MagicMock(),
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=MagicMock(),
            enable_overlap=enable_overlap,
            max_running_requests=1,
            future_map=MagicMock(),
            policy=MagicMock(),
            ngram_embedding_manager=MagicMock(),
            output_streamer=MagicMock(),
            device="cpu",
            core_lists=scheduler.parts.lists,
        )
        return scheduler

    def test_waiting_queue_is_sorted_before_admission(self):
        scheduler = self._new_scheduler(enable_overlap=False)
        owner = scheduler.disagg_decode
        original_ready = [MagicMock(rid="low"), MagicMock(rid="high")]
        owner.ready = original_ready
        original_ready[0].priority = 1
        original_ready[1].priority = 10
        owner.enable_priority_scheduling = True
        owner.policy.calc_priority.side_effect = lambda ready, _: ready.sort(
            key=lambda req: -req.priority
        )

        # Admission reads the published disagg config
        # (disaggregation_decode_enable_radix_cache).
        with get_context().override_server_args(
            disaggregation_decode_enable_radix_cache=False
        ):
            admitted = owner.pop_reqs_to_admit(ScheduleSlot())

        owner.policy.calc_priority.assert_called_once_with(
            original_ready, scheduler.parts.lists.decoding
        )
        self.assertEqual([req.rid for req in admitted], ["high"])
        self.assertEqual([req.rid for req in owner.ready], ["low"])

    def test_admission_joins_unfinished_reqs_to_decoding(self):
        scheduler = self._new_scheduler(enable_overlap=False)
        owner = scheduler.disagg_decode
        owner.req_to_token_pool = MagicMock(size=4)
        owner.max_running_requests = 4
        running = MagicMock(rid="running")
        scheduler.parts.lists.decoding = [running]
        kept, finished = MagicMock(rid="kept"), MagicMock(rid="finished")
        owner.ready = [kept, finished]
        # The handoff token finished one of them at admission.
        owner.commit_prefill_handoff = MagicMock()
        kept.finished.return_value = False
        finished.finished.return_value = True
        scheduler.hub.step_planner = MagicMock(spec=StepPlanner)
        scheduler.hub.step_planner.get_decode_step.return_value = None
        scheduler.hub.request_intake = MagicMock(spec=RequestIntake)
        scheduler.parts.dp_attn_adapter = MagicMock()
        scheduler.parts.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = (
            lambda step, need_sync=None: step
        )
        scheduler.parts.metrics_sink = SimpleNamespace(enable_fpm=False)
        scheduler.parts.scripted_scheduler_hook = None
        scheduler.parts.ngram_embedding_manager = MagicMock()
        scheduler.parts.ngram_embedding_manager.prepare_for_forward.side_effect = (
            lambda step, chunked_req: step
        )
        scheduler.scheduler_stage_metrics = None
        # The PD decode schedule stages of a flat driver, without its intake.
        scheduler.mode.stages = ModeStages(
            commit_extend=MagicMock(),
            drain_hicache=owner.check_hicache_events,
            join_ready=owner.join_ready,
            select_decode=scheduler._select_decode,
        )

        with get_context().override_server_args(
            disaggregation_decode_enable_radix_cache=False
        ):
            step = scheduler.schedule_step(ScheduleSlot())

        self.assertIsNone(step)
        owner.commit_prefill_handoff.assert_called_once_with([kept, finished])
        self.assertEqual(scheduler.parts.lists.decoding, [running, kept])
        self.assertEqual(owner.ready, [])

    def test_overlap_seeds_admitted_reqs_without_waiting_on_the_forward(self):
        # Their rows came from the pools, which get a released row back only
        # at the commit of the last pending step that writes it.
        scheduler = self._new_scheduler(enable_overlap=True)
        req = MagicMock(
            rid="request",
            origin_input_ids=[1, 2, 3],
            output_ids=[7],
            prefix_len=3,
            already_computed=0,
            cached_tokens=0,
            cached_tokens_device=0,
            retracted_stain=False,
            grammar=None,
            return_logprob=False,
        )
        req.kv.req_pool_idx = 0
        req.finished.return_value = False

        owner = scheduler.disagg_decode
        call_order = []
        owner.future_map.stash.side_effect = lambda *_: call_order.append("seed")

        with (
            patch("sglang.srt.disaggregation.decode.checkpoint_kv_cache"),
            get_context().override_server_args(
                disaggregation_decode_enable_radix_cache=False
            ),
        ):
            owner.commit_prefill_handoff([req])

        self.assertEqual(call_order, ["seed"])
        self.assertEqual(req.already_computed, 3)
        owner.penalty_store.alloc.assert_called_once_with(req)
        owner.output_streamer.stream_output.assert_called_once_with([req], False)


if __name__ == "__main__":
    unittest.main()
