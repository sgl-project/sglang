"""Cancellation receipts wait for real request resource and host-step retirement."""

import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    AbortReq,
    CancellationRetirementOutput,
    CancellationRetirementReq,
)
from sglang.srt.managers.req_types import FINISH_ABORT  # noqa: E402
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.request_retirer import (  # noqa: E402
    RequestRetirer,
)
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator  # noqa: E402
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixCache  # noqa: E402
from sglang.srt.mem_cache.registry import (  # noqa: E402
    TreeCacheBuildContext,
    default_radix_cache_factory,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.runtime_context import (  # noqa: E402
    get_memory,
    get_parallel,
    get_server_args,
)
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang.test.scheduler_test_utils import empty_request_lists  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_cpu_scheduler(send_output, *, cache_enabled=False, legacy_cache=False):
    scheduler = Scheduler.__new__(Scheduler)
    empty_request_lists(scheduler)
    scheduler.req_to_token_pool = ReqToTokenPool(8, 32, "cpu", False)
    scheduler.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
        size=32, dtype=torch.int64, device="cpu", kvcache=None, need_sort=False
    )
    params = CacheInitParams(
        disable=not cache_enabled,
        req_to_token_pool=scheduler.req_to_token_pool,
        token_to_kv_pool_allocator=scheduler.token_to_kv_pool_allocator,
        page_size=1,
    )
    scheduler.model_config = SimpleNamespace(is_multimodal=False)
    if legacy_cache:
        scheduler.tree_cache = RadixCache(params)
    else:
        scheduler.tree_cache = default_radix_cache_factory(
            TreeCacheBuildContext(
                server_args=get_server_args(),
                params=params,
                is_hybrid_swa=False,
                is_hybrid_ssm=False,
                enable_hierarchical_cache=False,
                disable_radix_cache=not cache_enabled,
                tp_worker=None,
                model_config=scheduler.model_config,
                tp_size=1,
                tp_rank=0,
                tp_group=None,
            )
        )
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    for attr in (
        "enable_hierarchical_cache",
        "enable_hicache_storage",
        "enable_lmcache",
        "enable_session_radix_cache",
        "is_hybrid_swa",
        "is_hybrid_ssm",
        "enable_hisparse",
        "enable_continuous_input_polling",
    ):
        setattr(scheduler, attr, False)
    scheduler.decode_offload_manager = None
    scheduler.rust_server = None
    scheduler.mm_receiver = None
    scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
    scheduler.running_batch = ScheduleBatch(reqs=[])
    scheduler.last_batch = None
    scheduler.result_queue = deque()
    scheduler._pending_chunked_abort_req = None
    scheduler.invariant_checker = Mock()
    scheduler.beam_coordinator = Mock()
    scheduler.output_streamer = Mock()
    scheduler.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(send_output=send_output)
    )
    scheduler.request_retirer = RequestRetirer(
        tree_cache=scheduler.tree_cache,
        beam_coordinator=scheduler.beam_coordinator,
        hisparse_coordinator=None,
        uocr_runtime=None,
        send_abort_output=scheduler.send_abort_output,
    )
    return scheduler


def allocate_cpu_req(scheduler, rid="target", instance="instance-a"):
    req = Req(
        rid,
        "",
        array("q", [11, 12, 13]),
        SamplingParams(max_new_tokens=4),
        request_instance_id=instance,
    )
    row = scheduler.req_to_token_pool.alloc([req])[0]
    tokens = scheduler.token_to_kv_pool_allocator.alloc(3)
    scheduler.req_to_token_pool.write((row, slice(0, 3)), tokens.to(torch.int32))
    req.kv.set_row_len(3)
    return req


class TestSchedulerCancellationRetirement(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())
        enter_scope(self, envs.SGLANG_ENABLE_CANCEL_RETIREMENT_RECEIPTS.override(True))
        self.outputs = []
        self.abort_observer = None

    def _scheduler(self, *, cache_enabled=False):
        return make_cpu_scheduler(self._send_output, cache_enabled=cache_enabled)

    def _send_output(self, output, req):
        if isinstance(output, CancellationRetirementOutput):
            self.outputs.append((output, req))
        elif self.abort_observer is not None:
            self.abort_observer(req)

    def _req(self, scheduler, rid="target", instance="instance-a"):
        return allocate_cpu_req(scheduler, rid, instance)

    def _cancel(self, scheduler, req, operation="operation-a", **kwargs):
        control = CancellationRetirementReq(
            rid=req.rid,
            request_instance_id=req.request_instance_id,
            cancellation_id=operation,
            **kwargs,
        )
        scheduler.handle_cancellation_retirement(control)
        return control

    def test_queued_output_precedes_release_but_receipt_follows_real_release(self):
        scheduler = self._scheduler()
        # UnifiedRadixCache is the default even with static, non-unified pools.
        self.assertIs(type(scheduler.tree_cache), UnifiedRadixCache)
        self.assertFalse(get_memory().enable_unified_memory)
        target = self._req(scheduler)
        neighbor = self._req(scheduler, "target-neighbor", "instance-b")
        scheduler.waiting_queue.join([target, neighbor])

        def on_terminal(req):
            self.assertIs(req, target)
            self.assertTrue(req.finished())
            self.assertTrue(req.kv.holds_kv)
            self.assertEqual(scheduler.req_to_token_pool.available_size(), 6)
            self.assertEqual(scheduler.token_to_kv_pool_allocator.available_size(), 26)
            scheduler._check_cancellation_retirements()
            self.assertEqual(self.outputs, [])

        self.abort_observer = on_terminal
        control = self._cancel(scheduler, target)

        self.assertEqual(
            [(out.status, route) for out, route in self.outputs], [("retired", control)]
        )
        self.assertFalse(target.kv.holds_kv)
        self.assertTrue(target.kv.is_kv_released)
        self.assertEqual(scheduler.req_to_token_pool.available_size(), 7)
        self.assertEqual(scheduler.token_to_kv_pool_allocator.available_size(), 29)
        self.assertFalse(neighbor.finished())
        self.assertTrue(neighbor.kv.holds_kv)
        self.assertEqual(list(scheduler.waiting_queue), [neighbor])

    def test_running_receipt_waits_for_each_launched_step_and_all_live_holders(self):
        scheduler = self._scheduler(cache_enabled=True)
        req = self._req(scheduler)
        first, second = ScheduleBatch(reqs=[req]), ScheduleBatch(reqs=[req])
        first.add_inflight_step()
        second.add_inflight_step()
        scheduler.running_batch = second
        scheduler.last_batch = first
        scheduler.running_reqs.join([req])
        self._cancel(scheduler, req)
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertEqual(self.outputs, [])

        # Use the same real release path as finished result processing. The
        # prefix may remain cached, but the request row must be returned.
        req.finish_outside_step(FINISH_ABORT())
        scheduler.request_retirer.release_finished(req, checkpoint=True)
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(scheduler.req_to_token_pool.available_size(), 8)
        self.assertEqual(scheduler.tree_cache.evictable_size(), 3)
        self.assertEqual(scheduler.token_to_kv_pool_allocator.available_size(), 29)
        scheduler._on_step_processed(first)
        self.assertEqual(req.inflight_steps, 1)
        self.assertEqual(self.outputs, [])

        # An unexpected live holder still forbids a positive receipt, even
        # when both step counters and allocation predicates are clear.
        scheduler.grammar_manager.grammar_queue.join([req])
        scheduler._on_step_processed(second)
        self.assertEqual(req.inflight_steps, 0)
        self.assertEqual(self.outputs, [])
        scheduler.grammar_manager.grammar_queue.leave([req])
        scheduler._check_cancellation_retirements()
        self.assertEqual(self.outputs[-1][0].status, "retired")
        self.assertIsNone(scheduler._cancellation_retirements["operation-a"].req)

    def test_parked_chunked_abort_releases_real_resources_before_receipt(self):
        scheduler = self._scheduler()
        req = self._req(scheduler)
        scheduler.chunked_req = req
        self._cancel(scheduler, req)
        self.assertEqual(self.outputs, [])
        self.assertIs(scheduler._pending_chunked_abort_req, req)

        scheduler.process_pending_chunked_abort()

        self.assertTrue(req.finished())
        self.assertIsNone(scheduler.chunked_req)
        self.assertIsNone(scheduler._pending_chunked_abort_req)
        self.assertFalse(req.kv.holds_kv)
        self.assertEqual(scheduler.req_to_token_pool.available_size(), 8)
        self.assertEqual(scheduler.token_to_kv_pool_allocator.available_size(), 32)
        self.assertEqual(self.outputs[-1][0].status, "retired")

    def test_unknown_instance_and_cached_operation_do_not_abort_reused_rid(self):
        scheduler = self._scheduler()
        req = self._req(scheduler)
        scheduler.waiting_queue.join([req])
        control = CancellationRetirementReq(
            rid=req.rid, request_instance_id="stale", cancellation_id="same-operation"
        )
        scheduler.handle_cancellation_retirement(control)
        self.assertEqual(self.outputs[-1][0].status, "unknown")
        req.request_instance_id = "stale"
        scheduler.handle_cancellation_retirement(control)
        self.assertEqual(self.outputs[-1][0].status, "unknown")
        self.assertFalse(req.finished())
        self.assertTrue(req.kv.holds_kv)

    def test_unsupported_topology_and_grammar_queue_leave_request_untouched(self):
        scheduler = self._scheduler()
        req = self._req(scheduler)
        scheduler.grammar_manager.grammar_queue.join([req])
        self._cancel(scheduler, req)
        self.assertEqual(self.outputs[-1][0].reason, "unsupported_request_queue")
        with get_parallel().override(dp_size=2):
            self._cancel(scheduler, req, "distributed-operation")
        self.assertEqual(self.outputs[-1][0].reason, "distributed_scheduler")
        with envs.SGLANG_ENABLE_CANCEL_RETIREMENT_RECEIPTS.override(False):
            self._cancel(scheduler, req, "disabled-operation")
        self.assertEqual(self.outputs[-1][0].reason, "feature_disabled")
        self.assertFalse(req.finished())
        self.assertTrue(req.kv.holds_kv)

    def test_unified_receipts_reject_other_cache_owners_and_custom_cores(self):
        scheduler = self._scheduler()
        req = self._req(scheduler)
        scheduler.waiting_queue.join([req])
        cache = scheduler.tree_cache
        allocator = scheduler.token_to_kv_pool_allocator
        unsupported = (
            (cache, "cache_controller", object()),
            (cache, "linker", object()),
            (cache, "enable_session_radix_cache", True),
            (cache, "components", {}),
            (cache, "_tree_core_backend", "custom"),
            (cache, "tree_core", object()),
            (allocator, "storage", object()),
            (allocator, "state", object()),
            (scheduler, "tree_cache", object()),
        )
        for index, (owner, attr, value) in enumerate(unsupported):
            with self.subTest(owner=type(owner).__name__, attr=attr):
                with patch.object(owner, attr, value):
                    self._cancel(scheduler, req, f"unsupported-{index}")
                self.assertEqual(self.outputs[-1][0].status, "unsupported")
                self.assertFalse(req.finished())
                self.assertTrue(req.kv.holds_kv)
        self.assertEqual(scheduler.req_to_token_pool.available_size(), 7)
        self.assertEqual(allocator.available_size(), 29)

    def test_legacy_static_radix_cache_still_returns_resources(self):
        scheduler = make_cpu_scheduler(self._send_output, legacy_cache=True)
        req = self._req(scheduler)
        scheduler.waiting_queue.join([req])
        self._cancel(scheduler, req)
        self.assertEqual(self.outputs[-1][0].status, "retired")
        self.assertEqual(scheduler.req_to_token_pool.available_size(), 8)
        self.assertEqual(scheduler.token_to_kv_pool_allocator.available_size(), 32)

    def test_pending_duplicates_expiry_and_capacity_never_fabricate_release(self):
        scheduler = self._scheduler()
        req = self._req(scheduler)
        scheduler.running_reqs.join([req])
        scheduler.running_batch = ScheduleBatch(reqs=[req])
        scheduler.running_batch.add_inflight_step()
        self._cancel(scheduler, req, http_worker_ipc="worker-a")
        self._cancel(scheduler, req, http_worker_ipc="worker-b")
        self.assertEqual(self.outputs, [])
        self.assertEqual(len(scheduler._cancellation_retirements), 1)
        entry = scheduler._cancellation_retirements["operation-a"]
        with patch(
            "sglang.srt.managers.scheduler.CANCELLATION_RETIREMENT_MAX_ENTRIES", 1
        ):
            self._cancel(scheduler, req, "overflow")
        self.assertEqual(self.outputs[-1][0].reason, "receipt_capacity")
        entry.expires_at = 0
        scheduler._check_cancellation_retirements()
        self.assertEqual([out.status for out, _ in self.outputs], ["unknown"] * 3)
        self.assertEqual(
            {route.http_worker_ipc for _, route in self.outputs[1:]},
            {"worker-a", "worker-b"},
        )
        self.assertEqual(self.outputs[-1][0].reason, "receipt_expired")
        self.assertTrue(req.kv.holds_kv)
        self.assertEqual(req.inflight_steps, 1)

    def test_legacy_prefix_abort_and_targeted_chunk_retry_remain_separate(self):
        scheduler = self._scheduler()
        old = self._req(scheduler)
        replacement = self._req(scheduler, instance="new-instance")
        scheduler.chunked_req = old
        self._cancel(scheduler, old)
        scheduler.chunked_req = None
        scheduler.running_batch = ScheduleBatch(reqs=[replacement])
        scheduler.running_reqs.join([replacement])
        scheduler.process_pending_chunked_abort()
        self.assertIsNone(replacement.to_finish)
        scheduler.waiting_queue.join(
            [old, self._req(scheduler, "target-suffix", "other")]
        )
        scheduler.abort_request(AbortReq(rid="target"))
        self.assertTrue(old.finished())
        self.assertEqual(list(scheduler.waiting_queue), [])
        self.assertIsInstance(replacement.to_finish, FINISH_ABORT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
