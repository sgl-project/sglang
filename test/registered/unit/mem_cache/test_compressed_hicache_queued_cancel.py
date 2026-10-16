"""Queued abort -> cache finish -> L2 ACK regression, without CUDA/RDMA.

The pool, leases, completion and lifecycle are real. Scheduler/cache entry
methods are executed from source with controlled dependencies; this is not a
full serving/GPU test. Set SGLANG_COMPRESSION_STANDALONE_TEST=1 on a CPU-only
workstation to bypass package initializers, as in the compression test kit.
"""

# Imports follow the optional CPU package bootstrap below.
# ruff: noqa: E402

import ast
import dataclasses
import logging
import os
import sys
import types
import unittest
from collections import defaultdict
from concurrent.futures import Future
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace as NS
from typing import NamedTuple
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
if os.environ.get("SGLANG_COMPRESSION_STANDALONE_TEST") == "1":
    for name, relative in (
        ("sglang", "python/sglang"),
        ("sglang.srt", "python/sglang/srt"),
        ("sglang.srt.mem_cache", "python/sglang/srt/mem_cache"),
        (
            "sglang.srt.mem_cache.unified_cache",
            "python/sglang/srt/mem_cache/unified_cache",
        ),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT / relative)]
            sys.modules[name] = module

from sglang.srt.kv_compression.store import CompressedHostKVCache
from sglang.srt.kv_compression.types import (
    BufferDrainError,
    CompressionCapacityError,
    EncodedPage,
    new_page_refs,
)
from sglang.srt.mem_cache.hicache_lifecycle import HiCacheLifecycleMixin
from sglang.srt.mem_cache.l2_completion import RestoreTransferResult, TransferCompletion
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def source_definition(file, name, owner=None, **namespace):
    """Run the actual entry methods without importing GPU scheduler modules."""
    path = SRT / file
    nodes = ast.parse(path.read_text()).body
    if owner:
        nodes = next(
            n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner
        ).body
    node = next(n for n in nodes if getattr(n, "name", None) == name)
    if isinstance(node, ast.FunctionDef):
        node.decorator_list = []  # Rank consensus is outside this single-rank test.
    module = ast.parse("from __future__ import annotations")
    module.body.append(node)
    namespace.setdefault("logger", logging.getLogger(__name__))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


CacheRequestHandle = source_definition(
    "mem_cache/base_prefix_cache.py", "CacheRequestHandle", dataclasses=dataclasses
)
CacheRequestOutcome = source_definition(
    "mem_cache/base_prefix_cache.py", "CacheRequestOutcome", Enum=Enum, auto=auto
)
HiCacheAck = source_definition(
    "managers/cache_controller.py", "HiCacheAck", NamedTuple=NamedTuple
)


class CacheHarness(HiCacheLifecycleMixin):
    finish = source_definition(
        "mem_cache/base_prefix_cache.py",
        "finish",
        "BasePrefixCache",
        CacheRequestOutcome=CacheRequestOutcome,
    )
    release_aborted_request = source_definition(
        "mem_cache/unified_radix_cache.py",
        "release_aborted_request",
        "UnifiedRadixCache",
    )
    loading_check = source_definition(
        "mem_cache/unified_radix_cache.py", "loading_check", "UnifiedRadixCache"
    )


class SchedulerHarness:
    _abort_prefill_sender = source_definition(
        "disaggregation/prefill.py",
        "_abort_prefill_sender",
        "SchedulerDisaggregationPrefillMixin",
        BufferDrainError=BufferDrainError,
    )
    release_aborted_prefill_waiting_req = source_definition(
        "disaggregation/prefill.py",
        "release_aborted_prefill_waiting_req",
        "SchedulerDisaggregationPrefillMixin",
        maybe_release_metadata_buffer=lambda *a: None,
        release_kv_cache=lambda *a, **kw: None,
    )
    clear_pending_chunk_send = lambda self, req: None
    _is_prefill_abort_quarantined = source_definition(
        "disaggregation/prefill.py",
        "_is_prefill_abort_quarantined",
        "SchedulerDisaggregationPrefillMixin",
    )
    _release_aborted_request = source_definition(
        "managers/scheduler.py",
        "_release_aborted_request",
        "Scheduler",
        CacheRequestOutcome=CacheRequestOutcome,
    )
    abort_request = source_definition(
        "managers/scheduler.py",
        "abort_request",
        "Scheduler",
        get_parallel=lambda: NS(pp_size=1),
        DisaggregationMode=NS(PREFILL="prefill", DECODE="decode"),
        maybe_release_metadata_buffer=lambda *a: None,
        _make_abort_req=source_definition(
            "managers/scheduler.py",
            "_make_abort_req",
            AbortReq=lambda **kw: NS(**kw),
            get_serving=lambda: NS(weight_version="test"),
            compute_weight_version_spans=lambda *a, **kw: [],
        ),
    )


class QueuedCancelTests(unittest.TestCase):
    def setUp(self):
        self.pool = CompressedHostKVCache(4096, 4 * 1024**2, pin_memory=False)
        self.handles = self.pool.alloc(3)
        refs = new_page_refs(3)
        stage = torch.zeros((3, self.pool.reservation_bytes), dtype=torch.uint8)
        write = self.pool.prepare_write(self.handles, [4096] * 3)
        try:
            write.scatter(stage)
            write.publish(
                refs,
                [EncodedPage(stage[i, :4096], "raw", 4096) for i in range(3)],
                [None] * 3,
            )
        finally:
            write.close()
        self.future = Future()
        self.completion = TransferCompletion(future=self.future)
        self.state = NS(
            restore_ticket=None,
            stats=defaultdict(int),
            quarantined=[],
            pool=self.pool,
            runtime=NS(verify=False),
        )
        self.req = NS(
            release_mm_inputs_on_abort=Mock(),
            session=None,
            rid="queued",
            cache_request_handle=CacheRequestHandle("queued", 0),
            finished_reason=None,
            to_finish=None,
            last_node=0,
            pending_bootstrap=False,
            kv=NS(holds_kv=False, holds_mamba=False),
            output_ids=[],
            weight_version_events=[],
            set_finish_with_abort=Mock(),
        )
        self.params = NS(req=self.req, best_match_node=1, mem_quota=None)
        self.node = NS(
            key=[1, 2, 3],
            component_data={
                ComponentType.FULL: NS(
                    host_value=self.handles.clone(),
                    metadata={"compression_page_refs": refs},
                )
            },
        )
        self.cache = c = CacheHarness()
        c.cache_controller = NS(
            async_l2=self.state,
            ack_load_queue=[],
            load_fence_stream=None,
            l2_transfer_engine=NS(
                submit_async_restore=Mock(return_value=self.completion)
            ),
            pp_prefetch_command_group=None,
        )
        c.tree_core = NS(
            node_by_id=lambda n: self.node,
            build_load_back_spec=Mock(
                return_value=(NS(host_indices=self.handles, nodes_to_load=[1]), {})
            ),
            commit_load_back=Mock(return_value=[]),
            finish_load_back=Mock(),
            _update_duplicate_tracking=Mock(),
            _update_evictable_leaf_sets=Mock(),
        )
        c.load_back_threshold = 1
        c.inc_host_lock_ref = Mock(return_value=NS(to_dec_params=lambda: "host"))
        c.inc_lock_ref = Mock(return_value=NS(delta=0, to_dec_params=lambda: "device"))
        c.dec_host_lock_ref = Mock()
        c.dec_lock_ref = Mock()
        c.token_to_kv_pool_allocator = NS(
            available_size=lambda: 10,
            alloc=Mock(return_value=torch.tensor([9, 11, 5])),
            free=Mock(),
        )
        c._apply_cache_actions = Mock()
        c.linker = None
        c.prefetch_loaded_tokens_by_reqid = {}
        c.prefetch_loaded_storage_start_by_reqid = {}
        c.storage_prefetch_retries = NS(cancel=Mock())
        c.buffer_pipeline = None
        c.ongoing_prefetch = {}
        c.discard_storage_prefetch_accounting = Mock()
        self.scheduler = s = SchedulerHarness()
        s.tree_cache = c
        s.chunked_req = s.mm_receiver = s.dllm_config = None
        s.waiting_queue = [self.req]
        s.beam_coordinator = NS(retire_group=Mock())
        s.ipc_channels = NS(send_to_tokenizer=NS(send_output=Mock()))
        s.disaggregation_mode = "prefill"
        s.req_to_metadata_buffer_idx_allocator = None
        s.grammar_manager = NS(abort_requests=Mock())
        s.disagg_prefill_bootstrap_queue = NS(queue=[])
        s.disagg_prefill_inflight_queue = []
        s.collect_inflight_reqs = lambda: set()
        self.begin()
        self.ticket = self.state.restore_ticket
        self.assertIsNotNone(self.ticket)
        self.assertEqual(self.pool.snapshot()["active_readers"], 3)

    def begin(self):
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.managers.cache_controller": NS(HiCacheAck=HiCacheAck),
                "sglang.srt.mem_cache.base_prefix_cache": NS(EvictParams=NS),
            },
        ):
            return self.cache.init_async_load_back(self.params)

    def cancel(self, **kwargs):
        self.scheduler.abort_request(
            NS(
                rid=kwargs.get("rid", self.req.rid),
                abort_all=kwargs.get("abort_all", False),
                finished_reason=None,
                abort_message=None,
            )
        )

    def complete(self, error=None):
        if error is None:
            self.future.set_result(RestoreTransferResult(3, 12288, 12288, 0, 0))
        else:
            self.future.set_exception(error)
        self.cache.loading_check(finish_count=1)

    def assert_retained(self):
        self.assertIs(self.state.restore_ticket, self.ticket)
        self.cache.token_to_kv_pool_allocator.free.assert_not_called()
        self.cache.dec_host_lock_ref.assert_not_called()
        self.cache.dec_lock_ref.assert_not_called()
        self.assertEqual(self.pool.snapshot()["active_readers"], 3)
        self.assertFalse(self.future.cancelled())

    def assert_released(self):
        self.assertIsNone(self.state.restore_ticket)
        self.assertTrue(self.ticket.consumed)
        self.assertEqual(self.pool.snapshot()["active_readers"], 0)
        self.cache.token_to_kv_pool_allocator.free.assert_called_once_with(
            self.ticket.device_indices
        )
        self.cache.dec_lock_ref.assert_called_once_with(1, "device")
        self.cache.dec_host_lock_ref.assert_called_once_with(1, "host")
        self.cache.tree_core.commit_load_back.assert_not_called()

    def test_actual_queued_abort_waits_for_ack_then_releases_once(self):
        self.cancel()
        self.assertEqual(self.scheduler.waiting_queue, [])
        self.scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()
        self.assertTrue(self.ticket.abandoned)
        self.assertIsNone(self.req.to_finish)  # Cache cancellation owns the marker.
        self.assert_retained()
        self.cache._reap_cancelled_restore()
        self.assert_retained()
        self.complete()
        self.cache._reap_cancelled_restore()
        self.cache.finish(self.req.cache_request_handle, CacheRequestOutcome.ABORT)
        self.assert_released()
        self.assertEqual(self.state.stats["cancelled_restores"], 1)
        self.assertEqual(self.state.stats["completed_restore_pages"], 3)
        self.assertEqual(self.state.stats["restored_pages"], 0)
        self.assertEqual(self.pool.snapshot()["used_pages"], 3)
        self.assertIsNotNone(self.node.component_data[ComponentType.FULL].host_value)

    def test_running_restore_is_not_freed_on_cancel(self):
        self.future.set_running_or_notify_cancel()
        self.cancel()
        self.assert_retained()
        self.complete()
        self.assert_released()

    def test_ready_ticket_cancel_releases_without_another_ack(self):
        self.complete()
        self.cache._complete_async_load_ack(
            HiCacheAck(None, None, [1], completion=self.completion)
        )
        self.assertEqual(self.state.stats["completed_restore_pages"], 3)
        self.cancel()
        self.assert_released()

    def test_repeated_abort_before_completion_is_idempotent(self):
        self.cancel()
        for _ in range(3):
            self.cache.finish(self.req.cache_request_handle, CacheRequestOutcome.ABORT)
            self.cancel()
        self.assert_retained()
        self.complete()
        self.assert_released()
        self.assertEqual(self.state.stats["cancelled_restores"], 1)

    def test_abort_all_marks_matching_ticket(self):
        self.cancel(rid="unrelated", abort_all=True)
        self.complete()
        self.assert_released()

    def test_abort_different_request_does_not_cancel_ticket(self):
        self.cache.finish(CacheRequestHandle("other", 0), CacheRequestOutcome.ABORT)
        self.assertFalse(self.ticket.abandoned)
        self.assert_retained()

    def test_abort_same_rid_different_attempt_does_not_cancel_ticket(self):
        self.cache.finish(
            CacheRequestHandle(self.req.rid, 1), CacheRequestOutcome.ABORT
        )
        self.assertFalse(self.ticket.abandoned)
        self.assert_retained()

    def test_ticket_retains_original_attempt_when_request_handle_advances(self):
        old = self.req.cache_request_handle
        self.req.cache_request_handle = CacheRequestHandle(self.req.rid, 1)
        self.cache.finish(self.req.cache_request_handle, CacheRequestOutcome.ABORT)
        self.assertFalse(self.ticket.abandoned)
        self.cache.finish(old, CacheRequestOutcome.ABORT)
        self.assertTrue(self.ticket.abandoned)
        self.complete()
        self.assert_released()

    def test_success_finish_leaves_restore_available_for_adoption(self):
        self.cache.finish(self.req.cache_request_handle, CacheRequestOutcome.SUCCESS)
        self.complete()
        adopted, anchor = self.begin()
        self.assertEqual(anchor, 1)
        self.assertEqual(adopted, 3)
        self.assertEqual(self.pool.snapshot()["active_readers"], 0)
        self.cache.token_to_kv_pool_allocator.free.assert_not_called()
        self.cache.tree_core.commit_load_back.assert_called_once()
        self.assertEqual(self.state.stats["restored_pages"], 3)

    def test_other_request_cannot_consume_abandoned_pending_ticket(self):
        self.cancel()
        self.params.req = NS(rid="other")
        self.assertIsNone(self.begin())
        self.assert_retained()
        self.complete()
        self.assert_released()

    def test_drained_capacity_failure_reclaims_without_invalidating_host(self):
        self.cancel()
        self.complete(CompressionCapacityError("busy"))
        self.assert_released()
        self.assertEqual(self.pool.snapshot()["used_pages"], 3)
        self.assertEqual(self.state.stats["restore_admission_retries"], 1)

    def test_uncertain_completion_quarantines_without_releasing(self):
        self.cancel()
        self.complete(BufferDrainError("scatter did not drain"))
        self.assertIsNone(self.state.restore_ticket)
        self.assertEqual(self.state.quarantined, [self.ticket])
        self.assertEqual(self.pool.snapshot()["active_readers"], 3)
        self.cache.token_to_kv_pool_allocator.free.assert_not_called()
        self.cache.dec_lock_ref.assert_not_called()
        self.cache.dec_host_lock_ref.assert_not_called()
        self.cache.tree_core.commit_load_back.assert_not_called()

    def test_partial_release_failure_keeps_existing_quarantine_semantics(self):
        self.cancel()
        self.cache.dec_lock_ref.side_effect = RuntimeError("pin release failed")
        with self.assertRaises(BufferDrainError):
            self.complete()
        self.assertEqual(self.state.quarantined, [self.ticket])
        self.assertIsNone(self.state.restore_ticket)
        self.cache.token_to_kv_pool_allocator.free.assert_called_once()
        self.cache.dec_host_lock_ref.assert_not_called()
        self.cache._reap_cancelled_restore()
        self.cache.token_to_kv_pool_allocator.free.assert_called_once()

    def test_native_cache_without_async_l2_keeps_abort_cleanup(self):
        self.cache.cache_controller.async_l2 = None
        self.cache.finish(self.req.cache_request_handle, CacheRequestOutcome.ABORT)
        self.cache.storage_prefetch_retries.cancel.assert_called_once_with(self.req.rid)

    def test_done_future_still_waits_for_original_ack(self):
        self.future.set_result(RestoreTransferResult(3, 12288, 12288, 0, 0))
        self.cancel()
        self.assert_retained()
        self.cache.loading_check(finish_count=1)
        self.assert_released()

    def test_storage_prefetch_early_return_still_marks_restore_abandoned(self):
        self.cache.buffer_pipeline = NS(release_staged_hold=Mock(return_value=True))
        self.cancel()
        self.assertTrue(self.ticket.abandoned)
        self.assert_retained()
        self.complete()
        self.assert_released()

    def test_abandoned_ready_ticket_cannot_publish_even_if_consumption_is_retried(self):
        self.complete()
        self.ticket.abandoned = True
        self.assertIsNone(self.cache._consume_restore_ticket(self.ticket, self.params))
        self.assert_released()

    def test_wrong_completion_cannot_reclaim_cancelled_ticket(self):
        self.cancel()
        wrong = TransferCompletion(future=Future())
        wrong.future.set_result(RestoreTransferResult(3, 12288, 12288, 0, 0))
        with self.assertRaisesRegex(RuntimeError, "does not own"):
            self.cache._complete_async_load_ack(
                HiCacheAck(None, None, [1], completion=wrong)
            )
        self.assert_retained()
        self.complete()
        self.assert_released()

    def test_drained_restore_failure_preserves_original_invalidation_and_cleanup(self):
        self.cancel()
        self.complete(ValueError("verification failed after drain"))
        self.assert_released()
        self.assertEqual(self.pool.snapshot()["used_pages"], 0)
        self.assertIsNone(self.node.component_data[ComponentType.FULL].host_value)
        self.assertEqual(self.state.stats["restore_failures"], 1)

    def test_next_attempt_restores_and_adopts_after_cancelled_ticket_drains(self):
        old_handle = self.req.cache_request_handle
        self.cancel()
        self.complete()
        self.assert_released()
        self.req.release_mm_inputs_on_abort = Mock()
        self.req.cache_request_handle = CacheRequestHandle(self.req.rid, 1)
        self.params.req = self.req
        self.future = Future()
        self.completion = TransferCompletion(future=self.future)
        self.cache.cache_controller.l2_transfer_engine.submit_async_restore.return_value = self.completion
        self.begin()
        next_ticket = self.state.restore_ticket
        self.cache.finish(old_handle, CacheRequestOutcome.ABORT)
        self.assertFalse(next_ticket.abandoned)
        self.complete()
        adopted, anchor = self.begin()
        self.assertEqual(anchor, 1)
        self.assertEqual(adopted, 3)
        self.assertEqual(self.pool.snapshot()["active_readers"], 0)
        self.assertEqual(self.state.stats["restored_pages"], 3)
        self.cache.token_to_kv_pool_allocator.free.assert_called_once()


if __name__ == "__main__":
    unittest.main()
