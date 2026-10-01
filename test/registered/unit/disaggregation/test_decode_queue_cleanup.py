import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.decode_hicache_mixin import DecodePrefixMatch
from sglang.srt.disaggregation.fake.conn import FakeKVManager, FakeKVReceiver
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.separate_buffer_allocator_double import (
    bind_separate_buffer_capacity,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class FakeReceiver:
    def __init__(self):
        self.clear_called = False
        self.conclude_state = None

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


class TestDecodeQueueCleanup(CustomTestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_paged_swa_retraction_resume_uses_physical_page_budget(self):
        # resume_retracted_reqs reads the retraction backend off the disagg
        # bag, so the case publishes a config instead of injecting one.
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor"
        )
        override.install()
        self.addCleanup(override.restore)

        page_size = 128
        fill_len = 574
        physical_tokens_per_req = 5 * page_size
        physical_available = 18 * page_size

        reqs = [
            SimpleNamespace(
                rid=f"req-{i}",
                origin_input_ids=[0] * fill_len,
                output_ids=[],
                is_retracted=True,
                retraction_backup=None,
                load_kv_cache=MagicMock(),
            )
            for i in range(4)
        ]

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.retracted_queue = reqs.copy()
        queue.num_reserved_decode_tokens = 0
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: len(reqs))
        queue.token_to_kv_pool_allocator = MagicMock(page_size=page_size)
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.tree_cache = MagicMock()
        queue.scheduler = SimpleNamespace(
            sliding_window_size=2047,
            server_args=SimpleNamespace(disable_radix_cache=True),
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(physical_available, physical_available)
        )
        queue._allocatable_token_budgets = MagicMock(
            side_effect=lambda **_: physical_available
        )
        queue._swa_tail_allocatable_token_budget = MagicMock(
            side_effect=lambda **_: physical_available
        )

        def pre_alloc(_req):
            nonlocal physical_available
            self.assertGreaterEqual(physical_available, physical_tokens_per_req)
            physical_available -= physical_tokens_per_req

        queue._pre_alloc = MagicMock(side_effect=pre_alloc)

        resumed = queue.resume_retracted_reqs()

        self.assertEqual(resumed, reqs[:3])
        self.assertEqual(queue.retracted_queue, reqs[3:])
        self.assertEqual(physical_available, 3 * page_size)
        self.assertEqual(queue._pre_alloc.call_count, 3)

    def test_declined_l3_prefetch_is_excluded_from_transfer_promise(self):
        """A positive probe cannot omit transfer rows when prefetch is declined."""
        override = get_context().override_server_args(
            disaggregation_decode_enable_radix_cache=True
        )
        override.install()
        self.addCleanup(override.restore)
        for admission in (
            "accepted",
            "declined",
            "error",
            "state_full",
            "state_reclaim",
            "state_pending",
        ):
            with self.subTest(admission=admission):
                req = SimpleNamespace(
                    rid="prefetch-admission",
                    cache_request_handle=CacheRequestHandle("prefetch-admission", 0),
                    origin_input_ids=list(range(8)),
                    output_ids=[],
                    finished_reason=None,
                    sampling_params=SimpleNamespace(max_new_tokens=1),
                    kv=SimpleNamespace(
                        req_pool_idx=0,
                        cache_protected_len=0,
                        holds_mamba=False,
                        mamba_ping_pong_track_buffer=None,
                    ),
                    extra_key=None,
                    cache_salt=None,
                )
                dr = SimpleNamespace(
                    req=req, waiting_for_input=True, is_rebootstrap=False
                )
                pm = DecodePrefixMatch(
                    prefix_indices=torch.tensor([10, 11]),
                    l2_host_hit_length=2,
                    l3_storage_hit_length=2,
                    last_device_node=11,
                    last_host_node=22,
                )
                queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
                queue.pp_size = 1
                queue.queue = [dr]
                queue.pending_reqs = []
                queue.retracted_queue = []
                queue.num_reserved_decode_tokens = 0
                queue._resolve_pending_reqs = lambda: None
                queue._update_handshake_waiters = lambda *_: None
                queue._uses_swa_tail_prealloc = lambda: False
                queue._uses_swa_reservation = lambda: False
                queue._allocatable_token_budgets = lambda **_: 1024
                queue._hicache_pending_restore_tokens = lambda: 0
                queue._match_prefix_and_lock = lambda _: pm
                queue.scheduler = SimpleNamespace(
                    running_batch=SimpleNamespace(reqs=[]),
                    enable_priority_scheduling=False,
                    enable_hisparse=False,
                    enable_lora=False,
                    enable_decode_hicache=True,
                )
                queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
                    available_size=lambda: 1
                )
                queue.req_to_token_pool = SimpleNamespace(
                    available_size=lambda: 1,
                    mamba_allocator=None,
                    req_to_token=torch.arange(10, 18).reshape(1, 8),
                )
                queue.token_to_kv_pool_allocator = SimpleNamespace(
                    page_size=1, translate_kv_indices_for_transfer=lambda x: x
                )
                bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
                queue.kv_manager = SimpleNamespace(
                    kv_args=SimpleNamespace(state_types=[])
                )
                queue.tree_cache = SimpleNamespace(
                    hicache_storage_pass_prefix_keys=False,
                    get_last_hash_value=lambda _: "h2",
                    prefetch_from_storage=MagicMock(
                        side_effect=RuntimeError("store unavailable")
                        if admission == "error"
                        else None
                    ),
                    has_ongoing_prefetch=lambda _: admission == "accepted",
                )
                queue.transfer_queue = SimpleNamespace(queue=[])
                state_free = [1]
                if admission.startswith("state_"):
                    queue.req_to_token_pool.mamba_allocator = SimpleNamespace(
                        available_size=lambda: state_free[0]
                    )
                    queue.req_to_token_pool.enable_mamba_extra_buffer = False
                    queue.tree_cache.supports_mamba = lambda: True

                    def evict(params):
                        if admission == "state_reclaim":
                            state_free[0] += params.mamba_num

                    queue.tree_cache.evict_for_alloc = evict
                    queue.tree_cache.has_ongoing_prefetch = lambda _: True
                    if admission == "state_pending":
                        state_free[0] = 2
                        queue.transfer_queue.queue = [
                            SimpleNamespace(
                                prefix_match=DecodePrefixMatch(
                                    torch.tensor([]), 2, 0, 0
                                ),
                                hicache_restore_status=HiCacheRestoreResult.PENDING,
                                hicache_restored_node=None,
                            )
                        ]
                allocated_prefix = []
                sent = []

                def allocate(_req, _indices, _l1, total):
                    allocated_prefix.append(total)
                    return torch.arange(10 + total, 18)

                def send(_dr, indices, _page, _state, **metadata):
                    sent.append((indices.tolist(), metadata["decode_prefix_len"]))

                queue._pre_alloc = allocate
                queue._send_kv_metadata = send
                admitted, failed = queue.pop_preallocated()
                expected = {
                    "accepted": 6,
                    "declined": 4,
                    "error": 4,
                    "state_full": 2,
                    "state_reclaim": 6,
                    "state_pending": 2,
                }[admission]
                self.assertEqual(admitted, [dr])
                self.assertEqual(failed, [])
                self.assertEqual(allocated_prefix, [expected])
                self.assertEqual(sent, [(list(range(10 + expected, 18)), expected)])
                self.assertEqual(req.kv.cache_protected_len, expected)
                self.assertEqual(pm.decode_prefix_len, expected)
                if expected == 2:
                    queue.tree_cache.prefetch_from_storage.assert_not_called()

    def test_mamba_preallocation_reserves_tracking_slots_before_prefix_match(self):
        # (free, extra buffer, lazy, overlap slots, holds state, holds buffer,
        #  reclaimable, admitted, slots requested from eviction)
        cases = [
            (2, True, False, 2, False, False, 1, True, 1),
            (2, True, False, 2, False, False, 0, False, 1),
            (0, True, False, 2, False, False, 1, False, 3),
            (2, True, False, 1, False, False, 0, True, 0),
            (2, True, True, 2, False, False, 0, True, 0),
            (1, False, False, 2, False, False, 0, True, 0),
            (2, True, False, 2, True, False, 0, True, 0),
            (1, True, False, 2, False, True, 0, True, 0),
        ]
        for (
            free,
            extra,
            lazy,
            slots,
            state,
            buffer,
            reclaimable,
            admitted,
            evicted,
        ) in cases:
            with self.subTest(
                free=free,
                extra=extra,
                lazy=lazy,
                slots=slots,
                state=state,
                buffer=buffer,
                reclaimable=reclaimable,
            ):
                req = SimpleNamespace(
                    finished_reason=None,
                    kv=SimpleNamespace(
                        holds_mamba=state,
                        mamba_ping_pong_track_buffer=object() if buffer else None,
                    ),
                )
                dr = SimpleNamespace(req=req, waiting_for_input=True)
                queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
                queue.pp_size = 1
                queue.queue = [dr]
                queue.pending_reqs = []
                queue._resolve_pending_reqs = lambda: None
                queue._update_handshake_waiters = lambda *_: None
                queue._uses_swa_tail_prealloc = lambda: False
                queue._uses_swa_reservation = lambda: False
                queue._allocatable_token_budgets = lambda **_: 1024
                queue._hicache_pending_restore_tokens = lambda: 0
                queue.scheduler = SimpleNamespace(
                    running_batch=SimpleNamespace(reqs=[]),
                    enable_priority_scheduling=False,
                    enable_hisparse=False,
                    enable_lora=False,
                    enable_decode_hicache=False,
                )
                queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
                    available_size=lambda: 1
                )
                queue.req_to_token_pool = SimpleNamespace(
                    available_size=lambda: 1,
                    mamba_allocator=SimpleNamespace(available_size=lambda: free),
                    enable_mamba_extra_buffer=extra,
                    enable_mamba_extra_buffer_lazy=lazy,
                    mamba_ping_pong_track_buffer_size=slots,
                )

                def evict(params):
                    nonlocal_free[0] = min(params.mamba_num, reclaimable)

                nonlocal_free = [0]
                queue.req_to_token_pool.mamba_allocator.available_size = lambda: (
                    free + nonlocal_free[0]
                )
                reclaim = MagicMock(side_effect=evict)
                queue.tree_cache = SimpleNamespace(
                    supports_mamba=lambda: True, evict=reclaim, evict_for_alloc=reclaim
                )
                # Stop at the first operation after admission, before matching
                # can COW state or acquire a prefix lock.
                queue._rebootstrap_prefill_len = MagicMock(side_effect=StopIteration)
                if admitted:
                    with self.assertRaises(StopIteration):
                        queue.pop_preallocated()
                else:
                    self.assertEqual(queue.pop_preallocated(), ([], []))
                    queue._rebootstrap_prefill_len.assert_not_called()
                self.assertEqual(queue.queue, [dr])
                if evicted:
                    self.assertEqual(reclaim.call_args.args[0].mamba_num, evicted)
                    self.assertEqual(reclaim.call_count, 1)
                else:
                    reclaim.assert_not_called()

    def test_prealloc_abort_clears_receiver_before_removing_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="abort-prealloc",
            bootstrap_room=42,
            kv=ReqKvInfo(),
            finished_reason=None,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req, kv_receiver=receiver, waiting_for_input=True
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.tp_rank = 0
        queue.gloo_group = object()
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        # `_uses_swa_reservation` consults the allocator once tail prealloc is
        # off, so this abort path needs one even though it never allocates.
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue.req_to_token_pool = SimpleNamespace(mamba_allocator=None)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.metrics_reporter.enable_metrics = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ) as poll:
            preallocated, failed = queue.pop_preallocated()

        poll.assert_called_once_with([receiver], queue.gloo_group)
        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_prealloc_abort_also_drops_from_pending_reqs(self):
        # Same DecodeRequest lives in both queue and pending_reqs (add() slow
        # path). Aborting must drop it from both, and compare by identity since
        # DecodeRequest's dataclass __eq__ would compare the tensor receiver.
        class BadEqReceiver(FakeReceiver):
            def __eq__(self, other):
                raise TypeError("use identity comparison, not value equality")

            __hash__ = object.__hash__

        receiver = BadEqReceiver()
        req = SimpleNamespace(
            rid="abort-shared",
            finished_reason=FINISH_ABORT("aborted"),
            kv=ReqKvInfo(),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]  # same object, dual ownership
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        # `_uses_swa_reservation` consults the allocator once tail prealloc is
        # off, so this abort path needs one even though it never allocates.
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue.req_to_token_pool = SimpleNamespace(mamba_allocator=None)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        # Must not raise on the receiver __eq__ above.
        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(all(r is not decode_req for r in queue.pending_reqs))
        self.assertIsNone(decode_req.kv_receiver)

    def test_swa_reclaim_failure_rejects_only_request(self):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="swa-reclaim-failed",
            origin_input_ids=[1, 2, 3],
            kv=ReqKvInfo(),
            output_ids=[],
            finished_reason=None,
            return_logprob=False,
            sampling_params=SimpleNamespace(max_new_tokens=1),
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            waiting_for_input=True,
            is_rebootstrap=False,
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(1024, 1024)
        )
        queue._prealloc_required_tokens = MagicMock(return_value=(3, 3))
        queue._prealloc_kv_lens = MagicMock(return_value=(3, 3))
        queue._reclaim_swa_tail_capacity = MagicMock(
            return_value=(
                "SWA eviction insufficient: needed=64, available=0, "
                "req=swa-reclaim-failed"
            )
        )
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._pre_alloc = MagicMock()
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.tree_cache = MagicMock()
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        # Non-hybrid pools have no mamba allocator; MagicMock would otherwise
        # auto-create one and break the `available_size() <= 0` comparison in
        # pop_preallocated.
        queue.req_to_token_pool.mamba_allocator = None
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.server_args.disaggregation_decode_enable_radix_cache = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        queue._pre_alloc.assert_not_called()
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        # A req whose kv_receiver was already cleared must not crash on .abort().
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

        cleared_req = SimpleNamespace(
            req=SimpleNamespace(rid="cleared"), kv_receiver=None
        )
        addr_to_reqs = {"127.0.0.1:11500": [cleared_req]}

        ready, remaining = queue._ensure_prefill_info(addr_to_reqs)

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])

    def test_prefetches_prefill_dp_rank_query(self):
        addr = "127.0.0.1:11500"
        executor = MagicMock()
        future = Future()
        future.set_result({"7": 1})
        executor.submit.return_value = future

        def decode_req(room):
            return SimpleNamespace(
                req=SimpleNamespace(
                    bootstrap_host="127.0.0.1",
                    bootstrap_port=11500,
                    bootstrap_room=room,
                ),
                kv_receiver=MagicMock(),
            )

        first = decode_req(7)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pending_reqs = [first]
        queue._prefill_dp_rank_queries = {}
        queue.kv_manager = SimpleNamespace(
            prefill_info_table={addr: object()},
            _ensure_prefill_recompute_executor=lambda: executor,
        )
        queue._resolve_prefill_dp_rank = MagicMock(return_value=None)
        queue._ensure_prefill_info = lambda groups: (groups, [])

        queue.prefetch_prefill_dp_rank_queries()
        tail = decode_req(8)
        queue.pending_reqs.append(tail)
        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks",
            return_value={"8": 2},
        ) as query:
            queue._resolve_pending_reqs()

        _, called_addr, called_rooms = executor.submit.call_args.args
        self.assertEqual((called_addr, called_rooms), (addr, [7]))
        query.assert_called_once_with(addr, [8])
        first.kv_receiver.init.assert_called_once_with(1)
        tail.kv_receiver.init.assert_called_once_with(2)
        self.assertEqual(queue.pending_reqs, [])

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_transfer_failure_cleanup_respects_deferred_release_gates(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        receiver = FakeReceiver()
        req = SimpleNamespace(
            rid="failed-transfer",
            bootstrap_room=7,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_deferred_kv_release = False
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler

        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

        # A successful P→D transfer cannot commit a failed local L3 promise.
        receiver = FakeReceiver()
        decode_req.kv_receiver = receiver
        decode_req.hicache_restore_status = HiCacheRestoreResult.FAILED
        queue.queue = [decode_req]
        queue.req_to_metadata_buffer_idx_allocator.reset_mock()
        queue._clean_hicache_prefetch_resources.reset_mock()
        mock_prepare_abort.reset_mock()
        mock_release_kv_cache.reset_mock()
        with patch.object(
            queue, "_poll_with_metadata_gate", return_value=[KVPoll.Success]
        ):
            self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        queue._clean_hicache_prefetch_resources.assert_called_once_with(decode_req)
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )
        decode_req.hicache_restore_status = HiCacheRestoreResult.READY

        receiver = FakeReceiver()
        receiver.kv_mgr = FakeKVManager.__new__(FakeKVManager)
        decode_req.kv_receiver = receiver
        queue.queue = [decode_req]
        queue.enable_deferred_kv_release = True
        queue.req_to_metadata_buffer_idx_allocator.reset_mock()
        mock_release_kv_cache.reset_mock()

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

        receiver = MagicMock()
        decode_req.kv_receiver = receiver
        decode_req.host_staged = True
        queue.enable_host_receive = True
        queue.enable_deferred_kv_release = False
        queue._defer_release = MagicMock()
        queue.queue = [decode_req]
        queue.req_to_metadata_buffer_idx_allocator.reset_mock()
        mock_release_kv_cache.reset_mock()
        self.assertEqual(queue.pop_transferred(), [])
        receiver.abort.assert_called_once_with()
        queue._defer_release.assert_called_once_with(decode_req)
        receiver.clear.assert_not_called()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
        mock_release_kv_cache.assert_not_called()

        queue.queue = [decode_req]
        decode_req.req.finished_reason = FINISH_ABORT("cancelled")
        with (
            patch.object(
                queue, "_poll_with_metadata_gate", return_value=[KVPoll.Success]
            ),
            patch(
                "sglang.srt.disaggregation.decode.discard_kv_cache_backup"
            ) as discard,
        ):
            self.assertEqual(queue.pop_transferred(), [])
            discard.assert_called_once_with(req, queue.tree_cache, "host_pool")
        receiver.clear.assert_called_once_with()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        mock_release_kv_cache.assert_not_called()

    def test_fake_receiver_initializes_deferred_release_state(self):
        manager = MagicMock()
        receiver = FakeKVReceiver(manager, "")

        self.assertIs(receiver.kv_mgr, manager)
        self.assertFalse(receiver.abort_notified)

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_failed_fake_transfer_releases_at_once_under_deferred_release(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        # A health-check request gets a FakeKVReceiver over the real transfer
        # manager, so a failed one reaches the deferred-release path.
        receiver = FakeKVReceiver(MagicMock(enable_deferred_decode_kv_release=True), "")
        req = SimpleNamespace(
            rid="health-check", bootstrap_room=0, return_logprob=False
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_host_receive = False
        queue.enable_deferred_kv_release = True
        queue._defer_release = MagicMock()
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()
        queue.scheduler = MagicMock(enable_decode_hicache=False, enable_hisparse=False)
        queue.scheduler.metrics_reporter.enable_metrics = False
        mock_poll.return_value = [KVPoll.Failed]

        self.assertEqual(queue.pop_transferred(), [])
        self.assertFalse(receiver.abort_notified)
        queue._defer_release.assert_not_called()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.enable_overlap = False
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[object()]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_lmcache = False

        self.assertFalse(scheduler.is_fully_idle())


if __name__ == "__main__":
    unittest.main()
