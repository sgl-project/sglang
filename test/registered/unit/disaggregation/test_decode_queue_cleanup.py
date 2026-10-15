import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.fake.conn import FakeKVManager, FakeKVReceiver
from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.disaggregation.req_state import DisaggReqState
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    poll_and_all_reduce_with_staging,
)
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.schedule_batch import FINISH_ABORT, ReqKvInfo
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.engine_state import (
    EngineFlags,
    LaunchCounters,
)
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.srt.managers.scheduler_components.mode_stages import ModeStages
from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import decode_owner_shell, tree_release
from sglang.test.scheduler_request_lists import (
    assert_one_list_per_request,
    request_registry_of,
)
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
                disagg=DisaggReqState(),
            )
            for i in range(4)
        ]

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.request_retirer = MagicMock()
        queue.retracted_queue = reqs.copy()
        queue.num_reserved_decode_tokens = 0
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: len(reqs))
        queue.token_to_kv_pool_allocator = MagicMock(page_size=page_size)
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.tree_cache = MagicMock()
        queue.sliding_window_size = 2047
        queue.server_args = SimpleNamespace(disable_radix_cache=True)
        queue.enable_overlap = False
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            side_effect=lambda _decoding, **_: (physical_available, physical_available)
        )

        def pre_alloc(_req):
            nonlocal physical_available
            self.assertGreaterEqual(physical_available, physical_tokens_per_req)
            physical_available -= physical_tokens_per_req

        queue._pre_alloc = MagicMock(side_effect=pre_alloc)

        resumed = queue.resume_retracted_reqs([])

        self.assertEqual(resumed, reqs[:3])
        self.assertEqual(queue.retracted_queue, reqs[3:])
        self.assertEqual(physical_available, 3 * page_size)
        self.assertEqual(queue._pre_alloc.call_count, 3)

    def test_overlap_resume_restores_after_inflight_forward(self):
        # Under overlap, a request finished at the last commit can still have
        # its redundant decode step in flight, writing a slot the resumed
        # request may reuse: the restore waits for that forward first.
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor"
        )
        override.install()
        self.addCleanup(override.restore)

        events = []
        reqs = [
            SimpleNamespace(
                rid=f"req-{i}",
                origin_input_ids=[0] * 8,
                output_ids=[],
                is_retracted=True,
                load_kv_cache=MagicMock(
                    side_effect=lambda *_, i=i: events.append(("restore", i))
                ),
            )
            for i in range(2)
        ]
        forward_stream = object()
        schedule_stream = MagicMock()
        schedule_stream.wait_stream.side_effect = lambda s: events.append(("wait", s))

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.request_retirer = MagicMock()
        queue.retracted_queue = reqs.copy()
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: len(reqs))
        queue.token_to_kv_pool_allocator = MagicMock(page_size=1)
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.tree_cache = MagicMock()
        queue.enable_overlap = True
        queue.schedule_stream = schedule_stream
        queue.forward_stream = forward_stream
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(4096, None)
        )
        queue._prealloc_required_tokens = MagicMock(return_value=(8, 0))
        queue._prealloc_reservation_fits = MagicMock(return_value=True)
        queue._pre_alloc = MagicMock()

        self.assertEqual(queue.resume_retracted_reqs([]), reqs)
        self.assertEqual(
            events, [("wait", forward_stream), ("restore", 0), ("restore", 1)]
        )

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
        queue.request_retirer = MagicMock()
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
        queue._swa_aware_allocatable_token_budgets = MagicMock(return_value=(0, None))
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        queue.enable_priority_scheduling = False
        queue.enable_hisparse = False
        queue.enable_lora = False
        queue.metrics_sink = SimpleNamespace(enable_metrics=False)
        queue.output_streamer = MagicMock()

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ) as poll:
            preallocated, failed = queue.pop_preallocated([])

        poll.assert_called_once_with([receiver], queue.gloo_group)
        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        queue.output_streamer.stream_output.assert_called_once_with(
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
        queue.request_retirer = MagicMock()
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
        queue._swa_aware_allocatable_token_budgets = MagicMock(return_value=(0, None))
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        queue.enable_priority_scheduling = False
        queue.enable_hisparse = False
        queue.enable_lora = False
        queue.output_streamer = MagicMock()

        # Must not raise on the receiver __eq__ above.
        preallocated, failed = queue.pop_preallocated([])

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
            disagg=DisaggReqState(),
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            waiting_for_input=True,
            is_rebootstrap=False,
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.request_retirer = MagicMock()
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
        # SWA tail pre-allocation runs on a paged pool.
        queue.token_to_kv_pool_allocator = MagicMock(page_size=64)
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

        queue.enable_priority_scheduling = False
        queue.enable_hisparse = False
        queue.enable_lora = False
        queue.output_streamer = MagicMock()

        preallocated, failed = queue.pop_preallocated([])

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        queue._pre_alloc.assert_not_called()
        queue.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        # A req whose kv_receiver was already cleared must not crash on .abort().
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.request_retirer = MagicMock()
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
        queue.request_retirer = MagicMock()
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

    @patch("sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache")
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
            host_staged=False,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_deferred_kv_release = False
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.request_retirer = SimpleNamespace(retire_finished=tree_release(queue))
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()

        queue.enable_decode_hicache = False
        queue.enable_hisparse = False
        queue.output_streamer = MagicMock()
        queue.metrics_sink = SimpleNamespace(enable_metrics=False)

        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred([])

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        queue.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, checkpoint=False
        )

        receiver = FakeReceiver()
        receiver.kv_mgr = FakeKVManager.__new__(FakeKVManager)
        decode_req.kv_receiver = receiver
        queue.queue = [decode_req]
        queue.enable_deferred_kv_release = True
        queue.req_to_metadata_buffer_idx_allocator.reset_mock()
        mock_release_kv_cache.reset_mock()

        transferred = queue.pop_transferred([])

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, checkpoint=False
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
        self.assertEqual(queue.pop_transferred([]), [])
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
            self.assertEqual(queue.pop_transferred([]), [])
            discard.assert_called_once_with(req, queue.tree_cache, "host_pool")
        receiver.clear.assert_called_once_with()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        mock_release_kv_cache.assert_not_called()

    def test_fake_receiver_initializes_deferred_release_state(self):
        manager = MagicMock()
        receiver = FakeKVReceiver(manager, "")

        self.assertIs(receiver.kv_mgr, manager)
        self.assertFalse(receiver.abort_notified)

    @patch("sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache")
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
            host_staged=False,
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
        queue.request_retirer = SimpleNamespace(retire_finished=tree_release(queue))
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()
        queue.enable_decode_hicache = False
        queue.enable_hisparse = False
        queue.output_streamer = MagicMock()
        queue.metrics_sink = SimpleNamespace(enable_metrics=False)
        mock_poll.return_value = [KVPoll.Failed]

        self.assertEqual(queue.pop_transferred([]), [])
        self.assertFalse(receiver.abort_notified)
        queue._defer_release.assert_not_called()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, checkpoint=False
        )

    def _decode_scheduler(self, *, retracted=(), held=(), releasing=()):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.mode_owners = ModeOwners()
        scheduler.counters = LaunchCounters()
        scheduler.flags = EngineFlags()
        scheduler.lists = CoreLists()
        scheduler.lists.decoding = []
        scheduler.pending = PendingSteps()
        scheduler.lists.chunked_req = None
        scheduler.lists.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(
            grammar_queue=[], abort_requests=MagicMock()
        )
        scheduler.mode_owners.dllm = None
        scheduler.mode_owners.hisparse_coordinator = None
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.flags._engine_paused = False
        scheduler.mode_owners.disagg_prefill = None
        scheduler.mode_owners.disagg_decode = decode_owner_shell(
            prealloc_queue=SimpleNamespace(
                queue=[],
                retracted_queue=list(retracted),
                held_rebootstrap_reqs=list(held),
            ),
            transfer_queue=SimpleNamespace(queue=[]),
            releasing=list(releasing),
        )
        scheduler.decode_offload_manager = None
        return scheduler

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        scheduler = self._decode_scheduler(retracted=[object()])
        registry = request_registry_of(scheduler)

        self.assertFalse(registry.is_fully_idle())

    def test_held_rebootstrap_requests_keep_scheduler_non_idle(self):
        scheduler = self._decode_scheduler(held=[MagicMock(**{"kv.holds_kv": False})])
        registry = request_registry_of(scheduler)

        self.assertFalse(registry.is_fully_idle())
        # The paused flush ignores waiting requests; held ones own no KV.
        self.assertTrue(registry.is_fully_idle(ignore_waiting=True))

    def _offloading(self, scheduler, *reqs):
        """Give the decode owner an offload manager with each request's copies
        in flight: (rid, finished, ack_ids)."""
        manager = DecodeKVCacheOffloadManager.__new__(DecodeKVCacheOffloadManager)
        manager.ongoing_offload = {}
        manager.ongoing_backup = {}
        manager.tp_world_size = 1
        manager.cache_controller = MagicMock(ack_write_queue=[])
        manager.cache_controller.ack_backup_queue.qsize.return_value = 0
        manager._trigger_backup = MagicMock()
        out = []
        for rid, finished, ack_ids in reqs:
            req = MagicMock(rid=rid, disagg=DisaggReqState())
            req.finished.return_value = finished
            if ack_ids:
                req.disagg.offload = OffloadedState(ack_ids=list(ack_ids))
            for ack_id in ack_ids:
                manager.ongoing_offload[ack_id] = (rid, None, [], [], 0.0, [])
            out.append(req)
        owner = scheduler.disagg_decode
        owner.decode_offload_manager = manager
        owner.request_retirer = MagicMock()
        scheduler.decode_offload_manager = manager
        scheduler.stages = ModeStages(
            schedule=MagicMock(), hold_finished=owner.hold_finished
        )
        return manager, out

    @staticmethod
    def _land(manager, *ack_ids):
        manager.cache_controller.ack_write_queue = [
            SimpleNamespace(
                finish_event=SimpleNamespace(synchronize=lambda: None),
                node_ids=[ack_id],
            )
            for ack_id in ack_ids
        ]

    def test_finished_request_releases_after_its_last_offload_copy(self):
        scheduler = self._decode_scheduler()
        owner = scheduler.disagg_decode
        manager, (done, live, plain) = self._offloading(
            scheduler, ("done", True, [1, 2]), ("live", False, [3]), ("plain", True, [])
        )
        scheduler.lists.decoding = [done, live, plain]
        registry = scheduler.request_registry = request_registry_of(scheduler)

        # The commit: finished requests leave decoding; the one whose release
        # waits on a copy moves into releasing, the live one keeps decoding.
        scheduler.leave_finished_reqs()
        self.assertEqual(scheduler.lists.decoding, [live])
        self.assertEqual(owner.releasing, [done])
        assert_one_list_per_request(scheduler)
        self.assertFalse(registry.is_fully_idle())

        # Its first copy and the live request's land: it waits for the second.
        self._land(manager, 1, 3)
        owner.check_offload_progress()
        self.assertEqual(owner.releasing, [done])
        owner.request_retirer.retire_offloaded.assert_not_called()

        self._land(manager, 2)
        owner.check_offload_progress()
        self.assertEqual(owner.releasing, [])
        owner.request_retirer.retire_offloaded.assert_called_once_with(done)
        assert_one_list_per_request(scheduler)

    def test_abort_leaves_releasing_requests_to_their_ack(self):
        scheduler = self._decode_scheduler()
        owner = scheduler.disagg_decode
        manager, (done,) = self._offloading(scheduler, ("done", True, [1]))
        owner.releasing = [done]
        scheduler.mm_receiver = None
        scheduler.grammar_manager = MagicMock()
        registry = request_registry_of(scheduler)

        registry.abort_request(AbortReq(abort_all=True))

        self.assertEqual(owner.releasing, [done])
        owner.request_retirer.retire_offloaded.assert_not_called()
        self._land(manager, 1)
        owner.check_offload_progress()
        self.assertEqual(owner.releasing, [])

    def test_releasing_requests_keep_scheduler_non_idle(self):
        # A failed transfer held until prefill stops writing its pages.
        scheduler = self._decode_scheduler(releasing=[object()])
        registry = request_registry_of(scheduler)

        self.assertFalse(registry.is_fully_idle())
        # The paused flush would clear pages a release frees later.
        self.assertFalse(registry.is_fully_idle(ignore_waiting=True))
        # Releasing memory drops the held releases with the pool.
        self.assertTrue(registry.is_fully_idle(ignore_releasing=True))
        # The health check reads running requests only.
        self.assertTrue(registry.is_fully_idle(for_health_check=True))

    def test_abort_reaches_held_rebootstrap_requests(self):
        held, kept = MagicMock(rid="held-1"), MagicMock(rid="other")
        # A held request owns no KV: the retraction released it.
        held.kv.holds_kv = held.kv.holds_mamba = False
        scheduler = self._decode_scheduler(held=[held, kept])
        scheduler.mm_receiver = None
        scheduler.grammar_manager = MagicMock()
        scheduler.mode_owners.dllm = None
        scheduler.ipc_channels = MagicMock()
        release_aborted_request = MagicMock()
        retirer = RequestRetirer(
            kv=MagicMock(),
            beam_coordinator=MagicMock(),
            hisparse_coordinator=None,
            decode_offload_manager=None,
            output_streamer=MagicMock(),
            ipc_channels=scheduler.ipc_channels,
        )
        retirer.release_aborted_request = release_aborted_request
        scheduler.disagg_decode.request_retirer = retirer
        registry = request_registry_of(scheduler)

        registry.abort_request(AbortReq(rid="held"))

        self.assertEqual(
            scheduler.disagg_decode.prealloc_queue.held_rebootstrap_reqs, [kept]
        )
        release_aborted_request.assert_called_once_with(held)
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()


class _GateReceiver:
    def __init__(self, poll):
        self._poll = poll
        self.require_staging = False

    def poll(self):
        return self._poll

    def clear(self):
        pass

    def failure_exception(self):
        raise KVTransferError(
            7, "Failed due to an unknown reason from another rank", True
        )


def _gated_decode_req(poll, restore_status):
    return SimpleNamespace(
        req=SimpleNamespace(rid="r", bootstrap_room=7, return_logprob=False),
        kv_receiver=_GateReceiver(poll),
        metadata_buffer_index=3,
        hicache_restore_status=restore_status,
        host_staged=False,
    )


class TestDecodeHiCacheRestoreGate(CustomTestCase):
    def test_gate_folds_restore_outcome_into_terminal_poll_only(self):
        cases = [
            (KVPoll.Success, HiCacheRestoreResult.READY, KVPoll.Success),
            (KVPoll.Success, HiCacheRestoreResult.PENDING, KVPoll.Transferring),
            (KVPoll.Success, HiCacheRestoreResult.FAILED, KVPoll.Failed),
            (KVPoll.Transferring, HiCacheRestoreResult.FAILED, KVPoll.Transferring),
            (
                KVPoll.WaitingForInput,
                HiCacheRestoreResult.FAILED,
                KVPoll.WaitingForInput,
            ),
            (KVPoll.Failed, HiCacheRestoreResult.PENDING, KVPoll.Failed),
        ]
        for poll, status, expected in cases:
            with self.subTest(poll=poll, status=status):
                gated = HiCacheRestoreGatedKVReceiver(_gated_decode_req(poll, status))
                self.assertEqual(gated.poll(), expected)

    def test_failed_restore_keeps_reserving_admission_budget(self):
        def dr(status, lock):
            return SimpleNamespace(
                prefix_match=SimpleNamespace(restore_token_count=3000),
                hicache_restore_status=status,
                hicache_restore_lock=lock,
            )

        prealloc = DecodePreallocQueue.__new__(DecodePreallocQueue)
        prealloc.enable_decode_hicache = True
        prealloc.transfer_queue = SimpleNamespace(
            queue=[
                dr(HiCacheRestoreResult.PENDING, None),
                dr(HiCacheRestoreResult.FAILED, None),
                dr(HiCacheRestoreResult.PENDING, object()),
                dr(HiCacheRestoreResult.READY, object()),
            ]
        )
        self.assertEqual(prealloc._hicache_pending_restore_tokens(), 6000)

    def test_staging_poll_uses_gated_pollers(self):
        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.PENDING)
        with patch(
            "sglang.srt.disaggregation.utils._all_reduce_polls",
            side_effect=lambda polls, group: polls,
        ):
            polls = poll_and_all_reduce_with_staging(
                [decode_req],
                staging_handler=MagicMock(),
                gloo_group=MagicMock(),
                pollers=[HiCacheRestoreGatedKVReceiver(decode_req)],
            )
        self.assertEqual(polls, [KVPoll.Transferring])

    @patch("sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_pop_transferred_branches_on_reduced_poll_only(
        self, mock_prepare_abort, mock_release_kv_cache
    ):
        def make_queue(decode_req, peer_poll):
            queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
            queue.queue = [decode_req]
            queue.enable_staging = False
            queue.enable_host_receive = False
            queue.enable_deferred_kv_release = False
            queue.gloo_group = MagicMock()
            queue.req_to_metadata_buffer_idx_allocator = MagicMock()
            queue.tp_rank = 0
            queue.tree_cache = MagicMock()
            queue.request_retirer = SimpleNamespace(retire_finished=tree_release(queue))
            queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
            queue._clean_hicache_prefetch_resources = MagicMock()
            queue._commit_transfer_to_req = MagicMock()
            queue._record_transfer_failure = MagicMock()
            queue.enable_decode_hicache = True
            queue.enable_hisparse = False
            queue.output_streamer = MagicMock()
            queue.metrics_sink = SimpleNamespace(enable_metrics=False)

            def reduce_with_peer(
                pollers, group, decode_reqs=None, metadata_buffers=None
            ):
                return [min(int(p.poll()), int(peer_poll)) for p in pollers]

            patcher = patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                side_effect=reduce_with_peer,
            )
            patcher.start()
            self.addCleanup(patcher.stop)
            return queue

        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.FAILED)
        queue = make_queue(decode_req, peer_poll=KVPoll.Failed)
        self.assertEqual(queue.pop_transferred([]), [])
        self.assertEqual(queue.queue, [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with("HiCache restore failed")
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.READY)
        queue = make_queue(decode_req, peer_poll=KVPoll.Failed)
        self.assertEqual(queue.pop_transferred([]), [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with(
            "Failed due to an unknown reason from another rank"
        )
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        mock_release_kv_cache.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Transferring, HiCacheRestoreResult.FAILED)
        queue = make_queue(decode_req, peer_poll=KVPoll.Success)
        self.assertEqual(queue.pop_transferred([]), [])
        self.assertEqual(queue.queue, [decode_req])
        mock_prepare_abort.assert_not_called()
        mock_release_kv_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
