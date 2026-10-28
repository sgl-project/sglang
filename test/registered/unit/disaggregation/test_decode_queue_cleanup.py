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
from sglang.srt.disaggregation.kv_events import OffloadedState
from sglang.srt.disaggregation.req_state import DisaggReqState
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
)
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.request_lists import CoreLists
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
from sglang.test.scheduler_request_lists import partial_scheduler  # noqa: E402
from sglang.test.scheduler_request_lists import (
    assert_one_list_per_request,
    request_registry_of,
)
from sglang.test.separate_buffer_allocator_double import (
    bind_separate_buffer_capacity,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


def _run_as_single_rank(test: unittest.TestCase) -> None:
    # The transfer queue sizes its release collectives from the gloo group;
    # these tests have no process group, so answer as one rank.
    world_size = patch("torch.distributed.get_world_size", return_value=1)
    world_size.start()
    test.addCleanup(world_size.stop)


class TestDecodeQueueCleanup(CustomTestCase):
    def setUp(self):
        _run_as_single_rank(self)
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_overlap_resume_restores_without_waiting_on_the_forward(self):
        # A released request's slot and written pages go back only at the
        # commit of the last pending step that carries it, so the slot a
        # resumed request takes is written by no forward in flight.
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
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.request_retirer = MagicMock()
        queue.retracted_queue = reqs.copy()
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: len(reqs))
        queue.token_to_kv_pool_allocator = MagicMock(page_size=1)
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.tree_cache = MagicMock()
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(4096, None)
        )
        queue._prealloc_required_tokens = MagicMock(return_value=(8, 0))
        queue._prealloc_reservation_fits = MagicMock(return_value=True)
        queue._pre_alloc = MagicMock()

        self.assertEqual(queue.resume_retracted_reqs([]), reqs)
        self.assertEqual(events, [("restore", 0), ("restore", 1)])

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

    def _decode_scheduler(self, *, retracted=(), held=(), releasing=()):
        scheduler = partial_scheduler()
        scheduler.parts.mode_owners = ModeOwners()
        scheduler.parts.counters = LaunchCounters()
        scheduler.parts.flags = EngineFlags()
        scheduler.parts.lists = CoreLists()
        scheduler.parts.lists.decoding = []
        scheduler.parts.pending = PendingSteps()
        scheduler.parts.lists.chunked_req = None
        scheduler.parts.lists.waiting_queue = []
        scheduler.parts.grammar_manager = SimpleNamespace(
            grammar_queue=[], abort_requests=MagicMock()
        )
        scheduler.parts.mode_owners.dllm = None
        scheduler.parts.mode_owners.hisparse_coordinator = None
        scheduler.mode.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.parts.flags._engine_paused = False
        scheduler.parts.mode_owners.disagg_prefill = None
        scheduler.parts.mode_owners.disagg_decode = decode_owner_shell(
            prealloc_queue=SimpleNamespace(
                queue=[],
                retracted_queue=list(retracted),
                held_rebootstrap_reqs=list(held),
            ),
            transfer_queue=SimpleNamespace(queue=[]),
            releasing=list(releasing),
        )
        scheduler.parts.decode_offload_manager = None
        return scheduler

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
        scheduler.parts.decode_offload_manager = manager
        scheduler.mode.stages = ModeStages(
            commit_extend=MagicMock(),
            hold_finished=owner.hold_finished,
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
        scheduler.parts.lists.decoding = [done, live, plain]
        registry = scheduler.hub.request_registry = request_registry_of(scheduler)

        # The commit: finished requests leave decoding; the one whose release
        # waits on a copy moves into releasing, the live one keeps decoding.
        scheduler.leave_finished_reqs()
        self.assertEqual(scheduler.parts.lists.decoding, [live])
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
        scheduler.parts.grammar_manager = MagicMock()
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
        scheduler.parts.grammar_manager = MagicMock()
        scheduler.parts.mode_owners.dllm = None
        release_aborted_request = MagicMock()
        retirer = RequestRetirer(
            kv=MagicMock(),
            beam_coordinator=MagicMock(),
            hisparse_coordinator=None,
            decode_offload_manager=None,
            pending=PendingSteps(),
            output_streamer=MagicMock(),
        )
        retirer.release_aborted_request = release_aborted_request
        scheduler.disagg_decode.request_retirer = retirer
        registry = request_registry_of(scheduler)

        registry.abort_request(AbortReq(rid="held"))

        self.assertEqual(
            scheduler.disagg_decode.prealloc_queue.held_rebootstrap_reqs, [kept]
        )
        release_aborted_request.assert_called_once_with(held)
        retirer.output_streamer.send_abort_output.assert_called_once()


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
    def setUp(self):
        super().setUp()
        _run_as_single_rank(self)

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
            queue.metadata_buffers.clear_slot = lambda idx: (
                queue.metadata_buffers.bootstrap_room.__setitem__(idx, 0)
            )
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
