import contextlib
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeRequest,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
    _PendingPublish,
)
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.utils import ReqToMetadataIdxAllocator
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    ListKind,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import FakeReq, decode_queue
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

    def test_an_empty_queue_reads_no_capacity(self):
        """Polling an empty preallocation queue returns before the capacity
        budgets, whose reads after an allocation search the layout again."""
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = IndexedReqList(ListKind.PD_PREALLOC)
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        for budget in (
            "_allocatable_token_budgets",
            "_window_aware_allocatable_token_budgets",
            "_uses_swa_reservation",
        ):
            setattr(queue, budget, MagicMock(side_effect=AssertionError(budget)))

        self.assertEqual(queue.pop_preallocated(), ([], []))
        queue._resolve_pending_reqs.assert_called_once()

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


class TestHeldDestinationPublication(CustomTestCase):
    def setUp(self):
        super().setUp()
        _run_as_single_rank(self)
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def make_queue(self):
        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.gloo_group = object()
        receiver = SimpleNamespace(
            poll=MagicMock(return_value=KVPoll.WaitingForInput),
            send_metadata=MagicMock(),
        )
        event = SimpleNamespace(query=MagicMock(return_value=True))
        entry = SimpleNamespace(
            req=FakeReq(),
            kv_receiver=receiver,
            pending_metadata=(event, ("pages", 3, None), {}),
            pending_receive=None,
            is_rebootstrap=False,
        )
        queue.queue = decode_queue(ListKind.PD_RECEIVING, [entry])
        return queue, entry, event, receiver

    def test_waits_for_every_rank_and_sends_once(self):
        queue, entry, event, receiver = self.make_queue()
        with (
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_reduce") as reduce,
        ):
            # This rank finished its forward; another rank is still writing.
            reduce.side_effect = lambda flags, **_: flags.zero_()
            queue.publish_pending_metadata()
            receiver.send_metadata.assert_not_called()
            self.assertIsNotNone(entry.pending_metadata)

            # The local forward is now the one that still needs to finish.
            reduce.side_effect = None
            event.query.return_value = False
            queue.publish_pending_metadata()
            receiver.send_metadata.assert_not_called()

            event.query.return_value = True
            queue.publish_pending_metadata()
            queue.publish_pending_metadata()
            receiver.send_metadata.assert_called_once_with("pages", 3, None)
            self.assertIsNone(entry.pending_metadata)

    def test_failed_receiver_keeps_destination_unpublished(self):
        queue, entry, _, receiver = self.make_queue()
        receiver.poll.return_value = KVPoll.Failed
        queue.publish_pending_metadata()
        receiver.send_metadata.assert_not_called()
        self.assertIsNotNone(entry.pending_metadata)

    def test_publishes_while_retracted_requests_wait(self):
        queue, _, _, receiver = self.make_queue()
        queue.resolve_deferred_releases = MagicMock()
        prealloc = SimpleNamespace(
            publish_pending_destinations=MagicMock(),
            resume_retracted_reqs=lambda: [],
            retracted_queue=[object()],
        )
        scheduler = SimpleNamespace(
            scheduler_stage_metrics=None,
            tree_cache=SimpleNamespace(linker=None),
            enable_decode_hicache=False,
            disagg_decode_prealloc_queue=prealloc,
            disagg_decode_transfer_queue=queue,
            waiting_queue=IndexedReqList(ListKind.WAITING),
        )
        SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)
        receiver.send_metadata.assert_called_once_with("pages", 3, None)


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
        req=FakeReq(rid="r", bootstrap_room=7, return_logprob=False),
        kv_receiver=_GateReceiver(poll),
        metadata_buffer_index=3,
        hicache_restore_status=restore_status,
        pending_metadata=None,
        pending_receive=None,
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

    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    def test_pop_transferred_branches_on_reduced_poll_only(self, mock_prepare_abort):
        def make_queue(decode_req, peer_poll):
            queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
            queue.queue = decode_queue(ListKind.PD_RECEIVING, [decode_req])
            queue.enable_staging = False
            queue.enable_host_receive = False
            queue.enable_deferred_kv_release = False
            queue.gloo_group = MagicMock()
            queue.req_to_metadata_buffer_idx_allocator = MagicMock()
            queue.tp_rank = 0
            queue.tree_cache = MagicMock()
            queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
            queue.metadata_buffers.clear_slot = lambda idx: (
                queue.metadata_buffers.bootstrap_room.__setitem__(idx, 0)
            )
            queue._clean_hicache_prefetch_resources = MagicMock()
            queue._commit_transfer_to_req = MagicMock()
            queue._record_transfer_failure = MagicMock()
            scheduler = MagicMock()
            scheduler.enable_decode_hicache = True
            scheduler.enable_hisparse = False
            scheduler.metrics_reporter.enable_metrics = False
            queue.scheduler = scheduler

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
        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(list(queue.queue.items()), [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with("HiCache restore failed")
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Success, HiCacheRestoreResult.READY)
        queue = make_queue(decode_req, peer_poll=KVPoll.Failed)
        self.assertEqual(queue.pop_transferred(), [])
        queue._commit_transfer_to_req.assert_not_called()
        queue._record_transfer_failure.assert_called_once_with(
            "Failed due to an unknown reason from another rank"
        )
        mock_prepare_abort.assert_called_once()

        mock_prepare_abort.reset_mock()
        decode_req = _gated_decode_req(KVPoll.Transferring, HiCacheRestoreResult.FAILED)
        queue = make_queue(decode_req, peer_poll=KVPoll.Success)
        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(list(queue.queue.items()), [decode_req])
        mock_prepare_abort.assert_not_called()
        queue.scheduler.request_retirer.release.assert_not_called()


class _PollingReceiver:
    def __init__(self, poll=KVPoll.WaitingForInput):
        self.clear_called = False
        self.conclude_state = None
        self._poll = poll
        self.send_metadata = MagicMock()

    def poll(self):
        return self._poll

    def clear(self):
        self.clear_called = True

    def failure_exception(self):
        return None


def _local_reduce(flags, group, op):
    return flags


class _HoldingStorage:
    """The storage surface a deferred publish uses: the move count, the
    receive-row hold, and network-transfer recording (none: no rows move)."""

    owner = "full"

    def __init__(self, epochs=(0,)):
        self._epochs = iter(epochs)
        self._epoch = 0
        self.holds = {}
        self.ended = []
        self.recorded = []

    def move_epoch(self):
        self._epoch = next(self._epochs, self._epoch)
        return self._epoch

    def hold_receive_rows(self, req, *, token_start, token_end, signal):
        self.holds[req.rid] = (token_start, token_end, signal)

    def end_receive_hold(self, req):
        self.holds.pop(req.rid, None)
        self.ended.append(req.rid)

    def wire_entries(self):
        return ()

    def record_network_transfer(self, req, *, pages, page_size, states, writes, peer):
        self.recorded.append((req.rid, self.owner, list(np.asarray(pages))))


class TestDeferredDestinationPublish(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        for target in ("stream", "current_stream", "Event"):
            patcher = patch(f"sglang.srt.disaggregation.decode.torch.cuda.{target}")
            patcher.start()
            self.addCleanup(patcher.stop)
        self.storage = _HoldingStorage()
        patcher = patch(
            "sglang.srt.disaggregation.decode.kv_storage_of",
            side_effect=lambda *_: self.storage,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _deferring_queue(self, decode_reqs):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.gloo_group = object()
        queue.queue = decode_queue(ListKind.PD_PREALLOC, decode_reqs)
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._num_published_destinations = 0
        queue._defer_publish = True
        queue._pending_publish = []
        queue._publish_stream = MagicMock()
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._uses_swa_reservation = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=1000)
        queue._prealloc_reservation_fits = MagicMock(return_value=True)
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 4
        queue.req_to_token_pool.holds_state_slots = False
        queue.req_to_token_pool.req_to_token = torch.arange(32).reshape(4, 8)

        def pre_alloc(req, *args):
            req.kv.req_pool_idx = 1
            return queue.req_to_token_pool.req_to_token[1, : len(req.origin_input_ids)]

        queue._pre_alloc = MagicMock(side_effect=pre_alloc)
        queue.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(4)
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=1,
            state=None,
            translate_kv_indices_for_transfer=lambda indices: indices,
            window_pool_names=lambda: (),
        )
        queue.kv_manager = SimpleNamespace(
            kv_args=SimpleNamespace(state_types=[], state_sub_pools=[])
        )
        queue.transfer_queue = SimpleNamespace(
            queue=decode_queue(ListKind.PD_RECEIVING),
            enable_staging=False,
            enable_deferred_kv_release=False,
        )
        queue.tree_cache = MagicMock()
        queue.scheduler = MagicMock(
            enable_priority_scheduling=False,
            enable_hisparse=False,
            enable_lora=False,
            enable_decode_hicache=False,
        )
        queue.scheduler.running_batch.reqs = []
        return queue

    @staticmethod
    def _decode_req(receiver):
        req = FakeReq(
            rid="deferred",
            origin_input_ids=[1] * 5,
            output_ids=[],
            kv=ReqKvInfo(),
            finished_reason=None,
            sampling_params=SimpleNamespace(max_new_tokens=4),
            time_stats=MagicMock(),
        )
        return DecodeRequest(req=req, kv_receiver=receiver, waiting_for_input=True)

    @staticmethod
    def _pending(decode_req):
        decode_req.awaiting_publish = True
        return _PendingPublish(
            allocated=MagicMock(),
            decode_req=decode_req,
            dst_kv_indices=torch.arange(4),
            prefix_len=0,
            total_prefix_len=0,
            fill_len=4,
        )

    def test_deferred_destination_keeps_its_metadata_slot_until_published(self):
        # The transfer queue frees the metadata slot of any request it fails, so
        # a deferred request must own one before joining the queue, and the
        # publish must send that slot instead of taking a second one.
        receiver = _PollingReceiver()
        decode_req = self._decode_req(receiver)
        queue = self._deferring_queue([decode_req])

        self.assertEqual(queue.pop_preallocated(), ([decode_req], []))

        receiver.send_metadata.assert_not_called()
        self.assertTrue(decode_req.awaiting_publish)
        self.assertEqual(decode_req.metadata_buffer_index, 0)
        self.assertEqual(queue.req_to_metadata_buffer_idx_allocator.available_size(), 3)
        entry_time = decode_req.req.time_stats.set_decode_transfer_queue_entry_time
        entry_time.assert_called_once()
        # transfer_queue.extend() takes the request over.
        queue.note_destinations_queued(1)

        allocated = queue._pending_publish[0].allocated
        with patch(
            "sglang.srt.disaggregation.decode.all_reduce_flags",
            side_effect=_local_reduce,
        ):
            queue.publish_pending_destinations()

        queue._publish_stream.wait_event.assert_called_once_with(allocated)
        receiver.send_metadata.assert_called_once()
        entry_time.assert_called_once()
        page_indices, metadata_index, state_indices = (
            receiver.send_metadata.call_args.args
        )
        self.assertEqual(page_indices.tolist(), [8, 9, 10, 11, 12])
        self.assertEqual(metadata_index, 0)
        self.assertEqual(queue.req_to_metadata_buffer_idx_allocator.available_size(), 3)
        self.assertFalse(decode_req.awaiting_publish)
        self.assertEqual(queue._pending_publish, [])
        self.assertFalse(queue.has_published_destinations)

    def test_deferred_publish_waits_for_a_page_move_since_its_allocation(self):
        """A page move between the allocation and the deferred publish rebinds
        the destination ids on the schedule stream, so the publish reads them
        back after that stream instead of after the allocation alone."""
        receiver = _PollingReceiver()
        decode_req = self._decode_req(receiver)
        queue = self._deferring_queue([decode_req])
        # A relief that could not keep the held rows moved pages in between.
        self.storage = _HoldingStorage(epochs=(0, 1))
        self.assertEqual(queue.pop_preallocated(), ([decode_req], []))
        queue.note_destinations_queued(1)
        with patch(
            "sglang.srt.disaggregation.decode.all_reduce_flags",
            side_effect=_local_reduce,
        ):
            queue.publish_pending_destinations()

        queue._publish_stream.wait_event.assert_not_called()
        queue._publish_stream.wait_stream.assert_called_once()
        receiver.send_metadata.assert_called_once()
        self.assertEqual(self.storage.ended, ["deferred"])

    def test_inline_publish_holds_the_send_until_the_forward_is_done(self):
        """Where the deferred publish is off (staging, HiSparse, no WAR
        barrier), overlap still reuses pages the in-flight forward may write, so
        the inline publish records an event on the forward stream and holds the
        send for the transfer queue."""
        receiver = _PollingReceiver()
        decode_req = self._decode_req(receiver)
        queue = self._deferring_queue([decode_req])
        queue._defer_publish = False
        recorded = []
        ready = SimpleNamespace(record=lambda stream: recorded.append(stream))
        queue.scheduler.enable_overlap = True
        queue.scheduler.device_module = SimpleNamespace(Event=lambda: ready)
        queue.scheduler.forward_stream = object()

        self.assertEqual(queue.pop_preallocated(), ([decode_req], []))

        receiver.send_metadata.assert_not_called()
        self.assertEqual(recorded, [queue.scheduler.forward_stream])
        self.assertFalse(decode_req.awaiting_publish)
        event, args, kwargs = decode_req.pending_metadata
        self.assertIs(event, ready)
        self.assertEqual(args[0].tolist(), [8, 9, 10, 11, 12])
        self.assertEqual(args[1], decode_req.metadata_buffer_index)
        self.assertEqual(kwargs, {"decode_prefix_len": 0})
        self.assertEqual(decode_req.metadata_buffer_index, 0)
        self.assertTrue(queue.has_published_destinations)

    def test_staging_readback_preserves_payload_and_waits_for_both_events(self):
        payloads = []
        original_empty, original_copy = torch.empty, torch.Tensor.copy_
        for master, staging in ((False, True), (True, False), (True, True)):
            receiver = _PollingReceiver()
            receiver.require_staging = True
            entry = self._decode_req(receiver)
            entry.req.bootstrap_room, entry.req.bootstrap_host = 17, "127.0.0.1"
            queue = self._deferring_queue([entry])
            queue._defer_publish = False
            queue.scheduler.enable_overlap = True
            forward_done, readback_done = MagicMock(), MagicMock()
            forward_done.query.return_value = readback_done.query.return_value = False
            queue.scheduler.device_module = SimpleNamespace(Event=lambda: forward_done)
            queue.scheduler.forward_stream = object()
            queue.transfer_queue.enable_staging = staging
            queue.transfer_queue.staging_handler = MagicMock()
            queue.token_to_kv_pool_allocator.page_size = 2
            queue.token_to_kv_pool_allocator.state_ids = lambda _: SimpleNamespace(
                ids=lambda x: x + 100
            )
            queue.req_to_token_pool.req_index_to_mamba_index_mapping = torch.arange(
                4
            ).reshape(4, 1)
            queue.kv_manager.kv_args.state_types = [StateType.MAMBA]
            queue.kv_manager.kv_args.state_sub_pools = [None]
            # A held send re-reads its rows' ids at publish.
            queue.scheduler.disagg_decode_prealloc_queue = queue

            def pinned_empty(shape, *, dtype, pin_memory):
                return original_empty(shape, dtype=dtype)

            def copy(host, device, *, non_blocking):
                self.assertTrue(non_blocking)
                return original_copy(host, device)

            with (
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(master),
                patch(
                    "sglang.srt.disaggregation.decode.torch.cuda.is_available",
                    return_value=True,
                ),
                patch(
                    "sglang.srt.disaggregation.decode.torch.cuda.Event",
                    return_value=readback_done,
                ),
                patch(
                    "sglang.srt.disaggregation.decode.torch.empty",
                    side_effect=pinned_empty,
                ),
                patch.object(torch.Tensor, "copy_", copy),
            ):
                queue.pop_preallocated()
            transfer = DecodeTransferQueue.__new__(DecodeTransferQueue)
            transfer.queue = decode_queue(ListKind.PD_RECEIVING, [entry])
            transfer.gloo_group = object()
            transfer.scheduler = queue.scheduler
            with patch(
                "sglang.srt.disaggregation.decode.all_reduce_flags",
                side_effect=_local_reduce,
            ):
                transfer.publish_pending_metadata()
                receiver.send_metadata.assert_not_called()
                forward_done.query.return_value = True
                transfer.publish_pending_metadata()
                self.assertEqual(
                    receiver.send_metadata.call_count, int(not (master and staging))
                )
                readback_done.query.return_value = True
                transfer.publish_pending_metadata()
            receiver.send_metadata.assert_called_once()
            pages, slot, states = receiver.send_metadata.call_args.args
            payloads.append(
                (
                    pages.tolist(),
                    slot,
                    [[x.tolist() for x in part] for part in states],
                    receiver.send_metadata.call_args.kwargs,
                )
            )
            self.assertIsNone(entry.pending_metadata)
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[1], payloads[2])
        self.assertEqual(
            payloads[1], ([4, 5, 6], 0, [[[101]]], {"decode_prefix_len": 0})
        )

    def test_inline_publish_holds_its_rows_until_the_send(self):
        """REGRESSION: the inline publish holds its rows, not records them as
        the peer's, until the send, which re-reads their ids after a page move."""
        receiver = _PollingReceiver()
        decode_req = self._decode_req(receiver)
        queue = self._deferring_queue([decode_req])
        queue._defer_publish = False
        ready = MagicMock()
        ready.query.return_value = True
        queue.scheduler.enable_overlap = True
        queue.scheduler.device_module = SimpleNamespace(Event=lambda: ready)
        queue.scheduler.disagg_decode_prealloc_queue = queue
        # A page moves between the hold and the send.
        self.storage = _HoldingStorage(epochs=(0, 1))

        self.assertEqual(queue.pop_preallocated(), ([decode_req], []))
        self.assertIn(decode_req.req.rid, self.storage.holds)
        self.assertEqual(self.storage.recorded, [])
        receiver.send_metadata.assert_not_called()

        moved = queue.token_to_kv_pool_allocator
        moved.translate_kv_indices_for_transfer = lambda indices: indices + 100
        transfer = DecodeTransferQueue.__new__(DecodeTransferQueue)
        transfer.queue = decode_queue(ListKind.PD_RECEIVING, [decode_req])
        transfer.gloo_group = object()
        transfer.scheduler = queue.scheduler
        with patch(
            "sglang.srt.disaggregation.decode.all_reduce_flags",
            side_effect=_local_reduce,
        ):
            transfer.publish_pending_metadata()

        receiver.send_metadata.assert_called_once()
        page_indices = receiver.send_metadata.call_args.args[0]
        self.assertEqual(page_indices.tolist(), [108, 109, 110, 111, 112])
        self.assertEqual(
            self.storage.recorded,
            [(decode_req.req.rid, "full", [108, 109, 110, 111, 112])],
        )
        self.assertEqual(self.storage.ended, [decode_req.req.rid])
        self.assertIsNone(decode_req.pending_receive)

    def test_recompute_waits_for_held_metadata_to_be_sent(self):
        receiver = _PollingReceiver()
        decode_req = self._decode_req(receiver)
        decode_req.is_rebootstrap = True
        queue = self._deferring_queue([decode_req])
        queue._defer_publish = False
        ready = MagicMock()
        ready.query.return_value = False
        queue.scheduler.enable_overlap = True
        queue.scheduler.device_module = SimpleNamespace(Event=lambda: ready)
        queue.scheduler.disagg_decode_prealloc_queue = queue
        queue._submit_prefill_recompute = MagicMock(
            side_effect=lambda _: receiver.send_metadata.assert_called_once()
        )

        queue.pop_preallocated()
        queue._submit_prefill_recompute.assert_not_called()
        transfer = DecodeTransferQueue.__new__(DecodeTransferQueue)
        transfer.queue = decode_queue(ListKind.PD_RECEIVING, [decode_req])
        transfer.gloo_group = object()
        transfer.scheduler = queue.scheduler
        with patch(
            "sglang.srt.disaggregation.decode.all_reduce_flags",
            side_effect=_local_reduce,
        ):
            transfer.publish_pending_metadata()
            queue._submit_prefill_recompute.assert_not_called()
            ready.query.return_value = True
            transfer.publish_pending_metadata()
        queue._submit_prefill_recompute.assert_called_once_with(decode_req)

    def test_deferred_destination_holds_its_rows_until_published(self):
        """A later allocation in the same pop must not move a deferred
        request's rows before its publish reads their ids back. Unified memory
        keeps no pool-wide move gate: the deferral holds the request's receive
        rows in place, which a relief keeps whenever it can, until the publish
        has recorded the rows' transfer. The destination still counts as
        published until the transfer queue takes it."""
        queue = self._deferring_queue([self._decode_req(_PollingReceiver())])

        queue.pop_preallocated()

        self.assertTrue(queue.has_published_destinations)
        start, end, signal = self.storage.holds["deferred"]
        self.assertEqual((start, end), (0, 5))
        self.assertIs(signal, queue._pending_publish[0].allocated)
        queue.note_destinations_queued(1)
        self.assertFalse(queue.has_published_destinations)
        with patch(
            "sglang.srt.disaggregation.decode.all_reduce_flags",
            side_effect=_local_reduce,
        ):
            queue.publish_pending_destinations()
        self.assertEqual(self.storage.holds, {})
        self.assertEqual(self.storage.ended, ["deferred"])

    def test_released_receiver_keeps_the_publish_reduction_aligned(self):
        """A pending request the transfer queue released keeps its slot in the
        all-reduced polls, polls as failed and is not published."""
        released = self._decode_req(None)
        live = self._decode_req(_PollingReceiver())
        queue = self._deferring_queue([])
        queue._pending_publish = [self._pending(released), self._pending(live)]
        queue._publish_destination = MagicMock()

        with patch(
            "sglang.srt.disaggregation.decode.all_reduce_flags",
            side_effect=_local_reduce,
        ) as reduce:
            queue.publish_pending_destinations()

        self.assertEqual(
            reduce.call_args.kwargs["flags"],
            [int(KVPoll.Failed), int(KVPoll.WaitingForInput)],
        )
        queue._publish_destination.assert_called_once()
        self.assertIs(queue._publish_destination.call_args.kwargs["decode_req"], live)
        self.assertTrue(released.awaiting_publish)
        self.assertFalse(live.awaiting_publish)
        # Both holds end: the released one unpublished, the live one published.
        self.assertEqual(self.storage.ended, ["deferred", "deferred"])


class TestPublishedDestinationHandoff(CustomTestCase):
    def test_transfer_queue_takes_published_destinations_over(self):
        """REGRESSION: `DecodeTransferQueue.extend` takes published destinations
        over, so the host backing's peer-write gate reopens once they drain."""
        prealloc = DecodePreallocQueue.__new__(DecodePreallocQueue)
        prealloc._num_published_destinations = 2
        transfer = DecodeTransferQueue.__new__(DecodeTransferQueue)
        transfer.queue = decode_queue(ListKind.PD_RECEIVING)
        transfer._deferred_releases = IndexedReqList(
            ListKind.PD_RELEASING, req_of=lambda entry: entry[0].req
        )
        transfer.scheduler = SimpleNamespace(disagg_decode_prealloc_queue=prealloc)
        prealloc.transfer_queue = transfer
        decode_reqs = [
            SimpleNamespace(req=FakeReq(rid=f"r{i}"), host_staged=True)
            for i in range(2)
        ]

        transfer.extend(decode_reqs)
        self.assertFalse(prealloc.has_published_destinations)
        self.assertFalse(prealloc._host_destinations_quiescent())

        # Both receives complete and leave the queue.
        transfer.queue = decode_queue(ListKind.PD_RECEIVING)
        self.assertTrue(prealloc._host_destinations_quiescent())


def _admit_req(**overrides):
    fields = dict(
        grammar=None,
        output_dsa_topk_indices=None,
        kv=SimpleNamespace(retraction_backup=None),
        hidden_states_tensor=object(),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestAdmissionOverlap(CustomTestCase):
    def test_scope_uses_admitted_requests_and_keeps_host_read_paths_stock(self):
        scheduler = SimpleNamespace(
            enable_overlap=True,
            spec_algorithm=SimpleNamespace(is_eagle=lambda: True),
            enable_hisparse=False,
        )
        disagg = SimpleNamespace(
            disaggregation_decode_host_receive_threshold=1,
            disaggregation_decode_enable_radix_cache=False,
        )
        with patch("sglang.srt.disaggregation.decode.get_disagg", return_value=disagg):
            for enabled in (False, True):
                with envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(enabled):
                    scope = (
                        SchedulerDisaggregationDecodeMixin._nonblocking_admit_in_scope
                    )
                    self.assertEqual(
                        scope(scheduler, SimpleNamespace(reqs=[_admit_req()])), enabled
                    )
                    self.assertFalse(
                        scope(
                            scheduler,
                            SimpleNamespace(reqs=[_admit_req(grammar=object())]),
                        )
                    )

    def test_seed_is_ordered_on_forward_stream_without_schedule_wait(self):
        for nonblocking in (False, True):
            batch = MagicMock()
            batch.is_empty.return_value = False
            calls = []

            @contextlib.contextmanager
            def forward_context():
                calls.append("enter_forward")
                yield
                calls.append("leave_forward")

            scheduler = SimpleNamespace(
                _get_new_prebuilt_batch=lambda _: batch,
                _nonblocking_admit_in_scope=lambda _: nonblocking,
                _verify_kv_checksums=lambda _: calls.append("verify"),
                ngram_embedding_manager=MagicMock(),
                chunked_req=None,
                future_map=object(),
                enable_overlap=True,
                schedule_stream=MagicMock(),
                forward_stream=MagicMock(),
                forward_stream_ctx=forward_context(),
            )
            batch.process_prebuilt.side_effect = lambda _: calls.append("seed")
            result = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(
                scheduler, object()
            )
            self.assertIs(result, batch)
            if nonblocking:
                self.assertEqual(
                    calls, ["verify", "enter_forward", "seed", "leave_forward"]
                )
                scheduler.schedule_stream.wait_stream.assert_not_called()
                scheduler.forward_stream.wait_stream.assert_called_once_with(
                    scheduler.schedule_stream
                )
                batch.req_pool_indices.record_stream.assert_called_once_with(
                    scheduler.forward_stream
                )
                batch.seq_lens.record_stream.assert_called_once_with(
                    scheduler.forward_stream
                )
            else:
                self.assertEqual(calls, ["verify", "seed"])
                scheduler.schedule_stream.wait_stream.assert_called_once_with(
                    scheduler.forward_stream
                )
                scheduler.forward_stream.wait_stream.assert_not_called()


class TestPreallocRequired(CustomTestCase):
    def _queue(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=4)
        queue.num_reserved_decode_tokens = 8
        queue._window_tails = lambda fill_len: {"window_16": min(fill_len, 16)}
        return queue

    def test_owner_fill_and_window_tails_in_whole_pages(self):
        """BUG REGRESSION. A PD decode preallocation sizes the owner's fill and
        each window's tail in whole pages, plus the decode reservation."""
        from sglang.srt.runtime_context import get_context

        # Ten prompt tokens and three outputs: the last output is the pending
        # decode input, so twelve tokens of KV are filled.
        req = SimpleNamespace(origin_input_ids=[0] * 10, output_ids=[1, 2, 3])
        for disable_radix_cache, window_reserved in ((False, 8), (True, 0)):
            with self.subTest(disable_radix_cache=disable_radix_cache):
                override = get_context().override_server_args(
                    disable_radix_cache=disable_radix_cache
                )
                override.install()
                try:
                    full, windows = self._queue()._prealloc_required(req)
                finally:
                    override.restore()
                self.assertEqual(full, 12 + 8)
                self.assertEqual(windows, {"window_16": 12 + window_reserved})


if __name__ == "__main__":
    unittest.main()
