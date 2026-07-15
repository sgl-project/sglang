import concurrent.futures
import threading
import unittest
from collections import defaultdict
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager, PrefillServerInfo
from sglang.srt.disaggregation.common.staging_handler import DecodeStagingHandler
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk
from sglang.srt.disaggregation.decode_hicache_mixin import (
    HiCacheRestoreGatedKVReceiver,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.mooncake.conn import (
    ABORT_RETRY_INTERVAL_S,
    QUIESCE_CAPABILITY_BYTES,
    KVArgsRegisterInfo,
    MooncakeKVManager,
    MooncakeKVReceiver,
    RoomTransferLifetime,
    TransferInfo,
    _has_kv_registration_capability,
    _has_transfer_metadata_capability,
    _submit_transfer_futures,
    _wait_transfer_futures,
)
from sglang.srt.disaggregation.utils import (
    defer_chunked_prefill_abort,
    poll_and_all_reduce,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRoomTransferLifetime(unittest.TestCase):
    def test_abort_waits_for_native_worker_before_ack(self):
        lifetime = RoomTransferLifetime()
        self.assertTrue(lifetime.acquire())
        lifetime.abort()
        ack_sent = threading.Event()

        waiter = threading.Thread(target=lambda: (lifetime.wait(), ack_sent.set()))
        waiter.start()
        self.assertFalse(ack_sent.wait(0.05))
        self.assertFalse(lifetime.acquire())

        lifetime.release()
        self.assertTrue(ack_sent.wait(1))
        waiter.join()


class TestExecutorDrain(unittest.TestCase):
    def test_failure_waits_for_running_sibling(self):
        sibling_started = threading.Event()
        release_sibling = threading.Event()
        returned = threading.Event()

        def sibling():
            sibling_started.set()
            release_sibling.wait()
            return 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(lambda: -1),
                executor.submit(sibling),
            ]
            self.assertTrue(sibling_started.wait(1))
            result = []
            waiter = threading.Thread(
                target=lambda: (
                    result.append(_wait_transfer_futures(futures)),
                    returned.set(),
                )
            )
            waiter.start()
            self.assertFalse(returned.wait(0.05))
            release_sibling.set()
            self.assertTrue(returned.wait(1))
            waiter.join()
        self.assertEqual(result, [-1])

    def test_submit_failure_waits_for_already_running_future(self):
        started = threading.Event()
        release = threading.Event()
        returned = threading.Event()
        errors = []

        def blocked():
            started.set()
            release.wait()
            return 0

        class RaiseOnSecondSubmit:
            def __init__(self):
                self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                self.calls = 0

            def submit(self, fn, *args):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("submit failed")
                return self.pool.submit(fn, *args)

        executor = RaiseOnSecondSubmit()

        def submit_all():
            try:
                _submit_transfer_futures(
                    executor,
                    [(blocked, ()), (lambda: 0, ())],
                )
            except Exception as error:
                errors.append(error)
            finally:
                returned.set()

        waiter = threading.Thread(target=submit_all)
        waiter.start()
        self.assertTrue(started.wait(1))
        self.assertFalse(returned.wait(0.05))
        release.set()
        self.assertTrue(returned.wait(1))
        waiter.join()
        executor.pool.shutdown()
        self.assertEqual(str(errors[0]), "submit failed")

    def test_ascend_custom_pool_submit_failure_drains_running_future(self):
        started = threading.Event()
        release = threading.Event()
        returned = threading.Event()
        errors = []

        manager = AscendKVManager.__new__(AscendKVManager)
        manager.pp_size = 1
        manager.enable_custom_mem_pool = True
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[100, 200],
            kv_item_lens=[8, 8],
        )

        def transfer(*_args):
            started.set()
            release.wait()
            return 0

        manager._transfer_data = transfer

        class RaiseOnSecondSubmit:
            def __init__(self):
                self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                self.calls = 0

            def submit(self, fn, *args):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("ascend submit failed")
                return self.pool.submit(fn, *args)

        executor = RaiseOnSecondSubmit()

        def send():
            try:
                manager.send_kvcache(
                    "session",
                    np.array([0], dtype=np.int32),
                    [300, 400],
                    np.array([0], dtype=np.int32),
                    executor,
                )
            except Exception as error:
                errors.append(error)
            finally:
                returned.set()

        waiter = threading.Thread(target=send)
        waiter.start()
        self.assertTrue(started.wait(1))
        self.assertFalse(returned.wait(0.05))
        release.set()
        self.assertTrue(returned.wait(1))
        waiter.join()
        executor.pool.shutdown()
        self.assertEqual(str(errors[0]), "ascend submit failed")


class TestCollectiveFailureQuiescence(unittest.TestCase):
    def test_failed_is_delayed_until_every_rank_is_quiesced(self):
        poller = SimpleNamespace(
            poll=Mock(return_value=KVPoll.Transferring),
            begin_failure_quiescence=Mock(),
            is_transfer_quiesced=Mock(return_value=True),
        )
        collective_call = 0

        def all_reduce(tensor, **_kwargs):
            nonlocal collective_call
            if collective_call % 2 == 0:
                tensor[0] = KVPoll.Failed
            elif collective_call == 1:
                tensor[0] = 0
            collective_call += 1

        with patch(
            "sglang.srt.disaggregation.utils.dist.all_reduce", side_effect=all_reduce
        ):
            self.assertEqual(
                poll_and_all_reduce([poller], object()), [KVPoll.Transferring]
            )
            self.assertEqual(poll_and_all_reduce([poller], object()), [KVPoll.Failed])

        self.assertEqual(poller.begin_failure_quiescence.call_count, 2)

    def test_hicache_receiver_delegates_failure_quiescence(self):
        receiver = SimpleNamespace(
            poll=Mock(return_value=KVPoll.Failed),
            begin_failure_quiescence=Mock(return_value="begun"),
            is_transfer_quiesced=Mock(return_value=False),
            is_failure_quiescing=Mock(return_value=True),
        )
        gated = HiCacheRestoreGatedKVReceiver(
            SimpleNamespace(
                kv_receiver=receiver,
                hicache_restore_status=HiCacheRestoreResult.PENDING,
            )
        )

        self.assertEqual(gated.poll(), KVPoll.Failed)
        self.assertEqual(gated.begin_failure_quiescence(), "begun")
        self.assertFalse(gated.is_transfer_quiesced())
        self.assertTrue(gated.is_failure_quiescing())
        receiver.begin_failure_quiescence.assert_called_once_with()
        receiver.is_transfer_quiesced.assert_called_once_with()
        receiver.is_failure_quiescing.assert_called_once_with()


class TestAbortAckTokens(unittest.TestCase):
    @staticmethod
    def _make_receiver(bootstrap_infos):
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.bootstrap_room = 9
        receiver.bootstrap_infos = bootstrap_infos
        receiver.kv_mgr = SimpleNamespace(local_ip="127.0.0.1", rank_port=1234)
        receiver._expected_abort_acks = set()
        receiver._received_abort_acks = set()
        receiver._abort_targets = []
        receiver._abort_ack_lock = threading.Lock()
        receiver._last_abort_send = float("-inf")
        return receiver

    def test_abort_sends_a_unique_token_to_every_participant(self):
        receiver = self._make_receiver([{"rank": 0}, {"rank": 1}])
        sockets = [Mock(), Mock()]
        receiver._connect_to_bootstrap_server = Mock(
            side_effect=[(socket, threading.Lock()) for socket in sockets]
        )

        receiver._send_abort_notification()

        tokens = [socket.send_multipart.call_args.args[0][4] for socket in sockets]
        self.assertEqual(len(set(tokens)), 2)
        self.assertEqual(receiver._expected_abort_acks, set(tokens))

    def test_abort_retries_missing_ack_with_stable_token(self):
        infos = [{"rank": 0}, {"rank": 1}]
        receiver = self._make_receiver(infos)
        sockets = {0: Mock(), 1: Mock()}
        receiver._connect_to_bootstrap_server = lambda info: (
            sockets[info["rank"]],
            threading.Lock(),
        )

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic",
            side_effect=[0, ABORT_RETRY_INTERVAL_S / 2, ABORT_RETRY_INTERVAL_S],
        ):
            receiver._send_abort_notification()
            tokens = {
                rank: sockets[rank].send_multipart.call_args.args[0][4]
                for rank in sockets
            }
            receiver._send_abort_notification()
            receiver._record_abort_ack(tokens[0])
            receiver._send_abort_notification()

        self.assertEqual(sockets[0].send_multipart.call_count, 1)
        self.assertEqual(sockets[1].send_multipart.call_count, 2)
        retried = sockets[1].send_multipart.call_args_list[1].args[0][4]
        self.assertEqual(retried, tokens[1])

    def test_every_token_is_required_and_legacy_ack_is_ignored(self):
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver._expected_abort_acks = {b"rank-0", b"rank-1"}
        receiver._received_abort_acks = set()
        receiver._abort_ack_lock = threading.Lock()

        receiver._record_abort_ack(None)
        receiver._record_abort_ack(b"rank-0")
        self.assertFalse(receiver._abort_acks_complete())
        receiver._record_abort_ack(b"rank-1")
        self.assertTrue(receiver._abort_acks_complete())

    def test_missing_bootstrap_info_is_safe_before_transfer(self):
        receiver = self._make_receiver(None)
        receiver.abort_notified = False
        receiver.conclude_state = None
        receiver.require_staging = False
        receiver._staging_registered = False
        receiver.kv_mgr = SimpleNamespace(
            aborting_rooms=set(),
            record_failure=Mock(),
            update_status=Mock(),
            _staging_handler=None,
            local_ip="127.0.0.1",
            rank_port=1234,
        )

        receiver.begin_failure_quiescence()

        self.assertTrue(receiver.is_transfer_quiesced())
        self.assertEqual(receiver._expected_abort_acks, set())


class TestDecodeStagingAbort(unittest.TestCase):
    def _make_handler(self, decode_req):
        allocator = SimpleNamespace(
            free=Mock(), get_watermark=Mock(return_value=(0, 0))
        )
        manager = SimpleNamespace(
            _chunk_writer_counts={7: {0: ["writer"]}},
            _staging_ctx=SimpleNamespace(
                room_bootstrap={7: ["bootstrap"]},
                room_receivers={7: decode_req.kv_receiver},
            ),
        )
        handler = DecodeStagingHandler.__new__(DecodeStagingHandler)
        handler._abort_lock = threading.RLock()
        handler._aborting_rooms = set()
        handler._abort_finalized_rooms = set()
        handler._abort_finalizing_rooms = set()
        handler._scatter_submitting = defaultdict(int)
        handler._room_to_decode_req = {7: decode_req}
        handler._wm_subscribers = {}
        handler.kv_manager = manager
        handler.staging_allocator = allocator
        handler.decode_tp = 1
        return handler, manager, allocator

    def test_abort_finalization_waits_and_frees_every_allocation_once(self):
        event = Mock()
        event.query.return_value = False
        receiver = SimpleNamespace(
            chunk_staging_infos=[
                (2, 100, 0, 0, 1),
                (3, 200, 0, 0, 1),
            ]
        )
        decode_req = SimpleNamespace(
            req=SimpleNamespace(bootstrap_room=7),
            kv_receiver=receiver,
            _chunk_events=[(event, 1)],
            _staging_last_scatter_submitted=False,
        )
        handler, manager, allocator = self._make_handler(decode_req)
        event.query.side_effect = lambda: (
            self.assertFalse(handler._abort_lock._is_owned()) or False
        )
        allocator.free.side_effect = lambda _alloc_id: self.assertFalse(
            handler._abort_lock._is_owned()
        )

        handler.begin_abort(7)
        self.assertFalse(handler.finalize_abort(7, decode_req))
        allocator.free.assert_not_called()
        counts = defaultdict(lambda: defaultdict(list))
        self.assertFalse(handler.handle_chunk_arrived(7, 0, 0, 1, "rank", counts))
        self.assertEqual(counts, {})

        event.query.side_effect = lambda: (
            self.assertFalse(handler._abort_lock._is_owned()) or True
        )
        self.assertTrue(handler.finalize_abort(7, decode_req))
        self.assertTrue(handler.finalize_abort(7, decode_req))
        self.assertEqual(
            [call.args[0] for call in allocator.free.call_args_list], [1, 2, 3]
        )
        self.assertEqual(receiver.chunk_staging_infos, [])
        self.assertNotIn(7, manager._chunk_writer_counts)
        self.assertNotIn(7, manager._staging_ctx.room_bootstrap)
        self.assertNotIn(7, manager._staging_ctx.room_receivers)
        self.assertNotIn(7, handler._room_to_decode_req)

    def test_staging_allocation_losing_abort_race_is_freed_without_response(self):
        assign_started = threading.Event()
        release_assign = threading.Event()
        socket = Mock()

        class BlockingAllocator:
            total_size = 1024

            def __init__(self):
                self.free = Mock()

            def assign(self, _required):
                assign_started.set()
                release_assign.wait()
                return 11, 64, 0

        receiver = SimpleNamespace(
            chunk_staging_infos=[],
            prefill_info=SimpleNamespace(attn_tp_size=2),
            _connect_to_bootstrap_server=Mock(return_value=(socket, threading.Lock())),
        )
        decode_req = SimpleNamespace(kv_receiver=receiver)
        manager = SimpleNamespace(
            _staging_ctx=SimpleNamespace(
                room_receivers={7: receiver},
                room_bootstrap={7: [{"rank": 0}]},
            )
        )
        handler = DecodeStagingHandler.__new__(DecodeStagingHandler)
        handler._abort_lock = threading.RLock()
        handler._aborting_rooms = set()
        handler._abort_finalized_rooms = set()
        handler._abort_finalizing_rooms = set()
        handler._scatter_submitting = defaultdict(int)
        handler._room_to_decode_req = {7: decode_req}
        handler.kv_manager = manager
        handler.staging_allocator = BlockingAllocator()
        kv_args = SimpleNamespace(
            page_size=1,
            kv_item_lens=[8, 8],
            total_kv_head_num=1,
            kv_head_num=1,
            engine_rank=0,
        )
        msg = [b"STAGING_REQ", b"7", b"0", b"1", b"session"]
        result = []
        worker = threading.Thread(
            target=lambda: result.append(
                handler.handle_staging_req(msg, kv_args, 1, None)
            )
        )
        worker.start()
        self.assertTrue(assign_started.wait(1))
        handler.begin_abort(7)
        release_assign.set()
        worker.join()

        self.assertEqual(result, [False])
        handler.staging_allocator.free.assert_called_once_with(11)
        self.assertEqual(receiver.chunk_staging_infos, [])
        socket.send_multipart.assert_not_called()

    def test_success_unregister_clears_state_and_late_request_cannot_allocate(self):
        receiver = SimpleNamespace(
            chunk_staging_infos=[],
            prefill_info=SimpleNamespace(attn_tp_size=2),
        )
        decode_req = SimpleNamespace(kv_receiver=receiver)
        handler, manager, allocator = self._make_handler(decode_req)
        allocator.assign = Mock(return_value=(12, 64, 0))

        handler.unregister_decode_req(7)

        self.assertNotIn(7, handler._room_to_decode_req)
        self.assertNotIn(7, manager._staging_ctx.room_receivers)
        self.assertNotIn(7, manager._staging_ctx.room_bootstrap)
        self.assertNotIn(7, manager._chunk_writer_counts)
        msg = [b"STAGING_REQ", b"7", b"0", b"1", b"session"]
        kv_args = SimpleNamespace(
            page_size=1,
            kv_item_lens=[8, 8],
            total_kv_head_num=1,
            kv_head_num=1,
            engine_rank=0,
        )
        self.assertFalse(handler.handle_staging_req(msg, kv_args, 1, None))
        allocator.assign.assert_not_called()

    def test_chunk_ready_deduplicates_writers_and_rejects_conflicts(self):
        receiver = SimpleNamespace(
            chunk_staging_infos=[(1, 64, 0, 72, 1)],
            prefill_info=SimpleNamespace(attn_tp_size=2),
        )
        decode_req = SimpleNamespace(kv_receiver=receiver)
        handler, manager, _allocator = self._make_handler(decode_req)
        handler.submit_chunk_scatter = Mock(return_value=True)
        counts = defaultdict(lambda: defaultdict(list))

        self.assertFalse(handler.handle_chunk_arrived(7, 0, 4, 1, "rank-0", counts))
        self.assertFalse(handler.handle_chunk_arrived(7, 0, 4, 1, "rank-0", counts))
        self.assertFalse(handler.handle_chunk_arrived(7, 0, 5, 1, "rank-1", counts))
        self.assertEqual(len(counts[7][0]), 1)
        handler.submit_chunk_scatter.assert_not_called()

        self.assertTrue(handler.handle_chunk_arrived(7, 0, 4, 1, "rank-1", counts))
        handler.submit_chunk_scatter.assert_called_once_with(7, 0, 4, 1)


class TestMooncakeCapabilities(unittest.TestCase):
    @staticmethod
    def _prefill_info(capabilities):
        return PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            pp_size=1,
            page_size=1,
            kv_cache_dtype="auto",
            follow_bootstrap_room=True,
            capabilities=capabilities,
        )

    def test_new_decode_rejects_prefill_without_capability(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.prefill_info_table = {"old-prefill": self._prefill_info([])}
        with patch.object(
            CommonKVManager, "try_ensure_parallel_info", return_value=True
        ):
            self.assertFalse(manager.try_ensure_parallel_info("old-prefill"))
        self.assertNotIn("old-prefill", manager.prefill_info_table)

    def test_new_prefill_rejects_decode_messages_without_capability(self):
        self.assertFalse(_has_kv_registration_capability([b""] * 14))
        self.assertFalse(_has_transfer_metadata_capability([b""] * 9))
        self.assertTrue(
            _has_kv_registration_capability([b""] * 14 + [QUIESCE_CAPABILITY_BYTES])
        )
        self.assertTrue(
            _has_transfer_metadata_capability([b""] * 9 + [QUIESCE_CAPABILITY_BYTES])
        )

    def test_old_wire_parsers_ignore_appended_capability(self):
        registration = [
            b"None",
            b"127.0.0.1",
            b"1",
            b"session",
            b"",
            b"",
            b"",
            b"0",
            b"1",
            b"1",
            b"",
            b"",
            b"",
            b"",
            QUIESCE_CAPABILITY_BYTES,
        ]
        self.assertEqual(
            KVArgsRegisterInfo.from_zmq(registration).mooncake_session_id, "session"
        )

        metadata = [
            b"7",
            b"127.0.0.1",
            b"1",
            b"session",
            b"",
            b"",
            b"",
            b"1",
            b"0",
            QUIESCE_CAPABILITY_BYTES,
        ]
        self.assertEqual(TransferInfo.from_zmq(metadata).room, 7)


class TestTransferWorkerQuiescence(unittest.TestCase):
    def test_native_transfer_delays_abort_ack(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingEngine:
            def batch_transfer_sync(self, *_args):
                entered.set()
                release.wait()
                return 0

        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.enable_trace = False
        manager.enable_staging = False
        manager.room_lifetimes = {}
        manager.room_lifetimes_lock = threading.Lock()
        manager._abort_ack_inflight = set()
        manager._abort_ack_lock = threading.Lock()
        manager._endpoint_send_locks = {}
        manager._endpoint_send_locks_lock = threading.Lock()
        manager.request_status = {7: KVPoll.Transferring}
        manager.check_status = lambda room: manager.request_status[room]
        manager.transfer_infos = {
            7: {
                "session:1": SimpleNamespace(
                    room=7,
                    endpoint="127.0.0.1",
                    dst_port=1,
                    mooncake_session_id="session:1",
                    dst_kv_indices=np.array([0], dtype=np.int32),
                    is_dummy=False,
                )
            }
        }
        manager.decode_kv_args_table = {
            "session:1": SimpleNamespace(
                dst_attn_tp_size=1,
                dst_kv_ptrs=[1],
            )
        }
        manager.session_lock = threading.Lock()
        manager.failed_sessions = set()
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.attn_tp_size = 1
        manager.attn_tp_rank = 0
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.engine = BlockingEngine()
        manager.req_to_decode_prefix_len = {}
        manager.send_kvcache = MethodType(
            lambda self, *_args: self.engine.batch_transfer_sync(), manager
        )

        queue = FastQueue()
        worker = threading.Thread(
            target=manager.transfer_worker,
            args=(queue, Mock()),
            daemon=True,
        )
        worker.start()
        queue.put(
            TransferKVChunk(
                room=7,
                prefill_kv_indices=np.array([0], dtype=np.int32),
                index_slice=slice(0, 1),
                is_last_chunk=False,
                prefill_aux_index=None,
                state_indices=None,
                trace_ctx=Mock(),
            )
        )
        self.assertTrue(entered.wait(1))

        socket = Mock()
        manager._connect = Mock(return_value=socket)
        lifetime = manager._abort_room(7)
        manager._schedule_abort_ack(lifetime, 7, "127.0.0.1", 1, b"token")
        manager._schedule_abort_ack(lifetime, 7, "127.0.0.1", 1, b"token")
        self.assertFalse(socket.send_multipart.called)
        release.set()
        for _ in range(100):
            if socket.send_multipart.called:
                break
            threading.Event().wait(0.01)
        self.assertTrue(socket.send_multipart.called)
        self.assertEqual(socket.send_multipart.call_args.args[0][2], b"token")
        for _ in range(100):
            with manager._abort_ack_lock:
                if not manager._abort_ack_inflight:
                    break
            threading.Event().wait(0.01)
        manager._schedule_abort_ack(lifetime, 7, "127.0.0.1", 1, b"token")
        for _ in range(100):
            if socket.send_multipart.call_count == 2:
                break
            threading.Event().wait(0.01)
        self.assertEqual(socket.send_multipart.call_count, 2)
        self.assertEqual(socket.send_multipart.call_args.args[0][2], b"token")


class TestManagerMessageOrdering(unittest.TestCase):
    def test_abort_ack_cannot_overtake_status_on_same_endpoint(self):
        first_send_entered = threading.Event()
        release_first_send = threading.Event()
        sent = []

        class BlockingSocket:
            def send_multipart(self, parts):
                sent.append(parts[0])
                if len(sent) == 1:
                    first_send_entered.set()
                    release_first_send.wait()

        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._endpoint_send_locks = {}
        manager._endpoint_send_locks_lock = threading.Lock()
        manager._abort_ack_inflight = set()
        manager._abort_ack_lock = threading.Lock()
        manager._connect = Mock(return_value=BlockingSocket())

        status_thread = threading.Thread(
            target=manager.sync_status_to_decode_endpoint,
            args=("127.0.0.1", 1, 7, KVPoll.Failed, 0),
        )
        ack_thread = threading.Thread(
            target=manager._send_abort_ack_when_quiesced,
            args=(None, 7, "127.0.0.1", 1, b"token"),
        )
        status_thread.start()
        self.assertTrue(first_send_entered.wait(1))
        ack_thread.start()
        self.assertEqual(sent, [b"7"])
        release_first_send.set()
        status_thread.join()
        ack_thread.join()
        self.assertEqual(sent, [b"7", b"ABORT_ACK"])


class TestLateMessageRejection(unittest.TestCase):
    def test_chunk_ready_uses_prefill_rank_as_writer_id(self):
        staging_handler = SimpleNamespace(handle_chunk_arrived=Mock(return_value=True))
        manager = SimpleNamespace(
            aborting_rooms=set(),
            _staging_handler=staging_handler,
            _chunk_writer_counts=defaultdict(lambda: defaultdict(list)),
        )

        self.assertTrue(
            MooncakeKVManager._handle_chunk_ready(
                manager,
                [b"CHUNK_READY", b"7", b"0", b"0", b"1", b"session", b"17"],
            )
        )

        staging_handler.handle_chunk_arrived.assert_called_once_with(
            7, 0, 0, 1, "17", manager._chunk_writer_counts
        )

    def test_clear_preserves_decode_abort_tombstones(self):
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.bootstrap_room = 7
        staging_handler = SimpleNamespace(
            clear_abort=Mock(),
            handle_chunk_arrived=Mock(),
        )
        manager = SimpleNamespace(
            request_status={7: KVPoll.Failed},
            required_prefill_response_num_table={7: 1},
            prefill_response_tracker={7: set()},
            abort_receivers={7: receiver},
            _abort_receivers_lock=threading.Lock(),
            aborting_rooms={7},
            _staging_handler=staging_handler,
            _chunk_writer_counts=defaultdict(lambda: defaultdict(list)),
        )
        receiver.kv_mgr = manager

        receiver.clear()

        self.assertIn(7, manager.aborting_rooms)
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.AuxDataCodec.deserialize_data_to_buffer"
        ) as deserialize:
            manager.kv_args = Mock()
            MooncakeKVManager._handle_aux_data(
                manager,
                [b"AUX_DATA", b"7", b"0", b"0", b"\x00\x00\x00\x00", b""],
            )
        deserialize.assert_not_called()
        self.assertFalse(
            MooncakeKVManager._handle_chunk_ready(
                manager,
                [b"CHUNK_READY", b"7", b"0", b"0", b"1", b"session"],
            )
        )
        staging_handler.handle_chunk_arrived.assert_not_called()

    def test_decode_room_reuse_fails_before_bootstrap(self):
        manager = SimpleNamespace(
            get_session_id=Mock(return_value="session"),
            addr_to_rooms_tracker=defaultdict(set),
            update_status=Mock(),
            record_failure=Mock(),
            aborting_rooms={7},
            abort_receivers={},
            _abort_receivers_lock=threading.Lock(),
        )
        receiver = MooncakeKVReceiver(manager, "bootstrap", 7)

        with patch(
            "sglang.srt.disaggregation.mooncake.conn.CommonKVReceiver.init"
        ) as common_init:
            receiver.init(0)

        common_init.assert_not_called()
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)

    def test_duplicate_active_room_does_not_replace_ack_receiver(self):
        existing = SimpleNamespace(_record_abort_ack=Mock())
        manager = SimpleNamespace(
            abort_receivers={7: existing},
            _abort_receivers_lock=threading.Lock(),
            aborting_rooms=set(),
            request_status={7: KVPoll.Transferring},
            get_session_id=Mock(return_value="new-session"),
        )

        with self.assertRaisesRegex(RuntimeError, "already active"):
            MooncakeKVReceiver(manager, "bootstrap", 7)

        self.assertIs(manager.abort_receivers[7], existing)
        self.assertEqual(manager.request_status[7], KVPoll.Transferring)
        MooncakeKVManager._handle_abort_ack(
            manager, [b"ABORT_ACK", b"7", b"stable-token"]
        )
        existing._record_abort_ack.assert_called_once_with(b"stable-token")
        manager.get_session_id.assert_not_called()


class TestRoomLifetimeTokens(unittest.TestCase):
    def _manager(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.room_lifetimes = {}
        manager.room_abort_tokens = defaultdict(set)
        manager.room_lifetimes_lock = threading.Lock()
        return manager

    def test_unknown_and_completed_abort_do_not_create_lifetime(self):
        manager = self._manager()

        self.assertIsNone(manager._abort_room_for_token(7, b"unknown"))
        self.assertEqual(manager.room_lifetimes, {})

        manager._get_room_lifetime(7)
        manager._register_room_abort_token(7, b"old")
        manager._clear_room_lifetime(7)
        self.assertIsNone(manager._abort_room_for_token(7, b"old"))
        self.assertEqual(manager.room_lifetimes, {})

    def test_late_old_token_cannot_abort_reused_room(self):
        manager = self._manager()
        lifetime = manager._get_room_lifetime(7)
        manager._register_room_abort_token(7, b"new")

        self.assertIsNone(manager._abort_room_for_token(7, b"old"))
        self.assertTrue(lifetime.acquire())
        lifetime.release()

        self.assertIs(manager._abort_room_for_token(7, b"new"), lifetime)
        self.assertFalse(lifetime.acquire())


class TestChunkedAbortCleanup(unittest.TestCase):
    def test_source_kv_moves_to_inflight_without_early_release_or_output(self):
        req = SimpleNamespace(
            rid="chunked",
            disagg_kv_sender=Mock(),
        )
        inflight_queue = []

        defer_chunked_prefill_abort(req, inflight_queue)
        defer_chunked_prefill_abort(req, inflight_queue)

        self.assertEqual(inflight_queue, [req])
        self.assertEqual(req.disagg_kv_sender.abort.call_count, 2)


if __name__ == "__main__":
    unittest.main()
