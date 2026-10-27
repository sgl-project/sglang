"""Mori Fabric staging: slot sizing, chunk caps, pack validation and READY/ACK handling."""

import threading
import time
import types
import unittest
from collections import defaultdict
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from test_mori_deferred_kv_release import _abort_message, _chunk, _manager

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.common.dcp_pack import (
    pack_row_segments,
    row_segment,
    unpack_row_segments,
)
from sglang.srt.disaggregation.mori.conn import (
    FABRIC_CHUNK_ACK,
    FABRIC_CHUNK_NACK,
    FABRIC_CHUNK_READY,
    MoriKVReceiver,
    MoriKVSender,
    StatusCode,
)
from sglang.srt.disaggregation.mori.fabric_staging import (
    FabricSlotAcks,
    FabricStagingRoom,
    FabricStagingRx,
    default_slot_tokens,
    slot_bytes_for,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.environ import envs
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")

GUARD = 165
PACK = "sglang.srt.disaggregation.mori.conn.pack_row_segments"
UNPACK = "sglang.srt.disaggregation.mori.conn.unpack_row_segments"


class _Slots:
    """CPU stand-in for FabricStagingSlots with one slot."""

    num_slots = 1

    def __init__(self, slot_bytes, storage=None):
        self.slot_bytes, self.storage, self.desc = slot_bytes, storage, object()
        self.stream_mock = MagicMock()

    def offset(self, slot):
        if slot != 0:
            raise RuntimeError(f"Invalid Fabric staging slot {slot}")
        return 0

    def view(self, slot):
        self.offset(slot)
        return self.storage

    def stream(self, slot):
        return self.stream_mock


def _staging_manager(local=6500, remote=5000):
    ns = types.SimpleNamespace
    manager = _manager()
    # Two KV buffers of 64 pages and two state components; the decode peer
    # needs 128 + 2 terminal rows (1152 bytes).
    manager.kv_args = ns(
        page_size=256,
        kv_item_lens=[192, 64],
        kv_data_ptrs=[1, 2],
        kv_data_lens=[192 * 64, 64 * 64],
        state_types=[StateType.SWA_RING, StateType.DSV4_REQUEST_STATE],
        state_item_lens=[[3, 5], [48, 16]],
        state_data_ptrs=[[3, 4], [5, 6]],
        state_data_lens=[[3 * 128, 5 * 128], [48 * 8, 16 * 8]],
    )
    manager._fabric_staging = True
    manager._mori_backend = "fabric"
    manager._staging_slots = _Slots(local)
    manager._staging_acks = FabricSlotAcks(1)
    manager._staging_rx = FabricStagingRx()
    manager._dcp_pack_max_tokens = None
    manager.is_dummy_cp_rank = False
    manager.enable_all_cp_ranks_for_transfer = False
    manager.attn_cp_size = 1
    manager.attn_tp_rank = 0
    manager._stopped = False
    manager.request_status[11] = KVPoll.WaitingForInput
    manager.transfer_infos = {
        11: {
            "dummy": ns(is_dummy=True),
            "decode": ns(
                is_dummy=False,
                endpoint="10.0.0.2",
                dst_port=6001,
                dst_state_indices=[np.arange(128), np.array([5, 6])],
                dst_staging_epoch=b"epoch",
            ),
        }
    }
    manager.decode_kv_args_table = {
        "decode": ns(
            dst_staging_desc=object(),
            dst_staging_slots=1,
            dst_staging_slot_bytes=remote,
            requires_dcp_relayout=False,
            decode_tp_rank=0,
        )
    }
    return manager


def _sender(manager, pages):
    sender = MoriKVSender.__new__(MoriKVSender)
    sender.kv_mgr, sender.bootstrap_room, sender.curr_idx = manager, 11, 0
    sender._fabric_transfer_tokens = None
    sender._transfer_num_kv_indices = sender._transfer_num_state_indices = 0
    sender.init(pages, 3)
    return sender


def _ready(manager, storage_bytes=6500):
    storage = torch.full((storage_bytes + 32,), GUARD, dtype=torch.uint8)
    manager._staging_slots = _Slots(storage_bytes, storage[16:-16])
    room = FabricStagingRoom(np.arange(6), [], {("10.0.0.1", 6000)})
    manager._staging_rx.rooms = {11: room}
    replies = []
    manager._connect_threadsafe = lambda *a, **kw: types.SimpleNamespace(
        send_multipart=replies.append
    )
    # room, start, stop, last, kv_bytes, state_bytes, ip, port, rank, slot, gen, epoch
    msg = [FABRIC_CHUNK_READY, b"11", b"0", b"2", b"0", b"512", b"0"]
    msg += [b"10.0.0.1", b"6000", b"0", b"0", b"1", room.epoch]
    return msg, replies, storage


class TestFabricMemoryPool(CustomTestCase):
    def test_ple_state_uses_fabric_pool_only_without_staging(self):
        """Direct Fabric transfers require exportable PLE state allocations."""
        fabric_pool = object()
        with (
            get_context().override_server_args(
                disaggregation_mode="prefill", disaggregation_transfer_backend="mori"
            ),
            envs.SGLANG_MORI_BACKEND.override("fabric"),
            envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override(None),
            patch(
                "sglang.srt.disaggregation.mori.custom_mem_pool.get_mori_fabric_mem_pool",
                return_value=fabric_pool,
            ),
            patch("torch.cuda.use_mem_pool", side_effect=lambda _: nullcontext()),
        ):
            for staging in (False, True):
                with (
                    self.subTest(staging=staging),
                    envs.SGLANG_MORI_FABRIC_STAGING.override(staging),
                ):
                    pools = (
                        ShortConvPool(
                            size=2,
                            state_shape=(2, 3),
                            layer_ids=[0],
                            dtype=torch.float32,
                            device="cpu",
                        ),
                        NGramPool(size=2, context_len=3, eos_token_id=0, device="cpu"),
                    )
                    for pool in pools:
                        self.assertEqual(pool.enable_custom_mem_pool, not staging)
                        self.assertIs(
                            pool.custom_mem_pool, None if staging else fabric_pool
                        )


class TestFabricStagingSizing(CustomTestCase):
    def test_slot_follows_chunk_tokens(self):
        kv_args = _staging_manager().kv_args
        mib = 1 << 20
        kv_args.kv_item_lens = [mib, mib]
        with patch.object(
            envs.SGLANG_MORI_FABRIC_STAGING_STATE_MB, "get", return_value=3
        ):
            # 2048 tokens = 8 pages, plus one page for a mid-page start.
            self.assertEqual(slot_bytes_for(kv_args, 2048), (9 * 2 + 3) * mib)
        server_args = types.SimpleNamespace(chunked_prefill_size=1000)
        tokens = envs.SGLANG_MORI_FABRIC_STAGING_TOKENS
        with patch.object(tokens, "get", return_value=0):
            self.assertEqual(default_slot_tokens(server_args, 256), 768)
            server_args.chunked_prefill_size = -1
            self.assertEqual(default_slot_tokens(server_args, 256), 8192)
        with patch.object(tokens, "get", return_value=512):
            self.assertEqual(default_slot_tokens(server_args, 256), 512)

    def test_cap_is_the_smaller_slot_minus_terminal_state(self):
        for local, remote in ((6500, 5000), (5000, 6500)):
            sender = _sender(_staging_manager(local, remote), 1)
            # (5000 - 1152) // 256 pages.
            self.assertEqual(sender.get_max_transfer_tokens(), 15 * 256)
        for local, remote in ((0, 5000), (5000, 0), (1407, 6500)):
            with self.assertRaises(ValueError):
                _sender(_staging_manager(local, remote), 1).get_max_transfer_tokens()
        manager = _staging_manager()
        manager.decode_kv_args_table["decode"].dst_staging_desc = None
        with self.assertRaisesRegex(ValueError, "Missing decode staging slots"):
            _sender(manager, 1).get_max_transfer_tokens()
        manager = _staging_manager()
        manager._fabric_staging = False
        self.assertIsNone(_sender(manager, 1).get_max_transfer_tokens())

    def test_cached_prefix_is_split_and_state_rides_the_last_chunk(self):
        ns = types.SimpleNamespace
        for early in (False, True):
            with self.subTest(early=early):
                manager = _staging_manager()
                sender = _sender(manager, 0)
                cap = sender.get_max_transfer_tokens()
                cached, total = 4 * cap, 4 * cap + 519
                sender.init((total + 255) // 256, 3)
                manager.add_transfer_request = MagicMock()
                pool = ns(
                    unified_swa_ring_size=128,
                    unified_swa_window=128,
                    request_state_transfer_indices=lambda *_: np.array([5, 6]),
                )
                req = ns(
                    rid="prefix",
                    kv=ns(req_pool_idx=0),
                    origin_input_ids=range(total),
                    extend_end=total,
                    start_send_idx=0,
                    disagg_decode_prefix_len=0,
                    disagg_kv_sender=sender,
                    prefix_len=cached,
                    host_hit_length=0,
                    pending_bootstrap=False,
                )
                scheduler = ns(
                    enable_staging=False,
                    enable_overlap=False,
                    token_to_kv_pool_allocator=ns(
                        page_size=256,
                        translate_kv_indices_for_transfer=lambda x: x,
                        get_kvcache=lambda: pool,
                    ),
                    req_to_token_pool=ns(req_to_token=torch.arange(total).view(1, -1)),
                    disagg_metadata_buffers=MagicMock(),
                    kv_checksum_computer=None,
                    disagg_prefill_bootstrap_queue=ns(kv_manager=manager),
                    disagg_prefill_pending_chunk_rids=set(),
                )
                mixin = SchedulerDisaggregationPrefillMixin
                for name in ("send_kv_chunk", "_send_kv_chunk"):
                    setattr(
                        scheduler,
                        name,
                        types.MethodType(getattr(mixin, name), scheduler),
                    )
                with patch.object(
                    envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX,
                    "get",
                    return_value=early,
                ):
                    mixin.maybe_send_cached_prefix_chunk(scheduler, req)
                scheduler._send_kv_chunk(req, last_chunk=True)

                calls = manager.add_transfer_request.call_args_list
                self.assertEqual(len(calls), 5)
                np.testing.assert_array_equal(
                    np.concatenate([c.args[1] for c in calls]),
                    np.arange(sender.num_kv_indices),
                )
                self.assertEqual(sum(c.kwargs["num_kv_tokens"] for c in calls), total)
                for i, call in enumerate(calls):
                    self.assertLessEqual(call.kwargs["num_kv_tokens"], cap)
                    has_state = call.kwargs.get("state_indices") is not None
                    self.assertEqual(has_state, i == len(calls) - 1)
                self.assertEqual(req.start_send_idx, total)


class TestFabricStagingTransfer(CustomTestCase):
    def test_terminal_completion_is_independent_of_aux(self):
        for last_rank, aux_index in ((False, 3), (True, None), (True, 3)):
            with self.subTest(last_rank=last_rank, aux_index=aux_index):
                manager = _staging_manager()
                manager.local_ip, manager.rank_port = "10.0.0.1", 6000
                manager.pp_group = types.SimpleNamespace(is_last_rank=last_rank)
                manager.deferred_bootstrap = None
                manager.transfer_infos[11]["decode"].dst_aux_index = 5
                manager.send_aux = MagicMock()
                manager.send_kv_status_message = MagicMock()
                status = MagicMock()
                status.InProgress.return_value = False
                manager.engine = MagicMock()
                manager.engine.batch_write.return_value = [status]
                manager.engine.wait_all.return_value = StatusCode.SUCCESS
                ready = []

                def reply(msg):
                    self.assertEqual(msg[0], FABRIC_CHUNK_READY)
                    self.assertEqual(manager.request_status[11], KVPoll.Transferring)
                    manager.engine.wait_all.assert_called_with([status])
                    ready.append(msg)
                    manager._handle_fabric_chunk_ack(
                        [FABRIC_CHUNK_ACK, msg[1], msg[10], msg[11], msg[3], msg[9]]
                    )

                manager._connect_threadsafe = lambda *a, **kw: types.SimpleNamespace(
                    send_multipart=reply
                )
                for last_chunk in (False, True):
                    chunk = _chunk()
                    chunk.is_last_chunk = last_chunk
                    chunk.prefill_aux_index = aux_index
                    with patch(PACK, return_value=256):
                        self.assertTrue(manager._process_transfer_chunk(chunk))
                    self.assertEqual(len(ready), 1 + int(last_chunk))
                    self.assertEqual(ready[-1][4], b"1" if last_chunk else b"0")
                    self.assertEqual(
                        manager.request_status[11],
                        KVPoll.Success if last_chunk else KVPoll.Transferring,
                    )
                    if last_chunk:
                        self.assertEqual(
                            manager.send_kv_status_message.call_args.kwargs["status"],
                            KVPoll.Success,
                        )
                    else:
                        manager.send_kv_status_message.assert_not_called()
                        manager.send_aux.assert_not_called()
                self.assertEqual(
                    manager.send_aux.call_count,
                    int(last_rank and aux_index is not None),
                )

    def test_invalid_chunk_fails_before_any_copy(self):
        for rows, state, remote, error in (
            (np.arange(26), None, 7000, "overflow before gather"),
            (np.arange(20), None, 5000, "too small"),
            (np.array([-1]), None, 5000, "negative row"),
            (np.array([64]), None, 5000, "source overflow"),
            (
                np.array([0]),
                [np.array([128]), np.array([5, 6])],
                5000,
                "source overflow",
            ),
        ):
            with self.subTest(error=error):
                manager = _staging_manager(remote=remote)
                chunk = _chunk()
                chunk.prefill_kv_indices, chunk.state_indices = rows, state
                chunk.is_last_chunk = state is not None
                with patch(PACK) as pack:
                    with self.assertRaisesRegex((ValueError, RuntimeError), error):
                        manager._process_fabric_staging_chunk(chunk, 0)
                pack.assert_not_called()

    def test_failed_transfer_drains_writes_and_quarantines_the_slot(self):
        for phase in (
            "pack",
            "between_targets",
            "allocate_uid",
            "batch_write",
            "wait_all",
            "wait_failed",
        ):
            with self.subTest(phase=phase):
                manager = _staging_manager()
                info = manager.transfer_infos[11]["decode"]
                manager.transfer_infos[11] = {"decode": info, "second": info}
                peer = manager.decode_kv_args_table["decode"]
                manager.decode_kv_args_table["second"] = types.SimpleNamespace(
                    **{**vars(peer), "decode_tp_rank": 1}
                )
                submitted = [MagicMock(), MagicMock()]
                for status in submitted:
                    status.InProgress.return_value = True
                manager.engine = MagicMock()
                manager.send_kv_status_message = MagicMock()

                def abort():
                    manager._handle_abort_message(_abort_message())

                def pack(*args):
                    if phase == "pack":
                        abort()
                    return 256

                def allocate_uid():
                    if manager.engine.batch_write.call_count:
                        if phase == "between_targets":
                            abort()
                        elif phase == "allocate_uid":
                            raise RuntimeError(phase)
                    return 1

                def batch_write(*args):
                    index = manager.engine.batch_write.call_count - 1
                    if phase == "batch_write" and index == 1:
                        raise RuntimeError(phase)
                    return [submitted[index]]

                def wait_all(statuses):
                    if phase == "wait_all":
                        raise RuntimeError(phase)
                    return StatusCode.FAILED

                manager.engine.allocate_transfer_uid.side_effect = allocate_uid
                manager.engine.batch_write.side_effect = batch_write
                manager.engine.wait_all.side_effect = wait_all
                chunk = _chunk()
                queue = MagicMock()
                queue.get.side_effect = (chunk, None)
                with (
                    patch(PACK, side_effect=pack),
                    patch("sglang.srt.disaggregation.mori.conn.logger.exception"),
                ):
                    manager._transfer_worker(queue, 0)
                self.assertEqual(manager.request_status[11], KVPoll.Failed)
                with self.assertRaisesRegex(RuntimeError, "quarantined"):
                    manager._staging_acks.acquire(0)
                if phase not in ("pack", "between_targets"):
                    abort()
                if phase == "pack":
                    self.assertTrue(manager._drain_queue.empty())
                else:
                    self.assertEqual(manager._sent, [])
                    self.assertEqual(manager._staging_outstanding[11], 1)
                    queued_chunk, statuses = manager._drain_queue.get_nowait()
                    self.assertIs(queued_chunk, chunk)
                    self.assertEqual(
                        statuses, submitted[: 2 if phase.startswith("wait") else 1]
                    )

                    for status in statuses:

                        def drain(status=status):
                            self.assertEqual(manager._sent, [])
                            status.InProgress.return_value = False

                        status.Wait.side_effect = drain
                    manager._drain_transfer_statuses(queued_chunk, statuses)
                    self.assertFalse(any(s.InProgress() for s in statuses))
                self.assertNotIn(11, manager._staging_outstanding)
                self.assertEqual(len(manager._sent), 1)

    def test_slot_is_reused_only_after_a_terminal_reply(self):
        manager = _staging_manager()
        acks = manager._staging_acks
        key = (11, 0, 1, 2, 0)
        acks.expect(0, {key})
        with self.assertRaisesRegex(RuntimeError, "Timed out"):
            acks.wait({key}, time.monotonic() - 1, cancelled=lambda: False)
        # Non-terminal NACK and an ACK for another generation keep it quarantined.
        manager._handle_fabric_chunk_ack(
            [FABRIC_CHUNK_NACK, b"11", b"0", b"1", b"2", b"0", b"0", b"busy"]
        )
        manager._handle_fabric_chunk_ack(
            [FABRIC_CHUNK_ACK, b"11", b"0", b"2", b"2", b"0"]
        )
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            acks.acquire(0)
        with self.assertRaisesRegex(RuntimeError, "NACK"):
            acks.wait({key}, time.monotonic() + 300, cancelled=lambda: False)
        manager._handle_fabric_chunk_ack(
            [FABRIC_CHUNK_NACK, b"11", b"0", b"1", b"2", b"0", b"1", b"rejected"]
        )
        self.assertEqual(acks.acquire(0), 1)
        self.assertTrue(acks.is_idle())


class TestFabricStagingReceive(CustomTestCase):
    def test_ready_with_bad_identity_is_nacked_before_unpack(self):
        for field, value in (
            (2, b"-1"),  # start
            (3, b"99"),  # stop
            (5, b"6501"),  # kv_bytes
            (9, b"1"),  # decode rank
            (10, b"-1"),  # slot
            (11, b"0"),  # generation
            (12, b"old"),  # epoch
        ):
            with self.subTest(field=field):
                manager = _staging_manager()
                msg, replies, storage = _ready(manager)
                msg[field] = value
                with patch(UNPACK) as unpack:
                    manager._handle_fabric_chunk_ready(msg)
                unpack.assert_not_called()
                self.assertEqual(replies[0][0], FABRIC_CHUNK_NACK)
                self.assertEqual(replies[0][6], b"1")
                self.assertTrue(torch.all(storage == GUARD))

        manager = _staging_manager()
        msg, replies, _ = _ready(manager)
        with patch(UNPACK) as unpack:
            manager._handle_fabric_chunk_ready(msg)
            manager._handle_fabric_chunk_ready(msg)  # replayed generation
        self.assertEqual([r[0] for r in replies], [FABRIC_CHUNK_ACK, FABRIC_CHUNK_NACK])
        self.assertEqual(unpack.call_count, 1)

    def test_failed_unpack_poisons_the_slot(self):
        manager = _staging_manager()
        msg, replies, _ = _ready(manager)
        manager._staging_slots.stream_mock.synchronize.side_effect = RuntimeError
        with patch(UNPACK, side_effect=RuntimeError("copy failed")) as unpack:
            manager._handle_fabric_chunk_ready(msg)
            msg[11] = b"2"
            manager._handle_fabric_chunk_ready(msg)
        self.assertEqual(manager._staging_rx.poisoned, {0})
        self.assertEqual([r[6] for r in replies], [b"0", b"0"])
        self.assertEqual(unpack.call_count, 1)

    def test_clear_waits_for_unpack_and_rejects_late_ready(self):
        manager = _staging_manager()
        msg, replies, _ = _ready(manager)
        manager.defer_decode_allocation = False
        manager.required_prefill_response_num_table = {}
        manager.prefill_response_tracker = {}
        manager.addr_to_rooms_tracker = defaultdict(set)
        receiver = MoriKVReceiver.__new__(MoriKVReceiver)
        receiver.kv_mgr, receiver.bootstrap_room = manager, 11
        receiver.bootstrap_addr = "peer"
        receiver._lifecycle_lock = manager._staging_rx.lock
        entered, finish, cleared = (threading.Event() for _ in range(3))

        def unpack(*args):
            entered.set()
            self.assertTrue(finish.wait(2))

        def clear():
            receiver.clear()
            cleared.set()

        with patch(UNPACK, side_effect=unpack):
            worker = threading.Thread(
                target=manager._handle_fabric_chunk_ready, args=(msg,)
            )
            worker.start()
            self.assertTrue(entered.wait(1))
            cleanup = threading.Thread(target=clear)
            cleanup.start()
            self.assertFalse(cleared.wait(0.05))
            finish.set()
            worker.join(2)
            cleanup.join(2)
            self.assertTrue(cleared.is_set())
            manager._handle_fabric_chunk_ready(msg)
        self.assertNotIn(11, manager._staging_rx.rooms)
        self.assertEqual([r[0] for r in replies], [FABRIC_CHUNK_ACK, FABRIC_CHUNK_NACK])


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestRowSegmentPackGpu(CustomTestCase):
    def test_round_trip_keeps_rows_offsets_and_guards(self):
        kv = torch.arange(16 * 12, dtype=torch.uint8, device="cuda").view(16, 12)
        state = torch.arange(16, dtype=torch.uint8, device="cuda").view(4, 4)
        kv_dst, state_dst = torch.full_like(kv, GUARD), torch.full_like(state, GUARD)
        guard, nbytes = 16, 3 * 12 + 2 * 4
        tx = torch.full((nbytes + 2 * guard,), GUARD, dtype=torch.uint8, device="cuda")
        stream = torch.cuda.Stream()

        def segments(kv_buf, kv_rows, state_buf, state_rows):
            return [
                row_segment("kv", kv_rows, [kv_buf.data_ptr()], [kv_buf.nbytes], [12]),
                row_segment(
                    "state", state_rows, [state_buf.data_ptr()], [state_buf.nbytes], [4]
                ),
            ]

        slot = tx[guard:-guard]
        src = segments(kv, [15, 2, 7], state, [1, 3])
        self.assertEqual(pack_row_segments(slot, src, stream), nbytes)
        dst = segments(kv_dst, [0, 5, 9], state_dst, [0, 2])
        self.assertEqual(unpack_row_segments(slot, dst, stream), nbytes)

        torch.testing.assert_close(kv_dst[[0, 5, 9]], kv[[15, 2, 7]], rtol=0, atol=0)
        torch.testing.assert_close(state_dst[[0, 2]], state[[1, 3]], rtol=0, atol=0)
        self.assertEqual(int((kv_dst != GUARD).any(dim=1).sum()), 3)
        self.assertTrue(torch.all(tx[:guard] == GUARD))
        self.assertTrue(torch.all(tx[-guard:] == GUARD))


if __name__ == "__main__":
    unittest.main()
