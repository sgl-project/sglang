"""Mori Fabric staging: slot sizing, send validation, completion and READY handling."""

import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from test_mori_deferred_kv_release import _chunk, _manager

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.dcp_pack import (
    pack_row_segments,
    row_segment,
    unpack_row_segments,
)
from sglang.srt.disaggregation.mori.conn import (
    FABRIC_CHUNK_ACK,
    FABRIC_CHUNK_NACK,
    FABRIC_CHUNK_READY,
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
from sglang.srt.environ import envs
from sglang.srt.mem_cache.utils import maybe_init_custom_mem_pool
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd-mi35x")

PACK = "sglang.srt.disaggregation.mori.conn.pack_row_segments"
UNPACK = "sglang.srt.disaggregation.mori.conn.unpack_row_segments"


class _Slots:
    num_slots, desc = 1, object()

    def __init__(self, slot_bytes):
        self.slot_bytes, self.stream_mock = slot_bytes, MagicMock()

    def offset(self, slot):
        if slot != 0:
            raise RuntimeError(f"Invalid Fabric staging slot {slot}")
        return 0

    def view(self, slot):
        return self.offset(slot)

    def stream(self, slot):
        return self.stream_mock


def _staging_manager(local=6500, remote=5000):
    ns = types.SimpleNamespace
    manager = _manager()
    # Decode needs 128 + 2 terminal state rows: 1152 bytes.
    manager.kv_args = ns(
        page_size=256,
        kv_item_lens=[192, 64],
        kv_data_ptrs=[1, 2],
        kv_data_lens=[192 * 64, 64 * 64],
        state_item_lens=[[3, 5], [48, 16]],
        state_data_ptrs=[[3, 4], [5, 6]],
        state_data_lens=[[3 * 128, 5 * 128], [48 * 8, 16 * 8]],
    )
    manager._fabric_staging, manager._mori_backend = True, "fabric"
    manager._staging_slots = _Slots(local)
    manager._staging_acks, manager._staging_rx = FabricSlotAcks(1), FabricStagingRx()
    manager._dcp_pack_max_tokens, manager._stopped = None, False
    manager.is_dummy_cp_rank, manager.attn_tp_rank = False, 0
    manager.request_status[11] = KVPoll.WaitingForInput
    info = ns(
        is_dummy=False,
        endpoint="10.0.0.2",
        dst_port=6001,
        dst_aux_index=5,
        dst_state_indices=[np.arange(128), np.array([5, 6])],
        dst_staging_epoch=b"epoch",
    )
    manager.transfer_infos = {11: {"decode": info}}
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


def _cap(manager):
    sender = MoriKVSender.__new__(MoriKVSender)
    sender.kv_mgr, sender.bootstrap_room, sender.curr_idx = manager, 11, 0
    sender._fabric_transfer_tokens = None
    sender._transfer_num_kv_indices = sender._transfer_num_state_indices = 0
    sender.init(1, 3)
    return sender.get_max_transfer_tokens()


class TestFabricStaging(CustomTestCase):
    def test_slot_size_and_chunk_cap(self):
        mib = 1 << 20
        kv_args = types.SimpleNamespace(page_size=256, kv_item_lens=[mib, mib])
        with patch.object(
            envs.SGLANG_MORI_FABRIC_STAGING_STATE_MB, "get", return_value=3
        ):
            # 8 pages for 2048 tokens plus one for a mid-page start.
            self.assertEqual(slot_bytes_for(kv_args, 2048), (9 * 2 + 3) * mib)
        args = types.SimpleNamespace(chunked_prefill_size=1000)
        self.assertEqual(default_slot_tokens(args, 256), 768)
        # (min(6500, 5000) - 1152) // 256 pages.
        self.assertEqual(_cap(_staging_manager()), 15 * 256)
        with self.assertRaises(ValueError):
            _cap(_staging_manager(local=1407, remote=6500))

    def test_mori_rdma_keeps_mooncake_pool_selection(self):
        mooncake = (True, object(), "NVLINK")
        with (
            get_context().override_server_args(
                disaggregation_mode="decode", disaggregation_transfer_backend="mori"
            ),
            envs.SGLANG_MORI_BACKEND.override("rdma"),
            envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.override("NVLINK"),
            patch(
                "sglang.srt.disaggregation.mooncake.utils.init_mooncake_custom_mem_pool",
                return_value=mooncake,
            ),
        ):
            self.assertIs(maybe_init_custom_mem_pool("cpu"), mooncake)

    def test_invalid_chunk_fails_before_any_copy(self):
        for rows, remote, error in (
            (np.arange(26), 7000, "overflow before gather"),
            (np.arange(20), 5000, "too small"),
            (np.array([64]), 5000, "source overflow"),
        ):
            with self.subTest(error=error), patch(PACK) as pack:
                chunk = _chunk()
                chunk.prefill_kv_indices = rows
                with self.assertRaisesRegex((ValueError, RuntimeError), error):
                    _staging_manager(remote=remote)._process_fabric_staging_chunk(
                        chunk, 0
                    )
                pack.assert_not_called()

    def _send_last_chunk(self, manager, wait_rc):
        manager.local_ip, manager.rank_port = "10.0.0.1", 6000
        manager.pp_group = types.SimpleNamespace(is_last_rank=False)
        manager.deferred_bootstrap = None
        manager.send_aux, manager.send_kv_status_message = MagicMock(), MagicMock()
        status = MagicMock()
        status.InProgress.return_value = wait_rc != StatusCode.SUCCESS
        manager.engine = MagicMock()
        manager.engine.batch_write.return_value = [status]
        manager.engine.wait_all.return_value = wait_rc
        manager._connect_threadsafe = lambda *a, **kw: types.SimpleNamespace(
            send_multipart=lambda m: manager._handle_fabric_chunk_ack(
                [FABRIC_CHUNK_ACK, m[1], m[10], m[11], m[3], m[9]]
            )
        )
        chunk = _chunk()
        chunk.is_last_chunk, chunk.prefill_aux_index = True, 3
        queue = MagicMock()
        queue.get.side_effect = (chunk, None)
        with patch(PACK, return_value=256):
            manager._transfer_worker(queue, 0)
        return status

    def test_terminal_chunk_completes_on_a_non_last_pp_rank(self):
        manager = _staging_manager()
        self._send_last_chunk(manager, StatusCode.SUCCESS)
        self.assertEqual(manager.request_status[11], KVPoll.Success)
        manager.send_aux.assert_not_called()

    def test_failed_write_is_drained_before_the_slot_is_reused(self):
        manager = _staging_manager()
        status = self._send_last_chunk(manager, StatusCode.ERR_RDMA_OP)
        self.assertEqual(manager.request_status[11], KVPoll.Failed)
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            manager._staging_acks.acquire(0)
        queued_chunk, statuses = manager._drain_queue.get_nowait()
        self.assertEqual(statuses, [status])

    def test_bad_or_replayed_ready_is_nacked(self):
        def ready():
            manager = _staging_manager()
            room = FabricStagingRoom(np.arange(6), [], {("10.0.0.1", 6000)})
            manager._staging_rx.rooms = {11: room}
            replies = []
            manager._connect_threadsafe = lambda *a, **kw: types.SimpleNamespace(
                send_multipart=replies.append
            )
            # room, start, stop, last, kv/state bytes, ip, port, rank, slot, gen, epoch
            msg = [FABRIC_CHUNK_READY, b"11", b"0", b"2", b"0", b"512", b"0"]
            return (
                manager,
                msg + [b"10.0.0.1", b"6000", b"0", b"0", b"1", room.epoch],
                replies,
            )

        for field, value in ((3, b"99"), (10, b"-1"), (12, b"old")):
            manager, msg, replies = ready()
            msg[field] = value
            with self.subTest(field=field), patch(UNPACK) as unpack:
                manager._handle_fabric_chunk_ready(msg)
                unpack.assert_not_called()
                self.assertEqual(replies[0][0], FABRIC_CHUNK_NACK)
        manager, msg, replies = ready()
        with patch(UNPACK) as unpack:
            manager._handle_fabric_chunk_ready(msg)
            manager._handle_fabric_chunk_ready(msg)
        self.assertEqual([r[0] for r in replies], [FABRIC_CHUNK_ACK, FABRIC_CHUNK_NACK])
        self.assertEqual(unpack.call_count, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_gpu_round_trip_keeps_rows_offsets_and_guards(self):
        kv = torch.arange(16 * 12, dtype=torch.uint8, device="cuda").view(16, 12)
        state = torch.arange(16, dtype=torch.uint8, device="cuda").view(4, 4)
        kv_dst, state_dst = torch.full_like(kv, 165), torch.full_like(state, 165)
        nbytes = 3 * 12 + 2 * 4
        tx = torch.full((nbytes + 32,), 165, dtype=torch.uint8, device="cuda")
        stream = torch.cuda.Stream()

        def segments(kv_buf, kv_rows, state_buf, state_rows):
            return [
                row_segment("kv", kv_rows, [kv_buf.data_ptr()], [kv_buf.nbytes], [12]),
                row_segment(
                    "state", state_rows, [state_buf.data_ptr()], [state_buf.nbytes], [4]
                ),
            ]

        slot = tx[16:-16]
        self.assertEqual(
            pack_row_segments(slot, segments(kv, [15, 2, 7], state, [1, 3]), stream),
            nbytes,
        )
        dst = segments(kv_dst, [0, 5, 9], state_dst, [0, 2])
        self.assertEqual(unpack_row_segments(slot, dst, stream), nbytes)
        torch.testing.assert_close(kv_dst[[0, 5, 9]], kv[[15, 2, 7]], rtol=0, atol=0)
        torch.testing.assert_close(state_dst[[0, 2]], state[[1, 3]], rtol=0, atol=0)
        self.assertEqual(int((kv_dst != 165).any(dim=1).sum()), 3)
        self.assertTrue(torch.all(tx[:16] == 165) and torch.all(tx[-16:] == 165))


if __name__ == "__main__":
    unittest.main()
