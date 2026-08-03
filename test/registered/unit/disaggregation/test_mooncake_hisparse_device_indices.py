"""Unit tests for the HiSparse device-index plumbing in Mooncake's send loop.

`MooncakeKVManager.transfer_worker` carries a *second* destination index space
(`dst_device_kv_indices`) alongside the regular host page ids, so a HiSparse
decode peer can land C4/indexer layers in device pages while the compressed
layers land in host pages. The chunk/truncate arithmetic that keeps the two
index spaces aligned lives inline in the worker loop and had no coverage.
"""

import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.utils import TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    TransferInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOM = 7
SESSION_ID = "decode-session-0"
DST_KV_PTRS = [0xA000, 0xB000]


class _StopWorker(BaseException):
    """Sentinel used to unwind ``transfer_worker``'s ``while True`` loop.

    Deliberately derived from ``BaseException``: the worker body is wrapped in
    ``except Exception`` which re-raises everything as ``RuntimeError``, so an
    ``Exception`` sentinel would be swallowed and rewritten and we could no
    longer tell a clean exit from a real failure.
    """


class _ScriptedQueue:
    """Minimal ``FastQueue`` stand-in that feeds N chunks then stops the loop."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def get(self):
        if not self._chunks:
            raise _StopWorker
        return self._chunks.pop(0)


def _make_register_info(requires_dcp_relayout: bool) -> KVArgsRegisterInfo:
    return KVArgsRegisterInfo(
        room="None",
        endpoint="127.0.0.1",
        dst_port=30001,
        mooncake_session_id=SESSION_ID,
        dst_kv_ptrs=list(DST_KV_PTRS),
        dst_aux_ptrs=[0xC000],
        dst_state_data_ptrs=[],
        dst_tp_rank=0,
        dst_attn_tp_size=1,
        dst_kv_item_len=64,
        dst_state_item_lens=[],
        dst_state_dim_per_tensor=[],
        dst_kv_layer_ids=[],
        dst_state_layer_ids=[],
        requires_dcp_relayout=requires_dcp_relayout,
        dcp_token_item_lens=[64, 64] if requires_dcp_relayout else None,
    )


def _make_manager(requires_dcp_relayout: bool = False) -> MooncakeKVManager:
    """Build a manager with only the attributes the send loop actually reads.

    No engine, no RDMA, no GPU: ``send_kvcache`` runs for real (it is what maps
    device indices onto the device-resident destination pointers) but the
    transfer primitive underneath it is stubbed.
    """
    manager = object.__new__(MooncakeKVManager)
    manager.enable_trace = False
    manager.enable_staging = False
    manager.enable_custom_mem_pool = False
    manager.bootstrap_port = 8998
    manager.is_mla_backend = True
    manager.is_hybrid_mla_backend = False
    manager.attn_tp_rank = 0
    manager.attn_tp_size = 1
    manager.attn_cp_rank = 0
    manager.attn_cp_size = 1
    manager.pp_rank = 0
    manager.pp_size = 1
    manager.session_lock = threading.Lock()
    manager.failure_lock = threading.Lock()
    manager.failed_sessions = set()
    manager.session_failures = defaultdict(int)
    manager.failure_records = {}
    manager.request_status = {ROOM: KVPoll.WaitingForInput}
    manager.req_to_decode_prefix_len = {}
    manager.transfer_infos = {}
    # 2 KV entries: layer 0 is C4-compressed (host pages), layer 1 is the
    # uncompressed/indexer entry that HiSparse keeps in device pages.
    manager.kv_args = SimpleNamespace(
        kv_data_ptrs=[0x1000, 0x2000],
        kv_item_lens=[64, 64],
        kv_layer_ids=[],
        mla_compression_ratios=[4, 128],
        prefill_start_layer=0,
        prefill_end_layer=2,
        page_size=1,
        engine_rank=0,
    )
    manager.decode_kv_args_table = {
        SESSION_ID: _make_register_info(requires_dcp_relayout)
    }
    # Stub the lowest transfer layer so the real send_kvcache() logic still runs.
    manager._send_kvcache_generic = MagicMock(return_value=0)
    manager.send_kvcache_dcp = MagicMock(return_value=0)
    manager.send_kvcache_slice = MagicMock(return_value=0)
    manager.sync_status_to_decode_endpoint = MagicMock()
    return manager


def _make_transfer_info(dst_kv_indices, dst_device_kv_indices) -> TransferInfo:
    return TransferInfo(
        room=ROOM,
        endpoint="127.0.0.1",
        dst_port=30001,
        mooncake_session_id=SESSION_ID,
        dst_kv_indices=np.asarray(dst_kv_indices, dtype=np.int32),
        dst_aux_index=0,
        dst_state_indices=[],
        required_dst_info_num=1,
        is_dummy=False,
        decode_prefix_len=0,
        dst_device_kv_indices=(
            None
            if dst_device_kv_indices is None
            else np.asarray(dst_device_kv_indices, dtype=np.int32)
        ),
    )


def _make_chunk(prefill_kv_indices, index_slice) -> TransferKVChunk:
    return TransferKVChunk(
        room=ROOM,
        prefill_kv_indices=np.asarray(prefill_kv_indices, dtype=np.int32),
        index_slice=index_slice,
        is_last_chunk=False,
        prefill_aux_index=None,
        state_indices=None,
        num_kv_tokens=len(prefill_kv_indices),
    )


def _drive_send_loop(manager, chunk, req):
    """Run one iteration of transfer_worker's send loop, then unwind cleanly."""
    manager.transfer_infos = {ROOM: {SESSION_ID: req}}
    manager.transfer_worker(
        _ScriptedQueue([chunk]),
        executor=MagicMock(),
    )


class TestMooncakeHiSparseDeviceIndices(unittest.TestCase):
    def test_device_indices_follow_the_chunk_slice(self):
        manager = _make_manager()
        req = _make_transfer_info(
            dst_kv_indices=[10, 11, 12, 13, 14, 15],
            dst_device_kv_indices=[90, 91, 92, 93, 94, 95],
        )
        chunk = _make_chunk([500, 501, 502, 503], slice(2, 6))

        with self.assertRaises(_StopWorker):
            _drive_send_loop(manager, chunk, req)

        kwargs = manager._send_kvcache_generic.call_args.kwargs
        np.testing.assert_array_equal(
            kwargs["dst_data_indices"], np.array([12, 13, 14, 15], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            kwargs["dst_device_data_indices"],
            np.array([92, 93, 94, 95], dtype=np.int32),
        )
        # send_kvcache() must route the device index space onto the
        # non-C4 (device-resident) destination pointers only.
        self.assertEqual(kwargs["dst_device_data_ptrs"], {DST_KV_PTRS[1]})

    def test_device_indices_clamp_to_shorter_prefill_chunk(self):
        """Chunk slice is wider than the prefill payload -> device tail dropped."""
        manager = _make_manager()
        req = _make_transfer_info(
            dst_kv_indices=[10, 11, 12, 13, 14, 15],
            dst_device_kv_indices=[90, 91, 92, 93, 94, 95],
        )
        # index_slice yields 4 destination pages but only 3 prefill pages.
        chunk = _make_chunk([500, 501, 502], slice(2, 6))

        with self.assertRaises(_StopWorker):
            _drive_send_loop(manager, chunk, req)

        kwargs = manager._send_kvcache_generic.call_args.kwargs
        # The host index space is left as-is by the production code ...
        np.testing.assert_array_equal(
            kwargs["dst_data_indices"], np.array([12, 13, 14, 15], dtype=np.int32)
        )
        # ... while the device index space is clamped to the prefill length.
        np.testing.assert_array_equal(
            kwargs["dst_device_data_indices"], np.array([92, 93, 94], dtype=np.int32)
        )

    def test_device_indices_truncate_after_prefill_is_shortened(self):
        """The subtle ordering: prefill is shortened first, device clamped after.

        When ``len(chunked_dst_kv_indice) < len(prefill_kv_indices)`` the worker
        first shortens ``prefill_kv_indices`` to the destination length, and only
        then clamps the device indices to the *already shortened* prefill length.
        A device clamp computed before the prefill fixup would leave 4 entries
        here instead of 3 and silently mis-pair the two index spaces.
        """
        manager = _make_manager()
        req = _make_transfer_info(
            # Only 3 destination pages exist even though the slice asks for 4.
            dst_kv_indices=[10, 11, 12],
            dst_device_kv_indices=[90, 91, 92, 93, 94, 95],
        )
        chunk = _make_chunk([500, 501, 502, 503], slice(0, 4))

        with self.assertRaises(_StopWorker):
            _drive_send_loop(manager, chunk, req)

        kwargs = manager._send_kvcache_generic.call_args.kwargs
        np.testing.assert_array_equal(
            kwargs["prefill_data_indices"], np.array([500, 501, 502], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            kwargs["dst_data_indices"], np.array([10, 11, 12], dtype=np.int32)
        )
        self.assertEqual(len(kwargs["dst_device_data_indices"]), 3)
        np.testing.assert_array_equal(
            kwargs["dst_device_data_indices"], np.array([90, 91, 92], dtype=np.int32)
        )

    def test_non_hisparse_path_forwards_no_device_indices(self):
        manager = _make_manager()
        req = _make_transfer_info(
            dst_kv_indices=[10, 11, 12, 13],
            dst_device_kv_indices=None,
        )
        chunk = _make_chunk([500, 501], slice(0, 2))

        with self.assertRaises(_StopWorker):
            _drive_send_loop(manager, chunk, req)

        kwargs = manager._send_kvcache_generic.call_args.kwargs
        np.testing.assert_array_equal(
            kwargs["dst_data_indices"], np.array([10, 11], dtype=np.int32)
        )
        self.assertIsNone(kwargs["dst_device_data_indices"])
        self.assertIsNone(kwargs["dst_device_data_ptrs"])

    def test_dcp_relayout_uses_full_dst_indices_without_device_plan(self):
        manager = _make_manager(requires_dcp_relayout=True)
        req = _make_transfer_info(
            dst_kv_indices=[10, 11, 12, 13, 14, 15],
            dst_device_kv_indices=None,
        )
        chunk = _make_chunk([500, 501], slice(2, 4))

        with self.assertRaises(_StopWorker):
            _drive_send_loop(manager, chunk, req)

        manager._send_kvcache_generic.assert_not_called()
        args = manager.send_kvcache_dcp.call_args.args
        # DCP relayout re-derives its own token plan, so it consumes the FULL
        # destination page list rather than the per-chunk slice.
        np.testing.assert_array_equal(
            args[3], np.array([10, 11, 12, 13, 14, 15], dtype=np.int32)
        )
        self.assertNotIn(
            "dst_device_kv_indices", manager.send_kvcache_dcp.call_args.kwargs
        )

    def test_dcp_relayout_with_device_indices_kills_the_transfer_thread(self):
        """Document the *actual* behaviour of the DCP/HiSparse guard.

        The assertion is not a graceful rejection: it escapes into
        transfer_worker's blanket ``except Exception`` handler, which re-raises
        ``RuntimeError`` and so terminates the whole transfer thread. The room
        is left un-failed and the decode side is never notified.
        """
        manager = _make_manager(requires_dcp_relayout=True)
        req = _make_transfer_info(
            dst_kv_indices=[10, 11, 12, 13],
            dst_device_kv_indices=[90, 91, 92, 93],
        )
        chunk = _make_chunk([500, 501], slice(0, 2))

        with self.assertRaises(RuntimeError) as ctx:
            _drive_send_loop(manager, chunk, req)

        message = str(ctx.exception)
        self.assertIn(
            "DCP relayout does not support separate HiSparse device KV destinations",
            message,
        )
        self.assertIn("Transfer thread failed", message)

        # No transfer was attempted on either path.
        manager.send_kvcache_dcp.assert_not_called()
        manager._send_kvcache_generic.assert_not_called()
        # And the failure is *not* surfaced through the normal channels: the
        # room keeps its pre-existing status, nothing is recorded, and the
        # decode endpoint is never told. The thread simply dies.
        self.assertEqual(manager.request_status[ROOM], KVPoll.WaitingForInput)
        self.assertEqual(manager.failure_records, {})
        manager.sync_status_to_decode_endpoint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
