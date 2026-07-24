import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

pytest.importorskip("mori.io", exc_type=ImportError)

from mori.io import MemoryLocationType

from sglang.srt.disaggregation.mori.conn import (
    DEFAULT_HOST_REGISTRATION_CHUNK_BYTES,
    BatchTransferPlan,
    MoriKVManager,
    MoriKVReceiver,
    TransferInfo,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class TestMoriHiSparseTransfer(unittest.TestCase):
    def test_transfer_metadata_carries_host_indices(self):
        device_pages = np.array([7, 8], dtype=np.int32)
        host_rows = np.array([21, 22, 31], dtype=np.int32)

        info = TransferInfo.from_zmq(
            [
                b"9",
                b"127.0.0.1",
                b"12345",
                b"engine",
                device_pages.tobytes(),
                b"0",
                b"",
                b"1",
                b"0",
                host_rows.tobytes(),
            ]
        )

        np.testing.assert_array_equal(info.dst_kv_indices, device_pages)
        np.testing.assert_array_equal(info.dst_host_kv_indices, host_rows)

    def test_empty_host_indices_remain_distinct_from_absent_metadata(self):
        info = TransferInfo.from_zmq(
            [
                b"9",
                b"127.0.0.1",
                b"12345",
                b"engine",
                np.array([7], dtype=np.int32).tobytes(),
                b"0",
                b"",
                b"1",
                b"0",
                b"",
            ]
        )

        self.assertIsNotNone(info.dst_host_kv_indices)
        self.assertEqual(info.dst_host_kv_indices.size, 0)

    def test_registration_metadata_roundtrips_chunk_groups(self):
        payload = [
            b"None",
            b"127.0.0.1",
            b"12345",
            b"engine",
            b"",
            b"",
            b"state-groups",
            b"0",
            b"8",
            b"0",
            b"1024",
            b"",
            b"",
            b"mem-kinds",
            b"item-lens",
            b"kv-groups",
        ]
        with (
            patch(
                "sglang.srt.disaggregation.mori.conn.EngineDesc",
                SimpleNamespace(
                    unpack=MagicMock(return_value=SimpleNamespace(key="engine"))
                ),
            ),
            patch(
                "sglang.srt.disaggregation.mori.conn._unpack_mem_desc_list",
                side_effect=[[], []],
            ),
            patch(
                "sglang.srt.disaggregation.mori.conn._unpack_mem_desc_lists",
                side_effect=[[["c4-0", "c4-1"], ["indexer-0"]], []],
            ),
            patch(
                "sglang.srt.disaggregation.mori.conn._unpack_list",
                side_effect=[["DRAM", "VRAM"], [1024, 1024]],
            ),
        ):
            from sglang.srt.disaggregation.mori.conn import KVArgsRegisterInfo

            info = KVArgsRegisterInfo.from_zmq(payload)

        self.assertEqual(
            info.dst_kv_mem_desc_groups,
            [["c4-0", "c4-1"], ["indexer-0"]],
        )
        self.assertEqual(info.dst_kv_mem_descs, ["c4-0", "indexer-0"])
        self.assertEqual(info.dst_kv_mem_kinds, ["DRAM", "VRAM"])
        self.assertEqual(info.dst_kv_item_lens, [1024, 1024])

    def test_dram_buffers_register_as_cpu_memory(self):
        manager = object.__new__(MoriKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[1000, 2000],
            kv_data_lens=[100, 200],
            kv_item_lens=[10, 10],
            kv_data_mem_kinds=["DRAM", "VRAM"],
            gpu_id=3,
            aux_data_ptrs=[],
            aux_data_lens=[],
            state_data_ptrs=[],
            state_data_lens=[],
        )
        manager.engine = MagicMock()
        manager.engine.register_memory.side_effect = ["host-desc", "device-desc"]
        manager.kv_mem_descs = []
        manager.aux_mem_descs = []
        manager.state_mem_descs = []

        manager._register_local_buffers()

        self.assertEqual(manager.kv_mem_descs, ["host-desc", "device-desc"])
        self.assertEqual(manager.kv_mem_desc_groups, [["host-desc"], ["device-desc"]])
        manager.engine.register_memory.assert_has_calls(
            [
                call(1000, 100, -1, MemoryLocationType.CPU),
                call(2000, 200, 3, MemoryLocationType.GPU),
            ]
        )

    def test_large_dram_buffer_registration_is_chunked(self):
        manager = object.__new__(MoriKVManager)
        total_size = 2 * DEFAULT_HOST_REGISTRATION_CHUNK_BYTES + 8192
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[0x100000],
            kv_data_lens=[total_size],
            kv_item_lens=[4096],
            kv_data_mem_kinds=["DRAM"],
            gpu_id=3,
            aux_data_ptrs=[],
            aux_data_lens=[],
            state_data_ptrs=[],
            state_data_lens=[],
        )
        manager.engine = MagicMock()
        manager.engine.register_memory.side_effect = (
            lambda ptr, size, device_id, location: SimpleNamespace(
                data=ptr, size=size, device_id=device_id, loc=location
            )
        )
        manager.kv_mem_descs = []
        manager.kv_mem_desc_groups = []
        manager.aux_mem_descs = []
        manager.state_mem_descs = []

        manager._register_local_buffers()

        sizes = [call.args[1] for call in manager.engine.register_memory.call_args_list]
        self.assertEqual(
            sizes,
            [
                DEFAULT_HOST_REGISTRATION_CHUNK_BYTES,
                DEFAULT_HOST_REGISTRATION_CHUNK_BYTES,
                8192,
            ],
        )
        self.assertEqual(len(manager.kv_mem_desc_groups[0]), 3)
        self.assertIs(manager.kv_mem_descs[0], manager.kv_mem_desc_groups[0][0])

    def test_mixed_item_plan_scatters_fragmented_host_rows(self):
        manager = object.__new__(MoriKVManager)

        plan = manager._build_mixed_item_transfer_plan(
            np.array([2], dtype=np.int32),
            np.array([100, 101, 180, 181], dtype=np.int32),
            src_item_len=40,
            dst_item_len=10,
        )

        self.assertEqual(plan.local_offsets, [80, 100])
        self.assertEqual(plan.remote_offsets, [1000, 1800])
        self.assertEqual(plan.sizes, [20, 20])

    def test_mla_transfer_uses_host_and_device_index_spaces(self):
        manager = object.__new__(MoriKVManager)
        manager.is_mla_backend = True
        manager.kv_mem_descs = ["src-c4", "src-indexer"]
        manager.kv_args = SimpleNamespace(
            kv_item_lens=[40, 40],
            prefill_start_layer=0,
            prefill_end_layer=2,
            mla_compression_ratios=None,
        )
        manager._submit_batch_transfer_plan = MagicMock(return_value=[])
        peer_info = SimpleNamespace(
            dst_kv_mem_descs=["dst-c4", "dst-indexer"],
            dst_kv_mem_kinds=["DRAM", "VRAM"],
            dst_kv_item_lens=[10, 40],
        )

        manager.send_kvcache(
            peer_info,
            np.array([2], dtype=np.int32),
            np.array([7], dtype=np.int32),
            dst_host_kv_indices=np.array([100, 101, 180, 181], dtype=np.int32),
        )

        host_call, device_call = manager._submit_batch_transfer_plan.call_args_list
        self.assertEqual(host_call.args[:2], (["src-c4"], ["dst-c4"]))
        self.assertEqual(host_call.args[2].local_offsets, [80, 100])
        self.assertEqual(host_call.args[2].remote_offsets, [1000, 1800])
        self.assertEqual(host_call.args[2].sizes, [20, 20])
        self.assertEqual(device_call.args[:2], (["src-indexer"], ["dst-indexer"]))
        self.assertEqual(device_call.args[2].local_offsets, [80])
        self.assertEqual(device_call.args[2].remote_offsets, [280])
        self.assertEqual(device_call.args[2].sizes, [40])

    def test_dsv4_pp_slice_preserves_bucket_order(self):
        manager = object.__new__(MoriKVManager)
        manager.kv_mem_descs = ["src-c4", "src-indexer", "src-c128"]
        manager.kv_args = SimpleNamespace(
            prefill_start_layer=2,
            prefill_end_layer=4,
            mla_compression_ratios=[4, 4, 4, 128],
        )
        peer_info = SimpleNamespace(
            dst_kv_mem_descs=[f"dst-{i}" for i in range(7)],
            dst_kv_mem_kinds=["DRAM", "DRAM", "DRAM"] + ["VRAM"] * 4,
            dst_kv_item_lens=[10] * 7,
        )

        _, dst_descs, dst_kinds, dst_item_lens = manager._get_mla_mem_desc_slices(
            peer_info
        )

        self.assertEqual(dst_descs, [["dst-2"], ["dst-5"], ["dst-6"]])
        self.assertEqual(dst_kinds, ["DRAM", "VRAM", "VRAM"])
        self.assertEqual(dst_item_lens, [10, 10, 10])

    def test_c4less_pp_stage_ignores_request_host_metadata(self):
        manager = object.__new__(MoriKVManager)
        manager.kv_mem_descs = ["src-c128"]
        manager.kv_args = SimpleNamespace(
            kv_item_lens=[40],
            prefill_start_layer=1,
            prefill_end_layer=2,
            mla_compression_ratios=[4, 128],
        )
        peer_info = SimpleNamespace(
            dst_kv_mem_descs=["dst-c4", "dst-indexer", "dst-c128"],
            dst_kv_mem_kinds=["DRAM", "VRAM", "VRAM"],
            dst_kv_item_lens=[10, 40, 40],
        )

        rows_per_page = manager._host_rows_per_source_page(peer_info)

        self.assertIsNone(rows_per_page)

    def test_transfer_plan_splits_at_registration_boundaries(self):
        manager = object.__new__(MoriKVManager)
        manager.engine = MagicMock()
        manager.engine.allocate_transfer_uid.side_effect = [11, 12, 13]
        manager.engine.batch_write.side_effect = [["s0"], ["s1"], ["s2"]]
        src = SimpleNamespace(size=300)
        dst = [
            SimpleNamespace(size=100),
            SimpleNamespace(size=100),
            SimpleNamespace(size=100),
        ]

        statuses = manager._submit_batch_transfer_plan(
            [src],
            dst,
            BatchTransferPlan(
                local_offsets=[0],
                remote_offsets=[50],
                sizes=[200],
            ),
        )

        self.assertEqual(statuses, ["s0", "s1", "s2"])
        calls = manager.engine.batch_write.call_args_list
        self.assertEqual(calls[0].args[1:5], ([[0]], [dst[0]], [[50]], [[50]]))
        self.assertEqual(calls[1].args[1:5], ([[50]], [dst[1]], [[0]], [[100]]))
        self.assertEqual(calls[2].args[1:5], ([[150]], [dst[2]], [[0]], [[50]]))

    def test_receiver_serializes_host_indices(self):
        receiver = object.__new__(MoriKVReceiver)
        receiver.bootstrap_infos = [{"is_dummy": False}]
        receiver.bootstrap_room = 9
        receiver.required_dst_info_num = 1
        receiver.kv_mgr = SimpleNamespace(
            local_ip="127.0.0.1",
            rank_port=12345,
            engine_desc=SimpleNamespace(key="engine"),
        )
        socket = MagicMock()
        receiver._connect_to_bootstrap_server = MagicMock(
            return_value=(socket, threading.Lock())
        )
        host_rows = np.array([100, 101, 180], dtype=np.int32)

        receiver.send_metadata(
            np.array([7], dtype=np.int32),
            host_kv_indices=host_rows,
        )

        payload = socket.send_multipart.call_args.args[0]
        np.testing.assert_array_equal(
            np.frombuffer(payload[10], dtype=np.int32), host_rows
        )

    def test_receiver_preserves_present_but_empty_host_indices(self):
        receiver = object.__new__(MoriKVReceiver)
        receiver.bootstrap_infos = [{"is_dummy": False}]
        receiver.bootstrap_room = 9
        receiver.required_dst_info_num = 1
        receiver.kv_mgr = SimpleNamespace(
            local_ip="127.0.0.1",
            rank_port=12345,
            engine_desc=SimpleNamespace(key="engine"),
        )
        socket = MagicMock()
        receiver._connect_to_bootstrap_server = MagicMock(
            return_value=(socket, threading.Lock())
        )

        receiver.send_metadata(
            np.array([7], dtype=np.int32),
            host_kv_indices=np.array([], dtype=np.int32),
        )

        payload = socket.send_multipart.call_args.args[0]
        self.assertEqual(len(payload), 11)
        self.assertEqual(payload[10], b"")


if __name__ == "__main__":
    unittest.main()
