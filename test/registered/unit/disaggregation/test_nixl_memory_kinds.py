import struct
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.nixl.conn import (
    KVArgsRegisterInfo,
    NixlKVManager,
    TransferStatus,
    _KVXferPreparedSegment,
    _kv_xfer_mem_segments,
    _homogeneous_kv_mem_kind,
    _pack_kv_mem_kinds,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeNixlAgent:
    def __init__(self):
        self.register_calls = []
        self.prep_calls = []
        self.xfer_calls = []
        self.transfer_calls = []

    def register_memory(self, addrs, mem_kind):
        self.register_calls.append((list(addrs), mem_kind))
        return f"{mem_kind}-registration"

    def prep_xfer_dlist(self, peer_name, array, mem_kind):
        self.prep_calls.append((peer_name, np.array(array), mem_kind))
        return f"{peer_name}-{mem_kind}-prep"

    def make_prepped_xfer(
        self, op, src_handle, src_indices, dst_handle, dst_indices, notif
    ):
        self.xfer_calls.append(
            (
                op,
                src_handle,
                np.array(src_indices),
                dst_handle,
                np.array(dst_indices),
                notif,
            )
        )
        return f"xfer-{len(self.xfer_calls)}"

    def transfer(self, handle):
        self.transfer_calls.append(handle)
        return "DONE"


class TestNixlMemoryKinds(unittest.TestCase):
    def test_register_buffer_to_engine_buckets_kv_by_memory_kind(self):
        agent = _FakeNixlAgent()
        mgr = NixlKVManager.__new__(NixlKVManager)
        mgr.agent = agent
        mgr.kv_args = SimpleNamespace(
            kv_data_ptrs=[100, 200, 300],
            kv_data_lens=[10, 20, 30],
            kv_data_mem_kinds=["VRAM", "DRAM", "DRAM"],
            aux_data_ptrs=[400],
            aux_data_lens=[40],
            state_data_ptrs=[],
            state_data_lens=[],
            gpu_id=7,
        )

        NixlKVManager.register_buffer_to_engine(mgr)

        self.assertEqual(
            agent.register_calls,
            [
                ([(100, 10, 7, "")], "VRAM"),
                ([(200, 20, 0, ""), (300, 30, 0, "")], "DRAM"),
                ([(400, 40, 0, "")], "DRAM"),
            ],
        )
        self.assertEqual(mgr.kv_descs, ["VRAM-registration", "DRAM-registration"])

    def test_equal_tp_prep_handle_uses_dram_device_id_zero(self):
        agent = _FakeNixlAgent()
        mgr = NixlKVManager.__new__(NixlKVManager)
        mgr.agent = agent
        mgr.prep_handles = {}
        mgr.kv_args = SimpleNamespace(kv_item_lens=[8], kv_data_lens=[16])

        NixlKVManager._init_equal_tp_prep_handle(
            mgr,
            peer_name="decode-peer",
            kv_ptrs=[1000],
            gpu_id=3,
            num_slots=2,
            mem_kind="DRAM",
        )

        self.assertEqual(mgr.prep_handles["decode-peer"], "decode-peer-DRAM-prep")
        peer_name, array, mem_kind = agent.prep_calls[0]
        self.assertEqual(peer_name, "decode-peer")
        self.assertEqual(mem_kind, "DRAM")
        np.testing.assert_array_equal(
            array,
            np.array([[1000, 8, 0], [1008, 8, 0]], dtype=np.int64),
        )

    def test_register_info_defaults_old_messages_to_vram(self):
        msg = self._register_msg(kv_mem_kinds=None)

        info = KVArgsRegisterInfo.from_zmq(msg)

        self.assertEqual(info.dst_kv_ptrs, [111, 222])
        self.assertEqual(info.dst_kv_mem_kinds, ["VRAM", "VRAM"])

    def test_register_info_parses_explicit_memory_kinds(self):
        msg = self._register_msg(kv_mem_kinds=["DRAM", "DRAM"])

        info = KVArgsRegisterInfo.from_zmq(msg)

        self.assertEqual(info.dst_kv_mem_kinds, ["DRAM", "DRAM"])

    def test_mixed_memory_kind_guard_is_explicit(self):
        with self.assertRaises(NotImplementedError):
            _homogeneous_kv_mem_kind(["DRAM", "VRAM"], "destination")

    def test_mixed_memory_segments_group_contiguous_pairs(self):
        segments = _kv_xfer_mem_segments(
            ["VRAM", "VRAM", "VRAM", "VRAM"],
            ["DRAM", "DRAM", "VRAM", "VRAM"],
        )

        self.assertEqual(
            [
                (seg.start, seg.end, seg.src_mem_kind, seg.dst_mem_kind)
                for seg in segments
            ],
            [(0, 2, "VRAM", "DRAM"), (2, 4, "VRAM", "VRAM")],
        )

    def test_prepare_payload_xfer_builds_mixed_prepped_segments(self):
        agent = _FakeNixlAgent()
        mgr = NixlKVManager.__new__(NixlKVManager)
        mgr.agent = agent
        mgr.kv_args = SimpleNamespace(
            kv_data_ptrs=[1000, 2000, 3000],
            kv_data_lens=[16, 16, 16],
            kv_item_lens=[8, 8, 8],
            kv_data_mem_kinds=["VRAM", "VRAM", "VRAM"],
            gpu_id=3,
        )
        mgr.prep_handles = {}
        mgr.prep_handles_segment_src = {}
        mgr.is_mla_backend = True
        mgr.attn_tp_size = 1
        mgr._num_slots_src = 2
        mgr.src_mem_kind = "VRAM"

        peer = KVArgsRegisterInfo(
            room="0",
            endpoint="127.0.0.1",
            dst_port=1,
            agent_name="decode-peer",
            agent_metadata=b"",
            dst_kv_ptrs=[4000, 5000, 6000],
            dst_kv_mem_kinds=["DRAM", "DRAM", "VRAM"],
            dst_aux_ptrs=[],
            dst_state_data_ptrs=[],
            gpu_id=7,
            decode_tp_size=1,
            decode_tp_rank=0,
            dst_kv_item_len=8,
            dst_num_slots=2,
        )

        NixlKVManager._prepare_payload_xfer(mgr, peer)

        self.assertIsNone(peer.dst_homogeneous_mem_kind)
        self.assertIsNotNone(peer.kv_xfer_segments)
        self.assertEqual(len(peer.kv_xfer_segments), 2)
        self.assertEqual(
            [(call[0], call[2]) for call in agent.prep_calls],
            [
                ("", "VRAM"),
                ("decode-peer", "DRAM"),
                ("", "VRAM"),
                ("decode-peer", "VRAM"),
            ],
        )
        np.testing.assert_array_equal(
            agent.prep_calls[1][1][:, 2],
            np.zeros(4, dtype=np.int64),
        )

    def test_send_kvcache_mixed_posts_part_notifications(self):
        agent = _FakeNixlAgent()
        mgr = NixlKVManager.__new__(NixlKVManager)
        mgr.agent = agent
        mgr._num_slots_src = 4
        peer = KVArgsRegisterInfo(
            room="0",
            endpoint="127.0.0.1",
            dst_port=1,
            agent_name="decode-peer",
            agent_metadata=b"",
            dst_kv_ptrs=[],
            dst_kv_mem_kinds=[],
            dst_aux_ptrs=[],
            dst_state_data_ptrs=[],
            gpu_id=7,
            decode_tp_size=1,
            decode_tp_rank=0,
            dst_kv_item_len=8,
            dst_num_slots=4,
        )
        peer.kv_xfer_segments = [
            _KVXferPreparedSegment(0, 2, "src-dram", "dst-dram", 4),
            _KVXferPreparedSegment(2, 3, "src-vram", "dst-vram", 4),
        ]
        mgr.decode_kv_args_table = {"decode-peer": peer}

        handles = NixlKVManager.send_kvcache_mixed(
            mgr,
            "decode-peer",
            np.array([0, 1], dtype=np.int32),
            np.array([2, 3], dtype=np.int32),
            "9_kv_4_1_0",
        )

        self.assertEqual(handles, ["xfer-1", "xfer-2"])
        self.assertEqual(agent.transfer_calls, handles)
        self.assertEqual(
            [call[5] for call in agent.xfer_calls],
            [b"9_kv_4_1_0_part_0_2", b"9_kv_4_1_0_part_1_2"],
        )

    def test_kv_part_arrival_waits_for_all_segments(self):
        mgr = NixlKVManager.__new__(NixlKVManager)
        mgr.transfer_statuses = {1: TransferStatus()}
        mgr.required_prefill_response_num_table = {1: 1}
        mgr.enable_staging = False
        mgr._staging_handler = None

        NixlKVManager._track_kv_part_arrival(
            mgr,
            room=1,
            chunk_id=2,
            is_last_chunk=True,
            pp_rank=0,
            part_idx=0,
            num_parts=2,
        )
        self.assertNotIn(2, mgr.transfer_statuses[1].received_kvs_per_pp[0])

        NixlKVManager._track_kv_part_arrival(
            mgr,
            room=1,
            chunk_id=2,
            is_last_chunk=True,
            pp_rank=0,
            part_idx=1,
            num_parts=2,
        )
        self.assertIn(2, mgr.transfer_statuses[1].received_kvs_per_pp[0])
        self.assertEqual(mgr.transfer_statuses[1].expected_kvs_per_pp[0], 3)

    @staticmethod
    def _register_msg(kv_mem_kinds):
        msg = [
            b"None",
            b"127.0.0.1",
            b"1234",
            b"agent-a",
            b"agent-metadata",
            struct.pack("QQ", 111, 222),
            struct.pack("Q", 333),
            pack_int_lists([], "Q"),
            b"5",
            b"1",
            b"0",
            b"16",
            pack_int_lists([], "I"),
            pack_int_lists([], "I"),
            b"",
            b"",
            b"8",
        ]
        if kv_mem_kinds is not None:
            msg.append(_pack_kv_mem_kinds(kv_mem_kinds))
        return msg


if __name__ == "__main__":
    unittest.main()
