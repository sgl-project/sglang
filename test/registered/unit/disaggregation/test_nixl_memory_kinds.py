import struct
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.common.utils import pack_int_lists
from sglang.srt.disaggregation.nixl.conn import (
    KVArgsRegisterInfo,
    NixlKVManager,
    _homogeneous_kv_mem_kind,
    _pack_kv_mem_kinds,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeNixlAgent:
    def __init__(self):
        self.register_calls = []
        self.prep_calls = []

    def register_memory(self, addrs, mem_kind):
        self.register_calls.append((list(addrs), mem_kind))
        return f"{mem_kind}-registration"

    def prep_xfer_dlist(self, peer_name, array, mem_kind):
        self.prep_calls.append((peer_name, np.array(array), mem_kind))
        return f"{peer_name}-{mem_kind}-prep"


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
