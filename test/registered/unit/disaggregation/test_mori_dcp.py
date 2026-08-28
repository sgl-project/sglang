import importlib
import struct
import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.srt.disaggregation.prefill import (
    SchedulerDisaggregationPrefillMixin,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _import_mori_conn():
    class EngineDesc:
        def __init__(self, key="peer"):
            self.key = key

        @classmethod
        def unpack(cls, value):
            return cls(value.decode())

    class MemoryDesc:
        @classmethod
        def unpack(cls, value):
            return value

    mori = types.ModuleType("mori")
    cpp = types.ModuleType("mori.cpp")
    io = types.ModuleType("mori.io")
    cpp.TransferStatus = object
    io.EngineDesc = EngineDesc
    io.MemoryDesc = MemoryDesc
    io.IOEngine = object
    io.IOEngineConfig = object
    io.RdmaBackendConfig = object
    io.BackendType = SimpleNamespace(RDMA=1)
    io.MemoryLocationType = SimpleNamespace(GPU=1, CPU=2)
    io.PollCqMode = SimpleNamespace(POLLING=1)
    io.StatusCode = SimpleNamespace(IN_PROGRESS=1, SUCCESS=0)
    modules = {"mori": mori, "mori.cpp": cpp, "mori.io": io}
    old_modules = {name: sys.modules.get(name) for name in modules}
    try:
        sys.modules.update(modules)
        sys.modules.pop("sglang.srt.disaggregation.mori.conn", None)
        return importlib.import_module("sglang.srt.disaggregation.mori.conn")
    finally:
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


mori_conn = _import_mori_conn()


def _registration_payload(*extra):
    return [
        b"None",
        b"127.0.0.1",
        b"1234",
        b"peer",
        b"",
        b"",
        b"",
        b"0",
        b"8",
        b"3",
        b"4096",
        b"",
        b"",
        *extra,
    ]


class TestMoriDCP(unittest.TestCase):
    def test_registration_wire_is_backward_compatible(self):
        legacy = mori_conn.KVArgsRegisterInfo.from_zmq(_registration_payload())
        self.assertEqual((legacy.dst_dcp_size, legacy.dst_dcp_rank), (1, 0))
        self.assertEqual(legacy.dst_state_layer_ids, [])
        self.assertEqual(legacy.dst_kv_layer_ids, [])

        payload = _registration_payload(
            b"8",
            b"3",
            mori_conn.pack_int_lists([[17, 23]], "I"),
            struct.pack("<3I", 11, 17, 23),
        )
        info = mori_conn.KVArgsRegisterInfo.from_zmq(payload)
        self.assertEqual((info.dst_dcp_size, info.dst_dcp_rank), (8, 3))
        self.assertEqual(info.dst_state_layer_ids, [[17, 23]])
        self.assertEqual(info.dst_kv_layer_ids, [11, 17, 23])

    def test_manager_dispatches_dcp_with_full_destination_indices(self):
        manager = object.__new__(mori_conn.MoriKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.request_status = {7: mori_conn.KVPoll.WaitingForInput}
        manager.transfer_lock = threading.Lock()
        dst_indices = np.array([100, 200, 300], dtype=np.int32)
        transfer_info = mori_conn.TransferInfo(
            room=7,
            endpoint="127.0.0.1",
            dst_port=1,
            engine_key="peer",
            dst_kv_indices=dst_indices,
            dst_aux_index=-1,
            dst_state_indices=[],
            required_dst_info_num=1,
            is_dummy=False,
        )
        peer = SimpleNamespace(requires_dcp_relayout=True)
        manager.transfer_infos = {7: {"peer": transfer_info}}
        manager.decode_kv_args_table = {"peer": peer}
        manager.update_status = Mock()
        manager.send_kvcache_dcp = Mock(return_value=[])

        manager._submit_kv_transfer(
            7,
            np.array([9], dtype=np.int32),
            slice(2, 3),
            False,
            num_kv_tokens=13,
        )

        args, kwargs = manager.send_kvcache_dcp.call_args
        self.assertIs(args[2], dst_indices)
        self.assertEqual(kwargs["src_page_offset"], 2)
        self.assertEqual(kwargs["num_kv_tokens"], 13)

    def test_pp_local_kv_descriptors_map_by_global_layer(self):
        manager = object.__new__(mori_conn.MoriKVManager)
        manager.kv_mem_descs = ["src-17", "src-23"]
        manager.kv_args = SimpleNamespace(kv_layer_ids=[17, 23], prefill_start_layer=0)

        src, dst, count = manager._get_mla_mem_desc_slices(
            ["dst-11", "dst-17", "dst-23"], [11, 17, 23]
        )

        self.assertEqual(src, ["src-17", "src-23"])
        self.assertEqual(dst, ["dst-17", "dst-23"])
        self.assertEqual(count, 2)

    def test_dcp_tp_mismatch_requires_equal_token_item_lengths(self):
        manager = object.__new__(mori_conn.MoriKVManager)
        manager.dcp_size = 1
        manager.dcp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.kv_args = SimpleNamespace(page_size=64, kv_item_lens=[4096])
        manager.decode_kv_args_table = {}
        manager.engine = Mock()
        info = mori_conn.KVArgsRegisterInfo(
            endpoint="127.0.0.1",
            dst_port=1,
            engine_desc=SimpleNamespace(key="peer"),
            dst_kv_mem_descs=[],
            dst_aux_mem_descs=[],
            dst_state_mem_descs=[],
            gpu_id=0,
            decode_tp_size=8,
            decode_tp_rank=0,
            dst_kv_item_len=2048,
            dst_state_item_lens=[],
            dst_state_dim_per_tensor=[],
            dst_dcp_size=8,
        )

        with self.assertRaisesRegex(RuntimeError, "KV geometry differs"):
            manager._add_remote_peer(info)
        manager.engine.register_remote_engine.assert_not_called()

        info.dst_kv_item_len = 4096
        manager._add_remote_peer(info)
        self.assertTrue(info.requires_dcp_relayout)
        self.assertEqual(info.dcp_token_item_lens, [64])
        manager.engine.register_remote_engine.assert_called_once()

    def test_mamba_state_uses_layer_mapping_and_layout_slices(self):
        manager = object.__new__(mori_conn.MoriKVManager)
        manager.attn_tp_size = 1
        manager.pp_size = 8
        manager.kv_args = SimpleNamespace(engine_rank=0)
        manager.engine = Mock()
        manager.engine.allocate_transfer_uid.side_effect = [1, 2, 3]
        manager.engine.batch_write.return_value = []
        peer = SimpleNamespace(decode_tp_size=8, decode_tp_rank=3)

        manager._send_mamba_state(
            peer,
            np.array([2], dtype=np.int32),
            np.array([5], dtype=np.int32),
            ["src-17"],
            ["dst-11", "dst-17"],
            [96],
            [12, 12],
            [24],
            [3, 3],
            [[8, 8, 8]],
            [1],
            [17],
            [11, 17],
        )

        self.assertEqual(manager.engine.batch_write.call_count, 3)
        for call in manager.engine.batch_write.call_args_list:
            self.assertEqual(call.args[2], ["dst-17"])
            self.assertEqual(call.args[4], [[4]])

    def test_cached_prefix_early_send_skips_dcp_relayout(self):
        sender = SimpleNamespace(requires_dcp_relayout=Mock(return_value=True))
        req = SimpleNamespace(pending_bootstrap=False, disagg_kv_sender=sender)
        scheduler = SimpleNamespace(enable_staging=False, send_kv_chunk=Mock())

        with envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.override(True):
            SchedulerDisaggregationPrefillMixin.maybe_send_cached_prefix_chunk(
                scheduler, req
            )

        scheduler.send_kv_chunk.assert_not_called()


if __name__ == "__main__":
    unittest.main()
