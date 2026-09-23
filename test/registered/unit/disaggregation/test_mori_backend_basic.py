import struct
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.mori import conn as mori_conn
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=0,
    suite="base-a-test-cpu",
    disabled="run by the run-mori-pd workflow",
)


class _EngineDesc:
    key = "peer"

    @classmethod
    def unpack(cls, value):
        return cls()


class TestMoriDCP(unittest.TestCase):
    def test_registration_parses_dcp_metadata(self):
        payload = [
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
            b"8",
            b"3",
            mori_conn.pack_int_lists([[17, 23]], "I"),
            struct.pack("<3I", 11, 17, 23),
        ]

        with patch.object(mori_conn, "EngineDesc", _EngineDesc):
            info = mori_conn.KVArgsRegisterInfo.from_zmq(payload)

        self.assertEqual((info.dst_dcp_size, info.dst_dcp_rank), (8, 3))
        self.assertEqual(info.dst_state_layer_ids, [[17, 23]])
        self.assertEqual(info.dst_kv_layer_ids, [11, 17, 23])

    def test_dispatch_uses_full_destination_indices(self):
        manager = object.__new__(mori_conn.MoriKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.request_status = {7: mori_conn.KVPoll.WaitingForInput}
        manager.transfer_lock = threading.Lock()
        dst_indices = np.array([100, 200, 300], dtype=np.int32)
        manager.transfer_infos = {
            7: {
                "peer": mori_conn.TransferInfo(
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
            }
        }
        manager.decode_kv_args_table = {
            "peer": SimpleNamespace(requires_dcp_relayout=True)
        }
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

    def test_pp_local_descriptors_map_by_layer(self):
        manager = object.__new__(mori_conn.MoriKVManager)
        manager.kv_mem_descs = ["src-17", "src-23"]
        manager.kv_args = SimpleNamespace(kv_layer_ids=[17, 23])

        src, dst, count = manager._get_mla_mem_desc_slices(
            ["dst-11", "dst-17", "dst-23"], [11, 17, 23]
        )

        self.assertEqual(src, ["src-17", "src-23"])
        self.assertEqual(dst, ["dst-17", "dst-23"])
        self.assertEqual(count, 2)

    def test_tp_mismatch_requires_matching_token_geometry(self):
        manager = object.__new__(mori_conn.MoriKVManager)
        manager.dcp_size = 1
        manager.dcp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.kv_args = SimpleNamespace(
            page_size=64, kv_item_lens=[4096], num_draft_entries=0
        )
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

        info.dst_kv_item_len = 4096
        manager._add_remote_peer(info)
        self.assertTrue(info.requires_dcp_relayout)
        self.assertEqual(info.dcp_token_item_lens, [64])


if __name__ == "__main__":
    unittest.main()
