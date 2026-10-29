import struct
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from sglang.srt.disaggregation.mori import conn as mori_conn
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, suite="stage-a-test-1-gpu-small-amd")


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
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        manager.dcp_size = 1
        manager.dcp_rank = 0
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.kv_args = SimpleNamespace(
            page_size=64, kv_item_lens=[4096], num_draft_entries=0, aux_data_ptrs=[]
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


class TestMoriDsaStateElision(unittest.TestCase):
    """_send_swa_dsa_state must read the dst length of the paired dst entry.

    A prefill PP stage holds a slice of the layers while decode holds all of
    them, so src entry i pairs with dst entry j != i.
    """

    def _send(self, src_layers, dst_layers, src_lens, dst_lens):
        manager = object.__new__(mori_conn.MoriKVManager)
        manager.pp_size = 2
        manager.is_mla_backend = True
        manager.attn_tp_size = 1
        manager._build_contiguous_transfer_plan = Mock(return_value="plan")
        manager._submit_batch_transfer_plan = Mock(return_value=[])
        indices = np.array([1, 2], dtype=np.int32)

        manager._send_swa_dsa_state(
            SimpleNamespace(decode_tp_size=1),
            indices,
            indices,
            [f"s{layer}" for layer in src_layers],
            src_lens,
            [
                None if length == 0 else f"d{layer}"
                for layer, length in zip(dst_layers, dst_lens)
            ],
            dst_lens,
            "dsa",
            src_layers,
            dst_layers,
        )
        return manager._submit_batch_transfer_plan.call_args_list

    def test_dst_elision_is_read_at_the_paired_entry(self):
        # Decode elided layer 2 only. Both prefill layers must still be sent;
        # reading dst_lens[i] instead of dst_lens[j] drops layer 3.
        calls = self._send([3, 4], [2, 3, 4], [132, 132], [0, 132, 132])
        self.assertEqual(calls, [call("s3", "d3", "plan"), call("s4", "d4", "plan")])

    def test_elided_dst_entry_is_never_written(self):
        # Decode elided layer 4. Writing to it would use a None descriptor.
        calls = self._send([4, 5], [3, 4, 5], [132, 132], [132, 0, 132])
        self.assertEqual(calls, [call("s5", "d5", "plan")])


if __name__ == "__main__":
    unittest.main()
