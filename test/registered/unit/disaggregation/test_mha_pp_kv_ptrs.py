"""Pointer arithmetic and wire tests for Mooncake MHA PP transfers."""

import ast
import inspect
import struct
import textwrap
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    MooncakeKVReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SRC_K, _SRC_V = 1000, 1100
_DST_K, _DST_V, _DRAFT_K, _DRAFT_V = 2000, 3000, 4000, 5000


def _dst_ptrs(num_target, num_draft=0):
    return (
        [_DST_K + i for i in range(num_target)]
        + [_DST_V + i for i in range(num_target)]
        + [_DRAFT_K + i for i in range(num_draft)]
        + [_DRAFT_V + i for i in range(num_draft)]
    )


def _src_ptrs(start, num):
    return [_SRC_K + i for i in range(start, start + num)] + [
        _SRC_V + i for i in range(start, start + num)
    ]


def _mha_manager(start_layer):
    return SimpleNamespace(kv_args=SimpleNamespace(prefill_start_layer=start_layer))


class TestGetMhaKvPtrsWithPp(CustomTestCase):
    def test_uneven_7_8_of_15_uses_explicit_target_count(self):
        dst = _dst_ptrs(15)
        _, _, dst_k0, dst_v0, n0 = CommonKVManager.get_mha_kv_ptrs_with_pp(
            _mha_manager(0), _src_ptrs(0, 7), dst, num_dst_target_kv_layers=15
        )
        _, _, dst_k1, dst_v1, n1 = CommonKVManager.get_mha_kv_ptrs_with_pp(
            _mha_manager(7), _src_ptrs(7, 8), dst, num_dst_target_kv_layers=15
        )
        self.assertEqual((n0, n1), (7, 8))
        self.assertEqual(dst_k0, [_DST_K + i for i in range(7)])
        self.assertEqual(dst_v0, [_DST_V + i for i in range(7)])
        self.assertEqual(dst_k1, [_DST_K + i for i in range(7, 15)])
        self.assertEqual(dst_v1, [_DST_V + i for i in range(7, 15)])

    def test_equal_layout_is_unchanged(self):
        dst = _dst_ptrs(4)
        result = CommonKVManager.get_mha_kv_ptrs_with_pp(
            _mha_manager(0), _src_ptrs(0, 4), dst, num_dst_target_kv_layers=4
        )
        self.assertEqual(result[0], _src_ptrs(0, 4)[:4])
        self.assertEqual(result[1], _src_ptrs(0, 4)[4:])
        self.assertEqual(result[2], [_DST_K + i for i in range(4)])
        self.assertEqual(result[3], [_DST_V + i for i in range(4)])
        self.assertEqual(result[4], 4)

    def test_mtp_final_stage_keeps_draft_k_out_of_v_section(self):
        _, _, dst_k, dst_v, n = CommonKVManager.get_mha_kv_ptrs_with_pp(
            _mha_manager(11),
            _src_ptrs(11, 4),
            _dst_ptrs(15, num_draft=1),
            num_dst_target_kv_layers=15,
        )
        self.assertEqual(n, 4)
        self.assertEqual(dst_k, [_DST_K + i for i in range(11, 15)])
        self.assertEqual(dst_v, [_DST_V + i for i in range(11, 15)])
        self.assertNotIn(_DRAFT_K, dst_v)

    def test_none_and_minus_one_preserve_existing_fallback(self):
        src = _src_ptrs(0, 15)
        dst = _dst_ptrs(15, num_draft=1)
        omitted = CommonKVManager.get_mha_kv_ptrs_with_pp(_mha_manager(0), src, dst)
        none = CommonKVManager.get_mha_kv_ptrs_with_pp(
            _mha_manager(0), src, dst, num_dst_target_kv_layers=None
        )
        minus_one = CommonKVManager.get_mha_kv_ptrs_with_pp(
            _mha_manager(0), src, dst, num_dst_target_kv_layers=-1
        )
        self.assertEqual(omitted, none)
        self.assertEqual(none, minus_one)
        self.assertEqual(minus_one[3], [_DST_V + i for i in range(15)])


def _registration_frames():
    return [
        b"room",
        b"127.0.0.1",
        b"1234",
        b"session",
        struct.pack("Q", 0x1000),
        struct.pack("Q", 0x2000),
        b"",
        b"0",
        b"1",
        b"128",
        b"",
        b"",
        b"",
        b"",
        struct.pack("Q", 0x3000),
        b"4096",
        b"4",
        b"2",
    ]


class TestMooncakeRegistrationTargetCount(CustomTestCase):
    def test_old_and_current_staging_frames_keep_optional_count_compatible(self):
        old = _registration_frames()
        old_info = KVArgsRegisterInfo.from_zmq(old)
        self.assertEqual(len(old), 18)
        self.assertEqual(old_info.dst_num_target_kv_layers, -1)
        self.assertEqual(old_info.staging.slot_layer_ids, [])

        slot_ids = struct.pack("QQ", 7, 15)
        current = old + [slot_ids]
        current_info = KVArgsRegisterInfo.from_zmq(current)
        self.assertEqual(current_info.staging.slot_layer_ids, [7, 15])
        self.assertEqual(current_info.dst_num_target_kv_layers, -1)

        current += [struct.pack("QQ", 128, 64)]
        current_info = KVArgsRegisterInfo.from_zmq(current)
        self.assertEqual(current_info.dst_kv_item_lens, [128, 64])
        self.assertEqual(current_info.dst_num_target_kv_layers, -1)
        new_info = KVArgsRegisterInfo.from_zmq(current + [b"15"])
        self.assertEqual(new_info.staging.slot_layer_ids, [7, 15])
        self.assertEqual(new_info.dst_kv_item_lens, [128, 64])
        self.assertEqual(new_info.dst_num_target_kv_layers, 15)
        empty_info = KVArgsRegisterInfo.from_zmq(current + [b""])
        self.assertEqual(empty_info.dst_num_target_kv_layers, -1)


class _RecordingMhaManager:
    get_mha_kv_ptrs_with_pp = CommonKVManager.get_mha_kv_ptrs_with_pp

    def __init__(self, start_layer):
        self.is_mla_backend = False
        self.is_hybrid_mla_backend = False
        self.enable_custom_mem_pool = False
        self.max_transfer_batch_indices = 0
        self.pp_size = 2
        self.kv_args = SimpleNamespace(prefill_start_layer=start_layer)
        self.blocks = []

    def _transfer_data(self, session_id, blocks):
        self.blocks.extend(blocks)
        return 0


class TestMooncakeTargetCountForwarding(CustomTestCase):
    def test_generic_mha_edge_uses_target_count(self):
        manager = _RecordingMhaManager(start_layer=7)
        rc = MooncakeKVManager._send_kvcache_generic(
            manager,
            mooncake_session_id="session",
            src_data_ptrs=_src_ptrs(7, 8),
            dst_data_ptrs=_dst_ptrs(15),
            item_lens=[1] * 16,
            prefill_data_indices=np.array([0], dtype=np.int32),
            dst_data_indices=np.array([0], dtype=np.int32),
            executor=None,
            num_dst_target_kv_layers=15,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(
            manager.blocks,
            [(_SRC_K + i, _DST_K + i, 1) for i in range(7, 15)]
            + [(_SRC_V + i, _DST_V + i, 1) for i in range(7, 15)],
        )

    def test_send_kvcache_forwards_target_count(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[], kv_item_lens=[], kv_layer_ids=[]
        )
        manager._validate_envelope_kv_layout = MagicMock()
        manager._send_kvcache_generic = MagicMock(return_value=0)
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.get_memory",
            return_value=SimpleNamespace(enable_unified_memory=False),
        ):
            rc = MooncakeKVManager.send_kvcache(
                manager,
                "session",
                np.array([], dtype=np.int32),
                [],
                np.array([], dtype=np.int32),
                None,
                num_dst_target_kv_layers=15,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(
            manager._send_kvcache_generic.call_args.kwargs["num_dst_target_kv_layers"],
            15,
        )

    def test_send_kvcache_slice_forwards_target_count(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.kv_args = SimpleNamespace(
            engine_rank=0,
            kv_item_lens=[1, 1],
            kv_data_ptrs=[1000, 1100],
            kv_layer_ids=[],
            kv_head_num=1,
            total_kv_head_num=1,
            page_size=1,
        )
        manager.attn_tp_size = 1
        manager.pp_size = 2
        manager.get_mha_kv_ptrs_with_pp = MagicMock(return_value=([], [], [], [], 0))
        manager._await_transfer_futures = MagicMock(return_value=0)
        rc = MooncakeKVManager.send_kvcache_slice(
            manager,
            mooncake_session_id="session",
            prefill_kv_indices=np.array([], dtype=np.int32),
            dst_kv_ptrs=[],
            dst_kv_indices=np.array([], dtype=np.int32),
            dst_tp_rank=0,
            dst_attn_tp_size=1,
            dst_kv_item_len=1,
            executor=None,
            num_dst_target_kv_layers=15,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(
            manager.get_mha_kv_ptrs_with_pp.call_args.kwargs[
                "num_dst_target_kv_layers"
            ],
            15,
        )

    def test_transfer_worker_forwards_target_count(self):
        tree = ast.parse(
            textwrap.dedent(inspect.getsource(MooncakeKVManager.transfer_worker))
        )
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("send_kvcache", "send_kvcache_slice")
            and any(
                keyword.arg == "num_dst_target_kv_layers" for keyword in node.keywords
            )
        ]
        self.assertEqual(len(calls), 2)

    def test_registration_roundtrip_preserves_item_lengths_and_target_count(self):
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.bootstrap_infos = [{}]
        receiver.session_id = "session"
        receiver.kv_mgr = SimpleNamespace(
            kv_args=SimpleNamespace(
                kv_data_ptrs=_dst_ptrs(15, num_draft=1),
                aux_data_ptrs=[],
                state_data_ptrs=[],
                state_item_lens=[],
                state_dim_per_tensor=[],
                state_layer_ids=[],
                kv_layer_ids=[],
                engine_rank=0,
                kv_item_lens=[128] * 15 + [64] * 15 + [128, 64],
                num_target_kv_layers=15,
            ),
            attn_tp_size=1,
            dcp_size=4,
            dcp_rank=2,
            enable_staging=True,
            _staging_ctx=SimpleNamespace(
                allocator=SimpleNamespace(
                    get_base_ptr=lambda: 0x3000,
                    get_total_size=lambda: 4096,
                )
            ),
            kv_buffer_tensors={"slot_layer_ids": [7, 15]},
            local_ip="127.0.0.1",
            rank_port=1234,
        )
        socket = MagicMock()
        receiver._connect_to_bootstrap_server = MagicMock(
            return_value=(socket, threading.Lock())
        )
        self.assertTrue(receiver._register_kv_args())
        frames = socket.send_multipart.call_args.args[0]
        self.assertEqual(len(frames), 21)
        self.assertEqual(
            frames[19], struct.pack("32Q", *receiver.kv_mgr.kv_args.kv_item_lens)
        )
        info = KVArgsRegisterInfo.from_zmq(frames)
        self.assertEqual(info.dst_kv_item_lens, receiver.kv_mgr.kv_args.kv_item_lens)
        self.assertEqual(info.staging.slot_layer_ids, [7, 15])
        self.assertEqual((info.dst_dcp_size, info.dst_dcp_rank), (4, 2))
        self.assertEqual(info.dst_num_target_kv_layers, 15)
        _, _, dst_k, dst_v, _ = CommonKVManager.get_mha_kv_ptrs_with_pp(
            _mha_manager(11),
            _src_ptrs(11, 4),
            info.dst_kv_ptrs,
            num_dst_target_kv_layers=info.dst_num_target_kv_layers,
        )
        self.assertEqual(dst_k, [_DST_K + i for i in range(11, 15)])
        self.assertEqual(dst_v, [_DST_V + i for i in range(11, 15)])


if __name__ == "__main__":
    unittest.main()
