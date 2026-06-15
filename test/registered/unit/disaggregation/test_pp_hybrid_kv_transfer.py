"""Hybrid-linear PP KV transfer regressions."""

import ast
import inspect
import struct
import textwrap
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
)
from sglang.srt.disaggregation.prefill import _transfer_start_layer
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _full_attention_ids(num_layers=60, interval=4):
    return [
        layer_id
        for layer_id in range(num_layers)
        if layer_id % interval == interval - 1
    ]


def _hybrid_pool(start_layer):
    pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
    pool.start_layer = start_layer
    return pool


class TestHybridTransferStartAndPointers(CustomTestCase):
    def test_dense_starts_and_explicit_draft_target_count(self):
        full_ids = _full_attention_ids()
        config = SimpleNamespace(full_attention_layer_ids=full_ids)
        global_starts = [0, 15, 30, 45]
        expected_starts = [0, 3, 7, 11]
        local_counts = [3, 4, 4, 4]
        dst = (
            [1000 + i for i in range(15)]
            + [2000 + i for i in range(15)]
            + [3000]
            + [4000]
        )

        for global_start, expected_start, local_count in zip(
            global_starts, expected_starts, local_counts
        ):
            start = _transfer_start_layer(
                pool=_hybrid_pool(global_start), hf_text_config=config
            )
            self.assertEqual(start, expected_start)
            src = [5000 + i for i in range(local_count)] + [
                6000 + i for i in range(local_count)
            ]
            _, _, dst_k, dst_v, layers = CommonKVManager.get_mha_kv_ptrs_with_pp(
                SimpleNamespace(kv_args=SimpleNamespace(prefill_start_layer=start)),
                src,
                dst,
                num_dst_target_kv_layers=15,
            )
            self.assertEqual(layers, local_count)
            self.assertEqual(
                dst_k, [1000 + i for i in range(start, start + local_count)]
            )
            self.assertEqual(
                dst_v, [2000 + i for i in range(start, start + local_count)]
            )
            if expected_start == 11:
                self.assertNotIn(3000, dst_v)


class _RecordingKVManager:
    get_mha_kv_ptrs_with_pp = CommonKVManager.get_mha_kv_ptrs_with_pp

    def __init__(self, prefill_start_layer=0):
        self.is_mla_backend = False
        self.is_hybrid_mla_backend = False
        self.enable_custom_mem_pool = False
        self.pp_size = 2
        self.kv_args = SimpleNamespace(prefill_start_layer=prefill_start_layer)
        self.blocks = []

    def _transfer_data(self, _session, transfer_blocks):
        self.blocks.extend(transfer_blocks)
        return 0


class TestHybridLayerIdPairing(CustomTestCase):
    def test_generic_path_pairs_non_mla_hybrid_entries(self):
        model_ids = _full_attention_ids()
        stage_ids = model_ids[7:]
        num_stage = len(stage_ids)
        src_ptrs = [1000 + i for i in range(2 * num_stage)]
        dst_ptrs = [2000 + i for i in range(2 * len(model_ids))]
        item_lens = [10 + i for i in range(2 * num_stage)]
        manager = _RecordingKVManager()

        rc = MooncakeKVManager._send_kvcache_generic(
            manager,
            mooncake_session_id="session",
            src_data_ptrs=src_ptrs,
            dst_data_ptrs=dst_ptrs,
            item_lens=item_lens,
            prefill_data_indices=np.array([0], dtype=np.int32),
            dst_data_indices=np.array([0], dtype=np.int32),
            executor=None,
            src_layer_ids=stage_ids * 2,
            dst_layer_ids=model_ids * 2,
        )
        self.assertEqual(rc, 0)
        expected = [
            (src_ptrs[i], dst_ptrs[7 + i], item_lens[i]) for i in range(num_stage)
        ] + [
            (
                src_ptrs[num_stage + i],
                dst_ptrs[len(model_ids) + 7 + i],
                item_lens[num_stage + i],
            )
            for i in range(num_stage)
        ]
        self.assertEqual(manager.blocks, expected)

    def test_one_sided_source_ids_use_mha_target_count_with_draft(self):
        model_ids = _full_attention_ids()
        stage_ids = model_ids[7:]
        num_stage = len(stage_ids)
        src_ptrs = [1000 + i for i in range(2 * num_stage)]
        dst_ptrs = (
            [2000 + i for i in range(15)]
            + [3000 + i for i in range(15)]
            + [4000]
            + [5000]
        )
        item_lens = [10 + i for i in range(2 * num_stage)]
        manager = _RecordingKVManager(prefill_start_layer=7)

        rc = MooncakeKVManager._send_kvcache_generic(
            manager,
            mooncake_session_id="session",
            src_data_ptrs=src_ptrs,
            dst_data_ptrs=dst_ptrs,
            item_lens=item_lens,
            prefill_data_indices=np.array([0], dtype=np.int32),
            dst_data_indices=np.array([0], dtype=np.int32),
            executor=None,
            src_layer_ids=stage_ids * 2,
            dst_layer_ids=[],
            num_dst_target_kv_layers=15,
        )
        self.assertEqual(rc, 0)
        expected = [
            (src_ptrs[i], dst_ptrs[7 + i], item_lens[i]) for i in range(num_stage)
        ] + [
            (
                src_ptrs[num_stage + i],
                dst_ptrs[15 + 7 + i],
                item_lens[num_stage + i],
            )
            for i in range(num_stage)
        ]
        self.assertEqual(manager.blocks, expected)


def _registration_frames():
    return [
        b"1",
        b"127.0.0.1",
        b"1234",
        b"session",
        struct.pack("Q", 0x1000),
        b"",
        b"",
        b"0",
        b"1",
        b"16",
        b"",
        b"",
        b"",
        b"",
        b"",
        b"",
        b"1",
        b"0",
    ]


class TestMooncakeRegistrationCompatibility(CustomTestCase):
    def test_old_registration_has_no_target_count(self):
        info = KVArgsRegisterInfo.from_zmq(_registration_frames())
        self.assertEqual(info.dst_num_target_kv_layers, -1)

    def test_new_registration_reads_target_count(self):
        info = KVArgsRegisterInfo.from_zmq(_registration_frames() + [b"15"])
        self.assertEqual(info.dst_num_target_kv_layers, 15)


def _calls_with_keyword(method, callee, keyword):
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == callee
        and any(item.arg == keyword for item in node.keywords)
    ]


class TestMooncakeTargetCountForwarding(CustomTestCase):
    def test_all_mha_edges_forward_target_count(self):
        keyword = "num_dst_target_kv_layers"
        edges = [
            (MooncakeKVManager._send_kvcache_generic, "get_mha_kv_ptrs_with_pp"),
            (MooncakeKVManager.send_kvcache, "_send_kvcache_generic"),
            (MooncakeKVManager.send_kvcache_dcp, "get_mha_kv_ptrs_with_pp"),
            (MooncakeKVManager.send_kvcache_slice, "get_mha_kv_ptrs_with_pp"),
        ]
        for method, callee in edges:
            with self.subTest(method=method.__name__, callee=callee):
                self.assertTrue(_calls_with_keyword(method, callee, keyword))

        transfer_calls = sum(
            len(_calls_with_keyword(MooncakeKVManager.transfer_worker, callee, keyword))
            for callee in ("send_kvcache", "send_kvcache_dcp", "send_kvcache_slice")
        )
        self.assertEqual(transfer_calls, 3)


if __name__ == "__main__":
    unittest.main()
