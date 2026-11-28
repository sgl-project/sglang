"""CPU unit tests for the layer-split Ascend PD-transfer pointer remap.

A layer-split prefill reports owned-only buffer entries while a non-split
decode peer registers every layer; ``AscendKVManager.get_mla_kv_ptrs_with_pp``
must slice the destination pointers per buffer group onto the
``[prefill_start_layer, prefill_start_layer + src_layers)`` window so both
sides agree on which destination layer each source entry feeds.

Run directly:
    python -m pytest test/registered/unit/npu/disaggregation/test_npu_dsa_layer_split_transfer.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.mooncake_utils import attach_mooncake_transport, engine_writes
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=5, suite="extra-a-test-npu")


class _FakePPGroup:
    is_last_rank = True


class _FakeParallel:
    pp_group = _FakePPGroup()


def _make_conn(
    *, prefill_start_layer=0, kv_buf_groups=1, hidden_kv_layers=0, draft_kv_layers=0
):
    conn = AscendKVManager.__new__(AscendKVManager)
    conn.kv_args = SimpleNamespace(
        prefill_start_layer=prefill_start_layer,
        kv_buf_groups=kv_buf_groups,
        hidden_kv_layers=hidden_kv_layers,
        draft_kv_layers=draft_kv_layers,
    )
    return conn


class TestGetMlaKvPtrsWithPP(CustomTestCase):
    def setUp(self):
        patcher = patch(
            "sglang.srt.disaggregation.ascend.conn.get_parallel",
            lambda: _FakeParallel(),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_equal_counts_pass_dst_through(self):
        conn = _make_conn()
        src, dst = list(range(10)), list(range(100, 110))
        out_src, out_dst, layers = conn.get_mla_kv_ptrs_with_pp(src, dst)
        self.assertEqual(out_src, src)
        self.assertEqual(out_dst, dst)
        self.assertEqual(layers, 10)

    def test_owned_only_src_maps_onto_dst_window(self):
        # Layer-split prefill owns 5 of the decode peer's 10 layers,
        # starting at layer 3 (e.g. a PP stage-two shard).
        conn = _make_conn(prefill_start_layer=3)
        src = [f"s{i}" for i in range(5)]
        dst = [f"d{i}" for i in range(10)]
        _, out_dst, layers = conn.get_mla_kv_ptrs_with_pp(src, dst)
        self.assertEqual(layers, 5)
        self.assertEqual(out_dst, [f"d{i}" for i in range(3, 8)])

    def test_owned_only_src_at_layer_zero_maps_prefix_window(self):
        conn = _make_conn(prefill_start_layer=0)
        src = [f"s{i}" for i in range(5)]
        dst = [f"d{i}" for i in range(10)]
        _, out_dst, _ = conn.get_mla_kv_ptrs_with_pp(src, dst)
        self.assertEqual(out_dst, [f"d{i}" for i in range(5)])

    def test_multi_group_dst_slices_every_group(self):
        # NPU MLA layout is group-major: [group0 layers, group1 layers, ...].
        # Owned-only entries exist per group, so each group's destination
        # block must be sliced onto the same layer window.
        conn = _make_conn(prefill_start_layer=2, kv_buf_groups=2, hidden_kv_layers=10)
        src = [f"s{i}" for i in range(5 * 2)]
        dst = [f"d{i}" for i in range(2 * 10)]
        _, out_dst, layers = conn.get_mla_kv_ptrs_with_pp(src, dst)
        self.assertEqual(layers, 10)
        # layers_current_pp_stage counts raw src entries.
        self.assertEqual(len(out_dst), 5 * 2)
        self.assertEqual(
            out_dst,
            [f"d{i}" for i in range(2, 7)] + [f"d{i}" for i in range(12, 17)],
        )

    def test_last_rank_appends_draft_group_windows(self):
        conn = _make_conn(
            prefill_start_layer=1,
            kv_buf_groups=2,
            hidden_kv_layers=5,
            draft_kv_layers=2,
        )
        # src: 4 target layers + 2 draft layers, x2 groups = 12 entries.
        src = [f"s{i}" for i in range(12)]
        # dst: 2 groups x (5 target + 2 draft) = 14 entries.
        dst = [f"d{i}" for i in range(14)]
        _, out_dst, layers = conn.get_mla_kv_ptrs_with_pp(src, dst)
        self.assertEqual(layers, 12)
        self.assertEqual(len(out_dst), 12)
        # Target windows: [1, 5) per group; draft windows: full draft range.
        self.assertEqual(
            out_dst,
            [f"d{i}" for i in range(1, 5)]
            + [f"d{i}" for i in range(6, 10)]
            + [f"d{i}" for i in range(10, 12)]
            + [f"d{i}" for i in range(12, 14)],
        )


class TestAscendHybridLayerSplitSend(CustomTestCase):
    """Exercise the send entrypoint and real pairing, mocking only transport."""

    def _send(self, *, split=True, hybrid=True, pp_size=1, owns_layer=True):
        conn = object.__new__(AscendKVManager)
        conn.is_mla_backend = True
        conn.is_hybrid_mla_backend = hybrid
        conn.pp_size = pp_size
        conn.enable_custom_mem_pool = False
        conn.max_transfer_batch_indices = 0
        conn.kv_args = SimpleNamespace(
            kv_data_ptrs=[1000, 2000, 3000] if owns_layer else [],
            kv_item_lens=[16, 8, 4] if owns_layer else [],
            kv_layer_ids=[11, 11, 11] if owns_layer else [],
            kv_buf_groups=3,
            hidden_kv_layers=5,
            draft_kv_layers=0,
            prefill_start_layer=0,
        )
        transfer = Mock(return_value=0)
        dst_ptrs = [
            10000 * (group + 1) + 1000 * i for group in range(3) for i in range(5)
        ]
        attach_mooncake_transport(
            conn,
            engine_writes(transfer),
            kv=conn.kv_args.kv_data_ptrs,
            peer_kv=dst_ptrs,
            sessions=("decode",),
        )
        with (
            patch(
                "sglang.srt.disaggregation.ascend.conn.get_parallel",
                return_value=SimpleNamespace(
                    enable_dsa_cache_layer_split=split,
                    pp_group=SimpleNamespace(is_last_rank=True),
                ),
            ),
            patch(
                "sglang.srt.disaggregation.mooncake.conn.get_memory",
                return_value=SimpleNamespace(enable_unified_memory=False),
            ),
        ):
            status = conn.send_kvcache(
                mooncake_session_id="decode",
                prefill_kv_indices=np.array([1, 2], dtype=np.int32),
                dst_kv_ptrs=dst_ptrs,
                dst_kv_indices=np.array([4, 5], dtype=np.int32),
                executor=None,
                dst_layer_ids=[3, 7, 11, 15, 19] * 3,
            )
        self.assertEqual(status, 0)
        return transfer

    def test_pp1_split_pairs_each_buffer_group_by_model_layer(self):
        transfer = self._send()
        # Compact child layer 2 is model layer 11, in each K/V/Indexer group.
        transfer.assert_called_once_with(
            "decode", [(1016, 12064, 32), (2008, 22032, 16), (3004, 32016, 8)]
        )

    def test_hybrid_pp_keeps_model_layer_pairing_without_split(self):
        self._send(split=False, pp_size=2).assert_called_once_with(
            "decode", [(1016, 12064, 32), (2008, 22032, 16), (3004, 32016, 8)]
        )

    def test_other_layouts_keep_legacy_destination_windows(self):
        for split, hybrid in ((False, True), (True, False)):
            with self.subTest(split=split, hybrid=hybrid):
                self._send(split=split, hybrid=hybrid).assert_called_once_with(
                    "decode", [(1016, 10064, 32), (2008, 20032, 16), (3004, 30016, 8)]
                )

    def test_rank_without_owned_layers_transfers_no_bytes(self):
        transfer = self._send(owns_layer=False)
        for args in transfer.call_args_list:
            self.assertEqual(args.args, ("decode", []))


if __name__ == "__main__":
    unittest.main()
