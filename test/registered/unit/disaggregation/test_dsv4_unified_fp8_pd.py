import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.vpp import (
    check_vpp_transfer_layers,
    vpp_transfer_layers,
)
from sglang.srt.distributed.vpp_layout import VPPLayout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")


def _pp_mgr(start, end, ratios):
    mgr = object.__new__(CommonKVManager)
    mgr.kv_args = SimpleNamespace(
        prefill_start_layer=start,
        prefill_end_layer=end,
        mla_compression_ratios=ratios,
    )
    return mgr


class TestDSV4UnifiedFp8PpSlice(CustomTestCase):
    RATIOS = [4, 4, 128, 4]
    # kv_data groups: 2*c4 + c128 bf16, 3*c4 + 2*c128 under the fp8 two-pool
    BF16_KV_LEN = 7
    FP8_KV_LEN = 11

    def test_contiguous_v41_pp_maps_shared_sources_to_pp1(self):
        ownership = [list(range(20)), list(range(20, 40))]
        self.assertEqual(
            vpp_transfer_layers(
                SimpleNamespace(vpp_layout=None, pp_transfer_layers=ownership)
            ),
            ownership,
        )
        ratios = [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
        full = [20, 20, 2, 8, 14, 2, 8, 14]
        strides = [149760, 65536] + [74880] * 3 + [32768] * 3
        for rank in range(2):
            mgr = _pp_mgr(rank * 20, (rank + 1) * 20, ratios)
            mgr.kv_args.vpp_layers = ownership
            mgr.kv_args.pp_rank = rank
            mgr.kv_args.mla_kv_source_layers = [2, 8, 14, 20]
            expected = [i for i, layer in enumerate(full) if layer in ownership[rank]]
            local_strides = [strides[i] for i in expected]
            _, selected, count = mgr.get_mla_kv_ptrs_with_pp(local_strides, strides)
            self.assertEqual(selected, local_strides)
            self.assertEqual(count, len(expected))
            _, selected, _ = mgr.get_mla_kv_ptrs_with_pp(
                [100] * len(expected), list(range(len(full)))
            )
            self.assertEqual(selected, expected)

    def test_v41_producer_groups_tile_decode_without_exporting_replicas(self):
        ratios = [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
        layout = VPPLayout(40, 2, virtual_stages=2, partition=(9, 12, 8, 11))
        ownership = vpp_transfer_layers(SimpleNamespace(vpp_layout=layout))
        for state, groups in (
            (None, [[20], [20], [2, 8, 14], [2, 8, 14]]),
            (StateType.DSV4_REQUEST_STATE, [[2, 8, 14]]),
            (StateType.SWA, [list(range(40))]),
        ):
            full = [layer for group in groups for layer in group]
            covered = []
            for rank in range(2):
                mgr = _pp_mgr(0, 40, ratios)
                mgr.kv_args.vpp_layers = ownership
                mgr.kv_args.pp_rank = rank
                mgr.kv_args.mla_kv_source_layers = [2, 8, 14, 20]
                expected = [
                    i for i, layer in enumerate(full) if layer in ownership[rank]
                ]
                _, selected, count = mgr.get_mla_kv_ptrs_with_pp(
                    [100 + rank] * len(expected), list(range(len(full))), state
                )
                self.assertEqual(selected, expected)
                self.assertEqual(count, len(expected))
                covered.extend(selected)
            self.assertEqual(sorted(covered), list(range(len(full))))

    def _slice(self, dst, start, end, state_type=None, src=None):
        mgr = _pp_mgr(start, end, self.RATIOS)
        if src is None:
            src = list(range(100, 100 + 8))
        return mgr._mla_slice_ptrs_for_pp(src, dst, self.RATIOS, state_type)

    def test_mixed_fp8_and_bf16_peers_are_rejected(self):
        cases = (
            (None, self.BF16_KV_LEN, self.FP8_KV_LEN),
            (None, self.FP8_KV_LEN, self.BF16_KV_LEN),
            (StateType.SWA_RING, len(self.RATIOS), 2 * len(self.RATIOS)),
            (StateType.SWA_RING, 2 * len(self.RATIOS), len(self.RATIOS)),
        )
        for state_type, src_len, dst_len in cases:
            with self.subTest(state_type=state_type, src_len=src_len):
                with self.assertRaisesRegex(ValueError, "SGLANG_DSV4_UNIFIED_KV_FP8"):
                    self._slice(
                        list(range(dst_len)),
                        1,
                        3,
                        state_type,
                        src=list(range(100, 100 + src_len)),
                    )

    def test_vpp_groups_tile_full_decode_without_touching_other_ranks(self):
        """A bounding interval includes foreign layers between virtual stages."""
        ratios = [4, 128, 0, 4, 128, 4, 0, 128, 4, 128, 4, 0]
        layout = VPPLayout(12, 2, virtual_stages=3, partition=(1, 2, 3, 1, 2, 3))
        ownership = vpp_transfer_layers(SimpleNamespace(vpp_layout=layout))
        all_layers = list(range(12))
        c4 = [0, 3, 5, 8, 10]
        c128 = [1, 4, 7, 9]
        for state, groups in (
            (None, [c4, c4, c128]),
            (None, [c4, c4, c4, c128, c128]),
            (StateType.SWA, [all_layers, c4, c4]),
            (StateType.SWA_RING, [all_layers]),
            (StateType.SWA_RING, [all_layers, all_layers]),
            (StateType.DSV4_REQUEST_STATE, [c128]),
        ):
            with self.subTest(state=state, groups=len(groups)):
                full = [layer for group in groups for layer in group]
                covered = []
                for rank in range(2):
                    mgr = _pp_mgr(0, 12, ratios)
                    mgr.kv_args.vpp_layers = ownership
                    mgr.kv_args.pp_rank = rank
                    count = sum(layer in ownership[rank] for layer in full)
                    src = [1000 + rank] * count
                    actual_src, selected, n = mgr.get_mla_kv_ptrs_with_pp(
                        src, list(range(len(full))), state
                    )
                    self.assertEqual(actual_src, src)
                    self.assertEqual(n, count)
                    self.assertTrue(
                        all(full[index] in ownership[rank] for index in selected)
                    )
                    covered.extend(selected)
                self.assertEqual(sorted(covered), list(range(len(full))))

    def test_vpp_prefill_requires_pp1_decode_with_matching_tp(self):
        common = dict(prefill_tp=2, decode_tp=2, decode_pp_size=2)
        with self.assertRaisesRegex(RuntimeError, "PP1 decode peer"):
            check_vpp_transfer_layers([[0, 3], [1, 2]], **common)
        check_vpp_transfer_layers(None, **common)
        check_vpp_transfer_layers(
            [[0, 3], [1, 2]],
            prefill_tp=2,
            decode_tp=2,
            decode_pp_size=1,
        )
        with self.assertRaisesRegex(RuntimeError, "same TP"):
            check_vpp_transfer_layers(
                [[0, 3], [1, 2]],
                prefill_tp=2,
                decode_tp=1,
                decode_pp_size=1,
            )

    def test_vpp_target_buffers_exclude_extra_mtp_ratio_entries(self):
        # The real Flash 0731 configuration has 43 target layers and 3 MTP
        # entries. SWA has 85 target regions, split into 39 and 46 locally.
        ratios = [0, 0] + [4, 128] * 20 + [4] + [0, 0, 0]
        layout = VPPLayout(43, 2, virtual_stages=2)
        ownership = vpp_transfer_layers(SimpleNamespace(vpp_layout=layout))
        layers = list(range(43))
        c4 = [layer for layer in layers if ratios[layer] == 4]
        c128 = [layer for layer in layers if ratios[layer] == 128]
        for state, groups in (
            (None, [c4, c4, c128]),
            (None, [c4, c4, c4, c128, c128]),
            (StateType.SWA, [layers, c4, c4]),
            (StateType.SWA_RING, [layers]),
            (StateType.SWA_RING, [layers, layers]),
            (StateType.DSV4_REQUEST_STATE, [c128]),
        ):
            with self.subTest(state=state, groups=len(groups)):
                full = [layer for group in groups for layer in group]
                selected_by_rank = []
                for rank in range(2):
                    mgr = _pp_mgr(0, 43, ratios)
                    mgr.kv_args.vpp_layers = ownership
                    mgr.kv_args.pp_rank = rank
                    expected = [
                        index
                        for index, layer in enumerate(full)
                        if layer in ownership[rank]
                    ]
                    src = [1000 + rank] * len(expected)
                    actual_src, selected, count = mgr.get_mla_kv_ptrs_with_pp(
                        src, list(range(len(full))), state
                    )
                    self.assertEqual(actual_src, src)
                    self.assertEqual(selected, expected)
                    self.assertEqual(count, len(expected))
                    if state == StateType.SWA:
                        self.assertEqual(count, (39, 46)[rank])
                    selected_by_rank.extend(selected)
                self.assertEqual(sorted(selected_by_rank), list(range(len(full))))

    def test_bootstrap_rejects_pipelined_decode_for_vpp_prefill(self):
        from unittest.mock import patch

        manager = object.__new__(CommonKVManager)
        manager.prefill_info_table = {}
        manager.attn_tp_size = manager.pp_size = 2
        payload = dict(
            attn_tp_size=2,
            attn_cp_size=1,
            dp_size=1,
            pp_size=2,
            page_size=256,
            kv_cache_dtype="fp8_e4m3",
            follow_bootstrap_room=True,
            vpp_layers=[[0, 2], [1, 3]],
        )
        response = SimpleNamespace(status_code=200, json=lambda: payload)
        with patch(
            "sglang.srt.disaggregation.common.conn.requests.get",
            return_value=response,
        ):
            with self.assertRaisesRegex(RuntimeError, "PP1 decode peer"):
                manager.try_ensure_parallel_info("127.0.0.1:8765")

    def test_vpp_fp8_bf16_mismatch_is_rejected_before_slicing(self):
        mgr = _pp_mgr(0, 4, self.RATIOS)
        mgr.kv_args.vpp_layers = [[0, 2], [1, 3]]
        mgr.kv_args.pp_rank = 0
        with self.assertRaisesRegex(ValueError, "buffer layout differs"):
            mgr.get_mla_kv_ptrs_with_pp([100] * 5, list(range(self.BF16_KV_LEN)))


if __name__ == "__main__":
    unittest.main()
