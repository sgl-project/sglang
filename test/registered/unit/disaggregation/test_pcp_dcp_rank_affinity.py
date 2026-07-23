import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    CommonKVSender,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _prefill_info(**overrides):
    values = dict(
        attn_tp_size=1,
        attn_cp_size=2,
        dp_size=1,
        pp_size=1,
        page_size=128,
        kv_cache_dtype="fp8_e4m3",
        follow_bootstrap_room=True,
        enable_dsa_cache_layer_split=False,
        cp_strategy="interleave",
        disaggregation_pcp_dcp_rank_affinity=True,
    )
    values.update(overrides)
    return PrefillServerInfo(**values)


def _decode_manager(dcp_rank=0, affinity=True):
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.attn_tp_size = 2
    mgr.attn_cp_size = 1
    mgr.attn_cp_rank = 0
    mgr.dcp_size = 2
    mgr.dcp_rank = dcp_rank
    mgr.pp_size = 1
    mgr.pp_rank = 0
    mgr.is_mla_backend = True
    mgr.is_hybrid_mla_backend = True
    mgr.enable_all_cp_ranks_for_transfer = True
    mgr.enable_pcp_dcp_rank_affinity = affinity
    mgr.kv_args = SimpleNamespace(
        engine_rank=dcp_rank,
        mla_compression_ratios=[4, 128],
    )
    return mgr


class TestPCPDCPRankAffinity(CustomTestCase):
    def test_disabled_keeps_all_prefill_cp_fanout(self):
        mgr = _decode_manager(affinity=False)
        info = _prefill_info(disaggregation_pcp_dcp_rank_affinity=False)

        mgr._resolve_rank_mapping(info)

        self.assertEqual(info.target_cp_ranks, [0, 1])
        self.assertEqual(info.required_dst_info_num, 2)
        self.assertEqual(info.required_prefill_response_num, 2)

    def test_enabled_maps_each_dcp_rank_to_matching_prefill_cp_rank(self):
        for dcp_rank in (0, 1):
            with self.subTest(dcp_rank=dcp_rank):
                mgr = _decode_manager(dcp_rank=dcp_rank)
                info = _prefill_info()

                mgr._resolve_rank_mapping(info)

                self.assertEqual(info.target_cp_ranks, [dcp_rank])
                self.assertEqual(info.required_dst_info_num, 1)
                self.assertEqual(info.required_prefill_response_num, 1)

    def test_rejects_unsupported_topologies(self):
        cases = [
            (
                "prefill CP size",
                _prefill_info(attn_cp_size=4),
                _decode_manager(),
            ),
            (
                "cp-strategy interleave",
                _prefill_info(cp_strategy="zigzag"),
                _decode_manager(),
            ),
            (
                "DSA cache layer split",
                _prefill_info(enable_dsa_cache_layer_split=True),
                _decode_manager(),
            ),
            (
                "DeepSeek-V4",
                _prefill_info(),
                _decode_manager(),
            ),
        ]
        cases[-1][2].kv_args.mla_compression_ratios = None

        for error, info, mgr in cases:
            with self.subTest(error=error), self.assertRaisesRegex(RuntimeError, error):
                mgr._resolve_rank_mapping(info)

    def test_prefill_requires_unsharded_interleave_replica(self):
        mgr = CommonKVManager.__new__(CommonKVManager)
        mgr.enable_pcp_dcp_rank_affinity = True
        mgr.kv_args = SimpleNamespace(mla_compression_ratios=[4, 128])
        mgr.server_args = SimpleNamespace(
            enable_dsa_cache_layer_split=False,
            cp_strategy="interleave",
        )
        mgr.disaggregation_mode = DisaggregationMode.PREFILL
        mgr.attn_cp_size = 2
        mgr.dcp_size = 2

        with self.assertRaisesRegex(RuntimeError, "prefill dcp_size=1"):
            mgr._check_pcp_dcp_rank_affinity_local_topology()

    def test_affinity_sender_keeps_full_token_mapping(self):
        kv_indices = np.arange(8, dtype=np.int32)
        mgr = SimpleNamespace(
            enable_all_cp_ranks_for_transfer=True,
            enable_pcp_dcp_rank_affinity=True,
            attn_cp_size=2,
            attn_cp_rank=1,
            is_dummy_cp_rank=False,
            server_args=SimpleNamespace(enable_dsa_cache_layer_split=False),
        )
        sender = CommonKVSender.__new__(CommonKVSender)
        sender.kv_mgr = mgr
        sender.curr_idx = 0
        sender.num_kv_indices = len(kv_indices)

        actual, index_slice, is_last, should_skip, position_offset = (
            sender._prepare_send_indices(kv_indices, token_position_offset=17)
        )

        np.testing.assert_array_equal(actual, kv_indices)
        self.assertEqual(index_slice, slice(0, 8))
        self.assertTrue(is_last)
        self.assertFalse(should_skip)
        self.assertEqual(position_offset, 17)

    def test_affinity_c4_c128_slots_cover_full_destination_once(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        src_token_locs = np.arange(1024, 1536, dtype=np.int32)
        dst_token_locs = np.arange(4096, 4608, dtype=np.int32)

        for ratio in (4, 128):
            with self.subTest(ratio=ratio):
                src_slots = mgr._dsv4_bucket_locs_from_token_locs(src_token_locs, ratio)
                dst_slots = mgr._dsv4_bucket_locs_from_token_locs(dst_token_locs, ratio)
                selected_pairs = []
                reconstructed_dst = []
                for dcp_rank in (0, 1):
                    src, dst = mgr._dcp_transfer_index_pair(
                        src_slots,
                        dst_slots,
                        src_dcp_size=1,
                        src_dcp_rank=0,
                        dst_dcp_size=2,
                        dst_dcp_rank=dcp_rank,
                    )
                    dst_global = dst * 2 + dcp_rank
                    selected_pairs.extend(zip(src.tolist(), dst_global.tolist()))
                    reconstructed_dst.extend(dst_global.tolist())

                np.testing.assert_array_equal(
                    np.sort(reconstructed_dst), np.sort(dst_slots)
                )
                self.assertEqual(len(reconstructed_dst), len(set(reconstructed_dst)))
                expected_pairs = set(zip(src_slots.tolist(), dst_slots.tolist()))
                self.assertEqual(set(selected_pairs), expected_pairs)

    def test_affinity_nonzero_prefill_cp_sends_state(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.is_hybrid_mla_backend = True
        mgr.attn_tp_size = 1
        mgr.attn_cp_size = 2
        mgr.attn_cp_rank = 1
        mgr.enable_pcp_dcp_rank_affinity = True
        mgr.server_args = SimpleNamespace(enable_dsa_cache_layer_split=False)

        self.assertEqual(mgr._get_dsa_cache_transfer_skip_flags(None), (False, False))

        mgr.enable_pcp_dcp_rank_affinity = False
        self.assertEqual(mgr._get_dsa_cache_transfer_skip_flags(None), (False, True))


if __name__ == "__main__":
    unittest.main()
