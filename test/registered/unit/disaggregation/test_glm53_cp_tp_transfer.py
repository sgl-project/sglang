"""GLM CP/TP sharing must transfer each TP-owned state to its matching rank."""

import ctypes
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    CommonKVSender,
    PrefillServerInfo,
)
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGlm53CpTpTransfer(CustomTestCase):
    def setUp(self):
        publish(ServerArgs(model_path="dummy"), role="test")

    def tearDown(self):
        reset_context()

    def test_matching_physical_ranks_and_completion_counts(self):
        for rank in range(4):
            manager = CommonKVManager.__new__(CommonKVManager)
            manager.attn_tp_size = 4
            manager.attn_cp_size = 1
            manager.attn_cp_rank = 0
            manager.attn_dp_size = 1
            manager.pp_size = 1
            manager.pp_rank = 0
            manager.kv_args = SimpleNamespace(engine_rank=rank)
            manager.is_mla_backend = False
            manager.is_hybrid_mla_backend = True
            manager.enable_all_cp_ranks_for_transfer = True
            info = PrefillServerInfo(
                attn_tp_size=1,
                attn_cp_size=4,
                dp_size=1,
                pp_size=1,
                page_size=64,
                kv_cache_dtype="fp8_e4m3",
                follow_bootstrap_room=True,
                enable_cp_tp_group_sharing=True,
            )
            manager._resolve_rank_mapping(info)
            self.assertEqual(info.target_tp_ranks, [0])
            self.assertEqual(info.target_cp_ranks, [rank])
            self.assertEqual(info.required_dst_info_num, 1)
            self.assertEqual(info.required_prefill_response_num, 1)

            info.attn_cp_size = 8
            with self.assertRaisesRegex(ValueError, "matching prefill CP"):
                manager._resolve_rank_mapping(info)

    def test_every_rank_sends_all_pages_and_its_state(self):
        for rank in range(4):
            manager = CommonKVManager.__new__(CommonKVManager)
            manager.enable_cp_tp_group_sharing = True
            manager.enable_all_cp_ranks_for_transfer = True
            manager.is_dummy_cp_rank = False
            manager.attn_cp_size = 4
            manager.attn_cp_rank = rank
            self.assertFalse(manager._should_skip_cp_replicated_state_transfer())
            sender = SimpleNamespace(kv_mgr=manager, curr_idx=0, num_kv_indices=7)
            for pages, expected_slice, last in (
                ([7, 8, 9], slice(0, 3), False),
                ([10, 11, 12, 13], slice(3, 7), True),
            ):
                result, index_slice, is_last, skip = (
                    CommonKVSender._prepare_send_indices(
                        sender, np.array(pages, dtype=np.int32)
                    )
                )
                np.testing.assert_array_equal(result, pages)
                self.assertEqual(index_slice, expected_slice)
                self.assertEqual(is_last, last)
                self.assertFalse(skip)

    def test_tp_owned_conv_and_recurrent_state_bytes(self):
        # Three slots; only source slot 1 -> destination slot 2 may change.
        for rank in range(4):
            src = [
                np.full((3, n), 10 * rank + i + 1, np.uint8)
                for i, n in enumerate((12, 32))
            ]
            dst = [np.full_like(value, 255) for value in src]
            manager = MooncakeKVManager.__new__(MooncakeKVManager)
            manager.attn_tp_size = 1
            manager.attn_cp_size = 4
            manager.attn_cp_rank = rank
            manager.enable_cp_tp_group_sharing = True
            manager.pp_size = 1
            manager.kv_args = SimpleNamespace(
                engine_rank=rank,
                state_types=[StateType.MAMBA],
                state_data_ptrs=[[a.ctypes.data for a in src]],
                state_item_lens=[[12, 32]],
                state_dim_per_tensor=[[12, 32]],
                state_layer_ids=[[0, 0]],
            )

            def transfer(_session, blocks):
                for source, destination, size in blocks:
                    ctypes.memmove(destination, source, size)
                return 0

            manager._transfer_data = transfer
            req = SimpleNamespace(mooncake_session_id="test", dst_state_indices=[[2]])
            registration = SimpleNamespace(
                dst_tp_rank=rank,
                dst_attn_tp_size=4,
                dst_state_data_ptrs=[[a.ctypes.data for a in dst]],
                dst_state_item_lens=[[12, 32]],
                dst_state_dim_per_tensor=[[12, 32]],
                dst_state_layer_ids=[[0, 0]],
            )
            self.assertEqual(
                manager.maybe_send_extra(req, [[1]], None, registration), 0
            )
            for source, destination in zip(src, dst):
                np.testing.assert_array_equal(destination[2], source[1])
                self.assertTrue(np.all(destination[:2] == 255))

    def test_replicated_kv_kpool_and_tp_draft_pages_arrive_once(self):
        for rank in range(4):
            # Two replicated components and one rank-specific external draft.
            src = [np.arange(6 * n, dtype=np.uint8).reshape(6, n) for n in (4, 8, 12)]
            src[2] += rank * 30
            dst = [np.full_like(a, 255) for a in src]
            manager = MooncakeKVManager.__new__(MooncakeKVManager)
            manager.is_mla_backend = True
            manager.is_hybrid_mla_backend = True
            manager.enable_custom_mem_pool = False
            manager.max_transfer_batch_indices = 0
            manager.pp_size = 1
            writes = []

            def transfer(_session, blocks):
                for source, destination, size in blocks:
                    writes.extend(range(destination, destination + size))
                    ctypes.memmove(destination, source, size)
                return 0

            manager._transfer_data = transfer
            self.assertEqual(
                manager._send_kvcache_generic(
                    mooncake_session_id="test",
                    src_data_ptrs=[a.ctypes.data for a in src],
                    dst_data_ptrs=[a.ctypes.data for a in dst],
                    item_lens=[4, 8, 12],
                    prefill_data_indices=np.array([1, 2, 4], dtype=np.int32),
                    dst_data_indices=np.array([0, 3, 4], dtype=np.int32),
                    executor=None,
                    src_layer_ids=[0, 1, 100],
                    dst_layer_ids=[0, 1, 100],
                ),
                0,
            )
            self.assertEqual(len(writes), len(set(writes)))
            for source, destination in zip(src, dst):
                np.testing.assert_array_equal(destination[[0, 3, 4]], source[[1, 2, 4]])
                self.assertTrue(np.all(destination[[1, 2, 5]] == 255))


if __name__ == "__main__":
    unittest.main()
