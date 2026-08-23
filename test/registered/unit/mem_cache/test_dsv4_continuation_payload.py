import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.deepseek_v4_continuation import (
    DeepSeekV4ContinuationPool,
    _TensorGroup,
    dsv4_continuation_payload_bytes,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _deepseek_v4_continuation_num_host_slots,
)
from sglang.srt.model_executor.pool_configurator import (
    _dsv4_packed_draft_layer_num,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDSV4ContinuationPayload(unittest.TestCase):
    def test_formal_geometry_uses_three_packed_draft_layers(self):
        spec_algorithm = MagicMock()
        spec_algorithm.is_dspark.return_value = True
        kvc = SimpleNamespace(
            spec_algorithm=spec_algorithm,
            spec_aux_config=SimpleNamespace(dflash_target_layer_ids=[58, 59, 60]),
        )

        self.assertEqual(_dsv4_packed_draft_layer_num(kvc), 3)

    def test_host_capacity_tracks_full_host_pages(self):
        self.assertEqual(
            _deepseek_v4_continuation_num_host_slots(
                num_host_pages=256,
                continuation_num_slots=8,
                hicache_ratio=2,
            ),
            256,
        )

    def test_payload_bytes_use_four_c4_rows_and_no_c128_state(self):
        payload = dsv4_continuation_payload_bytes(
            target_layer_num=61,
            draft_layer_num=3,
            c4_layer_num=30,
            attention_head_dim=512,
            indexer_head_dim=128,
            c4_state_element_size=4,
        )

        ring = 64 * 128 * 512 * torch.bfloat16.itemsize
        c4_attention = 30 * 4 * 4 * 512 * 4
        c4_indexer = 30 * 4 * 4 * 128 * 4
        self.assertEqual(payload, ring + c4_attention + c4_indexer)
        self.assertEqual(payload, 9_617_408)

    def test_batched_group_copy_remaps_each_slot_independently(self):
        tensor = torch.arange(24, dtype=torch.float32).reshape(12, 2)
        slot_view = torch.zeros((4, 1, 2, 2), dtype=torch.float32)
        group = _TensorGroup(
            tensors=(tensor,),
            rows_per_slot=2,
            slot_view=slot_view,
        )
        slots = torch.tensor([1, 3], dtype=torch.int64)
        source_rows = torch.tensor([[0, 1], [8, 9]], dtype=torch.int64)

        DeepSeekV4ContinuationPool._copy_into_group_batch(group, slots, source_rows)
        tensor.zero_()
        destination_rows = torch.tensor([[4, 5], [10, 11]], dtype=torch.int64)
        DeepSeekV4ContinuationPool._copy_from_group_batch(
            group, slots, destination_rows
        )

        self.assertTrue(
            torch.equal(tensor[4:6], torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
        )
        self.assertTrue(
            torch.equal(tensor[10:12], torch.tensor([[16.0, 17.0], [18.0, 19.0]]))
        )
        self.assertTrue(torch.equal(tensor[0:4], torch.zeros((4, 2))))

    def test_restore_batch_uses_destination_mapping_and_clears_each_req(self):
        pool = object.__new__(DeepSeekV4ContinuationPool)
        pool.device = torch.device("cpu")
        pool.logical_page_size = 8
        pool.target_pool = SimpleNamespace(
            unified_swa_window=2,
            unified_swa_ring_size=16,
        )
        pool.draft_pools = ()
        ring_tensor = torch.zeros((64, 1), dtype=torch.float32)
        ring_slot_view = torch.zeros((4, 1, 2, 1), dtype=torch.float32)
        ring_slot_view[1, 0, :, 0] = torch.tensor([11.0, 12.0])
        ring_slot_view[2, 0, :, 0] = torch.tensor([21.0, 22.0])
        pool.ring_groups = (_TensorGroup((ring_tensor,), 2, ring_slot_view),)

        c4_attention = torch.zeros((32, 1), dtype=torch.float32)
        c4_indexer = torch.zeros((32, 1), dtype=torch.float32)
        c4_attention_slots = torch.zeros((4, 1, 4, 1), dtype=torch.float32)
        c4_indexer_slots = torch.zeros((4, 1, 4, 1), dtype=torch.float32)
        c4_attention_slots[1, 0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        c4_attention_slots[2, 0, :, 0] = torch.tensor([5.0, 6.0, 7.0, 8.0])
        c4_indexer_slots[1, 0, :, 0] = torch.tensor([9.0, 10.0, 11.0, 12.0])
        c4_indexer_slots[2, 0, :, 0] = torch.tensor([13.0, 14.0, 15.0, 16.0])
        pool.c4_attention_group = _TensorGroup((c4_attention,), 4, c4_attention_slots)
        pool.c4_indexer_group = _TensorGroup((c4_indexer,), 4, c4_indexer_slots)
        pool.wait_ready_indices = MagicMock()
        pool._ring_rows_batch = MagicMock(
            return_value=torch.tensor([[2, 3], [18, 19]], dtype=torch.int64)
        )
        pool._c4_state_rows_batch = MagicMock(
            return_value=torch.tensor(
                [[4, 5, 6, 7], [20, 21, 22, 23]], dtype=torch.int64
            )
        )
        pool._clear_c128_state = MagicMock()
        pool.record_ready_indices = MagicMock()

        slots = torch.tensor([1, 2], dtype=torch.int64)
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        pool.restore_batch(
            slots=slots,
            req_pool_indices=req_pool_indices,
            req_pool_indices_cpu=[7, 9],
            endpoints=[8, 16],
        )

        self.assertEqual(ring_tensor[2:4, 0].tolist(), [11.0, 12.0])
        self.assertEqual(ring_tensor[18:20, 0].tolist(), [21.0, 22.0])
        self.assertEqual(c4_attention[4:8, 0].tolist(), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(c4_attention[20:24, 0].tolist(), [5.0, 6.0, 7.0, 8.0])
        self.assertEqual(c4_indexer[4:8, 0].tolist(), [9.0, 10.0, 11.0, 12.0])
        self.assertEqual(c4_indexer[20:24, 0].tolist(), [13.0, 14.0, 15.0, 16.0])
        self.assertEqual(
            pool._clear_c128_state.call_args_list,
            [unittest.mock.call(7), unittest.mock.call(9)],
        )
        pool.record_ready_indices.assert_called_once_with(slots)


if __name__ == "__main__":
    unittest.main()
