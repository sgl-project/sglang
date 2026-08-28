"""CPU tests for 4D MoE LoRA page scatter.

These tests exercise rank-component placement, non-contiguous physical pages,
tail-page zeroing, partial page-in, and the single scaling point used by the
classic fused MoE LoRA kernel.
"""

import unittest

import torch

from sglang.srt.lora.paged_mem_pool import LoRAPagePool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_pool() -> LoRAPagePool:
    pool = LoRAPagePool.__new__(LoRAPagePool)
    pool.PAGE_RANK_SIZE = 4
    pool.base_model = None
    pool.experts_shared_outer_loras = False
    pool.moe_use_local_expert_ids = False
    pool.num_experts = 2
    pool.moe_ep_rank = 0
    pool.page_access_times = [0] * 5
    pool._access_counter = 0
    pool.A_pages = {
        "gate_up_proj_moe": [torch.ones(5, 2, 8, 3)],
        "down_proj_moe": [torch.ones(5, 2, 4, 6)],
    }
    pool.B_pages = {
        "gate_up_proj_moe": [torch.ones(5, 2, 6, 4)],
        "down_proj_moe": [torch.ones(5, 2, 3, 4)],
    }
    return pool


class TestPagedMoEScatter(CustomTestCase):
    def test_local_expert_mapping_filters_non_local_weights(self):
        pool = _make_pool()
        pool.moe_use_local_expert_ids = True
        pool.moe_ep_rank = 1

        weights = {
            0: torch.full((3, 4), -1.0),
            2: torch.full((3, 4), 2.0),
            3: torch.full((3, 4), 3.0),
        }
        local = list(pool._iter_local_expert_weights(weights))
        self.assertEqual([expert_id for expert_id, _ in local], [0, 1])
        torch.testing.assert_close(local[0][1], weights[2])
        torch.testing.assert_close(local[1][1], weights[3])
        self.assertIsNone(pool._global_to_local_expert_id(4))

        stacked = torch.arange(4 * 3 * 4, dtype=torch.float32).reshape(4, 3, 4)
        local = list(pool._iter_local_expert_weights(stacked))
        self.assertEqual([expert_id for expert_id, _ in local], [0, 1])
        torch.testing.assert_close(local[0][1], stacked[2])
        torch.testing.assert_close(local[1][1], stacked[3])

        with self.assertRaisesRegex(TypeError, "3D tensor"):
            list(pool._iter_local_expert_weights(torch.ones(3, 4)))

    def test_unpaged_expert_mapping_preserves_ids(self):
        pool = _make_pool()
        stacked = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2)
        mapped = list(pool._iter_local_expert_weights(stacked))
        self.assertEqual([expert_id for expert_id, _ in mapped], [0, 1, 2])
        self.assertEqual(pool._global_to_local_expert_id(7), 7)

    def test_shared_outer_a_accepts_single_expert_and_validates_shape(self):
        pool = _make_pool()
        pool.experts_shared_outer_loras = True
        weight = torch.arange(12 * 3, dtype=torch.float32).reshape(1, 12, 3)

        pool._scatter_moe_a_weight_to_pages(
            "gate_up_proj_moe", 0, weight, 6, phys_pages=[2, 3]
        )
        target = pool.A_pages["gate_up_proj_moe"][0]
        torch.testing.assert_close(target[2, 0, 0:4], weight[0, 0:4])
        self.assertEqual(torch.count_nonzero(target[[2, 3], 1]).item(), 0)

        with self.assertRaisesRegex(ValueError, "expert dimension 1"):
            pool._scatter_moe_a_weight_to_pages(
                "gate_up_proj_moe", 0, torch.ones(2, 12, 3), 6, [0, 1]
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            pool._scatter_moe_a_weight_to_pages(
                "gate_up_proj_moe",
                0,
                {0: weight[0], 1: weight[0]},
                6,
                [0, 1],
            )

    def test_shared_outer_b_accepts_single_expert_and_validates_shape(self):
        pool = _make_pool()
        pool.experts_shared_outer_loras = True
        weight = torch.arange(3 * 6, dtype=torch.float32).reshape(1, 3, 6)

        pool._scatter_moe_b_weight_to_pages(
            "down_proj_moe", 0, {0: weight[0]}, 6, 0.5, phys_pages=[2, 3]
        )
        target = pool.B_pages["down_proj_moe"][0]
        torch.testing.assert_close(target[2, 0], weight[0, :, :4] * 0.5)
        self.assertEqual(torch.count_nonzero(target[[2, 3], 1]).item(), 0)

        with self.assertRaisesRegex(ValueError, "expert dimension 1"):
            pool._scatter_moe_b_weight_to_pages(
                "down_proj_moe", 0, torch.ones(2, 3, 6), 6, 1.0, [0, 1]
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            pool._scatter_moe_b_weight_to_pages(
                "down_proj_moe",
                0,
                {0: weight[0], 1: weight[0]},
                6,
                1.0,
                [0, 1],
            )

    def test_none_weights_zero_pages_without_marking_access(self):
        pool = _make_pool()
        pool._scatter_moe_a_weight_to_pages("down_proj_moe", 0, None, 4, phys_pages=[1])
        pool._scatter_moe_b_weight_to_pages(
            "down_proj_moe", 0, None, 4, 1.0, phys_pages=[2]
        )
        self.assertEqual(
            torch.count_nonzero(pool.A_pages["down_proj_moe"][0][1]).item(), 0
        )
        self.assertEqual(
            torch.count_nonzero(pool.B_pages["down_proj_moe"][0][2]).item(), 0
        )

    def test_scatter_skips_out_of_range_experts_and_logical_pages(self):
        pool = _make_pool()
        original = pool.A_pages["down_proj_moe"][0][0].clone()
        pool._scatter_moe_a_weight_to_pages(
            "down_proj_moe",
            0,
            {7: torch.ones(4, 6)},
            4,
            phys_pages=[0],
        )
        # Physical pages are always cleared before scatter.
        self.assertEqual(
            torch.count_nonzero(pool.A_pages["down_proj_moe"][0][0]).item(), 0
        )
        self.assertGreater(torch.count_nonzero(original).item(), 0)

        pool._scatter_moe_b_weight_to_pages(
            "down_proj_moe",
            0,
            {0: torch.ones(3, 4)},
            4,
            1.0,
            phys_pages=[4],
            logic_page_indices=[2],
        )
        self.assertEqual(
            torch.count_nonzero(pool.B_pages["down_proj_moe"][0][4]).item(), 0
        )

    def test_stacked_a_uses_page_local_component_offsets(self):
        pool = _make_pool()
        rank = 6
        expert0 = torch.arange(12 * 3, dtype=torch.float32).reshape(12, 3)
        expert1 = expert0 + 1000

        pool._scatter_moe_a_weight_to_pages(
            "gate_up_proj_moe",
            0,
            {0: expert0, 1: expert1},
            rank,
            phys_pages=[4, 1],
        )
        target = pool.A_pages["gate_up_proj_moe"][0]

        torch.testing.assert_close(target[4, 0, 0:4], expert0[0:4])
        torch.testing.assert_close(target[4, 0, 4:8], expert0[6:10])
        torch.testing.assert_close(target[1, 1, 0:2], expert1[4:6])
        torch.testing.assert_close(target[1, 1, 4:6], expert1[10:12])
        self.assertEqual(torch.count_nonzero(target[1, :, 2:4]).item(), 0)
        self.assertEqual(torch.count_nonzero(target[1, :, 6:8]).item(), 0)

    def test_b_scales_once_and_clears_unfilled_experts_and_tail(self):
        pool = _make_pool()
        rank = 6
        weight = torch.arange(3 * rank, dtype=torch.float32).reshape(3, rank)

        pool._scatter_moe_b_weight_to_pages(
            "down_proj_moe",
            0,
            {0: weight},
            rank,
            scaling=0.25,
            phys_pages=[3, 0],
        )
        target = pool.B_pages["down_proj_moe"][0]

        torch.testing.assert_close(target[3, 0], weight[:, 0:4] * 0.25)
        torch.testing.assert_close(target[0, 0, :, 0:2], weight[:, 4:6] * 0.25)
        self.assertEqual(torch.count_nonzero(target[0, 0, :, 2:4]).item(), 0)
        self.assertEqual(torch.count_nonzero(target[[3, 0], 1]).item(), 0)

    def test_partial_reload_preserves_logical_page_index(self):
        pool = _make_pool()
        rank = 6
        weight = torch.arange(2 * 6 * rank, dtype=torch.float32).reshape(2, 6, rank)

        pool._scatter_moe_b_weight_to_pages(
            "gate_up_proj_moe",
            0,
            weight,
            rank,
            scaling=1.0,
            phys_pages=[2],
            logic_page_indices=[1],
        )
        target = pool.B_pages["gate_up_proj_moe"][0]

        torch.testing.assert_close(target[2, :, :, 0:2], weight[:, :, 4:6])
        self.assertEqual(torch.count_nonzero(target[2, :, :, 2:4]).item(), 0)
        # Only the reloaded physical page is touched.
        self.assertTrue(torch.all(target[0] == 1))


if __name__ == "__main__":
    unittest.main()
