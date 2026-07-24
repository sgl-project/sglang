"""Correctness tests for canonical SGL-LoRA virtual-expert routing."""

import unittest

import torch

from sglang.srt.lora.sgl_lora.routing import build_virtual_expert_routing
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class TestSglLoraRouting(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda:0"

    def _build(
        self,
        topk_ids,
        adapters,
        *,
        factor_count,
        max_loras=2,
        factor_map=None,
        block_size=16,
        dtype=torch.int32,
    ):
        return build_virtual_expert_routing(
            torch.tensor(topk_ids, dtype=dtype, device=self.device),
            torch.tensor(adapters, dtype=dtype, device=self.device),
            factor_expert_count=factor_count,
            max_loras=max_loras,
            block_size=block_size,
            routed_expert_to_factor_id=(
                None
                if factor_map is None
                else torch.tensor(factor_map, dtype=dtype, device=self.device)
            ),
        )

    def test_identity_and_explicit_factor_maps(self):
        cases = (
            ("local_identity", [[0, 1], [2, 3]], None, 4, [[0, 1], [6, 7]]),
            (
                "global_owned",
                [[0, 1], [2, 3]],
                [0, -1, 2, 3],
                4,
                [[0, -1], [6, 7]],
            ),
            (
                "global_to_local",
                [[4, 5], [6, 7]],
                [-1, -1, -1, -1, 0, 1, 2, 3],
                4,
                [[0, 1], [6, 7]],
            ),
            (
                "local_to_offset_factor",
                [[0, 1], [2, 3]],
                [4, 5, 6, 7],
                8,
                [[4, 5], [14, 15]],
            ),
        )
        for name, topk_ids, factor_map, factor_count, expected in cases:
            with self.subTest(name=name):
                route = self._build(
                    topk_ids,
                    [0, 1],
                    factor_count=factor_count,
                    factor_map=factor_map,
                )
                self.assertEqual(route.virtual_topk_ids.cpu().tolist(), expected)

    def test_invalid_adapter_expert_and_map_ids_become_one_sentinel(self):
        route = self._build(
            [[-2], [-1], [3], [4], [99], [0], [0]],
            [0, 0, 0, 0, 0, 2, 3],
            factor_count=4,
        )
        self.assertEqual(
            route.virtual_topk_ids.flatten().cpu().tolist(),
            [-1, -1, 3, -1, -1, -1, -1],
        )

        mapped = self._build(
            [[0, 1, 2, 3]],
            [0],
            factor_count=3,
            factor_map=[0, -1, 3, 99],
        )
        self.assertEqual(mapped.virtual_topk_ids.cpu().tolist(), [[0, -1, -1, -1]])

        live_blocks = route.num_pairs_post_padded.item() // route.block_size
        live_ids = route.block_virtual_expert_ids[:live_blocks]
        self.assertTrue(
            bool(
                (
                    (live_ids == -1)
                    | ((live_ids >= 0) & (live_ids < route.num_virtual_experts))
                )
                .all()
                .item()
            )
        )

    def test_int64_ids_preserve_dtype(self):
        route = self._build(
            [[0, 3], [4, -2]],
            [0, 1],
            factor_count=4,
            factor_map=[0, 1, 2, 3, -1],
            dtype=torch.int64,
        )
        self.assertEqual(route.virtual_topk_ids.cpu().tolist(), [[0, 3], [-1, -1]])
        self.assertEqual(route.virtual_topk_ids.dtype, torch.int64)

    def test_sentinel_bucket_is_included_in_capacity(self):
        route = self._build(
            [[0], [1], [2], [3], [0], [1], [2], [3], [-1]],
            [0, 0, 0, 0, 1, 1, 1, 1, 0],
            factor_count=4,
            block_size=4,
        )
        self.assertEqual(route.num_pairs_post_padded.item(), 9 * 4)
        self.assertGreaterEqual(route.sorted_pair_ids.numel(), 9 * 4)
        self.assertGreaterEqual(route.block_virtual_expert_ids.numel(), 9)

    def test_alignment_capability_boundaries(self):
        for factor_count in (1023, 1024, 8192):
            with self.subTest(factor_count=factor_count):
                route = self._build(
                    [[0], [factor_count - 1], [-1]],
                    [0, 0, 0],
                    factor_count=factor_count,
                    max_loras=1,
                    block_size=1 if factor_count > 8191 else 16,
                )
                self.assertEqual(
                    route.virtual_topk_ids.flatten().cpu().tolist(),
                    [0, factor_count - 1, -1],
                )
                self.assertEqual(route.num_virtual_experts, factor_count)


if __name__ == "__main__":
    unittest.main(verbosity=2)
