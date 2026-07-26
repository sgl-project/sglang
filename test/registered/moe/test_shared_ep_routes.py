import unittest

import torch

from sglang.kernels.ops.moe.shared_ep_route_prep import (
    _route_specialization_key,
    prepare_routes_cuda,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.shared_ep.kernels import prepare_routes
from sglang.srt.layers.moe.shared_ep.profiles import select_profile
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")


def _profile(hidden_size, top_k):
    return select_profile(
        MoeRunnerConfig(
            hidden_size=hidden_size,
            intermediate_size_per_partition=2048,
            top_k=top_k,
            num_experts=256,
            num_local_experts=32,
        ),
        capability=(9, 0),
        ep_size=8,
        block_shape=(128, 128),
        max_tokens_per_rank=32,
    )


def _ready(owner_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(owner_count * 4, dtype=torch.uint8, device="cuda"),
        torch.zeros(1, dtype=torch.int32, device="cuda"),
    )


class TestSharedEpRoutes(unittest.TestCase):
    def test_route_specialization_key_is_shape_based(self):
        """JIT variants must be selected by kernel shape, never model name."""
        glm_key = _route_specialization_key(
            (8, 32, 8),
            num_local_experts=32,
            block_size_m=16,
            num_threads=1024,
        )
        dsv4_key = _route_specialization_key(
            (8, 32, 6),
            num_local_experts=32,
            block_size_m=16,
            num_threads=1024,
        )

        self.assertEqual(glm_key, (8, 32, 8, 32, 16, 1024))
        self.assertEqual(dsv4_key, (8, 32, 6, 32, 16, 1024))
        self.assertNotEqual(glm_key, dsv4_key)

    def test_route_specialization_rejects_invalid_configs(self):
        invalid_configs = (
            ((8, 32), 32, 16, 1024),
            ((8, 0, 6), 32, 16, 1024),
            ((8, 32, 6), 0, 16, 1024),
            ((8, 32, 6), 32, 0, 1024),
            ((8, 32, 6), 32, 16, 1000),
            ((8, 32, 6), 32, 16, 1056),
            ((8, 32, 6), 32.0, 16, 1024),
            ((8, 32, 6), 32, 16, 1024.0),
        )
        for shape, num_local_experts, block_size_m, num_threads in invalid_configs:
            with self.subTest(
                shape=shape,
                num_local_experts=num_local_experts,
                block_size_m=block_size_m,
                num_threads=num_threads,
            ):
                with self.assertRaises(ValueError):
                    _route_specialization_key(
                        shape,
                        num_local_experts=num_local_experts,
                        block_size_m=block_size_m,
                        num_threads=num_threads,
                    )

    def test_cuda_route_prep_rejects_cpu_tensors_before_jit(self):
        ids = torch.zeros((8, 32, 6), dtype=torch.int32)
        weights = torch.zeros_like(ids, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "CUDA tensors"):
            prepare_routes_cuda(
                ids,
                weights,
                torch.zeros(8 * 4, dtype=torch.uint8),
                torch.zeros(1, dtype=torch.int32),
                local_expert_start=0,
                num_local_experts=32,
                block_size_m=16,
                num_threads=1024,
            )

    def test_production_route_prep_rejects_cpu_tensors(self):
        ids = torch.zeros((8, 32, 6), dtype=torch.int32)
        weights = torch.zeros_like(ids, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "CUDA tensors"):
            prepare_routes(
                _profile(4096, 6),
                ids,
                weights,
                torch.zeros(8 * 4, dtype=torch.uint8),
                torch.zeros(1, dtype=torch.int32),
                local_expert_start=0,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA JIT")
    def test_route_multiset_and_padding_for_glm_and_dsv4(self):
        """Every local route must be consumed once by its selected expert."""
        fixtures = (
            (_profile(6144, 8), 8),
            (_profile(4096, 6), 6),
        )
        for profile, top_k in fixtures:
            with self.subTest(profile=profile.name):
                ids = torch.arange(2 * 3 * top_k, dtype=torch.int32).view(
                    2,
                    3,
                    top_k,
                )
                ids = (ids * 13) % 256
                ids[0, 0, 0] = -1
                weights = torch.arange(ids.numel(), dtype=torch.float32).view_as(ids)
                ready_signals, ready_epoch = _ready(ids.shape[0])
                result = prepare_routes(
                    profile,
                    ids.cuda(),
                    weights.cuda(),
                    ready_signals,
                    ready_epoch,
                    local_expert_start=64,
                )

                expected_local = ids - 64
                expected_local = torch.where(
                    (expected_local >= 0) & (expected_local < 32),
                    expected_local,
                    -1,
                )
                self.assertTrue(torch.equal(result.local_ids.cpu(), expected_local))
                self.assertTrue(torch.equal(result.local_weights.cpu(), weights))

                flat_local = result.local_ids.flatten()
                padded = int(result.num_tokens_post_padded.item())
                self.assertEqual(padded % profile.block_size_m, 0)
                for block_start in range(0, padded, profile.block_size_m):
                    block = result.sorted_token_ids[
                        block_start : block_start + profile.block_size_m
                    ]
                    expert = int(result.expert_ids[block_start // profile.block_size_m])
                    valid = block[block < flat_local.numel()]
                    self.assertTrue(torch.all(flat_local[valid] == expert))

                consumed = result.sorted_token_ids[:padded]
                consumed = consumed[consumed < flat_local.numel()]
                expected = torch.nonzero(flat_local >= 0).flatten()
                self.assertEqual(
                    sorted(consumed.tolist()),
                    sorted(expected.tolist()),
                )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA JIT")
    def test_skew_and_empty_experts_do_not_create_fake_routes(self):
        """Invalid and non-local routes must not be encoded as a dummy expert."""
        profile = _profile(4096, 6)
        ids = torch.full((1, 32, 6), 255, dtype=torch.int32)
        ids[0, :7, 0] = 96
        ids[0, :3, 1] = 127
        weights = torch.ones_like(ids, dtype=torch.float32)
        ready_signals, ready_epoch = _ready(ids.shape[0])

        result = prepare_routes(
            profile,
            ids.cuda(),
            weights.cuda(),
            ready_signals,
            ready_epoch,
            local_expert_start=96,
        )

        padded = int(result.num_tokens_post_padded.item())
        self.assertEqual(padded, 32)
        self.assertEqual(result.expert_ids[:2].tolist(), [0, 31])
        self.assertTrue(torch.all(result.expert_ids[2:] == -1))
        consumed = result.sorted_token_ids[:padded]
        valid = consumed[consumed < ids.numel()]
        self.assertEqual(valid.numel(), 10)


if __name__ == "__main__":
    unittest.main()
