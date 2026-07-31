"""Tests for seeded, batch-invariant EAGLE draft proposal sampling."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative.spec_utils import fast_sample
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def _sample(probs, seeds, positions, *, draft_step=0):
    return fast_sample(
        probs,
        sampling_seed=torch.tensor(seeds, device="cuda", dtype=torch.int64),
        positions=torch.tensor(positions, device="cuda", dtype=torch.int64),
        draft_step=draft_step,
    )


class TestDeterministicFastSample(CustomTestCase):
    def setUp(self):
        torch.manual_seed(1234)
        weights = torch.rand((8, 4096), device="cuda", dtype=torch.float32)
        self.probs = weights / weights.sum(dim=-1, keepdim=True)
        self.seeds = [101, 202, 303, 404, 505, 606, 707, 808]
        self.positions = [11, 12, 13, 14, 15, 16, 17, 18]

    def test_replay_is_bitwise_reproducible(self):
        p_a, index_a = _sample(self.probs, self.seeds, self.positions)
        p_b, index_b = _sample(self.probs, self.seeds, self.positions)

        self.assertTrue(torch.equal(index_a, index_b))
        self.assertTrue(torch.equal(p_a, p_b))

    def test_cuda_graph_replay_and_input_updates(self):
        seeds = torch.tensor(self.seeds, device="cuda", dtype=torch.int64)
        positions = torch.tensor(self.positions, device="cuda", dtype=torch.int64)

        # Warm Triton/PyTorch kernels before capture.
        fast_sample(self.probs, sampling_seed=seeds, positions=positions)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_p, graph_index = fast_sample(
                self.probs, sampling_seed=seeds, positions=positions
            )

        graph.replay()
        first_p = graph_p.clone()
        first_index = graph_index.clone()
        graph.replay()
        self.assertTrue(torch.equal(first_p, graph_p))
        self.assertTrue(torch.equal(first_index, graph_index))

        seeds.add_(1)
        positions.add_(2)
        graph.replay()
        updated_p = graph_p.clone()
        updated_index = graph_index.clone()
        expected_p, expected_index = fast_sample(
            self.probs, sampling_seed=seeds, positions=positions
        )
        self.assertTrue(torch.equal(updated_p, expected_p))
        self.assertTrue(torch.equal(updated_index, expected_index))

    def test_batch_permutation_and_split_are_invariant(self):
        p_ref, index_ref = _sample(self.probs, self.seeds, self.positions)

        permutation = torch.tensor([6, 2, 7, 0, 4, 1, 5, 3], device="cuda")
        inverse = torch.argsort(permutation)
        p_permuted, index_permuted = _sample(
            self.probs[permutation],
            [self.seeds[i] for i in permutation.cpu().tolist()],
            [self.positions[i] for i in permutation.cpu().tolist()],
        )
        self.assertTrue(torch.equal(index_ref, index_permuted[inverse]))
        self.assertTrue(torch.equal(p_ref, p_permuted[inverse]))

        split_indices = []
        split_probs = []
        for start, end in ((0, 3), (3, 7), (7, 8)):
            sampled_p, sampled_index = _sample(
                self.probs[start:end],
                self.seeds[start:end],
                self.positions[start:end],
            )
            split_indices.append(sampled_index)
            split_probs.append(sampled_p)
        self.assertTrue(torch.equal(index_ref, torch.cat(split_indices)))
        self.assertTrue(torch.equal(p_ref, torch.cat(split_probs)))

    def test_seed_position_and_draft_step_select_separate_streams(self):
        uniform_probs = torch.full((64, 4096), 1 / 4096, device="cuda")
        seeds = list(range(64))
        positions = list(range(100, 164))

        _, base = _sample(uniform_probs, seeds, positions)
        _, changed_seed = _sample(
            uniform_probs, [seed + 1 for seed in seeds], positions
        )
        _, changed_position = _sample(
            uniform_probs, seeds, [position + 1 for position in positions]
        )
        _, changed_step = _sample(uniform_probs, seeds, positions, draft_step=1)

        self.assertFalse(torch.equal(base, changed_seed))
        self.assertFalse(torch.equal(base, changed_position))
        self.assertFalse(torch.equal(base, changed_step))

    def test_seeded_samples_follow_categorical_distribution(self):
        num_samples = 100_000
        expected = torch.tensor([0.1, 0.2, 0.7], device="cuda")
        probs = expected.expand(num_samples, -1)
        seeds = torch.arange(num_samples, device="cuda", dtype=torch.int64)
        positions = torch.full((num_samples,), 37, device="cuda", dtype=torch.int64)

        _, sampled = fast_sample(
            probs,
            sampling_seed=seeds,
            positions=positions,
        )
        observed = torch.bincount(sampled.view(-1), minlength=3) / num_samples

        self.assertTrue(
            torch.allclose(observed, expected, rtol=0, atol=0.01),
            f"Expected frequencies near {expected.tolist()}, got {observed.tolist()}",
        )

    def test_hash_endpoints_produce_finite_ordering(self):
        probs = torch.tensor(
            [[0.01, 0.09, 0.90], [0.80, 0.15, 0.05]],
            device="cuda",
            dtype=torch.float32,
        )

        for hash_value in (0, torch.iinfo(torch.uint32).max):
            with self.subTest(hash_value=hash_value):

                def _constant_hash(seed, positions, col_indices):
                    return torch.full(
                        (seed.shape[0], col_indices.shape[0]),
                        hash_value,
                        device=seed.device,
                        dtype=torch.uint32,
                    )

                with patch(
                    "sglang.kernels.ops.sampling.murmur_hash.murmur_hash32",
                    side_effect=_constant_hash,
                ):
                    _, sampled = _sample(probs, [1, 2], [3, 4])

                # Equal finite exponential variates leave probability ordering
                # to decide the winner. Endpoint infinities would violate this.
                self.assertTrue(
                    torch.equal(
                        sampled,
                        torch.tensor([[2], [0]], device="cuda", dtype=torch.int64),
                    )
                )

    def test_requires_matching_seed_position_rows(self):
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            fast_sample(
                self.probs[:2],
                sampling_seed=torch.tensor([1, 2], device="cuda"),
                positions=torch.tensor([3], device="cuda"),
            )


if __name__ == "__main__":
    unittest.main()
