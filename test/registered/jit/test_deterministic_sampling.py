"""Kernel-level regression tests for deterministic sampling."""

import unittest

import torch

from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.layers.utils.hash import murmur_hash32
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_UINT32_MASK = (1 << 32) - 1


def _rotate_left_32(value: int, shift: int) -> int:
    return ((value << shift) | (value >> (32 - shift))) & _UINT32_MASK


def _murmur3_mix(hash_value: int, block: int) -> int:
    block = (block * 0xCC9E2D51) & _UINT32_MASK
    block = _rotate_left_32(block, 15)
    block = (block * 0x1B873593) & _UINT32_MASK

    hash_value ^= block
    hash_value = _rotate_left_32(hash_value, 13)
    return (hash_value * 5 + 0xE6546B64) & _UINT32_MASK


def _finalize_murmur3_32(hash_value: int) -> int:
    hash_value ^= hash_value >> 16
    hash_value = (hash_value * 0x85EBCA6B) & _UINT32_MASK
    hash_value ^= hash_value >> 13
    hash_value = (hash_value * 0xC2B2AE35) & _UINT32_MASK
    hash_value ^= hash_value >> 16
    return hash_value


def _murmur_hash32_reference(seed: int, position: int, column: int) -> int:
    hash_value = 0
    for block in (
        seed & _UINT32_MASK,
        (seed >> 32) & _UINT32_MASK,
        position & _UINT32_MASK,
        column & _UINT32_MASK,
    ):
        hash_value = _murmur3_mix(hash_value, block)

    return _finalize_murmur3_32(hash_value ^ 16)


class TestMurmurHashReference(CustomTestCase):
    def test_known_uint32_max_vector(self):
        self.assertEqual(
            _murmur_hash32_reference(seed=7847, position=12345, column=1208),
            _UINT32_MASK,
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDeterministicSamplingKernels(CustomTestCase):
    def test_murmur_hash_matches_reference_at_block_boundaries(self):
        seed_values = [0, 42, (1 << 32) + 7, (1 << 64) - 1]
        position_values = [0, 1, 127, (1 << 32) + 9]
        seeds = torch.tensor(seed_values, dtype=torch.uint64, device="cuda")
        positions = torch.tensor(position_values, dtype=torch.int64, device="cuda")

        for width in (1, 1023, 1024, 1025):
            with self.subTest(width=width):
                columns = torch.arange(width, dtype=torch.int64, device="cuda")
                actual = murmur_hash32(seeds, positions, columns)
                expected = torch.tensor(
                    [
                        [
                            _murmur_hash32_reference(seed, position, column)
                            for column in range(width)
                        ]
                        for seed, position in zip(seed_values, position_values)
                    ],
                    dtype=torch.uint32,
                    device="cuda",
                )

                self.assertTrue(torch.equal(actual, expected))

    def test_murmur_hash_reaches_uint32_max(self):
        actual = murmur_hash32(
            torch.tensor([7847], dtype=torch.uint64, device="cuda"),
            torch.tensor([12345], dtype=torch.int64, device="cuda"),
            torch.tensor([1208], dtype=torch.int64, device="cuda"),
        )

        self.assertEqual(actual.item(), _UINT32_MASK)

    def test_seeded_sampling_is_batch_and_order_invariant(self):
        batch_size = 4
        vocab_size = 257
        values = torch.arange(
            batch_size * vocab_size, dtype=torch.float64, device="cuda"
        )
        logits = torch.sin(values * 0.37).reshape(batch_size, vocab_size)
        logprobs = torch.log_softmax(logits, dim=-1)
        seeds = torch.tensor(
            [0, 42, (1 << 32) + 7, -1],
            dtype=torch.int64,
            device="cuda",
        )
        positions = torch.tensor([0, 1, 127, 4096], device="cuda")

        batched = multinomial_with_seed(logprobs, seeds, positions)
        replayed = multinomial_with_seed(logprobs.clone(), seeds, positions)
        self.assertTrue(torch.equal(batched, replayed))

        chunked = torch.cat(
            [
                multinomial_with_seed(logprobs[:2], seeds[:2], positions[:2]),
                multinomial_with_seed(logprobs[2:], seeds[2:], positions[2:]),
            ]
        )
        self.assertTrue(torch.equal(chunked, batched))

        for index in range(batch_size):
            with self.subTest(index=index):
                single = multinomial_with_seed(
                    logprobs[index : index + 1],
                    seeds[index : index + 1],
                    positions[index : index + 1],
                )
                self.assertTrue(torch.equal(single, batched[index : index + 1]))

        permutation = torch.tensor([2, 0, 3, 1], device="cuda")
        permuted = multinomial_with_seed(
            logprobs[permutation], seeds[permutation], positions[permutation]
        )
        self.assertTrue(torch.equal(permuted, batched[permutation]))


if __name__ == "__main__":
    unittest.main()
