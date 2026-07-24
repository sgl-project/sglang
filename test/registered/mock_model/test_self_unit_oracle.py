from __future__ import annotations

import random
import unittest

import torch

from sglang.kernels.ops.kv_canary.consts import splitmix64
from sglang.srt.kv_canary.token_oracle.oracle import (
    HashOracle,
    _splitmix64_tensor,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=60, suite="extra-a-test-1-gpu-small-amd")


_U64_MASK: int = (1 << 64) - 1


def _signed_to_unsigned_i64(value: int) -> int:
    return value & _U64_MASK


def _call(oracle: HashOracle, *, generalized_req_id: int, position: int) -> int:
    out = oracle.expected_tokens(
        generalized_req_ids=torch.tensor([generalized_req_id], dtype=torch.int64),
        positions=torch.tensor([position], dtype=torch.int64),
    )
    return int(out.tolist()[0])


class TestHashOracle(CustomTestCase):
    def test_hash_oracle_is_deterministic_for_same_inputs(self) -> None:
        """Verify HashOracle returns the same token for identical inputs."""
        oracle = HashOracle(vocab_size=32000)

        first = _call(oracle, generalized_req_id=7, position=42)
        second = _call(oracle, generalized_req_id=7, position=42)

        self.assertEqual(first, second)

    def test_hash_oracle_output_in_vocab_range(self) -> None:
        """Verify HashOracle outputs stay within the configured vocabulary range."""
        vocab_size = 1024
        oracle = HashOracle(vocab_size=vocab_size)

        generalized_req_ids = torch.arange(0, 64, dtype=torch.int64).repeat_interleave(
            64
        )
        positions = torch.arange(0, 64, dtype=torch.int64).repeat(64)
        tokens = oracle.expected_tokens(
            generalized_req_ids=generalized_req_ids, positions=positions
        ).tolist()

        for token in tokens:
            self.assertTrue(0 <= token < vocab_size)


class TestSplitmix64Tensor(CustomTestCase):
    def test_splitmix64_tensor_matches_scalar_ref_on_random_inputs(self) -> None:
        """Verify tensor SplitMix64 matches the scalar reference on random inputs."""
        rng = random.Random(0)
        num_cases = 1000
        unsigned_inputs: list[int] = [
            rng.randrange(0, 1 << 64) for _ in range(num_cases)
        ]

        signed_inputs = [
            value if value < (1 << 63) else value - (1 << 64)
            for value in unsigned_inputs
        ]
        actual = _splitmix64_tensor(torch.tensor(signed_inputs, dtype=torch.int64))

        actual_unsigned = [_signed_to_unsigned_i64(v) for v in actual.tolist()]
        expected_unsigned = [splitmix64(v) for v in unsigned_inputs]

        self.assertEqual(actual_unsigned, expected_unsigned)

    def test_splitmix64_tensor_known_vectors(self) -> None:
        """Verify tensor SplitMix64 matches scalar reference values for known inputs."""
        inputs = [0, 1, -1, 1 << 32, (1 << 63) - 1, -(1 << 63)]
        expected_unsigned = [splitmix64(_signed_to_unsigned_i64(v)) for v in inputs]

        actual = _splitmix64_tensor(torch.tensor(inputs, dtype=torch.int64))
        actual_unsigned = [_signed_to_unsigned_i64(v) for v in actual.tolist()]

        self.assertEqual(actual_unsigned, expected_unsigned)

    def test_splitmix64_tensor_preserves_shape_and_dtype(self) -> None:
        """Verify tensor SplitMix64 preserves input shape and int64 dtype."""
        shape = (3, 4, 5)
        rng = torch.Generator().manual_seed(42)
        inputs = torch.randint(
            low=-(1 << 62),
            high=(1 << 62),
            size=shape,
            dtype=torch.int64,
            generator=rng,
        )

        out = _splitmix64_tensor(inputs)

        self.assertEqual(out.shape, inputs.shape)
        self.assertEqual(out.dtype, torch.int64)

    def test_splitmix64_tensor_is_deterministic(self) -> None:
        """Verify tensor SplitMix64 returns stable values for repeated calls."""
        inputs = torch.tensor([0, 1, 2, 3, 1 << 40, -7], dtype=torch.int64)

        first = _splitmix64_tensor(inputs.clone()).tolist()
        second = _splitmix64_tensor(inputs.clone()).tolist()

        self.assertEqual(first, second)

    def test_splitmix64_tensor_is_injective_on_distinct_inputs(self) -> None:
        """Verify tensor SplitMix64 maps distinct sampled inputs to distinct outputs."""
        inputs = torch.arange(-1000, 1000, dtype=torch.int64)

        out = _splitmix64_tensor(inputs).tolist()

        self.assertEqual(len(set(out)), len(out))


if __name__ == "__main__":
    unittest.main()
