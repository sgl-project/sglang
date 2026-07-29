from __future__ import annotations

import random
import unittest

import torch

from sglang.kernels.ops.kv_canary.verify_ref import splitmix64
from sglang.srt.kv_canary.token_oracle.oracle import HashOracle
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=30, suite="extra-a-test-1-gpu-small-amd")


class TestHashOracleTorchVsRef(CustomTestCase):
    def test_hash_oracle_matches_scalar_splitmix64_ref(self) -> None:
        """Verify single-item HashOracle calls match the scalar SplitMix64 reference."""
        rng = random.Random(0)
        vocab_size = 32000
        num_cases = 1000

        generalized_req_ids: list[int] = []
        positions: list[int] = []
        for _ in range(num_cases):
            generalized_req_ids.append(rng.randrange(0, 1 << 60))
            positions.append(rng.randrange(0, 1 << 60))

        ref_tokens: list[int] = [
            splitmix64(generalized_req_ids[i] ^ positions[i]) % vocab_size
            for i in range(num_cases)
        ]

        oracle = HashOracle(vocab_size=vocab_size)
        torch_tokens: list[int] = []
        generalized_req_ids_tensor = torch.tensor(
            generalized_req_ids, dtype=torch.int64
        )
        positions_tensor = torch.tensor(positions, dtype=torch.int64)
        for i in range(num_cases):
            out = oracle.expected_tokens(
                generalized_req_ids=generalized_req_ids_tensor[i : i + 1],
                positions=positions_tensor[i : i + 1],
            )
            torch_tokens.append(int(out.tolist()[0]))

        for i in range(num_cases):
            self.assertEqual(
                torch_tokens[i],
                ref_tokens[i],
                f"mismatch at case {i}: generalized_req_id={generalized_req_ids[i]} "
                f"position={positions[i]}: torch={torch_tokens[i]} ref={ref_tokens[i]}",
            )

    def test_hash_oracle_batched_matches_scalar_splitmix64_ref(self) -> None:
        """Verify batched HashOracle calls match the scalar SplitMix64 reference."""
        rng = random.Random(1)
        vocab_size = 32000
        num_cases = 1000

        generalized_req_ids = [rng.randrange(0, 1 << 60) for _ in range(num_cases)]
        positions = [rng.randrange(0, 1 << 60) for _ in range(num_cases)]
        ref_tokens = [
            splitmix64(generalized_req_ids[i] ^ positions[i]) % vocab_size
            for i in range(num_cases)
        ]

        oracle = HashOracle(vocab_size=vocab_size)
        out = oracle.expected_tokens(
            generalized_req_ids=torch.tensor(generalized_req_ids, dtype=torch.int64),
            positions=torch.tensor(positions, dtype=torch.int64),
        )
        torch_tokens = out.tolist()

        for i in range(num_cases):
            self.assertEqual(
                torch_tokens[i],
                ref_tokens[i],
                f"batched mismatch at case {i}: generalized_req_id={generalized_req_ids[i]} "
                f"position={positions[i]}: torch={torch_tokens[i]} ref={ref_tokens[i]}",
            )


if __name__ == "__main__":
    unittest.main()
