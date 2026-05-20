from __future__ import annotations

import unittest

import torch

from sglang.srt.kv_canary.token_oracle.oracle import HashOracle
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")


def _call(oracle: HashOracle, *, req_id: int, position: int) -> int:
    out = oracle.expected_tokens(
        req_ids=torch.tensor([req_id], dtype=torch.int64),
        positions=torch.tensor([position], dtype=torch.int64),
    )
    return int(out.tolist()[0])


class TestHashOracle(CustomTestCase):
    def test_hash_oracle_is_deterministic_for_same_inputs(self) -> None:
        oracle = HashOracle(seed=12345, vocab_size=32000)

        first = _call(oracle, req_id=7, position=42)
        second = _call(oracle, req_id=7, position=42)

        self.assertEqual(first, second)

    def test_hash_oracle_output_in_vocab_range(self) -> None:
        vocab_size = 1024
        oracle = HashOracle(seed=0xCAFEBABE, vocab_size=vocab_size)

        req_ids = torch.arange(0, 64, dtype=torch.int64).repeat_interleave(64)
        positions = torch.arange(0, 64, dtype=torch.int64).repeat(64)
        tokens = oracle.expected_tokens(req_ids=req_ids, positions=positions).tolist()

        for token in tokens:
            self.assertTrue(0 <= token < vocab_size)


if __name__ == "__main__":
    unittest.main()
