"""Bitwise-equality cross-check: vectorized HashOracle.expected_tokens vs scalar splitmix64 ref."""

from __future__ import annotations

import random

import torch

from sglang.jit_kernel.kv_canary.verify_ref import splitmix64
from sglang.srt.kv_canary.mock_model.oracle import HashOracle
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="extra-a-1-gpu-large")


def test_hash_oracle_matches_scalar_splitmix64_ref() -> None:
    rng = random.Random(0)
    vocab_size = 32000
    num_cases = 1000

    seeds: list[int] = []
    req_ids: list[int] = []
    positions: list[int] = []
    for _ in range(num_cases):
        seeds.append(rng.randrange(0, 1 << 60))
        req_ids.append(rng.randrange(0, 1 << 60))
        positions.append(rng.randrange(0, 1 << 60))

    ref_tokens: list[int] = [
        splitmix64(seeds[i] ^ req_ids[i] ^ positions[i]) % vocab_size
        for i in range(num_cases)
    ]

    torch_tokens: list[int] = []
    req_ids_tensor = torch.tensor(req_ids, dtype=torch.int64)
    positions_tensor = torch.tensor(positions, dtype=torch.int64)
    for i in range(num_cases):
        oracle = HashOracle(seed=seeds[i], vocab_size=vocab_size)
        out = oracle.expected_tokens(
            req_ids=req_ids_tensor[i : i + 1],
            positions=positions_tensor[i : i + 1],
        )
        torch_tokens.append(int(out.tolist()[0]))

    for i in range(num_cases):
        assert torch_tokens[i] == ref_tokens[i], (
            f"mismatch at case {i}: seed={seeds[i]} req_id={req_ids[i]} "
            f"position={positions[i]}: torch={torch_tokens[i]} ref={ref_tokens[i]}"
        )


def test_hash_oracle_batched_matches_scalar_splitmix64_ref() -> None:
    rng = random.Random(1)
    vocab_size = 32000
    num_cases = 1000
    seed = 0xC0FFEE

    req_ids = [rng.randrange(0, 1 << 60) for _ in range(num_cases)]
    positions = [rng.randrange(0, 1 << 60) for _ in range(num_cases)]
    ref_tokens = [
        splitmix64(seed ^ req_ids[i] ^ positions[i]) % vocab_size
        for i in range(num_cases)
    ]

    oracle = HashOracle(seed=seed, vocab_size=vocab_size)
    out = oracle.expected_tokens(
        req_ids=torch.tensor(req_ids, dtype=torch.int64),
        positions=torch.tensor(positions, dtype=torch.int64),
    )
    torch_tokens = out.tolist()

    for i in range(num_cases):
        assert torch_tokens[i] == ref_tokens[i], (
            f"batched mismatch at case {i}: req_id={req_ids[i]} "
            f"position={positions[i]}: torch={torch_tokens[i]} ref={ref_tokens[i]}"
        )
