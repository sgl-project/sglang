"""HashOracle determinism."""

from __future__ import annotations

import pytest
import torch

from sglang.srt.kv_canary.token_oracle.oracle import HashOracle, TokenOracle
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")


def _call(oracle: HashOracle, *, req_id: int, position: int) -> int:
    out = oracle.expected_tokens(
        req_ids=torch.tensor([req_id], dtype=torch.int64),
        positions=torch.tensor([position], dtype=torch.int64),
    )
    return int(out.tolist()[0])


def test_hash_oracle_is_deterministic_for_same_inputs() -> None:
    oracle = HashOracle(seed=12345, vocab_size=32000)

    first = _call(oracle, req_id=7, position=42)
    second = _call(oracle, req_id=7, position=42)

    assert first == second


def test_hash_oracle_output_in_vocab_range() -> None:
    vocab_size = 1024
    oracle = HashOracle(seed=0xCAFEBABE, vocab_size=vocab_size)

    req_ids = torch.arange(0, 64, dtype=torch.int64).repeat_interleave(64)
    positions = torch.arange(0, 64, dtype=torch.int64).repeat(64)
    tokens = oracle.expected_tokens(req_ids=req_ids, positions=positions).tolist()

    for token in tokens:
        assert 0 <= token < vocab_size


def test_hash_oracle_different_seeds_give_different_streams() -> None:
    a = HashOracle(seed=1, vocab_size=32000)
    b = HashOracle(seed=2, vocab_size=32000)

    req_ids = torch.zeros(64, dtype=torch.int64)
    positions = torch.arange(0, 64, dtype=torch.int64)
    a_out = a.expected_tokens(req_ids=req_ids, positions=positions).tolist()
    b_out = b.expected_tokens(req_ids=req_ids, positions=positions).tolist()

    differences = sum(1 for i in range(64) if a_out[i] != b_out[i])

    assert differences >= 50


def test_hash_oracle_different_req_ids_give_different_tokens() -> None:
    oracle = HashOracle(seed=42, vocab_size=32000)

    left_ids = torch.arange(0, 64, dtype=torch.int64)
    right_ids = torch.arange(1, 65, dtype=torch.int64)
    positions = torch.zeros(64, dtype=torch.int64)
    left = oracle.expected_tokens(req_ids=left_ids, positions=positions).tolist()
    right = oracle.expected_tokens(req_ids=right_ids, positions=positions).tolist()

    differences = sum(1 for i in range(64) if left[i] != right[i])

    assert differences >= 50


def test_hash_oracle_satisfies_oracle_protocol() -> None:
    oracle: TokenOracle = HashOracle(seed=0, vocab_size=100)

    out = oracle.expected_tokens(
        req_ids=torch.tensor([0], dtype=torch.int64),
        positions=torch.tensor([0], dtype=torch.int64),
    )
    assert out.dtype == torch.int32
    assert int(out.tolist()[0]) is not None


def test_hash_oracle_is_frozen_dataclass() -> None:
    oracle = HashOracle(seed=1, vocab_size=100)

    with pytest.raises(Exception):
        oracle.seed = 2  # type: ignore[misc]
