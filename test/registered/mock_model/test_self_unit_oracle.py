"""HashOracle determinism."""

from __future__ import annotations

import pytest

from sglang.srt.kv_canary.mock_model.oracle import HashOracle, Oracle
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")


def test_hash_oracle_is_deterministic_for_same_inputs() -> None:
    oracle = HashOracle(seed=12345, vocab_size=32000)

    first = oracle.expected_token(req_id=7, position=42)
    second = oracle.expected_token(req_id=7, position=42)

    assert first == second


def test_hash_oracle_output_in_vocab_range() -> None:
    vocab_size = 1024
    oracle = HashOracle(seed=0xCAFEBABE, vocab_size=vocab_size)

    for req_id in range(0, 64):
        for position in range(0, 64):
            token = oracle.expected_token(req_id=req_id, position=position)
            assert 0 <= token < vocab_size


def test_hash_oracle_different_seeds_give_different_streams() -> None:
    a = HashOracle(seed=1, vocab_size=32000)
    b = HashOracle(seed=2, vocab_size=32000)

    differences = sum(
        1
        for i in range(64)
        if a.expected_token(req_id=0, position=i)
        != b.expected_token(req_id=0, position=i)
    )

    assert differences >= 50


def test_hash_oracle_different_req_ids_give_different_tokens() -> None:
    oracle = HashOracle(seed=42, vocab_size=32000)

    differences = sum(
        1
        for r in range(64)
        if oracle.expected_token(req_id=r, position=0)
        != oracle.expected_token(req_id=r + 1, position=0)
    )

    assert differences >= 50


def test_hash_oracle_satisfies_oracle_protocol() -> None:
    oracle: Oracle = HashOracle(seed=0, vocab_size=100)

    assert oracle.expected_token(req_id=0, position=0) is not None


def test_hash_oracle_is_frozen_dataclass() -> None:
    oracle = HashOracle(seed=1, vocab_size=100)

    with pytest.raises(Exception):
        oracle.seed = 2  # type: ignore[misc]
