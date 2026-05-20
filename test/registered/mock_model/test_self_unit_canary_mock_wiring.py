"""TokenIdOracleManager.fill_expected_inputs writes oracle-derived (token, position) into canary
placeholders, and mock_model_engine_kwargs returns the right Engine kwargs for mock-model tests.
"""

from __future__ import annotations

import dataclasses
import json
import os

import torch

from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.mock_model.oracle import HashOracle
from sglang.srt.kv_canary.mock_model.sampler import install_oracle_sampler
from sglang.test.ci.ci_register import register_cuda_ci

from .utils import mock_model_engine_kwargs

register_cuda_ci(est_time=60, suite="extra-a-1-gpu-large")


@dataclasses.dataclass
class _StubForwardMode:
    extend: bool

    def is_extend(self) -> bool:
        return self.extend


@dataclasses.dataclass
class _StubForwardBatch:
    input_ids: torch.Tensor
    positions: torch.Tensor
    req_pool_indices: torch.Tensor
    forward_mode: _StubForwardMode
    extend_seq_lens: object


def test_fill_expected_inputs_decode_one_token_per_req() -> None:
    oracle = HashOracle(seed=1, vocab_size=32000)
    hook = install_oracle_sampler(oracle=oracle)

    fb = _StubForwardBatch(
        input_ids=torch.tensor([0, 0], dtype=torch.int64),
        positions=torch.tensor([10, 20], dtype=torch.int64),
        req_pool_indices=torch.tensor([5, 7], dtype=torch.int64),
        forward_mode=_StubForwardMode(extend=False),
        extend_seq_lens=None,
    )
    expected_inputs = ExpectedInputs.allocate(capacity=8, device=torch.device("cpu"))

    hook.fill_expected_inputs(
        forward_batch=fb,
        expected_inputs_out=expected_inputs,
    )

    assert expected_inputs.tokens[:2].tolist() == [
        oracle.expected_token(req_id=5, position=10),
        oracle.expected_token(req_id=7, position=20),
    ]
    assert expected_inputs.positions[:2].tolist() == [10, 20]


def test_fill_expected_inputs_extend_uses_extend_seq_lens() -> None:
    oracle = HashOracle(seed=2, vocab_size=32000)
    hook = install_oracle_sampler(oracle=oracle)

    fb = _StubForwardBatch(
        input_ids=torch.tensor([0, 0, 0, 0], dtype=torch.int64),
        positions=torch.tensor([0, 1, 2, 0], dtype=torch.int64),
        req_pool_indices=torch.tensor([5, 7], dtype=torch.int64),
        forward_mode=_StubForwardMode(extend=True),
        extend_seq_lens=torch.tensor([3, 1], dtype=torch.int64),
    )
    expected_inputs = ExpectedInputs.allocate(capacity=8, device=torch.device("cpu"))

    hook.fill_expected_inputs(
        forward_batch=fb,
        expected_inputs_out=expected_inputs,
    )

    assert expected_inputs.tokens[:4].tolist() == [
        oracle.expected_token(req_id=5, position=0),
        oracle.expected_token(req_id=5, position=1),
        oracle.expected_token(req_id=5, position=2),
        oracle.expected_token(req_id=7, position=0),
    ]
    assert expected_inputs.positions[:4].tolist() == [0, 1, 2, 0]


def test_fill_expected_inputs_zero_tokens_is_noop_but_stashes_req_pool() -> None:
    hook = install_oracle_sampler(oracle=HashOracle(seed=0, vocab_size=100))

    fb = _StubForwardBatch(
        input_ids=torch.empty(0, dtype=torch.int64),
        positions=torch.empty(0, dtype=torch.int64),
        req_pool_indices=torch.tensor([5, 7], dtype=torch.int64),
        forward_mode=_StubForwardMode(extend=False),
        extend_seq_lens=None,
    )
    expected_inputs = ExpectedInputs.allocate(capacity=4, device=torch.device("cpu"))

    hook.fill_expected_inputs(
        forward_batch=fb,
        expected_inputs_out=expected_inputs,
    )

    assert hook._req_pool_indices_per_row == [5, 7]


def test_mock_model_engine_kwargs_returns_defaults() -> None:
    os.environ.pop("SGLANG_KV_CANARY_INPUT_CHECK", None)

    kwargs = mock_model_engine_kwargs()

    assert kwargs["load_format"] == "dummy"
    assert json.loads(kwargs["json_model_override_args"]) == {"num_hidden_layers": 1}
    assert kwargs["sampling_backend"] == "oracle"
    assert kwargs["kv_canary"] == "raise"
    assert os.environ["SGLANG_KV_CANARY_INPUT_CHECK"] == "1"


def test_mock_model_engine_kwargs_overrides_win() -> None:
    kwargs = mock_model_engine_kwargs(
        kv_canary="log",
        sampling_backend="pytorch",
        tp_size=2,
    )

    assert kwargs["kv_canary"] == "log"
    assert kwargs["sampling_backend"] == "pytorch"
    assert kwargs["tp_size"] == 2
    assert kwargs["load_format"] == "dummy"


def test_mock_model_engine_kwargs_merges_json_override() -> None:
    kwargs = mock_model_engine_kwargs(
        json_model_override_args='{"rope_theta": 1000.0}',
    )

    merged = json.loads(kwargs["json_model_override_args"])
    assert merged["num_hidden_layers"] == 1
    assert merged["rope_theta"] == 1000.0
