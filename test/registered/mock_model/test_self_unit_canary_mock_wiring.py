"""OracleSamplerHook.fill_expected_inputs writes oracle-derived (token, position) into canary
placeholders, and mock_model_engine_kwargs returns the right Engine kwargs for mock-model tests.
"""

from __future__ import annotations

import dataclasses
import json

import torch

from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.mock_model.oracle import ScriptedOracle
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
    table = {(5, 10): 111, (7, 20): 333}
    hook = install_oracle_sampler(oracle=ScriptedOracle(table=table))

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

    assert expected_inputs.tokens[:2].tolist() == [111, 333]
    assert expected_inputs.positions[:2].tolist() == [10, 20]


def test_fill_expected_inputs_extend_uses_extend_seq_lens() -> None:
    table = {(5, 0): 11, (5, 1): 22, (5, 2): 33, (7, 0): 99}
    hook = install_oracle_sampler(oracle=ScriptedOracle(table=table))

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

    assert expected_inputs.tokens[:4].tolist() == [11, 22, 33, 99]
    assert expected_inputs.positions[:4].tolist() == [0, 1, 2, 0]


def test_fill_expected_inputs_zero_tokens_is_noop_but_stashes_req_pool() -> None:
    hook = install_oracle_sampler(oracle=ScriptedOracle(table={}))

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
    kwargs = mock_model_engine_kwargs()

    assert kwargs["load_format"] == "dummy"
    assert json.loads(kwargs["json_model_override_args"]) == {"num_hidden_layers": 1}
    assert kwargs["sampling_backend"] == "oracle"
    assert kwargs["kv_canary"] == "raise"
    assert kwargs["kv_canary_input_check"] is True


def test_mock_model_engine_kwargs_overrides_win() -> None:
    kwargs = mock_model_engine_kwargs(
        kv_canary="on",
        sampling_backend="pytorch",
        tp_size=2,
    )

    assert kwargs["kv_canary"] == "on"
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
