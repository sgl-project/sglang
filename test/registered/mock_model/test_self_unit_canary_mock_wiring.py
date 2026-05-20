"""OracleSamplerHook.fill_expected_inputs writes oracle-derived (token, position) into canary
placeholders, and apply_mock_model_defaults pre-fills the dependent ServerArgs flags when
mock_model is opted in.
"""

from __future__ import annotations

import dataclasses

import torch

from sglang.srt.kv_canary.mock_model.args_modifier import (
    apply_mock_model_defaults,
)
from sglang.srt.kv_canary.mock_model.oracle import ScriptedOracle
from sglang.srt.kv_canary.mock_model.sampler import install_oracle_sampler
from sglang.test.ci.ci_register import register_cuda_ci

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
    tokens_out = torch.zeros(8, dtype=torch.int32)
    positions_out = torch.zeros(8, dtype=torch.int32)

    hook.fill_expected_inputs(
        forward_batch=fb,
        expected_input_tokens_out=tokens_out,
        expected_input_positions_out=positions_out,
    )

    assert tokens_out[:2].tolist() == [111, 333]
    assert positions_out[:2].tolist() == [10, 20]


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
    tokens_out = torch.zeros(8, dtype=torch.int32)
    positions_out = torch.zeros(8, dtype=torch.int32)

    hook.fill_expected_inputs(
        forward_batch=fb,
        expected_input_tokens_out=tokens_out,
        expected_input_positions_out=positions_out,
    )

    assert tokens_out[:4].tolist() == [11, 22, 33, 99]
    assert positions_out[:4].tolist() == [0, 1, 2, 0]


def test_fill_expected_inputs_zero_tokens_is_noop_but_stashes_req_pool() -> None:
    hook = install_oracle_sampler(oracle=ScriptedOracle(table={}))

    fb = _StubForwardBatch(
        input_ids=torch.empty(0, dtype=torch.int64),
        positions=torch.empty(0, dtype=torch.int64),
        req_pool_indices=torch.tensor([5, 7], dtype=torch.int64),
        forward_mode=_StubForwardMode(extend=False),
        extend_seq_lens=None,
    )
    tokens_out = torch.zeros(4, dtype=torch.int32)
    positions_out = torch.zeros(4, dtype=torch.int32)

    hook.fill_expected_inputs(
        forward_batch=fb,
        expected_input_tokens_out=tokens_out,
        expected_input_positions_out=positions_out,
    )

    assert hook._req_pool_indices_per_row == [5, 7]


def test_apply_mock_model_defaults_no_op_when_disabled() -> None:
    from sglang.srt.server_args import ServerArgs

    original = ServerArgs(model_path="dummy", mock_model_enabled=False)
    result = apply_mock_model_defaults(original)

    assert result is original
    assert result.load_format == "auto"


def test_apply_mock_model_defaults_fills_holes_when_enabled() -> None:
    from sglang.srt.server_args import ServerArgs

    original = ServerArgs(model_path="dummy", mock_model_enabled=True)
    result = apply_mock_model_defaults(original)

    assert result.load_format == "dummy"
    assert result.num_hidden_layers_override == 1
    assert result.sampling_backend == "oracle"
    assert result.kv_canary == "raise"
    assert result.kv_canary_input_check_mode == "ON"


def test_apply_mock_model_defaults_preserves_user_overrides() -> None:
    from sglang.srt.server_args import ServerArgs

    original = ServerArgs(
        model_path="dummy",
        mock_model_enabled=True,
        num_hidden_layers_override=4,
        kv_canary="on",
    )
    result = apply_mock_model_defaults(original)

    assert result.num_hidden_layers_override == 4
    assert result.kv_canary == "on"
    assert result.load_format == "dummy"


def test_apply_mock_model_defaults_is_idempotent() -> None:
    from sglang.srt.server_args import ServerArgs

    original = ServerArgs(model_path="dummy", mock_model_enabled=True)
    first = apply_mock_model_defaults(original)
    second = apply_mock_model_defaults(first)

    assert first.load_format == second.load_format
    assert first.sampling_backend == second.sampling_backend
    assert first.kv_canary == second.kv_canary
