from __future__ import annotations

import dataclasses
import json
import os
import unittest

import torch
from utils import mock_model_engine_kwargs

from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.token_oracle.oracle import HashOracle
from sglang.srt.kv_canary.token_oracle.sampler import install_oracle_sampler
from sglang.srt.model_executor.forward_batch_info import _stable_hash_rid_i64
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

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
    rids_hashed: torch.Tensor


def _scalar_expected_token(oracle: HashOracle, *, req_id: int, position: int) -> int:
    out = oracle.expected_tokens(
        req_ids=torch.tensor([req_id], dtype=torch.int64),
        positions=torch.tensor([position], dtype=torch.int64),
    )
    return int(out.tolist()[0])


class TestFillExpectedInputs(CustomTestCase):
    def test_fill_expected_inputs_decode_one_token_per_req(self) -> None:
        oracle = HashOracle(vocab_size=32000)
        hook = install_oracle_sampler(oracle=oracle)

        rid_a = "req-a"
        rid_b = "req-b"
        fb = _StubForwardBatch(
            input_ids=torch.tensor([0, 0], dtype=torch.int64),
            positions=torch.tensor([10, 20], dtype=torch.int64),
            req_pool_indices=torch.tensor([5, 7], dtype=torch.int64),
            forward_mode=_StubForwardMode(extend=False),
            extend_seq_lens=None,
            rids_hashed=torch.tensor(
                [_stable_hash_rid_i64(rid_a), _stable_hash_rid_i64(rid_b)],
                dtype=torch.int64,
            ),
        )
        expected_inputs = ExpectedInputs.allocate(
            capacity=8, device=torch.device("cpu")
        )

        hook.fill_expected_inputs(
            forward_batch=fb,
            expected_inputs_out=expected_inputs,
        )

        self.assertEqual(
            expected_inputs.tokens[:2].tolist(),
            [
                _scalar_expected_token(
                    oracle, req_id=_stable_hash_rid_i64(rid_a), position=10
                ),
                _scalar_expected_token(
                    oracle, req_id=_stable_hash_rid_i64(rid_b), position=20
                ),
            ],
        )
        self.assertEqual(expected_inputs.positions[:2].tolist(), [10, 20])

    def test_fill_expected_inputs_extend_uses_extend_seq_lens(self) -> None:
        oracle = HashOracle(vocab_size=32000)
        hook = install_oracle_sampler(oracle=oracle)

        rid_a = "req-a"
        rid_b = "req-b"
        hashed_a = _stable_hash_rid_i64(rid_a)
        hashed_b = _stable_hash_rid_i64(rid_b)
        fb = _StubForwardBatch(
            input_ids=torch.tensor([0, 0, 0, 0], dtype=torch.int64),
            positions=torch.tensor([0, 1, 2, 0], dtype=torch.int64),
            req_pool_indices=torch.tensor([5, 7], dtype=torch.int64),
            forward_mode=_StubForwardMode(extend=True),
            extend_seq_lens=torch.tensor([3, 1], dtype=torch.int64),
            rids_hashed=torch.tensor([hashed_a, hashed_b], dtype=torch.int64),
        )
        expected_inputs = ExpectedInputs.allocate(
            capacity=8, device=torch.device("cpu")
        )

        hook.fill_expected_inputs(
            forward_batch=fb,
            expected_inputs_out=expected_inputs,
        )

        self.assertEqual(
            expected_inputs.tokens[:4].tolist(),
            [
                _scalar_expected_token(oracle, req_id=hashed_a, position=0),
                _scalar_expected_token(oracle, req_id=hashed_a, position=1),
                _scalar_expected_token(oracle, req_id=hashed_a, position=2),
                _scalar_expected_token(oracle, req_id=hashed_b, position=0),
            ],
        )
        self.assertEqual(expected_inputs.positions[:4].tolist(), [0, 1, 2, 0])

    def test_fill_expected_inputs_zero_tokens_is_noop(
        self,
    ) -> None:
        hook = install_oracle_sampler(oracle=HashOracle(vocab_size=100))

        rid_a = "req-a"
        rid_b = "req-b"
        fb = _StubForwardBatch(
            input_ids=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.tensor([5, 7], dtype=torch.int64),
            forward_mode=_StubForwardMode(extend=False),
            extend_seq_lens=None,
            rids_hashed=torch.tensor(
                [_stable_hash_rid_i64(rid_a), _stable_hash_rid_i64(rid_b)],
                dtype=torch.int64,
            ),
        )
        expected_inputs = ExpectedInputs.allocate(
            capacity=4, device=torch.device("cpu")
        )
        initial_tokens = expected_inputs.tokens.clone()

        hook.fill_expected_inputs(
            forward_batch=fb,
            expected_inputs_out=expected_inputs,
        )

        self.assertEqual(expected_inputs.tokens.tolist(), initial_tokens.tolist())


class TestMockModelEngineKwargs(CustomTestCase):
    def setUp(self) -> None:
        self._prior_input_check = os.environ.get("SGLANG_KV_CANARY_INPUT_CHECK")

    def tearDown(self) -> None:
        if self._prior_input_check is None:
            os.environ.pop("SGLANG_KV_CANARY_INPUT_CHECK", None)
        else:
            os.environ["SGLANG_KV_CANARY_INPUT_CHECK"] = self._prior_input_check

    def test_mock_model_engine_kwargs_merges_json_override(self) -> None:
        kwargs = mock_model_engine_kwargs(
            json_model_override_args='{"rope_theta": 1000.0}',
        )

        merged = json.loads(kwargs["json_model_override_args"])
        self.assertEqual(merged["num_hidden_layers"], 1)
        self.assertEqual(merged["rope_theta"], 1000.0)

    def test_mock_model_engine_kwargs_speculative_disables_input_check(self) -> None:
        os.environ["SGLANG_KV_CANARY_INPUT_CHECK"] = "1"
        kwargs = mock_model_engine_kwargs(speculative_algorithm="EAGLE")
        self.assertEqual(kwargs["speculative_algorithm"], "EAGLE")
        self.assertEqual(os.environ["SGLANG_KV_CANARY_INPUT_CHECK"], "0")


if __name__ == "__main__":
    unittest.main()
