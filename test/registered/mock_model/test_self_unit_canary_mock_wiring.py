from __future__ import annotations

import dataclasses
import unittest

import torch

from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.token_oracle.oracle import HashOracle
from sglang.srt.kv_canary.token_oracle.sampler import install_oracle_sampler
from sglang.srt.model_executor.forward_batch_info import _stable_hash_rid_i64
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mock_model_utils import mock_model_server_args, mock_model_server_env
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
    rids_int: torch.Tensor


def _scalar_expected_token(oracle: HashOracle, *, req_id: int, position: int) -> int:
    out = oracle.expected_tokens(
        req_ids=torch.tensor([req_id], dtype=torch.int64),
        positions=torch.tensor([position], dtype=torch.int64),
    )
    return int(out.tolist()[0])


class TestFillExpectedInputs(CustomTestCase):
    def test_fill_expected_inputs_decode_one_token_per_req(self) -> None:
        """Verify decode mode fills one expected token per request."""
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
            rids_int=torch.tensor(
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
        """Verify extend mode fills expected tokens from per-request sequence lengths."""
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
            rids_int=torch.tensor([hashed_a, hashed_b], dtype=torch.int64),
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
        """Verify filling zero expected tokens leaves the output buffer unchanged."""
        hook = install_oracle_sampler(oracle=HashOracle(vocab_size=100))

        rid_a = "req-a"
        rid_b = "req-b"
        fb = _StubForwardBatch(
            input_ids=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.tensor([5, 7], dtype=torch.int64),
            forward_mode=_StubForwardMode(extend=False),
            extend_seq_lens=None,
            rids_int=torch.tensor(
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


class TestMockModelServerLaunchHelpers(CustomTestCase):
    def test_mock_model_server_args_adds_canary_defaults(self) -> None:
        """Verify mock model launch args include KV canary defaults before user args."""
        args = mock_model_server_args("--tp", "2")

        self.assertIn("--load-format", args)
        self.assertIn("dummy", args)
        self.assertIn("--sampling-backend", args)
        self.assertIn("token_oracle", args)
        self.assertIn("--kv-canary", args)
        self.assertIn("raise", args)
        self.assertEqual(args[-2:], ["--tp", "2"])

    def test_mock_model_server_env_enables_input_check_by_default(self) -> None:
        """Verify mock model launch env enables canary input checking by default."""
        env = mock_model_server_env()

        self.assertEqual(env["SGLANG_KV_CANARY_INPUT_CHECK"], "1")
        self.assertEqual(env["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"], "1")

    def test_mock_model_server_env_can_disable_input_check(self) -> None:
        """Verify mock model launch env can disable canary input checking."""
        env = mock_model_server_env(input_check_enabled=False)

        self.assertEqual(env["SGLANG_KV_CANARY_INPUT_CHECK"], "0")
        self.assertEqual(env["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"], "1")


if __name__ == "__main__":
    unittest.main()
