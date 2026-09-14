from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.kv_canary.expected_inputs import ExpectedInputs
from sglang.srt.kv_canary.token_oracle.oracle import HashOracle
from sglang.srt.kv_canary.token_oracle.oracle_manager import TokenOracleManager
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kv_canary.fixtures import DEFAULT_DEVICE
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=1, suite="extra-a-test-1-gpu-small-amd")


class TestTokenOracleManager(CustomTestCase):
    def setUp(self) -> None:
        self.device = DEFAULT_DEVICE

    def test_fill_expected_inputs_expands_draft_extend_generalized_req_ids_per_token(
        self,
    ) -> None:
        """Verify EAGLE draft extend maps one request row to every draft token."""
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            spec_info=SimpleNamespace(num_tokens_per_req=4),
            rids_int=torch.tensor([3, 7], dtype=torch.int64, device=self.device),
            bootstrap_room_ids_int=None,
            input_ids=torch.tensor(
                [101, 102, 103, 104, 201, 202, 203, 204],
                dtype=torch.int64,
                device=self.device,
            ),
            positions=torch.arange(8, dtype=torch.int64, device=self.device),
            extend_seq_lens=torch.tensor([1, 1], dtype=torch.int64, device=self.device),
        )
        expected_inputs = ExpectedInputs.allocate(capacity=8, device=self.device)
        manager = TokenOracleManager(oracle=HashOracle(vocab_size=32000))

        manager.fill_expected_inputs(
            forward_batch=forward_batch,
            expected_inputs_out=expected_inputs,
        )

        # draft-extend-v2 is not an extend mode, so fill_expected_inputs records
        # deterministic oracle tokens (not input_ids). The req row [3, 7] expands
        # to num_tokens_per_req=4 draft tokens each -> one row per draft token.
        expected_generalized_req_ids = torch.tensor(
            [3, 3, 3, 3, 7, 7, 7, 7], dtype=torch.int64, device=self.device
        )
        expected_tokens = manager.oracle.expected_tokens(
            generalized_req_ids=expected_generalized_req_ids,
            positions=forward_batch.positions.to(torch.int64),
        )
        self.assertTrue(
            torch.equal(expected_inputs.tokens[:8], expected_tokens.to(torch.int64))
        )
        self.assertTrue(
            torch.equal(expected_inputs.positions[:8], forward_batch.positions)
        )


if __name__ == "__main__":
    unittest.main()
