"""Unit tests for NGRAM speculative decoding mamba state rollback.

Tests that `last_correct_step_indices` is correctly computed from
`accept_index` and `accept_lens`, and that the shared post-verify mamba
state commit helper updates hybrid linear-attention models.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNgramLastCorrectStepIndices(CustomTestCase):
    """Test the last_correct_step_indices computation logic."""

    def _compute_last_correct_step_indices(
        self,
        accept_indices: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor:
        """Replicate the computation from ngram_info.py verify()."""
        bs = accept_indices.shape[0]
        req_idx = torch.arange(bs, dtype=torch.int64, device=accept_indices.device)
        accept_indices_offset = (req_idx * draft_token_num).to(accept_indices.dtype)
        last_correct_step_indices = (
            accept_indices[req_idx, num_correct_drafts.to(torch.int64)]
            - accept_indices_offset
        )
        return last_correct_step_indices

    def test_linear_chain_all_accepted(self):
        """All draft tokens accepted in a linear chain (topk=1)."""
        bs, draft_token_num = 3, 5
        # Linear chain: accept_indices[i] = [i*5, i*5+1, ..., i*5+4]
        accept_indices = torch.stack(
            [
                torch.arange(
                    i * draft_token_num,
                    i * draft_token_num + draft_token_num,
                    dtype=torch.int32,
                )
                for i in range(bs)
            ]
        )
        # All 5 drafts accepted (last accepted = index 4)
        num_correct_drafts = torch.tensor([4, 4, 4], dtype=torch.int32)

        result = self._compute_last_correct_step_indices(
            accept_indices, num_correct_drafts, draft_token_num
        )
        # Last accepted step within each request's window should be 4
        expected = torch.tensor([4, 4, 4], dtype=torch.int32)
        self.assertTrue(torch.equal(result, expected))

    def test_linear_chain_partial_accept(self):
        """Partial acceptance in a linear chain with correct kernel output format.

        verify_tree_greedy fills accept_indices with the accepted positions:
        - num_correct_drafts[i] correct drafts + 1 bonus = num_accept_tokens
        - The first (num_correct_drafts[i] + 1) entries are valid global indices
        - The remaining entries are -1
        """
        bs, draft_token_num = 3, 5
        # Request 0: 2 correct drafts + bonus = 3 accepted (steps 0,1,2 in window)
        # Request 1: 0 correct drafts + bonus = 1 accepted (step 0 in window)
        # Request 2: 4 correct drafts + bonus = 5 accepted (steps 0,1,2,3,4 in window)
        accept_indices = torch.tensor(
            [
                [0, 1, 2, -1, -1],  # req 0: global indices 0,1,2 accepted
                [5, -1, -1, -1, -1],  # req 1: global index 5 accepted (bonus only)
                [10, 11, 12, 13, 14],  # req 2: all 5 accepted
            ],
            dtype=torch.int32,
        )
        num_correct_drafts = torch.tensor([2, 0, 4], dtype=torch.int32)

        result = self._compute_last_correct_step_indices(
            accept_indices, num_correct_drafts, draft_token_num
        )
        # Request 0: accept_indices[0, 2] = 2, offset = 0*5 = 0, step = 2-0 = 2
        # Request 1: accept_indices[1, 0] = 5, offset = 1*5 = 5, step = 5-5 = 0
        # Request 2: accept_indices[2, 4] = 14, offset = 2*5 = 10, step = 14-10 = 4
        expected = torch.tensor([2, 0, 4], dtype=torch.int32)
        self.assertTrue(torch.equal(result, expected))

    def test_tree_structure_non_sequential(self):
        """Tree structure (topk > 1) where accepted path is non-sequential."""
        bs, draft_token_num = 2, 6
        # Request 0: tree path goes through positions 0, 2, 5 (skipping linear chain)
        # Request 1: tree path goes through positions 6, 7, 10
        accept_indices = torch.tensor(
            [
                [0, 2, 5, -1, -1, -1],  # 2 correct drafts + bonus
                [6, 7, 10, -1, -1, -1],  # 2 correct drafts + bonus
            ],
            dtype=torch.int32,
        )
        num_correct_drafts = torch.tensor([2, 2], dtype=torch.int32)

        result = self._compute_last_correct_step_indices(
            accept_indices, num_correct_drafts, draft_token_num
        )
        # Request 0: accept_indices[0, 2] = 5, offset = 0, step = 5
        # Request 1: accept_indices[1, 2] = 10, offset = 6, step = 4
        expected = torch.tensor([5, 4], dtype=torch.int32)
        self.assertTrue(torch.equal(result, expected))

    def test_single_request_zero_drafts(self):
        """Edge case: zero correct drafts (only bonus token accepted)."""
        bs, draft_token_num = 1, 4
        accept_indices = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32)
        num_correct_drafts = torch.tensor([0], dtype=torch.int32)

        result = self._compute_last_correct_step_indices(
            accept_indices, num_correct_drafts, draft_token_num
        )
        # accept_indices[0, 0] = 0, offset = 0, step = 0
        expected = torch.tensor([0], dtype=torch.int32)
        self.assertTrue(torch.equal(result, expected))


class TestNgramMambaVerifyUpdate(CustomTestCase):
    """Test that NGRAM verify commits mamba states through the shared helper."""

    def _make_mock_target_worker(self, has_mambaish_config: bool):
        """Create a mock target worker with necessary model runner attributes."""
        target_worker = MagicMock()
        target_worker.model_runner.model = MagicMock()
        target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify = (
            MagicMock()
        )

        target_worker.model_runner.mambaish_config = (
            {"some": "config"} if has_mambaish_config else None
        )
        return target_worker

    def test_mamba_verify_update_called_with_correct_indices(self):
        """commit_mamba_states_after_verify passes correct last_correct_step_indices."""
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = self._make_mock_target_worker(has_mambaish_config=True)
        batch = MagicMock()
        batch.forward_mode.is_idle.return_value = False
        batch.mamba_track_indices = None
        batch.seq_lens = torch.tensor([10, 20, 30], dtype=torch.int32)
        accept_lens = torch.tensor([3, 1, 5], dtype=torch.int32)
        accept_index = torch.tensor(
            [
                [0, 1, 2, -1, -1],
                [5, -1, -1, -1, -1],
                [10, 11, 12, 13, 14],
            ],
            dtype=torch.int32,
        )

        commit_mamba_states_after_verify(
            target_worker,
            batch,
            accept_lens,
            accept_index,
            draft_token_num=5,
        )

        update_call = (
            target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify
        )
        update_call.assert_called_once()
        call_kwargs = update_call.call_args[1]
        self.assertTrue(
            torch.equal(
                call_kwargs["last_correct_step_indices"],
                torch.tensor([2, 0, 4], dtype=torch.int32),
            )
        )
        self.assertIsNone(call_kwargs["mamba_track_indices"])
        self.assertIsNone(call_kwargs["mamba_steps_to_track"])

    def test_mamba_verify_update_not_called_for_non_mamba_model(self):
        """Verify the commit helper is a no-op when mambaish_config is None."""
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = self._make_mock_target_worker(has_mambaish_config=False)
        batch = MagicMock()
        batch.forward_mode.is_idle.return_value = False
        batch.mamba_track_indices = None
        accept_lens = torch.tensor([1], dtype=torch.int32)
        accept_index = torch.tensor([[0, -1, -1, -1, -1]], dtype=torch.int32)

        commit_mamba_states_after_verify(
            target_worker,
            batch,
            accept_lens,
            accept_index,
            draft_token_num=5,
        )

        update_call = (
            target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify
        )
        update_call.assert_not_called()

    def test_mamba_verify_update_with_track_indices(self):
        """commit_mamba_states_after_verify computes mamba_steps_to_track."""
        from unittest.mock import patch

        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = self._make_mock_target_worker(has_mambaish_config=True)
        batch = MagicMock()
        batch.forward_mode.is_idle.return_value = False
        batch.mamba_track_indices = torch.tensor([100, 200], dtype=torch.int64)
        # seq_lens before verify; helper adds accept_lens to compute the post-verify lengths.
        batch.seq_lens = torch.tensor([253, 128], dtype=torch.int32)
        accept_lens = torch.tensor([4, 3], dtype=torch.int32)
        accept_index = torch.tensor(
            [
                [0, 1, 2, 3, -1],
                [5, 6, 7, -1, -1],
            ],
            dtype=torch.int32,
        )

        with patch(
            "sglang.srt.speculative.spec_utils.get_global_server_args",
            return_value=MagicMock(mamba_track_interval=256),
        ):
            commit_mamba_states_after_verify(
                target_worker,
                batch,
                accept_lens,
                accept_index,
                draft_token_num=5,
            )

        update_call = (
            target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify
        )
        update_call.assert_called_once()
        call_kwargs = update_call.call_args[1]
        self.assertTrue(
            torch.equal(
                call_kwargs["last_correct_step_indices"],
                torch.tensor([3, 2], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                call_kwargs["mamba_steps_to_track"],
                torch.tensor([2, -1], dtype=torch.int32),
            )
        )


if __name__ == "__main__":
    unittest.main()
