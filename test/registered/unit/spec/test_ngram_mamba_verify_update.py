import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNgramLastCorrectStepIndices(CustomTestCase):
    def _compute_last_correct_step_indices(
        self,
        accept_indices: torch.Tensor,
        num_correct_drafts: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor:
        bs = accept_indices.shape[0]
        req_idx = torch.arange(bs, dtype=torch.int64, device=accept_indices.device)
        accept_indices_offset = (req_idx * draft_token_num).to(accept_indices.dtype)
        last_correct_step_indices = (
            accept_indices[req_idx, num_correct_drafts.to(torch.int64)]
            - accept_indices_offset
        )
        return last_correct_step_indices

    def test_linear_chain_all_accepted(self):
        bs, draft_token_num = 3, 5
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
        num_correct_drafts = torch.tensor([4, 4, 4], dtype=torch.int32)

        result = self._compute_last_correct_step_indices(
            accept_indices, num_correct_drafts, draft_token_num
        )
        expected = torch.tensor([4, 4, 4], dtype=torch.int32)
        self.assertTrue(torch.equal(result, expected))

    def test_linear_chain_partial_accept(self):
        bs, draft_token_num = 3, 5
        accept_indices = torch.tensor(
            [
                [0, 1, 2, -1, -1],
                [5, -1, -1, -1, -1],
                [10, 11, 12, 13, 14],
            ],
            dtype=torch.int32,
        )
        num_correct_drafts = torch.tensor([2, 0, 4], dtype=torch.int32)

        result = self._compute_last_correct_step_indices(
            accept_indices, num_correct_drafts, draft_token_num
        )
        expected = torch.tensor([2, 0, 4], dtype=torch.int32)
        self.assertTrue(torch.equal(result, expected))

    def test_tree_structure_non_sequential(self):
        bs, draft_token_num = 2, 6
        accept_indices = torch.tensor(
            [
                [0, 2, 5, -1, -1, -1],
                [6, 7, 10, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        num_correct_drafts = torch.tensor([2, 2], dtype=torch.int32)

        result = self._compute_last_correct_step_indices(
            accept_indices, num_correct_drafts, draft_token_num
        )
        expected = torch.tensor([5, 4], dtype=torch.int32)
        self.assertTrue(torch.equal(result, expected))

    def test_single_request_zero_drafts(self):
        bs, draft_token_num = 1, 4
        accept_indices = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32)
        num_correct_drafts = torch.tensor([0], dtype=torch.int32)

        result = self._compute_last_correct_step_indices(
            accept_indices, num_correct_drafts, draft_token_num
        )
        expected = torch.tensor([0], dtype=torch.int32)
        self.assertTrue(torch.equal(result, expected))


class TestNgramMambaVerifyUpdate(CustomTestCase):
    def _make_mock_target_worker(self):
        target_worker = MagicMock()
        target_worker.model_runner.model = MagicMock()
        target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify = (
            MagicMock()
        )
        return target_worker

    def test_mamba_verify_update_called_with_correct_indices(self):
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = self._make_mock_target_worker()
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

        with patch(
            "sglang.srt.speculative.spec_utils.mambaish_config",
            return_value={"some": "config"},
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
                torch.tensor([2, 0, 4], dtype=torch.int32),
            )
        )
        self.assertIsNone(call_kwargs["mamba_track_indices"])
        self.assertIsNone(call_kwargs["mamba_steps_to_track"])

    def test_mamba_verify_update_not_called_for_non_mamba_model(self):
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = self._make_mock_target_worker()
        batch = MagicMock()
        batch.forward_mode.is_idle.return_value = False
        batch.mamba_track_indices = None
        accept_lens = torch.tensor([1], dtype=torch.int32)
        accept_index = torch.tensor([[0, -1, -1, -1, -1]], dtype=torch.int32)

        with patch(
            "sglang.srt.speculative.spec_utils.mambaish_config",
            return_value=None,
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
        update_call.assert_not_called()

    def test_mamba_verify_update_with_track_indices(self):
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = self._make_mock_target_worker()
        batch = MagicMock()
        batch.forward_mode.is_idle.return_value = False
        batch.mamba_track_indices = torch.tensor([100, 200], dtype=torch.int64)
        # Only the first request crosses the 256-token tracking boundary.
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
            "sglang.srt.speculative.spec_utils.mambaish_config",
            return_value={"some": "config"},
        ), patch(
            "sglang.srt.speculative.spec_utils.get_server_args",
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


class TestConvWindowDedupLayout(CustomTestCase):
    """KDA stores conv_state as (K-1, channel), unlike GDN; partial-accept
    commits must preserve that layout in the overlapping view.
    """

    @staticmethod
    def _build_fixed_view(channel_dim, win_len, draft_tokens, window_major, device):
        shared_win = draft_tokens + win_len - 1
        L, S = 1, 1
        phys = torch.zeros(L, S, channel_dim, shared_win, device=device)
        # Encoding both coordinates makes axis aliasing observable.
        for c in range(channel_dim):
            for w in range(shared_win):
                phys[0, 0, c, w] = c * 1000 + w
        if not window_major:
            # GDN: view[l, s, step, d, w] = phys[l, s, d, step + w]
            view = phys.as_strided(
                (L, S, draft_tokens, channel_dim, win_len),
                (
                    phys.stride(0),
                    phys.stride(1),
                    phys.stride(3),
                    phys.stride(2),
                    phys.stride(3),
                ),
            )
        else:
            # KDA: view[l, s, step, w, d] = phys[l, s, d, step + w]
            view = phys.as_strided(
                (L, S, draft_tokens, win_len, channel_dim),
                (
                    phys.stride(0),
                    phys.stride(1),
                    phys.stride(3),
                    phys.stride(3),
                    phys.stride(2),
                ),
            )
        return view, phys

    @staticmethod
    def _build_buggy_kda_view(channel_dim, win_len, draft_tokens, device):
        """Preserve the former axis swap so the regression test distinguishes
        the corrected view from the broken one.
        """
        conv_shape = (win_len, channel_dim)
        conv_dim, win = conv_shape
        shared_win = draft_tokens + win - 1
        L, S = 1, 1
        phys = torch.zeros(L, S, conv_dim, shared_win, device=device)
        for c in range(conv_dim):
            for w in range(shared_win):
                phys[0, 0, c, w] = c * 1000 + w
        view = phys.as_strided(
            (L, S, draft_tokens, conv_dim, win),
            (
                phys.stride(0),
                phys.stride(1),
                phys.stride(3),
                phys.stride(2),
                phys.stride(3),
            ),
        )
        return view

    def test_kda_window_major_sliding_window(self):
        channel_dim, win_len, draft_tokens = 5, 3, 4
        view, _ = self._build_fixed_view(
            channel_dim, win_len, draft_tokens, window_major=True, device="cpu"
        )
        for t in range(draft_tokens):
            for w in range(win_len):
                for d in range(channel_dim):
                    got = int(view[0, 0, t, w, d].item())
                    self.assertEqual(
                        got,
                        d * 1000 + (t + w),
                        msg=f"KDA view alias at step={t} w={w} d={d}",
                    )

    def test_kda_channel_axis_independent(self):
        channel_dim, win_len, draft_tokens = 5, 3, 4
        view, _ = self._build_fixed_view(
            channel_dim, win_len, draft_tokens, window_major=True, device="cpu"
        )
        for t in range(draft_tokens):
            for w in range(win_len):
                for d in range(channel_dim):
                    self.assertEqual(int(view[0, 0, t, w, d].item()) // 1000, d)

    def test_kda_window_shifts_by_one_per_step(self):
        channel_dim, win_len, draft_tokens = 5, 3, 4
        view, _ = self._build_fixed_view(
            channel_dim, win_len, draft_tokens, window_major=True, device="cpu"
        )
        fixed_channel = 2
        for t in range(draft_tokens - 1):
            a = view[0, 0, t, :, fixed_channel].tolist()
            b = view[0, 0, t + 1, :, fixed_channel].tolist()
            self.assertEqual(a[1:], b[:-1])

    def test_gdn_channel_major_unchanged(self):
        channel_dim, win_len, draft_tokens = 5, 3, 4
        view, _ = self._build_fixed_view(
            channel_dim, win_len, draft_tokens, window_major=False, device="cpu"
        )
        for t in range(draft_tokens):
            for d in range(channel_dim):
                for w in range(win_len):
                    self.assertEqual(
                        int(view[0, 0, t, d, w].item()), d * 1000 + (t + w)
                    )

    def test_partial_accept_commit_reads_correct_window(self):
        channel_dim, win_len, draft_tokens = 5, 3, 4
        view, _ = self._build_fixed_view(
            channel_dim, win_len, draft_tokens, window_major=True, device="cpu"
        )
        n = 1
        committed = view[0, 0, n]
        for w in range(win_len):
            for d in range(channel_dim):
                self.assertEqual(int(committed[w, d].item()), d * 1000 + (n + w))

    def test_buggy_kda_view_aliases_step_onto_channel(self):
        channel_dim, win_len, draft_tokens = 5, 3, 4
        buggy = self._build_buggy_kda_view(
            channel_dim, win_len, draft_tokens, device="cpu"
        )
        self.assertEqual(buggy.shape[3], win_len)
        self.assertEqual(buggy.shape[4], channel_dim)
        aliased = False
        for c in range(buggy.shape[3]):
            if buggy[0, 0, 0, c, :].tolist() != buggy[0, 0, 1, c, :].tolist():
                aliased = True
                break
        self.assertTrue(
            aliased,
            "expected the buggy KDA view to alias the draft-step axis onto the "
            "channel axis",
        )


if __name__ == "__main__":
    unittest.main()
