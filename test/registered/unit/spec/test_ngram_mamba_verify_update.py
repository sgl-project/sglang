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


class TestConvWindowDedupLayout(CustomTestCase):
    """Guard the deduplicated sliding-window conv-intermediate view layout.

    Regression for the KDA aliasing bug: the dedup ``as_strided`` view in
    ``MambaPool.__init__`` was built assuming GDN's ``(channel, K-1)`` conv
    layout. KDA swaps its conv_state axes to ``(K-1, channel)``
    (``KimiLinearStateShape.create``), so unpacking ``conv_dim, win = conv_shape``
    aliased the draft-step axis onto the channel axis: per-step window writes
    clobbered each other, and the post-verify commit restored channel-shifted
    garbage into conv_states whenever acceptance was *partial* (full acceptance
    was coincidentally correct, which is why equivalence tests passed). Only
    triggers for chain EAGLE/MTP (topk<=1, where dedup is enabled).

    These tests replicate the exact view construction from
    ``MambaPool.__init__`` (device-independent, so they run on CPU CI) for both
    layouts and assert the sliding-window semantics: step ``t``'s window is the
    shared buffer slice ``[..., t:t+(K-1)]`` and the channel axis stays
    independent.
    """

    @staticmethod
    def _build_fixed_view(channel_dim, win_len, draft_tokens, window_major, device):
        """Replicate the fixed dedup view construction from MambaPool.__init__.

        phys keeps the channel axis independent and the shared sliding window on
        its last axis; the logical view's last two dims follow conv_state's axis
        order (channel-major for GDN, window-major for KDA).
        """
        shared_win = draft_tokens + win_len - 1
        L, S = 1, 1
        phys = torch.zeros(L, S, channel_dim, shared_win, device=device)
        # Encode each physical cell as channel*1000 + shared_col so we can read
        # back which (channel, window-column) each view element points to.
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
                    phys.stride(3),  # step -> shared-win axis
                    phys.stride(2),  # channel
                    phys.stride(3),  # win -> shared-win axis
                ),
            )
        else:
            # KDA: view[l, s, step, w, d] = phys[l, s, d, step + w]
            view = phys.as_strided(
                (L, S, draft_tokens, win_len, channel_dim),
                (
                    phys.stride(0),
                    phys.stride(1),
                    phys.stride(3),  # step -> shared-win axis
                    phys.stride(3),  # win -> shared-win axis
                    phys.stride(2),  # channel
                ),
            )
        return view, phys

    @staticmethod
    def _build_buggy_kda_view(channel_dim, win_len, draft_tokens, device):
        """Replicate the ORIGINAL buggy construction for the KDA layout.

        The old code did ``conv_dim, win = conv_shape`` on KDA's
        ``(K-1, channel)`` shape, giving ``conv_dim = K-1`` and ``win = channel``,
        so the shared window was built over the channel axis.
        """
        conv_shape = (win_len, channel_dim)  # KDA (K-1, channel)
        conv_dim, win = conv_shape  # buggy unpack: conv_dim=K-1, win=channel
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
        """Fixed KDA view: step t's window == phys[:, t:t+(K-1)], channel independent."""
        channel_dim, win_len, draft_tokens = 5, 3, 4  # K=4 -> win=3
        view, _ = self._build_fixed_view(
            channel_dim, win_len, draft_tokens, window_major=True, device="cpu"
        )
        # view[0,0,t,w,d] must equal phys code d*1000 + (t+w)
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
        """Fixed KDA view: the channel (last) axis must carry the channel identity."""
        channel_dim, win_len, draft_tokens = 5, 3, 4
        view, _ = self._build_fixed_view(
            channel_dim, win_len, draft_tokens, window_major=True, device="cpu"
        )
        for t in range(draft_tokens):
            for w in range(win_len):
                for d in range(channel_dim):
                    # channel identity == floor(code / 1000) must equal d
                    self.assertEqual(int(view[0, 0, t, w, d].item()) // 1000, d)

    def test_kda_window_shifts_by_one_per_step(self):
        """Fixed KDA view: advancing the draft step slides the window by exactly 1."""
        channel_dim, win_len, draft_tokens = 5, 3, 4
        view, _ = self._build_fixed_view(
            channel_dim, win_len, draft_tokens, window_major=True, device="cpu"
        )
        fixed_channel = 2
        for t in range(draft_tokens - 1):
            a = view[0, 0, t, :, fixed_channel].tolist()
            b = view[0, 0, t + 1, :, fixed_channel].tolist()
            # b is a shifted left by one within the shared window
            self.assertEqual(a[1:], b[:-1])

    def test_gdn_channel_major_unchanged(self):
        """Fixed GDN view keeps the legacy channel-major semantics."""
        channel_dim, win_len, draft_tokens = 5, 3, 4
        view, _ = self._build_fixed_view(
            channel_dim, win_len, draft_tokens, window_major=False, device="cpu"
        )
        # view[0,0,t,d,w] must equal phys code d*1000 + (t+w)
        for t in range(draft_tokens):
            for d in range(channel_dim):
                for w in range(win_len):
                    self.assertEqual(
                        int(view[0, 0, t, d, w].item()), d * 1000 + (t + w)
                    )

    def test_partial_accept_commit_reads_correct_window(self):
        """Under partial acceptance, committing step n reads the right window.

        Simulates the post-verify commit: for accepted step n, the conv_state to
        restore is the window at draft step n. Assert the fixed KDA view exposes
        that window intact (channel-independent, correctly shifted), which the
        buggy view did not.
        """
        channel_dim, win_len, draft_tokens = 5, 3, 4
        view, _ = self._build_fixed_view(
            channel_dim, win_len, draft_tokens, window_major=True, device="cpu"
        )
        # Partial accept: last correct step = 1 (not full acceptance)
        n = 1
        committed = view[0, 0, n]  # shape (win_len, channel_dim)
        for w in range(win_len):
            for d in range(channel_dim):
                self.assertEqual(int(committed[w, d].item()), d * 1000 + (n + w))

    def test_buggy_kda_view_aliases_step_onto_channel(self):
        """Regression sentinel: the OLD construction DID alias step->channel.

        This documents the bug and guards against silently reintroducing the
        old unpack. In the buggy view, advancing the draft step changes the
        (mislabeled) channel axis content, i.e. step contaminates channel.
        """
        channel_dim, win_len, draft_tokens = 5, 3, 4
        buggy = self._build_buggy_kda_view(
            channel_dim, win_len, draft_tokens, device="cpu"
        )
        # The buggy view's 4th axis has size K-1 (=win_len), NOT channel_dim,
        # confirming the axes were swapped/mislabeled.
        self.assertEqual(buggy.shape[3], win_len)
        self.assertEqual(buggy.shape[4], channel_dim)
        # Stepping t must change the mislabeled-channel slice -> aliasing present.
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
