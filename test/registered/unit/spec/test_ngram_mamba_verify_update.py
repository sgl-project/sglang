import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=23, suite="base-a-test-cpu")


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
        mamba_pool = target_worker.model_runner.req_to_token_pool.mamba_pool
        mamba_pool.replayssm_spec_fold = False
        mamba_pool.replayssm_cache_base = None
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

        with (
            patch(
                "sglang.srt.speculative.spec_utils.mambaish_config",
                return_value={"some": "config"},
            ),
            patch(
                "sglang.srt.speculative.spec_utils.mamba_track_grid",
                return_value=256,
            ),
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


class TestPPReplaySSMVerifySourceRows(CustomTestCase):
    @staticmethod
    def _spec_state():
        return SimpleNamespace(
            temporal=torch.empty((1, 8, 2, 2), dtype=torch.float32),
            replayssm_d=torch.empty((1, 8, 4, 2), dtype=torch.float32),
            replayssm_k=torch.empty((1, 8, 4, 2), dtype=torch.float32),
            replayssm_rawv=torch.empty((1, 8, 4, 2), dtype=torch.float32),
            replayssm_rawk=torch.empty((1, 8, 3, 4, 2), dtype=torch.float32),
            replayssm_g=torch.empty((1, 8, 4, 2), dtype=torch.float32),
            replayssm_beta=torch.empty((1, 8, 4), dtype=torch.float32),
            conv=[torch.empty((1, 8, 2, 3), dtype=torch.float32)],
            intermediate_conv_window=[
                torch.empty((1, 32, 4, 2, 3), dtype=torch.float32)
            ],
        )

    def test_fold_helpers_read_pp_request_rows(self):
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
            commit_gdn_replayssm_fold_after_verify,
        )
        from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
            commit_kda_replayssm_after_verify,
        )

        source_rows = torch.tensor([17, 23], dtype=torch.int64)
        destinations = torch.tensor([5, 7], dtype=torch.int32)
        steps = torch.tensor([2, 0], dtype=torch.int32)
        accept_lens = torch.tensor([3, 1], dtype=torch.int32)

        cases = (
            (
                commit_gdn_replayssm_fold_after_verify,
                "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold",
                "commit_gdn_replayssm_fold_all_layers",
            ),
            (
                commit_kda_replayssm_after_verify,
                "sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode",
                "commit_kda_replayssm_spec_all_layers",
            ),
        )
        for commit, module, fold_name in cases:
            with (
                self.subTest(module=module),
                patch(f"{module}.{fold_name}"),
                patch(
                    "sglang.kernels.ops.mamba.mamba_state_scatter_triton."
                    "fused_conv_window_scatter_with_mask"
                ) as scatter,
            ):
                commit(
                    spec_state=self._spec_state(),
                    state_batch_indices=destinations,
                    accept_lens=accept_lens,
                    last_correct_step_indices=steps,
                    source_indices_raw=source_rows,
                )

                scatter.assert_called_once()
                torch.testing.assert_close(scatter.call_args.args[4], source_rows)

    def test_circular_commit_reads_pp_request_rows(self):
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = MagicMock()
        req_pool = target_worker.model_runner.req_to_token_pool
        req_pool.mamba_pool = SimpleNamespace(
            replayssm_spec_fold=False,
            replayssm_is_kda=False,
            replayssm_cache_base=torch.empty(1),
            replayssm_spec_write_pos=torch.empty(1),
            replayssm_is_flush=torch.empty(1),
        )
        req_pool.get_mamba_indices.return_value = torch.tensor(
            [5, 7], dtype=torch.int32
        )
        req_pool.get_speculative_mamba2_params_all_layers.return_value = (
            self._spec_state()
        )
        batch = MagicMock()
        batch.forward_mode.is_idle.return_value = False
        batch.req_pool_indices = torch.tensor([17, 23], dtype=torch.int64)
        batch.mamba_track_indices = None
        batch.seq_lens = torch.tensor([10, 20], dtype=torch.int32)
        accept_lens = torch.tensor([2, 1], dtype=torch.int32)
        accept_index = torch.tensor([[0, 1, -1], [3, -1, -1]], dtype=torch.int32)

        with (
            patch(
                "sglang.srt.speculative.spec_utils.mambaish_config",
                return_value={"some": "config"},
            ),
            patch(
                "sglang.srt.speculative.spec_utils.envs.SGLANG_ENABLE_PP_SPEC.get",
                return_value=True,
            ),
            patch(
                "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode."
                "commit_gdn_replayssm_spec"
            ),
            patch(
                "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode."
                "commit_gdn_replayssm_circular"
            ),
            patch(
                "sglang.kernels.ops.mamba.mamba_state_scatter_triton."
                "fused_conv_window_scatter_with_mask"
            ) as scatter,
        ):
            commit_mamba_states_after_verify(
                target_worker,
                batch,
                accept_lens,
                accept_index,
                draft_token_num=3,
            )

        scatter.assert_called_once()
        torch.testing.assert_close(scatter.call_args.args[4], batch.req_pool_indices)


class TestDelayedMambaCommitBatchPairing(CustomTestCase):
    def test_pp_forward_snapshot_keeps_live_rows_for_non_mamba_model(self):
        from sglang.srt.managers.scheduler_pp_mixin import (
            _pp_snapshot_forward_batch,
        )

        batch = MagicMock()
        batch.spec_algorithm.is_none.return_value = False
        batch.req_pool_indices = torch.tensor([5, 2, 7], dtype=torch.int32)
        snapshot = SimpleNamespace(req_pool_indices=batch.req_pool_indices)
        batch.copy.return_value = snapshot

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.mambaish_config",
            return_value=None,
        ):
            result = _pp_snapshot_forward_batch(batch)

        self.assertIs(result.req_pool_indices, batch.req_pool_indices)

    def test_pp_forward_snapshot_owns_rows_for_mamba_model(self):
        from sglang.srt.managers.scheduler_pp_mixin import (
            _pp_snapshot_forward_batch,
        )

        batch = MagicMock()
        batch.spec_algorithm.is_none.return_value = False
        batch.req_pool_indices = torch.tensor([5, 2, 7], dtype=torch.int32)
        snapshot = SimpleNamespace(req_pool_indices=batch.req_pool_indices)
        batch.copy.return_value = snapshot

        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.mambaish_config",
            return_value={"some": "config"},
        ):
            result = _pp_snapshot_forward_batch(batch)

        self.assertIsNot(result.req_pool_indices, batch.req_pool_indices)
        torch.testing.assert_close(result.req_pool_indices, batch.req_pool_indices)

    def test_flashinfer_gdn_positional_scratch_moves_to_request_rows(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            copy_verify_intermediate_rows,
        )

        destination = torch.zeros((8, 2, 3), dtype=torch.float32)
        positional = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
        rows = torch.tensor([5, 2, 7], dtype=torch.int32)

        copy_verify_intermediate_rows(destination, positional, rows)

        torch.testing.assert_close(destination[rows.long()], positional)

    def test_ple_commit_reads_stable_request_rows(self):
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )

        dst = torch.zeros((1, 8, 2), dtype=torch.float32)
        src = torch.zeros((1, 16, 4, 2), dtype=torch.float32)
        src[0, 11, 2] = torch.tensor([3.0, 7.0])

        HybridLinearAttnBackend._scatter_speculative_state_with_mask(
            dst,
            src,
            torch.tensor([5], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([11], dtype=torch.int64),
        )

        torch.testing.assert_close(dst[0, 5], torch.tensor([3.0, 7.0]))

    def test_pp_verify_scratch_uses_stable_request_rows_for_all_backends(self):
        from sglang.srt.layers.attention.linear.utils import (
            select_verify_intermediate_state_indices,
        )

        default = torch.arange(8, dtype=torch.int32)
        req_rows = torch.tensor([17, 23, 31], dtype=torch.int64)
        cache_indices = torch.tensor([4, -1, 9], dtype=torch.int32)

        with patch(
            "sglang.srt.layers.attention.linear.utils."
            "pp_spec_stable_rows_enabled",
            return_value=True,
        ):
            result = select_verify_intermediate_state_indices(
                default, req_rows, cache_indices >= 0, pool_size=64
            )

        torch.testing.assert_close(
            result, torch.tensor([17, 64, 31], dtype=torch.int32)
        )

    def test_pp_commit_runs_without_verify_kv_address(self):
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        scheduler = MagicMock()
        scheduler.pp_group.is_last_rank = False
        scheduler.tp_worker = MagicMock()
        batch = MagicMock()
        batch.seq_lens = torch.tensor([11, 22], dtype=torch.int64)
        batch.tree_cache = MagicMock()
        fwd_batch = MagicMock()
        fwd_batch.forward_mode.is_idle.return_value = False
        fwd_batch.seq_lens_cpu = torch.tensor([10, 20], dtype=torch.int64)
        outputs = MagicMock()
        outputs.tensors = {
            "spec_accept_index": torch.tensor([[0, 1], [2, -1]], dtype=torch.int32)
        }
        outputs.__getitem__.return_value = torch.tensor([2, 1], dtype=torch.int32)

        with (
            patch(
                "sglang.srt.speculative.spec_utils.commit_mamba_states_after_verify"
            ) as commit,
            patch(
                "sglang.srt.managers.scheduler_pp_mixin.get_spec",
                return_value=SimpleNamespace(speculative_num_draft_tokens=2),
            ),
        ):
            SchedulerPPMixin._pp_spec_compact_accept_kv(
                scheduler,
                batch,
                fwd_batch,
                ["a", "b"],
                ["a", "b"],
                None,
                outputs,
            )

        commit.assert_called_once()
        torch.testing.assert_close(
            commit.call_args.args[2], outputs["spec_accept_lens"]
        )

    def test_pp_recomposed_batch_requires_seq_len_snapshot(self):
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        scheduler = MagicMock()
        batch = MagicMock()
        batch.seq_lens = torch.tensor([11], dtype=torch.int64)
        fwd_batch = MagicMock()
        fwd_batch.forward_mode.is_idle.return_value = False
        fwd_batch.seq_lens_cpu = None
        outputs = MagicMock()
        outputs.tensors = {"spec_accept_index": torch.tensor([[0]], dtype=torch.int32)}

        with self.assertRaisesRegex(AssertionError, "forward-time seq_lens_cpu"):
            SchedulerPPMixin._pp_spec_compact_accept_kv(
                scheduler,
                batch,
                fwd_batch,
                ["before"],
                ["after"],
                None,
                outputs,
            )

    def test_request_slots_override_stale_forward_metadata(self):
        """A delayed PP relay must commit the batch that produced the accept result."""
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )

        backend = object.__new__(HybridLinearAttnBackend)
        req_pool = MagicMock()
        req_pool.mamba_pool = SimpleNamespace(replayssm_is_kda=False)
        mamba_caches = MagicMock()
        req_pool.get_speculative_mamba2_params_all_layers.return_value = mamba_caches
        req_pool.get_mamba_indices.return_value = torch.tensor(
            [41, 42, 43], dtype=torch.int32
        )

        linear_backend = MagicMock()
        linear_backend.req_to_token_pool = req_pool
        linear_backend.forward_metadata.mamba_cache_indices = torch.tensor(
            [99], dtype=torch.int32
        )
        linear_backend._translate_mamba_indices.side_effect = lambda value: value + 100
        linear_backend.accept_lens_pool = None
        backend.linear_attn_backend = linear_backend
        backend._update_ple_state_after_mtp_verify = MagicMock()

        last_steps = torch.tensor([2, 0, 3], dtype=torch.int32)
        req_pool_indices = torch.tensor([7, 8, 9], dtype=torch.int32)
        with (
            patch(
                "sglang.srt.layers.attention.hybrid_linear_attn_backend."
                "scatter_mamba_states_after_mtp_verify"
            ) as scatter,
            patch(
                "sglang.srt.layers.attention.hybrid_linear_attn_backend."
                "envs.SGLANG_ENABLE_PP_SPEC.get",
                return_value=True,
            ),
        ):
            backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=last_steps,
                mamba_track_indices=None,
                mamba_steps_to_track=None,
                model=None,
                req_pool_indices=req_pool_indices,
            )

        req_pool.get_mamba_indices.assert_called_once()
        torch.testing.assert_close(
            req_pool.get_mamba_indices.call_args.args[0], req_pool_indices
        )
        scatter.assert_called_once()
        torch.testing.assert_close(
            scatter.call_args.args[1],
            torch.tensor([141, 142, 143], dtype=torch.int32),
        )
        torch.testing.assert_close(scatter.call_args.args[2], last_steps)
        torch.testing.assert_close(
            scatter.call_args.kwargs["source_indices_tensor"], req_pool_indices
        )

    def test_non_pp_commit_keeps_forward_metadata_fast_path(self):
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )

        backend = object.__new__(HybridLinearAttnBackend)
        req_pool = MagicMock()
        req_pool.mamba_pool = SimpleNamespace(replayssm_is_kda=False)
        req_pool.get_speculative_mamba2_params_all_layers.return_value = MagicMock()

        linear_backend = MagicMock()
        linear_backend.req_to_token_pool = req_pool
        linear_backend.forward_metadata.mamba_cache_indices = torch.tensor(
            [9, 10, 11], dtype=torch.int32
        )
        linear_backend.accept_lens_pool = None
        backend.linear_attn_backend = linear_backend
        backend._update_ple_state_after_mtp_verify = MagicMock()

        last_steps = torch.tensor([2, 0, 3], dtype=torch.int32)
        req_pool_indices = torch.tensor([7, 8, 9], dtype=torch.int32)
        with (
            patch(
                "sglang.srt.layers.attention.hybrid_linear_attn_backend."
                "scatter_mamba_states_after_mtp_verify"
            ) as scatter,
            patch(
                "sglang.srt.layers.attention.hybrid_linear_attn_backend."
                "envs.SGLANG_ENABLE_PP_SPEC.get",
                return_value=False,
            ),
        ):
            backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=last_steps,
                mamba_track_indices=None,
                mamba_steps_to_track=None,
                model=None,
                req_pool_indices=req_pool_indices,
            )

        req_pool.get_mamba_indices.assert_not_called()
        linear_backend._translate_mamba_indices.assert_not_called()
        scatter.assert_called_once()
        torch.testing.assert_close(
            scatter.call_args.args[1], torch.tensor([9, 10, 11], dtype=torch.int32)
        )
        self.assertIsNone(scatter.call_args.kwargs["source_indices_tensor"])


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


class TestMtpVerifyHookSignature(CustomTestCase):
    """Every ``update_mamba_state_after_mtp_verify`` override must accept the full
    keyword call the spec workers make, or it raises TypeError at verify time on
    whatever hardware it serves.

    Parses sources rather than importing: the accelerator backends defining
    overrides are exactly the ones whose deps are absent on most hosts, so an
    import-based check would skip the cases that matter.
    """

    CALL_KWARGS = {
        "last_correct_step_indices",
        "mamba_track_indices",
        "mamba_steps_to_track",
        "model",
        "req_pool_indices",
    }
    HOOK = "update_mamba_state_after_mtp_verify"

    def test_all_overrides_accept_the_call_kwargs(self):
        import ast
        import pathlib

        srt = pathlib.Path(next(iter(sglang.srt.__path__)))
        found = []
        for path in srt.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name != self.HOOK:
                    continue
                args = node.args
                if args.kwarg is not None:
                    continue  # **kwargs passthrough accepts everything
                names = {a.arg for a in args.args} | {a.arg for a in args.kwonlyargs}
                found.append((path.relative_to(srt), node.lineno, names))

        self.assertTrue(found, f"no {self.HOOK} definitions found under sglang.srt")
        for rel, lineno, names in found:
            missing = self.CALL_KWARGS - names
            self.assertFalse(
                missing,
                f"{rel}:{lineno} {self.HOOK} is missing {sorted(missing)}; "
                "the spec workers call this hook by keyword.",
            )


if __name__ == "__main__":
    unittest.main()
