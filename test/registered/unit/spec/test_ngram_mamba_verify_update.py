import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=34, suite="base-a-test-cpu")


class TestNgramMambaVerifyUpdate(CustomTestCase):
    def _make_mock_target_worker(self):
        target_worker = MagicMock()
        target_worker.model_runner.model = MagicMock()
        target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify = (
            MagicMock()
        )
        mamba_pool = MagicMock()
        # The runner's KV pool router binds the state pool.
        router = target_worker.model_runner.token_to_kv_pool.layer_router
        router.part_view.return_value = mamba_pool
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
        state = torch.empty((1, 8, 4, 2), dtype=torch.float32)
        spec_state = MagicMock()
        for name in "temporal replayssm_d replayssm_k replayssm_rawv replayssm_g replayssm_beta".split():  # noqa: SIM905
            setattr(spec_state, name, state)
        spec_state.replayssm_rawk = state.unsqueeze(2)
        spec_state.conv = spec_state.intermediate_conv_window = [state]
        return spec_state

    def test_fold_helpers_read_pp_request_rows(self):
        from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_fold import (
            commit_gdn_replayssm_fold_after_verify,
        )
        from sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode import (
            commit_kda_replayssm_after_verify,
        )

        cases = (
            (
                commit_gdn_replayssm_fold_after_verify,
                "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_fold",
                "commit_gdn_replayssm_fold_all_layers",
            ),
            (
                commit_kda_replayssm_after_verify,
                "sglang.kernels.ops.attention.linear.kda_replayssm_spec_decode",
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
                source_rows = torch.tensor([17, 23])
                commit(
                    spec_state=self._spec_state(),
                    state_batch_indices=torch.tensor([5, 7]),
                    accept_lens=torch.tensor([3, 1]),
                    last_correct_step_indices=torch.tensor([2, 0]),
                    src_indices_raw=source_rows,
                )

                scatter.assert_called_once()
                torch.testing.assert_close(scatter.call_args.args[4], source_rows)

    def test_circular_commit_reads_pp_request_rows(self):
        from sglang.srt.layers.attention.mamba.mamba2_metadata import (
            ForwardStateSlots,
        )
        from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

        target_worker = MagicMock()
        target_worker.model_runner.attn_backend.forward_state_slots.return_value = (
            ForwardStateSlots(
                torch.tensor([5, 7], dtype=torch.int32),
                ring=torch.tensor([2, 3], dtype=torch.int32),
            )
        )
        mamba_pool = MagicMock()
        router = target_worker.model_runner.token_to_kv_pool.layer_router
        router.part_view.return_value = mamba_pool
        mamba_pool.replayssm_spec_fold = False
        mamba_pool.replayssm_is_kda = False
        mamba_pool.replayssm_cache_base = torch.empty(1)
        mamba_pool.get_speculative_mamba2_params_all_layers.return_value = (
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
                "sglang.srt.speculative.spec_utils.pp_spec_stable_rows_enabled",
                return_value=True,
            ),
            patch(
                "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode."
                "commit_gdn_replayssm_spec"
            ),
            patch(
                "sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode."
                "commit_gdn_replayssm_circular"
            ),
            patch(
                "sglang.kernels.ops.mamba.mamba_state_scatter_triton."
                "fused_conv_window_scatter_with_mask"
            ) as scatter,
            patch(
                "sglang.srt.layers.attention.hybrid_linear_attn_backend."
                "commit_ple_states_after_verify"
            ),
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
            "sglang.srt.layers.attention.linear.utils.pp_spec_stable_rows_enabled",
            return_value=True,
        ):
            result = select_verify_intermediate_state_indices(
                default, req_rows, cache_indices >= 0, pool_size=64
            )

        torch.testing.assert_close(
            result, torch.tensor([17, 64, 31], dtype=torch.int32)
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
        "state_slots",
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


def _commit_inputs():
    return dict(
        checkpoint_state=torch.zeros(1, 1, 1, 128, 128, dtype=torch.bfloat16),
        d_cache=torch.zeros(1, 1, 1, 16, 128, dtype=torch.bfloat16),
        k_cache=torch.zeros(1, 1, 1, 16, 128, dtype=torch.bfloat16),
        g_cache=torch.zeros(1, 1, 1, 16),
        d_residual_cache=None,
        k_residual_cache=None,
        state_batch_indices=torch.zeros(1, dtype=torch.int64),
        replay_indices=torch.zeros(1, dtype=torch.int64),
        write_pos=torch.zeros(1, dtype=torch.int32),
        cache_base=torch.zeros(1, dtype=torch.int32),
        is_flush=torch.zeros(1, dtype=torch.int8),
        accept_lens=torch.ones(1, dtype=torch.int32),
    )


class TestGdnTiledCommitDispatch(CustomTestCase):
    def test_supported_shapes_launch_the_tiled_commit(self):
        """SGLANG_OPT_GDN_REPLAYSSM_TILED_COMMIT: supported shapes launch the tiled
        kernel with the stock kernel's cursor strides, others the stock kernel."""
        from sglang.kernels.ops.attention.linear import (
            gdn_replayssm_spec_decode as replay,
        )

        for tiled in (False, True):
            with (
                self.subTest(tiled=tiled),
                patch.object(replay, "_use_tiled_commit", return_value=tiled),
                patch.object(
                    replay, "_finish_gdn_replayssm_circular_fold_kernel", MagicMock()
                ) as finish,
                patch.object(
                    replay, "gdn_replayssm_compact_commit_kernel", MagicMock()
                ) as stock,
                patch.object(
                    replay, "gdn_replayssm_compact_commit_tiled_kernel", MagicMock()
                ) as tiled_kernel,
            ):
                replay.commit_gdn_replayssm_circular(**_commit_inputs())
                self.assertEqual(tiled_kernel.__getitem__.call_count, int(tiled))
                self.assertEqual(stock.__getitem__.call_count, int(not tiled))
                self.assertEqual(finish.__getitem__.call_count, 1)
                launched = tiled_kernel if tiled else stock
                kw = launched.__getitem__.return_value.call_args.kwargs
                self.assertEqual(
                    (kw["STRIDE_WP"], kw["STRIDE_CB"], kw["STRIDE_FL"]), (1, 1, 1)
                )
                if tiled:
                    # 32-row V tiles, one row per request
                    grid = tiled_kernel.__getitem__.call_args.args[0]
                    self.assertEqual(grid[:2], (128 // 32, 1))
                    self.assertEqual(
                        (
                            kw["BV"],
                            kw["num_warps"],
                            kw["num_stages"],
                            kw["HEADS_PER_PROGRAM"],
                        ),
                        (32, 1, 1, 4),
                    )


def _rollback_rows(*, track: bool, src: bool = False):
    # As served: int32 working slots, int64 track slots and steps.
    return dict(
        slots=torch.tensor([3, 7, -1], dtype=torch.int32),
        steps=torch.tensor([2, 0, 1]),
        track=torch.tensor([10, -1, 12]) if track else None,
        track_steps=torch.tensor([1, -1, 0]) if track else None,
        src=torch.tensor([2, 0, 1]) if src else None,
    )


def _row_sets(rows):
    sets = [(rows["slots"], rows["steps"])]
    if rows["track"] is not None:
        sets.append((rows["track"], rows["track_steps"]))
    return sets


class TestCommitMultiScatterDispatch(CustomTestCase):
    """SGLANG_OPT_SPEC_COMMIT_MULTI_SCATTER: off, the stock scatter per pair and row
    set on the rows as given; on, one multi-pair launch for the conv windows and one
    for the PLE side states, or, with PP source rows or pairs the multi-pair launch
    does not take, the per-pair scatters on int32 rows cast once."""

    def _conv(self, *, multi, track, src=False, dtype=torch.bfloat16):
        from sglang.kernels.ops.attention.linear import (
            gdn_replayssm_spec_decode as replay,
        )
        from sglang.kernels.ops.mamba import mamba_state_scatter_triton as scatter

        rows = _rollback_rows(track=track, src=src)
        state = SimpleNamespace(
            conv=[torch.zeros(2, 16, 6, 3, dtype=dtype)],
            intermediate_conv_window=[torch.zeros(2, 4, 3, 6, 3, dtype=dtype)],
            temporal=None,
            replayssm_d=torch.zeros(2, 4, 1, 16, 8),
            replayssm_k=None,
            replayssm_g=None,
            replayssm_rawv=None,
            replayssm_rawk=None,
        )
        with (
            patch.object(replay, "_COMMIT_MULTI_SCATTER", multi),
            patch.object(replay, "commit_gdn_replayssm_spec"),
            patch.object(replay, "commit_gdn_replayssm_circular"),
            patch.object(scatter, "fused_conv_window_scatter_with_mask") as per_pair,
            patch.object(scatter, "fused_conv_window_scatter_multi") as fused,
        ):
            replay.commit_gdn_replayssm_after_verify(
                spec_state=state,
                write_pos=None,
                cache_base=None,
                is_flush=None,
                state_batch_indices=rows["slots"],
                replay_indices=None,
                accept_lens=None,
                last_correct_step_indices=rows["steps"],
                fold_every_commit=True,
                mamba_track_indices=rows["track"],
                mamba_steps_to_track=rows["track_steps"],
                src_indices_raw=rows["src"],
            )
        return rows, state, per_pair, fused

    def _ple(self, *, multi, track, src=False, eligible=True):
        from sglang.srt.layers.attention import hybrid_linear_attn_backend as hla
        from sglang.srt.mem_cache.ple_state_pool import NGRAM_PART, SHORT_CONV_PART

        rows = _rollback_rows(track=track, src=src)
        short_conv = SimpleNamespace(
            conv_state=torch.zeros(1, 16, 8, 9, dtype=torch.bfloat16),
            intermediate_conv_state=torch.zeros(1, 4, 3, 8, 9, dtype=torch.bfloat16),
        )
        ngram = SimpleNamespace(
            context=torch.zeros(16, 2, dtype=torch.int64),
            intermediate_context=torch.zeros(4, 3, 2, dtype=torch.int64),
        )
        router = SimpleNamespace(
            part_view={SHORT_CONV_PART: short_conv, NGRAM_PART: ngram}.get
        )
        with (
            patch.object(hla, "_COMMIT_MULTI_SCATTER", multi),
            # CPU tensors: the multi-pair launch itself takes CUDA pairs only.
            patch.object(hla, "_state_multi_eligible", return_value=eligible),
            patch.object(
                hla.HybridLinearAttnBackend, "_scatter_speculative_state_with_mask"
            ) as per_pair,
            patch.object(hla, "fused_state_scatter_multi") as fused,
        ):
            hla.commit_ple_states_after_verify(
                router,
                rows["slots"],
                rows["steps"],
                rows["track"],
                rows["track_steps"],
                rows["src"],
            )
        return rows, (short_conv, ngram), per_pair, fused

    def _assert_stock_rows(self, per_pair, rows, *, pairs):
        """One call per pair and row set (working, then track), on the rows as given."""
        sets = _row_sets(rows)
        calls = per_pair.call_args_list
        self.assertEqual(len(calls), pairs * len(sets))
        for i, (slots, steps) in enumerate(sets):
            for call in calls[i :: len(sets)]:
                self.assertIs(call.args[2], slots)
                self.assertIs(call.args[3], steps)
                self.assertIsNone(call.args[4])

    def _assert_int32_rows(self, per_pair, rows, *, pairs, src):
        """One call per pair and row set, on that set's int32 rows built once, all
        sharing one int32 source-row tensor."""
        sets = _row_sets(rows)
        calls = per_pair.call_args_list
        self.assertEqual(len(calls), pairs * len(sets))
        want_src = (rows["src"] if src else torch.arange(3)).to(torch.int32)
        self.assertTrue(torch.equal(calls[0].args[4], want_src))
        for i, (slots, steps) in enumerate(sets):
            set_calls = calls[i :: len(sets)]
            for call in set_calls:
                self.assertIs(call.args[4], calls[0].args[4])
                for got, first, want in zip(
                    call.args[2:4], set_calls[0].args[2:4], (slots, steps)
                ):
                    self.assertIs(got, first)
                    self.assertEqual(got.dtype, torch.int32)
                    self.assertTrue(torch.equal(got, want.to(torch.int32)))

    def test_conv_windows_take_one_launch(self):
        for track in (False, True):
            with self.subTest(track=track):
                rows, _, per_pair, fused = self._conv(multi=False, track=track)
                self.assertEqual(fused.call_count, 0)
                self._assert_stock_rows(per_pair, rows, pairs=1)

                rows, state, per_pair, fused = self._conv(multi=True, track=track)
                self.assertEqual((per_pair.call_count, fused.call_count), (0, 1))
                (pairs, *index_sets), _ = fused.call_args
                self.assertEqual(len(pairs), 1)
                self.assertIs(pairs[0][0], state.conv[0])
                self.assertIs(pairs[0][1], state.intermediate_conv_window[0])
                for got, name in zip(
                    index_sets, ("slots", "steps", "track", "track_steps")
                ):
                    self.assertIs(got, rows[name])

    def test_conv_windows_fall_back_to_int32_rows(self):
        # PP source rows, or FP32 conv windows (the multi-pair conv launch is BF16).
        for src, dtype in ((True, torch.bfloat16), (False, torch.float32)):
            with self.subTest(src=src, dtype=dtype):
                rows, _, per_pair, fused = self._conv(
                    multi=True, track=True, src=src, dtype=dtype
                )
                self.assertEqual(fused.call_count, 0)
                self._assert_int32_rows(per_pair, rows, pairs=1, src=src)

    def test_ple_side_states_take_one_launch(self):
        for track in (False, True):
            with self.subTest(track=track):
                rows, _, per_pair, fused = self._ple(multi=False, track=track)
                self.assertEqual(fused.call_count, 0)
                self._assert_stock_rows(per_pair, rows, pairs=2)

                rows, (short_conv, ngram), per_pair, fused = self._ple(
                    multi=True, track=track
                )
                self.assertEqual((per_pair.call_count, fused.call_count), (0, 1))
                (pairs, *index_sets), _ = fused.call_args
                self.assertEqual(len(pairs), 2)
                self.assertIs(pairs[0][0], short_conv.conv_state)
                self.assertIs(pairs[0][1], short_conv.intermediate_conv_state)
                self.assertEqual(pairs[1][0].shape, (1, 16, 2))
                self.assertEqual(pairs[1][0].data_ptr(), ngram.context.data_ptr())
                self.assertEqual(
                    pairs[1][1].data_ptr(), ngram.intermediate_context.data_ptr()
                )
                for got, name in zip(
                    index_sets, ("slots", "steps", "track", "track_steps")
                ):
                    self.assertIs(got, rows[name])

    def test_ple_side_states_fall_back_to_int32_rows(self):
        for src, eligible in ((True, True), (False, False)):
            with self.subTest(src=src, eligible=eligible):
                rows, _, per_pair, fused = self._ple(
                    multi=True, track=True, src=src, eligible=eligible
                )
                self.assertEqual(fused.call_count, 0)
                self._assert_int32_rows(per_pair, rows, pairs=2, src=src)


if __name__ == "__main__":
    unittest.main()
