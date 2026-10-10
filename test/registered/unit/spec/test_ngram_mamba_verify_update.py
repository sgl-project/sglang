import unittest
from unittest.mock import MagicMock, patch

import torch

import sglang.srt
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=23, suite="base-a-test-cpu")


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
