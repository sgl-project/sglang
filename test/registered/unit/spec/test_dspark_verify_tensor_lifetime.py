import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
    TargetVerifyResult,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=1, stage="stage-b", runner_config="1-gpu-small-amd")


class TestDSparkVerifyTensorLifetime(unittest.TestCase):
    @staticmethod
    def _executor():
        executor = object.__new__(TargetVerifyExecutor)
        executor.verify_num_draft_tokens = 6
        executor.model_runner = SimpleNamespace(device=torch.device("cpu"))
        return executor

    @staticmethod
    def _batch(old_cache_loc):
        return SimpleNamespace(
            input_ids=torch.tensor([1], dtype=torch.int64),
            out_cache_loc=old_cache_loc,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([16], dtype=torch.int64),
            seq_lens_cpu=None,
            seq_lens_sum=16,
        )

    def test_non_compact_records_old_cache_before_rebind(self):
        executor = self._executor()
        executor._verify_backend_self_adds_seq_lens = MagicMock(return_value=True)
        old_cache_loc = torch.tensor([3], dtype=torch.int64)
        verify_cache_loc = torch.tensor([4, 5], dtype=torch.int64)
        batch = self._batch(old_cache_loc)
        expected = TargetVerifyResult(
            logits_output=object(),
            can_run_cuda_graph=True,
            verify_forward_batch=object(),
        )
        calls = []

        def record_prepared(batch_arg, verify_input, stream):
            self.assertIs(batch_arg.out_cache_loc, old_cache_loc)
            calls.append("record-old")

        def forward_prepared(**kwargs):
            self.assertIs(batch.out_cache_loc, verify_cache_loc)
            calls.append("prepare-forward")
            return expected

        executor._forward_prepared_verify = MagicMock(side_effect=forward_prepared)
        verify_window = SimpleNamespace(
            positions_2d=torch.tensor([[16, 17]], dtype=torch.int64),
            verify_cache_loc=verify_cache_loc,
        )

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "record_stream_for_v2_verify",
            side_effect=record_prepared,
        ), patch(
            "torch.get_device_module",
            return_value=SimpleNamespace(current_stream=lambda: object()),
        ):
            result = executor.run_non_compact(
                batch=batch,
                draft_input=SimpleNamespace(nxt_kv_lens_cpu=None),
                verify_ids_2d=torch.tensor([[7, 8]], dtype=torch.int64),
                verify_window=verify_window,
                sampling_info=None,
            )

        self.assertIs(result, expected)
        self.assertEqual(calls, ["record-old", "prepare-forward"])

    def test_ragged_records_old_cache_before_rebind(self):
        executor = self._executor()
        old_cache_loc = torch.tensor([3], dtype=torch.int64)
        verify_cache_loc = torch.tensor([4, 5], dtype=torch.int64)
        batch = self._batch(old_cache_loc)
        expected = TargetVerifyResult(
            logits_output=object(),
            can_run_cuda_graph=True,
            verify_forward_batch=object(),
        )
        calls = []

        def record_prepared(batch_arg, verify_input, stream):
            self.assertIs(batch_arg.out_cache_loc, old_cache_loc)
            calls.append("record-old")

        def forward_prepared(**kwargs):
            self.assertIs(batch.out_cache_loc, verify_cache_loc)
            calls.append("prepare-forward")
            return expected

        executor._forward_prepared_verify = MagicMock(side_effect=forward_prepared)
        layout = SimpleNamespace(verify_lens_cpu=None, verify_lens=torch.tensor([2]))
        ragged_window = SimpleNamespace(
            positions=torch.tensor([16, 17], dtype=torch.int64),
            verify_cache_loc=verify_cache_loc,
            verify_ids=torch.tensor([7, 8], dtype=torch.int64),
        )

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "record_stream_for_v2_verify",
            side_effect=record_prepared,
        ), patch(
            "torch.get_device_module",
            return_value=SimpleNamespace(current_stream=lambda: object()),
        ):
            result = executor._run_ragged(
                batch=batch,
                layout=layout,
                ragged_window=ragged_window,
                sampling_info=None,
            )

        self.assertIs(result, expected)
        self.assertEqual(calls, ["record-old", "prepare-forward"])

    def test_prepare_records_new_bindings_before_target_forward(self):
        executor = self._executor()
        forward_stream = object()
        new_input_ids = torch.tensor([7, 8], dtype=torch.int64)
        new_cache_loc = torch.tensor([4, 5], dtype=torch.int64)
        batch = self._batch(torch.tensor([3], dtype=torch.int64))
        batch.out_cache_loc = new_cache_loc
        verify_forward_batch = object()
        calls = []

        verify_input = MagicMock()

        def prepare_for_verify(batch_arg, target_worker):
            batch_arg.input_ids = new_input_ids
            calls.append("prepare")
            return verify_forward_batch, True

        verify_input.prepare_for_verify.side_effect = prepare_for_verify
        target_out = SimpleNamespace(logits_output=object(), can_run_cuda_graph=True)
        executor.target_worker = MagicMock()

        def target_forward(**kwargs):
            calls.append("target-forward")
            return target_out

        executor.target_worker.forward_batch_generation.side_effect = target_forward

        def record_new(tensors, stream):
            self.assertEqual(tensors, (new_input_ids, new_cache_loc))
            self.assertIs(stream, forward_stream)
            calls.append("record-new")

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "record_stream_each",
            side_effect=record_new,
        ), patch(
            "torch.get_device_module",
            return_value=SimpleNamespace(current_stream=lambda: forward_stream),
        ):
            result = executor._forward_prepared_verify(
                batch=batch,
                verify_input=verify_input,
                seq_lens_cpu_backup=None,
                seq_lens_sum_backup=16,
            )

        self.assertIs(result.logits_output, target_out.logits_output)
        self.assertTrue(result.can_run_cuda_graph)
        self.assertIs(result.verify_forward_batch, verify_forward_batch)
        self.assertEqual(calls, ["prepare", "record-new", "target-forward"])


if __name__ == "__main__":
    unittest.main()
