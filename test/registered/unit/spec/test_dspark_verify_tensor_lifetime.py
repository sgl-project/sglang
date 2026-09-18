import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.speculative.dspark_components.dspark_verify import (
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


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

    @patch("torch.get_device_module")
    @patch(
        "sglang.srt.speculative.dspark_components.dspark_verify."
        "record_stream_for_v2_verify"
    )
    def test_records_tensors_before_rebind(self, record_stream, get_device_module):
        executor = self._executor()
        executor._verify_backend_self_adds_seq_lens = MagicMock(return_value=True)
        old_cache_loc = torch.tensor([3], dtype=torch.int64)
        verify_cache_loc = torch.tensor([4, 5], dtype=torch.int64)
        batch = self._batch(old_cache_loc)
        expected = object()

        def assert_old_binding(batch_arg, *_):
            self.assertIs(batch_arg.out_cache_loc, old_cache_loc)

        def assert_new_binding(**_):
            self.assertIs(batch.out_cache_loc, verify_cache_loc)
            return expected

        record_stream.side_effect = assert_old_binding
        get_device_module.return_value.current_stream.return_value = object()
        executor._forward_prepared_verify = MagicMock(side_effect=assert_new_binding)

        result = executor.run_non_compact(
            batch=batch,
            draft_input=SimpleNamespace(nxt_kv_lens_cpu=None),
            verify_ids_2d=torch.tensor([[7, 8]]),
            verify_window=SimpleNamespace(
                positions_2d=torch.tensor([[16, 17]]),
                verify_cache_loc=verify_cache_loc,
            ),
            sampling_info=None,
        )

        self.assertIs(result, expected)
        record_stream.assert_called_once()

    def test_prepare_records_new_bindings_before_target_forward(self):
        executor = self._executor()
        forward_stream = object()
        new_input_ids = torch.tensor([7, 8], dtype=torch.int64)
        new_cache_loc = torch.tensor([4, 5], dtype=torch.int64)
        batch = self._batch(torch.tensor([3], dtype=torch.int64))
        batch.out_cache_loc = new_cache_loc
        verify_forward_batch = object()
        verify_input = MagicMock()

        def prepare_for_verify(batch_arg, target_worker):
            batch_arg.input_ids = new_input_ids
            return verify_forward_batch, True

        verify_input.prepare_for_verify.side_effect = prepare_for_verify
        target_out = SimpleNamespace(logits_output=object(), can_run_cuda_graph=True)
        executor.target_worker = MagicMock()
        executor.target_worker.forward_batch_generation.return_value = target_out

        with (
            patch(
                "sglang.srt.speculative.dspark_components.dspark_verify."
                "record_stream_each"
            ) as record_stream,
            patch(
                "torch.get_device_module",
                return_value=SimpleNamespace(current_stream=lambda: forward_stream),
            ),
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
        record_stream.assert_called_once_with(
            (new_input_ids, new_cache_loc), forward_stream
        )

    @patch(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2."
        "torch.get_device_module"
    )
    def test_wrapper_publishes_final_shared_read_event(self, get_device_module):
        for is_extend in (True, False):
            with self.subTest(is_extend=is_extend):
                calls = []
                event = SimpleNamespace(record=lambda: calls.append("record"))
                get_device_module.return_value = SimpleNamespace(Event=lambda: event)
                worker = object.__new__(DSparkWorkerV2)
                worker.device = torch.device("cpu")
                worker.model_runner = SimpleNamespace(shared_read_done_event=object())
                worker._verify_planner = MagicMock()
                worker._observers = MagicMock()
                worker._forward_prefill = MagicMock(
                    side_effect=lambda *_: calls.append("forward") or "prefill"
                )
                worker._forward_decode = MagicMock(
                    side_effect=lambda *_: calls.append("forward") or "decode"
                )
                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(is_extend=lambda: is_extend),
                    is_extend_in_batch=False,
                )

                result = worker.forward_batch_generation(batch)

                self.assertEqual(calls, ["forward", "record"])
                self.assertEqual(result, "prefill" if is_extend else "decode")
                self.assertIs(worker.model_runner.shared_read_done_event, event)


if __name__ == "__main__":
    unittest.main()
