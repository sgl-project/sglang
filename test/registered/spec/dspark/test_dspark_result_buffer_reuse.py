import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _make_epilogue(device):
    epilogue = DsparkVerifyEpilogue.__new__(DsparkVerifyEpilogue)
    epilogue.correct_len_buf = torch.zeros((1,), dtype=torch.int32, device=device)
    epilogue._result_copy_done_event = None
    epilogue._result_copy_pending = False
    return epilogue


class TestDsparkResultBufferReuse(CustomTestCase):
    def test_result_copy_event_is_consumed_once_at_reuse(self):
        epilogue = _make_epilogue("cpu")
        event = Mock()
        stream = Mock()
        device_module = Mock()
        device_module.Event.return_value = event
        device_module.current_stream.return_value = stream

        with patch("torch.get_device_module", return_value=device_module):
            epilogue.record_result_copy_done()
            event.record.assert_called_once_with()

            with self.assertRaisesRegex(RuntimeError, "copied again"):
                epilogue.record_result_copy_done()

            epilogue.wait_for_result_copy_done()
            stream.wait_event.assert_called_once_with(event)
            self.assertFalse(epilogue._result_copy_pending)

            epilogue.wait_for_result_copy_done()
            self.assertEqual(stream.wait_event.call_count, 1)

    def test_target_verify_waits_at_the_buffer_reuse_point(self):
        order = []
        executor = TargetVerifyExecutor.__new__(TargetVerifyExecutor)
        executor._target_is_dsv41 = False
        executor.verify_num_draft_tokens = 3
        executor.verify_epilogue = SimpleNamespace(
            wait_for_result_copy_done=lambda: order.append("wait")
        )
        executor.target_worker = SimpleNamespace(
            forward_batch_generation=lambda **kwargs: (
                order.append("forward")
                or SimpleNamespace(logits_output=Mock(), can_run_cuda_graph=True)
            )
        )
        verify_input = SimpleNamespace(
            live_seq_lens_cpu=torch.tensor([8]),
            prepare_for_verify=lambda batch, worker: (
                order.append("prepare") or Mock(),
                None,
            ),
        )
        batch = SimpleNamespace(reqs=[], seq_lens_cpu=torch.tensor([8]), seq_lens_sum=8)

        executor._forward_prepared_verify(
            batch=batch,
            verify_input=verify_input,
            seq_lens_cpu_backup=batch.seq_lens_cpu,
            seq_lens_sum_backup=batch.seq_lens_sum,
        )

        self.assertEqual(order, ["prepare", "wait", "forward"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_delayed_d2h_finishes_before_persistent_buffer_reuse(self):
        device = torch.device("cuda")
        epilogue = _make_epilogue(device)
        epilogue.out_tokens_buf = torch.tensor(
            [[11, 12, 13], [21, 22, 23]], dtype=torch.int64, device=device
        )
        epilogue.commit_lens_buf = torch.tensor(
            [2, 3], dtype=torch.int32, device=device
        )
        expected_tokens = epilogue.out_tokens_buf.cpu().reshape(-1)
        expected_lens = epilogue.commit_lens_buf.cpu()

        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=epilogue.out_tokens_buf.reshape(-1),
            accept_lens=epilogue.commit_lens_buf,
            copy_done=torch.cuda.Event(),
            persistent_result_copy_owner=epilogue,
        )
        forward_stream = torch.cuda.current_stream(device)
        copy_stream = torch.cuda.Stream(device=device)
        copy_stream.wait_stream(forward_stream)
        with torch.cuda.stream(copy_stream):
            torch.cuda._sleep(20_000_000)
            result.copy_to_cpu(return_logprob=False, return_hidden_states=False)

        epilogue.wait_for_result_copy_done()
        epilogue.out_tokens_buf.fill_(99)
        epilogue.commit_lens_buf.fill_(1)
        torch.cuda.synchronize(device)

        torch.testing.assert_close(result.next_token_ids, expected_tokens)
        torch.testing.assert_close(result.accept_lens, expected_lens)


if __name__ == "__main__":
    unittest.main()
