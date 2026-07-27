import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPrefillCudaGraphPadding(CustomTestCase):
    def _make_runner(self):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner._is_full_backend = False
        runner.prefill_backend_name = Backend.TC_PIECEWISE
        runner.has_mha_companion_layers = False
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 32]
        runner.max_num_tokens = 32
        runner.enable_tbo_prefill_graph = False
        return runner

    def _make_forward_batch(self, num_tokens):
        return SimpleNamespace(
            batch_size=1,
            input_embeds=None,
            replace_embeds=None,
            mm_inputs=None,
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            global_num_tokens_cpu=None,
            can_run_dp_breakable_cuda_graph=False,
            can_run_tbo=False,
            return_logprob=False,
            input_ids=list(range(num_tokens)),
        )

    def test_rejects_more_than_four_x_token_padding(self):
        runner = self._make_runner()

        self.assertFalse(runner.can_run_graph(self._make_forward_batch(5)))

    def test_accepts_four_x_token_padding(self):
        runner = self._make_runner()

        self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))

    def test_full_graph_request_slots_shrink_with_token_bucket(self):
        runner = self._make_runner()
        runner._is_full_backend = True
        runner._capture_req_slots = 64

        self.assertEqual(runner._capture_req_slots_for_tokens(4), 4)
        self.assertEqual(runner._capture_req_slots_for_tokens(32), 32)
        self.assertEqual(runner._capture_req_slots_for_tokens(128), 64)

    def test_nested_body_output_trims_only_token_axis_tensors(self):
        output = (
            (
                torch.arange(32),
                torch.tensor(7),
            ),
            [
                torch.arange(64).reshape(32, 2),
                torch.arange(5),
            ],
        )

        trimmed = PrefillCudaGraphRunner._trim_replayed_body_output(
            output,
            raw_num_tokens=5,
            static_num_tokens=32,
        )

        self.assertIsInstance(trimmed, tuple)
        self.assertEqual(trimmed[0][0].shape, (5,))
        self.assertEqual(trimmed[0][1].shape, ())
        self.assertIsInstance(trimmed[1], list)
        self.assertEqual(trimmed[1][0].shape, (5, 2))
        self.assertEqual(trimmed[1][1].shape, (5,))


if __name__ == "__main__":
    unittest.main()
