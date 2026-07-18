import unittest
from types import SimpleNamespace

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
        runner.capture_hidden_mode = CaptureHiddenMode.NULL
        runner.capture_num_tokens = [4, 16]
        runner.max_num_tokens = 16
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
            return_logprob=False,
            input_ids=list(range(num_tokens)),
        )

    def test_rejects_more_than_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertFalse(runner.can_run_graph(self._make_forward_batch(5)))

    def test_accepts_two_x_token_padding(self):
        runner = self._make_runner()

        self.assertTrue(runner.can_run_graph(self._make_forward_batch(8)))


if __name__ == "__main__":
    unittest.main()
