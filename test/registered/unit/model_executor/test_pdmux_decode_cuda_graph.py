import unittest

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Backend:
    def on_after_cuda_graph_warmup(self):
        pass


class TestPDMuxDecodeCudaGraph(unittest.TestCase):
    def test_post_warmup_hook_belongs_to_selected_decode_backend(self):
        prefill_backend = _Backend()
        decode_backend = _Backend()

        hook = DecodeCudaGraphRunner._get_post_warmup_hook(decode_backend)

        self.assertIs(hook.__self__, decode_backend)
        self.assertIsNot(hook.__self__, prefill_backend)


if __name__ == "__main__":
    unittest.main()
