import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDecodeCudaGraphOverflowWarning(unittest.TestCase):
    def test_warns_once_on_overflow(self):
        runner = SimpleNamespace()

        with self.assertLogs(
            "sglang.srt.model_executor.model_runner", level="WARNING"
        ) as logs:
            ModelRunner._maybe_warn_decode_cuda_graph_overflow(runner, 44, 32)
            ModelRunner._maybe_warn_decode_cuda_graph_overflow(runner, 51, 32)

        matching = [
            message
            for message in logs.output
            if "largest captured CUDA graph shape" in message
        ]
        self.assertEqual(len(matching), 1)
        self.assertIn("44", matching[0])
        self.assertIn("32", matching[0])

    def test_does_not_warn_within_capture_range(self):
        runner = SimpleNamespace()

        with self.assertNoLogs(
            "sglang.srt.model_executor.model_runner", level="WARNING"
        ):
            ModelRunner._maybe_warn_decode_cuda_graph_overflow(runner, 32, 32)


if __name__ == "__main__":
    unittest.main()
