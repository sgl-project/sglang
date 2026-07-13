import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeSpecAlgorithm:
    def __init__(self):
        self.calls = []

    def is_speculative(self):
        return True

    def get_num_tokens_per_bs_for_target_verify(
        self, num_draft_tokens, is_draft_worker
    ):
        self.calls.append((num_draft_tokens, is_draft_worker))
        return num_draft_tokens


class _FakeModelRunner:
    decode_num_tokens_per_bs = ModelRunner.decode_num_tokens_per_bs
    max_decode_logits_rows = ModelRunner.max_decode_logits_rows

    def __init__(self):
        self.server_args = SimpleNamespace(
            speculative_num_draft_tokens=4,
            max_speculative_num_draft_tokens=8,
        )
        self.spec_algorithm = _FakeSpecAlgorithm()
        self.is_draft_worker = False


class TestModelRunnerDecodeRows(unittest.TestCase):
    def test_max_decode_logits_rows_uses_adaptive_max_draft_tokens(self):
        runner = _FakeModelRunner()

        with patch(
            "sglang.srt.model_executor.model_runner.get_batch_sizes_to_capture",
            return_value=([1, 32], []),
        ) as mock_get_batch_sizes:
            self.assertEqual(runner.max_decode_logits_rows(), 256)

        self.assertEqual(runner.spec_algorithm.calls, [(8, False)])
        mock_get_batch_sizes.assert_called_once_with(runner, 8)


if __name__ == "__main__":
    unittest.main()
