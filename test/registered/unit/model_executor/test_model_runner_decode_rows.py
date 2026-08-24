import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeModelRunner:
    max_decode_logits_rows = ModelRunner.max_decode_logits_rows

    def __init__(
        self,
        *,
        config_path: str,
        initial_width: int,
        cuda_graph_bs: list[int],
        max_draft_tokens: int,
    ):
        self.initial_width = initial_width
        self.server_args = SimpleNamespace(
            speculative_adaptive=True,
            speculative_adaptive_config=config_path,
            max_speculative_num_draft_tokens=max_draft_tokens,
        )
        self.cuda_graph_bs = cuda_graph_bs

    def decode_num_tokens_per_req(self, *, num_draft_tokens=None):
        return self.initial_width if num_draft_tokens is None else num_draft_tokens


def _alignment_8_capture_bs(runner, width):
    return ([bs for bs in runner.cuda_graph_bs if bs * width % 8 == 0], [])


class TestModelRunnerDecodeRows(unittest.TestCase):
    def test_adaptive_sizing_covers_a_wider_candidate_width(self):
        """The shared logits buffer is sized for the widest adaptive candidate
        width: bs 12 at width 6 needs 72 rows."""
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            f.write('{"1":{"candidate_steps":[3,5]}}')
            f.flush()
            runner = _FakeModelRunner(
                config_path=f.name,
                initial_width=4,
                cuda_graph_bs=[4, 8, 12],
                max_draft_tokens=6,
            )
            with patch(
                "sglang.srt.model_executor.model_runner.get_batch_sizes_to_capture",
                side_effect=_alignment_8_capture_bs,
            ):
                self.assertEqual(runner.max_decode_logits_rows(), 72)


if __name__ == "__main__":
    unittest.main()
