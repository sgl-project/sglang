import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner import base_cuda_graph_runner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeModelRunner:
    max_decode_logits_rows = ModelRunner.max_decode_logits_rows

    def __init__(self, *, initial_width: int, cuda_graph_bs: list[int]):
        self.initial_width = initial_width
        self.cuda_graph_bs = cuda_graph_bs

    def decode_num_tokens_per_req(self, *, num_draft_tokens=None):
        return self.initial_width if num_draft_tokens is None else num_draft_tokens


def _alignment_8_capture_bs(runner, width):
    return ([bs for bs in runner.cuda_graph_bs if bs * width % 8 == 0], [])


class TestModelRunnerDecodeRows(unittest.TestCase):
    def test_adaptive_sizing_covers_a_wider_candidate_width(self):
        """The shared logits buffer is sized for the widest adaptive candidate
        width: bs 12 at width 6 needs 72 rows."""
        runner = _FakeModelRunner(initial_width=4, cuda_graph_bs=[4, 8, 12])
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            f.write('{"1":{"candidate_steps":[3,5]}}')
            f.flush()
            spec = SimpleNamespace(
                speculative_adaptive=True, speculative_adaptive_config=f.name
            )
            with (
                patch(
                    "sglang.srt.model_executor.model_runner.get_spec", return_value=spec
                ),
                patch(
                    "sglang.srt.model_executor.model_runner.max_speculative_num_draft_tokens",
                    return_value=6,
                ),
                patch(
                    "sglang.srt.model_executor.model_runner.get_batch_sizes_to_capture",
                    side_effect=_alignment_8_capture_bs,
                ),
            ):
                self.assertEqual(runner.max_decode_logits_rows(), 72)


class _FakeDllmModelRunner:
    max_decode_logits_rows = ModelRunner.max_decode_logits_rows

    def __init__(self, *, block_size: int):
        self.block_size = block_size
        self.server_args = SimpleNamespace()
        self.req_to_token_pool = SimpleNamespace(size=4096)

    def decode_num_tokens_per_req(self, *, num_draft_tokens=None):
        return self.block_size


class TestDllmDecodeRows(unittest.TestCase):
    def _max_rows(self, *, requires_separate_context_encoding: bool) -> int:
        runner = _FakeDllmModelRunner(block_size=256)
        dllm_config = SimpleNamespace(
            block_size=256,
            max_running_requests=2,
            requires_separate_context_encoding=requires_separate_context_encoding,
        )
        exec_ctx = SimpleNamespace(
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    decode=SimpleNamespace(bs=[1, 2, 4, 8, 256])
                )
            ),
            overlap=SimpleNamespace(enable_two_batch_overlap=False),
        )
        flags = SimpleNamespace(capture=SimpleNamespace(enable_torch_compile=False))
        spec = SimpleNamespace(speculative_adaptive=False)
        mr_module = "sglang.srt.model_executor.model_runner"
        with (
            patch(f"{mr_module}.get_spec", return_value=spec),
            patch(f"{mr_module}.max_speculative_num_draft_tokens", return_value=None),
            patch.object(base_cuda_graph_runner, "get_exec", return_value=exec_ctx),
            patch.object(base_cuda_graph_runner, "get_flags", return_value=flags),
            patch.object(
                base_cuda_graph_runner,
                "get_cuda_graph_batch_size_alignment",
                return_value=1,
            ),
            patch.object(
                base_cuda_graph_runner,
                "get_cuda_graph_max_batch_size",
                side_effect=lambda num_max_requests: num_max_requests,
            ),
            patch.object(
                base_cuda_graph_runner.DllmConfig,
                "from_server_args",
                return_value=dllm_config,
            ),
        ):
            return runner.max_decode_logits_rows()

    def test_separate_context_rows_follow_dllm_request_cap(self):
        """The shared logits buffer must not be sized for decode batches a
        separate-context dLLM never captures (a 64 GiB OOM on DiffusionGemma)."""
        self.assertEqual(self._max_rows(requires_separate_context_encoding=True), 512)

    def test_other_dllm_rows_keep_full_capture_range(self):
        self.assertEqual(
            self._max_rows(requires_separate_context_encoding=False), 256 * 256
        )


if __name__ == "__main__":
    unittest.main()
