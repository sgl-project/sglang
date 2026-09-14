import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor import model_runner as mr_mod
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.state_capturer import base as capture_base
from sglang.srt.state_capturer import routed_experts as capture_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeModelRunner:
    max_decode_logits_rows = ModelRunner.max_decode_logits_rows
    init_routed_experts_capturer = ModelRunner.init_routed_experts_capturer

    def __init__(self, *, initial_width: int, cuda_graph_bs: list[int]):
        self.initial_width = initial_width
        self.cuda_graph_bs = cuda_graph_bs

    def decode_num_tokens_per_req(self, *, num_draft_tokens=None):
        return self.initial_width if num_draft_tokens is None else num_draft_tokens


def _alignment_8_capture_bs(runner, width):
    return ([bs for bs in runner.cuda_graph_bs if bs * width % 8 == 0], [])


class TestModelRunnerDecodeRows(CustomTestCase):
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

    def test_routed_experts_buffer_fits_verification_tokens(self):
        cases = [
            # name, eager requests, graph requests, max width, prefill, DP, rows
            ("ordinary_decode", 64, 64, 1, 128, 1, 128),
            ("fixed_verify", 64, 64, 8, 128, 1, 512),
            ("adaptive_verify", 64, 64, 16, 128, 1, 1024),
            ("graph_padding", 3, 8, 8, 16, 1, 64),
            ("eager_padding", 8, 4, 8, 16, 1, 64),
            ("dp_verify", 64, 64, 8, 128, 2, 1024),
        ]
        for name, eager_bs, graph_bs, width, prefill, dp_size, rows in cases:
            with self.subTest(name=name):
                runner = _FakeModelRunner(
                    initial_width=min(width, 4), cuda_graph_bs=[1]
                )
                runner.is_draft_worker = False
                runner.max_running_requests = min(eager_bs, graph_bs)
                runner.req_to_token_pool = SimpleNamespace(
                    size=runner.max_running_requests
                )
                runner.max_token_pool_size = 4096
                runner.page_size = 1
                runner.device = "cpu"
                runner.model = SimpleNamespace()
                runner.model_config = SimpleNamespace(
                    hf_text_config=SimpleNamespace(
                        num_experts_per_tok=2, num_hidden_layers=2
                    )
                )
                capturers = []
                with (
                    patch.multiple(
                        mr_mod,
                        get_eager_max_batch_size=lambda _: eager_bs,
                        get_cuda_graph_max_batch_size=lambda _: graph_bs,
                        max_speculative_num_draft_tokens=lambda: width,
                        set_global_experts_capturer=capturers.append,
                    ),
                    patch.multiple(
                        capture_mod,
                        get_exec=lambda: SimpleNamespace(
                            features=SimpleNamespace(enable_return_routed_experts=True)
                        ),
                        get_schedule=lambda: SimpleNamespace(
                            chunked_prefill_size=prefill
                        ),
                        get_parallel=lambda: SimpleNamespace(dp_size=dp_size),
                        _is_scattered_a2a_backend=lambda: False,
                    ),
                    patch.object(capture_base, "BaseHostCache"),
                ):
                    runner.init_routed_experts_capturer()
                    routes = torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2)
                    capturer = capturers[0]
                    capturer.capture(1, routes)
                    torch.testing.assert_close(
                        capturer.device_cache.buffer[:, 1], routes
                    )


if __name__ == "__main__":
    unittest.main()
