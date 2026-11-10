"""CPU tests: BreakableCudaGraphBackend buffers for LogitsProcessorOutput."""

from __future__ import annotations

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _backend() -> BreakableCudaGraphBackend:
    return BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)


class TestBreakableCudaGraphLogitsBuffers(CustomTestCase):
    def test_alloc_slice_copy_keeps_python_side_fields(self):
        backend = _backend()
        warmup = LogitsProcessorOutput(
            next_token_logits=torch.arange(8, dtype=torch.float32).reshape(1, 8),
            hidden_states=torch.ones(1, 4),
            next_token_top_logprobs_val=[[0.1, 0.2]],
            customized_info={"k": ["v"]},
        )
        buf = backend._alloc_full_buffer(warmup, size=2)
        self.assertIsInstance(buf, LogitsProcessorOutput)
        self.assertEqual(tuple(buf.next_token_logits.shape), (2, 8))
        self.assertEqual(tuple(buf.hidden_states.shape), (2, 4))
        self.assertIs(buf.next_token_top_logprobs_val, warmup.next_token_top_logprobs_val)
        self.assertIs(buf.customized_info, warmup.customized_info)

        src = LogitsProcessorOutput(
            next_token_logits=torch.arange(8, 16, dtype=torch.float32).reshape(1, 8),
            hidden_states=torch.full((1, 4), 3.0),
        )
        backend._copy_output_to_buffer(src, buf, num_tokens=1)
        sliced = backend._slice_output(buf, num_tokens=1)
        self.assertTrue(torch.equal(sliced.next_token_logits, src.next_token_logits))
        self.assertTrue(torch.equal(sliced.hidden_states, src.hidden_states))
        self.assertEqual(backend._output_rows(src, cap=2), 1)

    def test_copy_output_and_eager_clone_lpo(self):
        from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
            _eager_clone_lpo,
        )
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
            _copy_output,
        )

        dst = LogitsProcessorOutput(
            next_token_logits=torch.zeros(1, 4),
            hidden_states=torch.zeros(1, 2),
        )
        src = LogitsProcessorOutput(
            next_token_logits=torch.arange(4, dtype=torch.float32).reshape(1, 4),
            hidden_states=torch.ones(1, 2),
        )
        _copy_output(dst, src)
        self.assertTrue(torch.equal(dst.next_token_logits, src.next_token_logits))
        cloned = _eager_clone_lpo(src)
        self.assertTrue(torch.equal(cloned.next_token_logits, src.next_token_logits))
        self.assertIsNot(cloned.next_token_logits, src.next_token_logits)


if __name__ == "__main__":
    import unittest

    unittest.main()
