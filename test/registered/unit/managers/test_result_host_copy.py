"""A step's host commit turns its logprobs into host values on a copy of the
logits output; the result keeps what the forward produced."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.logits_processor import LogitsProcessorOutput  # noqa: E402
from sglang.srt.managers.scheduler_components.batch_result_processor import (  # noqa: E402
    SchedulerBatchResultProcessor,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _CopyDone:
    def __init__(self):
        self.synchronized = 0

    def synchronize(self):
        self.synchronized += 1


class TestResultHostCopy(CustomTestCase):
    def test_logprobs_turn_into_host_values_on_a_copy(self):
        copy_done = _CopyDone()
        produced = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_logprobs=torch.tensor([-0.5, -1.0]),
            next_token_top_logprobs_val=[torch.tensor([-0.5]), torch.tensor([-1.0])],
            next_token_top_logprobs_idx=[torch.tensor([3]), torch.tensor([4])],
            input_token_logprobs=torch.tensor([-2.0, -3.0]),
            # One sequence whose input top-k rows arrive in two spans.
            input_top_logprobs_val=[[torch.tensor([[-2.0]]), torch.tensor([[-3.0]])]],
            input_logprobs_copy_done=copy_done,
        )
        spans = produced.input_top_logprobs_val[0]

        host = produced.host_copy()
        processor = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
        processor.move_logprobs_to_cpu(
            batch=SimpleNamespace(return_logprob=True), logits_output=host
        )

        self.assertEqual(host.next_token_logprobs, [-0.5, -1.0])
        self.assertEqual(host.next_token_top_logprobs_idx, [[3], [4]])
        self.assertEqual(host.input_token_logprobs, (-2.0, -3.0))
        self.assertEqual(host.input_top_logprobs_val, [[[-2.0], [-3.0]]])
        self.assertEqual(copy_done.synchronized, 1)

        # The produced output is as the forward left it.
        self.assertTrue(torch.is_tensor(produced.next_token_logprobs))
        self.assertTrue(torch.is_tensor(produced.next_token_top_logprobs_idx[0]))
        self.assertTrue(torch.is_tensor(produced.input_token_logprobs))
        self.assertIs(produced.input_top_logprobs_val[0], spans)
        self.assertIs(produced.input_logprobs_copy_done, copy_done)


if __name__ == "__main__":
    unittest.main()
