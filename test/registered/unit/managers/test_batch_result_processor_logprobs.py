"""Regression tests for scheduler logprob CPU normalization."""

from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBatchResultProcessorLogprobs(CustomTestCase):
    def test_move_logprobs_preserves_list_placeholders(self):
        """Mixed batches include [] for requests without token-id logprobs."""
        processor = object.__new__(SchedulerBatchResultProcessor)
        output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_token_ids_logprobs_val=[torch.tensor([-0.5]), []],
        )

        processor.move_logprobs_to_cpu(
            batch=SimpleNamespace(return_logprob=True), logits_output=output
        )

        self.assertEqual(output.next_token_token_ids_logprobs_val, [[-0.5], []])
