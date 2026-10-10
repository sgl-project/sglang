"""Unit tests for mixed host/device token-id logprob results."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _NoSpecAlgorithm:
    def is_none(self) -> bool:
        return True


class TestBatchResultProcessorLogprobs(CustomTestCase):
    def setUp(self):
        self.processor = object.__new__(SchedulerBatchResultProcessor)
        self.batch = SimpleNamespace(
            return_logprob=True,
            spec_algorithm=_NoSpecAlgorithm(),
        )

    def test_prefill_preserves_lists_and_converts_tensors(self):
        logits_output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_token_ids_logprobs_val=[
                [],
                [-0.1, -0.2],
                torch.tensor([-0.3, -0.4], dtype=torch.float64),
            ],
        )

        self.processor.move_logprobs_to_cpu(
            batch=self.batch,
            logits_output=logits_output,
        )

        self.assertEqual(
            logits_output.next_token_token_ids_logprobs_val,
            [[], [-0.1, -0.2], [-0.3, -0.4]],
        )

    def test_decode_preserves_lists_and_converts_tensors(self):
        logits_output = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_logprobs=torch.tensor([-0.5, -0.6], dtype=torch.float64),
            next_token_token_ids_logprobs_val=[
                [],
                [-0.1, -0.2],
                torch.tensor([-0.3, -0.4], dtype=torch.float64),
            ],
        )

        next_token_ids, next_token_logprobs = self.processor._normalize_decode_outputs(
            batch=self.batch,
            result=SimpleNamespace(),
            logits_output=logits_output,
            next_token_ids=torch.tensor([1, 2, 3]),
        )

        self.assertEqual(next_token_ids, [[1], [2], [3]])
        self.assertEqual(next_token_logprobs, [-0.5, -0.6])
        self.assertEqual(
            logits_output.next_token_token_ids_logprobs_val,
            [[], [-0.1, -0.2], [-0.3, -0.4]],
        )


if __name__ == "__main__":
    unittest.main()
