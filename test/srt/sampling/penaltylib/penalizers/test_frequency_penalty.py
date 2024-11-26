import unittest
from typing import List

import torch

from sglang.srt.sampling.penaltylib.penalizers.frequency_penalty import (
    BatchedFrequencyPenalizer,
)
from sglang.test.srt.sampling.penaltylib.utils import (
    BaseBatchedPenalizerTest,
    MockSamplingParams,
    Step,
    StepType,
    Subject,
)


class BaseBatchedFrequencyPenalizerTest(BaseBatchedPenalizerTest):
    Penalizer = BatchedFrequencyPenalizer
    frequency_penalty: float

    def setUp(self):
        if self.__class__ == BaseBatchedFrequencyPenalizerTest:
            self.skipTest("Base class for frequency_penalty tests")

        super().setUp()

    def _create_subject(self, frequency_penalty: float) -> Subject:
        return Subject(
            sampling_params=MockSamplingParams(
                frequency_penalty=frequency_penalty,
            ),
            steps=[
                Step(
                    type=StepType.INPUT,
                    token_ids=[0, 1, 2],
                    expected_tensors={
                        "frequency_penalties": self.tensor(
                            [[frequency_penalty] * self.vocab_size], dtype=torch.float32
                        ),
                        "cumulated_frequency_penalties": self.tensor(
                            [[0.0] * self.vocab_size], dtype=torch.float32
                        ),
                    },
                    expected_logits=self.tensor(
                        [[1] * self.vocab_size], dtype=torch.float32
                    ),
                ),
                Step(
                    type=StepType.OUTPUT,
                    token_ids=[
                        1,
                        2,
                        2,
                    ],  # This is the output ids of one request in three steps.
                    expected_tensors={
                        "frequency_penalties": self.tensor(
                            [[frequency_penalty] * self.vocab_size], dtype=torch.float32
                        ),
                        "cumulated_frequency_penalties": self.tensor(
                            [
                                [
                                    frequency_penalty * i if i in {1, 2} else 0.0
                                    for i in range(self.vocab_size)
                                ],
                            ],
                            dtype=torch.float32,
                        ),
                    },
                    expected_logits=self.tensor(
                        [
                            [
                                1.0 - frequency_penalty * i if i in {1, 2} else 1.0
                                for i in range(self.vocab_size)
                            ],
                        ],
                        dtype=torch.float32,
                    ),
                ),
            ],
        )

    def create_test_subjects(self) -> List[Subject]:
        self.enabled = self._create_subject(frequency_penalty=self.frequency_penalty)
        self.disabled = self._create_subject(frequency_penalty=0.0)


class TestBatchedFrequencyPenalizerPositiveValue(BaseBatchedFrequencyPenalizerTest):
    frequency_penalty = 0.12


class TestBatchedFrequencyPenalizerNegativeValue(BaseBatchedFrequencyPenalizerTest):
    frequency_penalty = -0.12


if __name__ == "__main__":
    unittest.main()
