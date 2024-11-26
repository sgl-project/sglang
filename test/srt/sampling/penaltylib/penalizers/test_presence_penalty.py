import unittest
from typing import List

import torch

from sglang.srt.sampling.penaltylib.penalizers.presence_penalty import (
    BatchedPresencePenalizer,
)
from sglang.test.srt.sampling.penaltylib.utils import (
    BaseBatchedPenalizerTest,
    MockSamplingParams,
    Step,
    StepType,
    Subject,
)


class BaseBatchedPresencePenalizerTest(BaseBatchedPenalizerTest):
    Penalizer = BatchedPresencePenalizer
    presence_penalty: float

    def setUp(self):
        if self.__class__ == BaseBatchedPresencePenalizerTest:
            self.skipTest("Base class for presence_penalty tests")

        super().setUp()

    def _create_subject(self, presence_penalty: float) -> Subject:
        return Subject(
            sampling_params=MockSamplingParams(
                presence_penalty=presence_penalty,
            ),
            steps=[
                Step(
                    type=StepType.INPUT,
                    token_ids=[0, 1, 2],
                    expected_tensors={
                        "presence_penalties": self.tensor(
                            [[presence_penalty] * self.vocab_size], dtype=torch.float32
                        ),
                        "cumulated_presence_penalties": self.tensor(
                            [[0.0] * self.vocab_size], dtype=torch.float32
                        ),
                    },
                    expected_logits=self.tensor(
                        [[1] * self.vocab_size], dtype=torch.float32
                    ),
                ),
                Step(
                    type=StepType.OUTPUT,
                    token_ids=[1, 2, 2],
                    expected_tensors={
                        "presence_penalties": self.tensor(
                            [[presence_penalty] * self.vocab_size], dtype=torch.float32
                        ),
                        "cumulated_presence_penalties": self.tensor(
                            [
                                [
                                    presence_penalty if i in {1, 2} else 0.0
                                    for i in range(self.vocab_size)
                                ],
                            ],
                            dtype=torch.float32,
                        ),
                    },
                    expected_logits=self.tensor(
                        [
                            [
                                1.0 - presence_penalty if i in {1, 2} else 1.0
                                for i in range(self.vocab_size)
                            ],
                        ],
                        dtype=torch.float32,
                    ),
                ),
            ],
        )

    def create_test_subjects(self) -> List[Subject]:
        self.enabled = self._create_subject(presence_penalty=self.presence_penalty)
        self.disabled = self._create_subject(presence_penalty=0.0)


class TestBatchedPresencePenalizerPositiveValue(BaseBatchedPresencePenalizerTest):
    presence_penalty = 0.12


class TestBatchedPresencePenalizerNegativeValue(BaseBatchedPresencePenalizerTest):
    presence_penalty = -0.12


if __name__ == "__main__":
    unittest.main()
