import typing
import unittest

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

PRESENCE_PENALTY = 0.12


class TestBatchedPresencePenalizer(BaseBatchedPenalizerTest):
    Penalizer = BatchedPresencePenalizer

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

    def create_test_subjects(self) -> typing.List[Subject]:
        self.enabled = self._create_subject(presence_penalty=PRESENCE_PENALTY)
        self.disabled = self._create_subject(presence_penalty=0.0)


if __name__ == "__main__":
    unittest.main()
