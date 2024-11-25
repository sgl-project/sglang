import unittest
from typing import List

import torch

from sglang.srt.sampling.penaltylib.penalizers.repetition_penalty import (
    BatchedRepetitionPenalizer,
)
from sglang.test.srt.sampling.penaltylib.utils import (
    BaseBatchedPenalizerTest,
    MockSamplingParams,
    Step,
    StepType,
    Subject,
)

REPETITION_PENALTY = 2.0


class TestBatchedRepetitionPenalizer(BaseBatchedPenalizerTest):
    Penalizer = BatchedRepetitionPenalizer

    def _create_subject(self, repetition_penalty: float) -> Subject:
        l = 1.0 / repetition_penalty
        return Subject(
            sampling_params=MockSamplingParams(
                repetition_penalty=repetition_penalty,
            ),
            steps=[
                Step(
                    type=StepType.INPUT,
                    token_ids=[0, 1, 2],
                    expected_tensors={
                        "repetition_penalties": self.tensor(
                            [[repetition_penalty] * self.vocab_size],
                            dtype=torch.float32,
                        ),
                        "cumulated_repetition_penalties": (
                            self.tensor(
                                [[2.0, 2.0, 2.0, 1.0, 1.0]], dtype=torch.float32
                            )
                            if repetition_penalty != 1.0
                            else self.tensor(
                                [[1.0] * self.vocab_size], dtype=torch.float32
                            )
                        ),
                    },
                    expected_logits=(
                        self.tensor([[l, l, l, 1.0, 1.0]], dtype=torch.float32)
                        if repetition_penalty != 1.0
                        else self.tensor([[1.0] * self.vocab_size], dtype=torch.float32)
                    ),
                ),
                Step(
                    type=StepType.OUTPUT,
                    token_ids=[0, 1, 3],
                    expected_tensors={
                        "repetition_penalties": self.tensor(
                            [[repetition_penalty] * self.vocab_size],
                            dtype=torch.float32,
                        ),
                        "cumulated_repetition_penalties": (
                            self.tensor(
                                [[2.0, 2.0, 2.0, 2.0, 1.0]], dtype=torch.float32
                            )
                            if repetition_penalty != 1.0
                            else self.tensor(
                                [[1.0] * self.vocab_size], dtype=torch.float32
                            )
                        ),
                    },
                    expected_logits=(
                        self.tensor([[l, l, l, l, 1.0]], dtype=torch.float32)
                        if repetition_penalty != 1.0
                        else self.tensor([[1.0] * self.vocab_size], dtype=torch.float32)
                    ),
                ),
            ],
        )

    def create_test_subjects(self) -> List[Subject]:
        self.enabled = self._create_subject(repetition_penalty=REPETITION_PENALTY)
        self.disabled = self._create_subject(repetition_penalty=1.0)


if __name__ == "__main__":
    unittest.main()
