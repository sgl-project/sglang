import unittest
from typing import List

import torch

from sglang.srt.sampling.penaltylib.penalizers.min_new_tokens import (
    BatchedMinNewTokensPenalizer,
)
from sglang.test.srt.sampling.penaltylib.utils import (
    BaseBatchedPenalizerTest,
    MockSamplingParams,
    Step,
    StepType,
    Subject,
)

MIN_NEW_TOKENS = 2
EOS_TOKEN_ID = 4
STOP_TOKEN_ID = 3

ALL_STOP_TOKEN_IDS = {STOP_TOKEN_ID, EOS_TOKEN_ID}


class TestBatchedMinNewTokensPenalizer(BaseBatchedPenalizerTest):
    Penalizer = BatchedMinNewTokensPenalizer

    def _create_subject(self, min_new_tokens: int) -> Subject:
        return Subject(
            eos_token_id=EOS_TOKEN_ID,
            sampling_params=MockSamplingParams(
                min_new_tokens=min_new_tokens,
                stop_token_ids={STOP_TOKEN_ID},
            ),
            steps=[
                Step(
                    type=StepType.INPUT,
                    token_ids=[0, 1, 2],
                    expected_tensors={
                        "min_new_tokens": self.tensor(
                            [[min_new_tokens]], dtype=torch.int32
                        ),
                        "stop_token_penalties": self.tensor(
                            [
                                [
                                    float("-inf") if i in ALL_STOP_TOKEN_IDS else 0
                                    for i in range(self.vocab_size)
                                ]
                            ],
                            dtype=torch.float32,
                        ),
                        "len_output_tokens": self.tensor([[0]], dtype=torch.int32),
                    },
                    expected_logits=(
                        self.tensor(
                            [
                                [
                                    float("-inf") if i in ALL_STOP_TOKEN_IDS else 1
                                    for i in range(self.vocab_size)
                                ]
                            ],
                            dtype=torch.float32,
                        )
                        if min_new_tokens > 0
                        else torch.ones(
                            (1, self.vocab_size),
                            dtype=torch.float32,
                            device=self.device,
                        )
                    ),
                ),
                Step(
                    type=StepType.OUTPUT,
                    token_ids=[0],
                    expected_tensors={
                        "min_new_tokens": self.tensor(
                            [[min_new_tokens]], dtype=torch.int32
                        ),
                        "stop_token_penalties": self.tensor(
                            [
                                [
                                    float("-inf") if i in ALL_STOP_TOKEN_IDS else 0
                                    for i in range(self.vocab_size)
                                ]
                            ],
                            dtype=torch.float32,
                        ),
                        "len_output_tokens": self.tensor([[1]], dtype=torch.int32),
                    },
                    expected_logits=(
                        self.tensor(
                            [
                                [
                                    float("-inf") if i in ALL_STOP_TOKEN_IDS else 1
                                    for i in range(self.vocab_size)
                                ]
                            ],
                            dtype=torch.float32,
                        )
                        if min_new_tokens > 1
                        else torch.ones(
                            (1, self.vocab_size),
                            dtype=torch.float32,
                            device=self.device,
                        )
                    ),
                ),
                Step(
                    type=StepType.OUTPUT,
                    token_ids=[0],
                    expected_tensors={
                        "min_new_tokens": self.tensor(
                            [[min_new_tokens]], dtype=torch.int32
                        ),
                        "stop_token_penalties": self.tensor(
                            [
                                [
                                    float("-inf") if i in ALL_STOP_TOKEN_IDS else 0
                                    for i in range(self.vocab_size)
                                ]
                            ],
                            dtype=torch.float32,
                        ),
                        "len_output_tokens": self.tensor([[2]], dtype=torch.int32),
                    },
                    expected_logits=(
                        self.tensor(
                            [
                                [
                                    float("-inf") if i in ALL_STOP_TOKEN_IDS else 1
                                    for i in range(self.vocab_size)
                                ]
                            ],
                            dtype=torch.float32,
                        )
                        if min_new_tokens > 2
                        else torch.ones(
                            (1, self.vocab_size),
                            dtype=torch.float32,
                            device=self.device,
                        )
                    ),
                ),
            ],
        )

    def create_test_subjects(self) -> List[Subject]:
        self.enabled = self._create_subject(min_new_tokens=MIN_NEW_TOKENS)
        self.disabled = self._create_subject(min_new_tokens=0.0)


if __name__ == "__main__":
    unittest.main()
