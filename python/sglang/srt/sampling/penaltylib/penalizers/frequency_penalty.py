import typing

import torch

from ..orchestrator import _BatchedPenalizer, _TokenIDs


class BatchedFrequencyPenalizer(_BatchedPenalizer):
    """
    Frequency penalizer penalizes tokens based on their frequency in the output.
    """

    frequency_penalties: torch.Tensor = None
    cumulated_frequency_penalties: torch.Tensor = None

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.frequency_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.cumulated_frequency_penalties = (
            torch.tensor(
                data=[0.0 for _ in self.orchestrator.reqs()],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
            .unsqueeze_(1)
            .repeat(1, self.orchestrator.vocab_size)
        )

        self.frequency_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.frequency_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
            .unsqueeze_(1)
            .expand_as(self.cumulated_frequency_penalties)
        )

    def _teardown(self):
        del self.frequency_penalties
        del self.cumulated_frequency_penalties

        self.frequency_penalties = None
        self.cumulated_frequency_penalties = None

    def _cumulate_input_tokens(self, input_ids: _TokenIDs):
        pass

    def _cumulate_output_tokens(self, output_ids: _TokenIDs):
        pass

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        logits -= self.cumulated_frequency_penalties
        return logits

    def _filter(
        self, indices_to_keep: typing.List[int], indices_tensor_to_keep: torch.Tensor
    ):
        self.frequency_penalties = self.frequency_penalties[indices_tensor_to_keep]
        self.cumulated_frequency_penalties = self.cumulated_frequency_penalties[
            indices_tensor_to_keep
        ]

    def _merge(self, their: "BatchedFrequencyPenalizer"):
        self.frequency_penalties = torch.cat(
            [self.frequency_penalties, their.frequency_penalties], dim=0
        )
        self.cumulated_frequency_penalties = torch.cat(
            [self.cumulated_frequency_penalties, their.cumulated_frequency_penalties],
            dim=0,
        )
