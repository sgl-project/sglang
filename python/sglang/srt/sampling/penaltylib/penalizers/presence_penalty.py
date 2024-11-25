from typing import List

import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer, _TokenIDs


class BatchedPresencePenalizer(_BatchedPenalizer):
    """
    Presence penalizer penalizes tokens based on their presence in the output.
    """

    presence_penalties: torch.Tensor = None
    cumulated_presence_penalties: torch.Tensor = None

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.presence_penalty != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.cumulated_presence_penalties = (
            torch.tensor(
                data=[0.0 for _ in self.orchestrator.reqs()],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
            .unsqueeze_(1)
            .repeat(1, self.orchestrator.vocab_size)
        )

        self.presence_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.presence_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
            .unsqueeze_(1)
            .expand_as(self.cumulated_presence_penalties)
        )

    def _teardown(self):
        self.presence_penalties = None
        self.cumulated_presence_penalties = None

    def _cumulate_input_tokens(self, input_ids: _TokenIDs):
        pass

    def _cumulate_output_tokens(self, output_ids: _TokenIDs):
        mask = output_ids.occurrence_count() > 0
        self.cumulated_presence_penalties[mask] = self.presence_penalties[mask]

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        logits -= self.cumulated_presence_penalties
        return logits

    def _filter(self, indices_to_keep: List[int], indices_tensor_to_keep: torch.Tensor):
        self.presence_penalties = self.presence_penalties[indices_tensor_to_keep]
        self.cumulated_presence_penalties = self.cumulated_presence_penalties[
            indices_tensor_to_keep
        ]

    def _merge(self, their: "BatchedPresencePenalizer"):
        self.presence_penalties = torch.cat(
            [self.presence_penalties, their.presence_penalties], dim=0
        )
        self.cumulated_presence_penalties = torch.cat(
            [self.cumulated_presence_penalties, their.cumulated_presence_penalties],
            dim=0,
        )
