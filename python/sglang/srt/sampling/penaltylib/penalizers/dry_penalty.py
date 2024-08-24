import typing
from collections import defaultdict
import torch

from ..orchestrator import _BatchedPenalizer, _TokenIDs

class BatchedDryPenalizer(_BatchedPenalizer):
    """
    DRY (Don't Repeat Yourself) penalizer penalizes tokens based on their repetition patterns in the input.
    """

    multipliers: torch.Tensor = None
    bases: torch.Tensor = None
    allowed_lengths: torch.Tensor = None
    sequence_breakers: typing.List[set[int]] = None
    ranges: torch.Tensor = None
    input_ids: torch.Tensor = None
    output_ids: torch.Tensor = None

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.dry_multiplier != 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.multipliers = torch.tensor(
            [req.sampling_params.dry_multiplier for req in self.orchestrator.reqs()],
            dtype=torch.float32,
            device=self.orchestrator.device
        )
        self.bases = torch.tensor(
            [req.sampling_params.dry_base for req in self.orchestrator.reqs()],
            dtype=torch.float32,
            device=self.orchestrator.device
        )
        self.allowed_lengths = torch.tensor(
            [req.sampling_params.dry_allowed_length for req in self.orchestrator.reqs()],
            dtype=torch.float32,  # Ensure this is float to match other tensors
            device=self.orchestrator.device
        )
        self.sequence_breakers = [
            [req.tokenizer.encode(f'a{prompt}', add_special_tokens=False)[-1] 
            for prompt in req.sampling_params.dry_sequence_breakers]
            for req in self.orchestrator.reqs()
        ]
        self.ranges = torch.tensor(
            [req.sampling_params.dry_penalty_last_n for req in self.orchestrator.reqs()],
            dtype=torch.int64,
            device=self.orchestrator.device
        )

    def _teardown(self):
        del self.multipliers
        del self.bases
        del self.allowed_lengths
        del self.sequence_breakers
        del self.ranges

        self.multipliers = None
        self.bases = None
        self.allowed_lengths = None
        self.sequence_breakers = None
        self.ranges = None

    def _cumulate_input_tokens(self, input_ids: _TokenIDs):
        self.input_ids = input_ids.token_ids

    def _cumulate_output_tokens(self, output_ids: _TokenIDs):
        self.output_ids = output_ids.token_ids

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = logits.shape[0], logits.shape[1]
        max_back_length = 50  # Limit the backward match to 50 to prevent overflow
        for i in range(batch_size):
            if self.output_ids is not None:
                input_ids = self.input_ids[i] = torch.cat(
                    [self.input_ids[i], self.output_ids], dim=0
                )
            else:
                input_ids = self.input_ids[i]
            input_ids = input_ids.tolist()
            range_limit = min(self.ranges[i].item(), len(input_ids))
            input_ids = input_ids[-range_limit:] if range_limit > 0 else input_ids
            last_token = input_ids[-1]
            if last_token in self.sequence_breakers[i]:
                continue

            match_indices = [idx for idx, val in enumerate(input_ids[:-1]) if val == last_token]
            match_lengths = defaultdict(int)

            for idx in match_indices:
                next_token = input_ids[idx + 1]
                if next_token in self.sequence_breakers[i]:
                    continue
                match_length = 1
                while match_length < max_back_length and idx - match_length >= 0:
                    previous_token = input_ids[-(match_length + 1)]
                    if input_ids[idx - match_length] != previous_token:
                        break
                    if previous_token in self.sequence_breakers[i]:
                        break
                    match_length += 1
                match_lengths[next_token] = max(match_length, match_lengths[next_token])

            for token, match_length in match_lengths.items():
                if match_length >= self.allowed_lengths[i].item():
                    penalty = self.multipliers[i].item() * self.bases[i].item() ** (match_length - self.allowed_lengths[i].item())
                    logits[i, token] -= penalty

        return logits

    def _filter(
        self, indices_to_keep: typing.List[int], indices_tensor_to_keep: torch.Tensor
    ):
        self.multipliers = self.multipliers[indices_tensor_to_keep]
        self.bases = self.bases[indices_tensor_to_keep]
        self.allowed_lengths = self.allowed_lengths[indices_tensor_to_keep]
        self.sequence_breakers = [self.sequence_breakers[i] for i in indices_to_keep]
        self.ranges = self.ranges[indices_tensor_to_keep]

    def _merge(self, their: "BatchedDryPenalizer"):
        self.multipliers = torch.cat([self.multipliers, their.multipliers], dim=0)
        self.bases = torch.cat([self.bases, their.bases], dim=0)
        self.allowed_lengths = torch.cat([self.allowed_lengths, their.allowed_lengths], dim=0)
        self.sequence_breakers.extend(their.sequence_breakers)
        self.ranges = torch.cat([self.ranges, their.ranges], dim=0)