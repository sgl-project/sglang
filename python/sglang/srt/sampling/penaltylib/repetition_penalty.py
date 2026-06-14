import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer
from sglang.srt.utils import get_compiler_backend, is_npu

_is_npu = is_npu()


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def apply_scaling_penalties(logits, scaling_penalties):
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )


class BatchedRepetitionPenalizer(_BatchedPenalizer):
    """
    Repetition penalizer applies a penalty to tokens that have appeared in the output.
    Implementation follows the standard repetition_penalty logic (e.g., HuggingFace):
    - If logit > 0: logit /= penalty
    - If logit < 0: logit *= penalty
    This reduces the probability of repeated tokens.
    """

    is_multiplicative: bool = True

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.repetition_penalty != 1.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.repetition_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.repetition_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                device=self.orchestrator.device,
            )
        ).unsqueeze_(1)

        self.seen_tokens = torch.zeros(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.bool,
            device=self.orchestrator.device,
        )

    def _cumulate_output_tokens(self, output_ids: torch.Tensor):
        if output_ids.dim() == 1:
            indices = output_ids
        else:
            indices = output_ids.squeeze(-1)

        self.seen_tokens.scatter_(
            dim=1,
            index=indices.unsqueeze(1),
            src=torch.ones_like(indices, dtype=torch.bool, device=self.orchestrator.device).unsqueeze(1),
        )

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        penalties = self.repetition_penalties  # (batch, 1)
        factors = torch.where(logits > 0, 1.0 / penalties, penalties)
        mask = self.seen_tokens
        multiplier = torch.where(mask, factors, 1.0)
        logits.mul_(multiplier)
        return logits

    def _filter(self, keep_indices: torch.Tensor):
        self.repetition_penalties = self.repetition_penalties[keep_indices]
        self.seen_tokens = self.seen_tokens[keep_indices]

    def _merge(self, their: "BatchedRepetitionPenalizer"):
        self.repetition_penalties = torch.cat(
            [self.repetition_penalties, their.repetition_penalties], dim=0
        )
        self.seen_tokens = torch.cat([self.seen_tokens, their.seen_tokens], dim=0)

    def _teardown(self) -> None:
        for name in ("repetition_penalties", "seen_tokens"):
            if hasattr(self, name):
                delattr(self, name)
