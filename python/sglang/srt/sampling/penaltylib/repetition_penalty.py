from typing import Optional

import torch

from sglang.srt.sampling.penaltylib.orchestrator import _BatchedPenalizer
from sglang.srt.utils import get_compiler_backend, is_npu
from sglang.srt.utils.device import is_pin_memory_available

_is_npu = is_npu()


@torch.compile(dynamic=True, backend=get_compiler_backend(), disable=_is_npu)
def apply_scaling_penalties(logits, scaling_penalties):
    logits[:] = torch.where(
        logits < 0,
        logits * scaling_penalties,
        logits / scaling_penalties,
    )


def apply_repetition_penalty_with_prefix(
    logits: torch.Tensor,
    history_scaling: torch.Tensor,
    repetition_penalties: torch.Tensor,
    verify_input_ids: torch.Tensor,
    committed_tokens: Optional[torch.Tensor] = None,
    committed_lens: Optional[torch.Tensor] = None,
) -> None:
    """Apply ordinary decode's repetition penalty to each causal verify row.

    Row j predicts the token *after* verify_input_ids[:, j]. Column zero is
    the already sampled anchor; the remaining columns are hypothetical draft
    tokens. Neither those tokens nor an unaccepted suffix may mutate history.
    The overlap relay supplies the previous committed block, which the CPU
    history can still be missing. Union is idempotent if it is already present.
    Only one [bs, vocab] scratch buffer is needed, independent of block width.
    """
    scaling = history_scaling.clone()
    penalties = repetition_penalties.reshape(-1, 1)
    if committed_tokens is not None:
        assert committed_lens is not None
        for column in range(committed_tokens.shape[1]):
            valid = column < committed_lens
            # Uncommitted entries may contain -1 or arbitrary draft garbage.
            index = torch.where(valid, committed_tokens[:, column], 0).long()[:, None]
            previous = scaling.gather(1, index)
            scaling.scatter_(1, index, torch.where(valid[:, None], penalties, previous))

    for position in range(verify_input_ids.shape[1]):
        index = verify_input_ids[:, position].long()[:, None]
        scaling.scatter_(1, index, penalties)
        apply_scaling_penalties(logits[:, position, :], scaling)


class BatchedRepetitionPenalizer(_BatchedPenalizer):
    """
    Repetition penalizer penalizes tokens based on their presence in the generated output.
    """

    is_multiplicative: bool = True

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.repetition_penalty != 1.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        pin_memory = is_pin_memory_available(self.orchestrator.device)
        self.cumulated_repetition_penalties = torch.ones(
            (len(self.orchestrator.reqs()), self.orchestrator.vocab_size),
            dtype=torch.float32,
            device=self.orchestrator.device,
        )
        self.repetition_penalties = (
            torch.tensor(
                data=[
                    req.sampling_params.repetition_penalty
                    for req in self.orchestrator.reqs()
                ],
                dtype=torch.float32,
                pin_memory=pin_memory,
            )
            .to(self.orchestrator.device, non_blocking=True)
            .unsqueeze_(1)
        )

    def _cumulate_output_tokens(
        self, output_ids: torch.Tensor, valid: Optional[torch.Tensor] = None
    ):
        index = output_ids.unsqueeze(1)
        src = self.repetition_penalties
        if valid is not None:
            # Padding entries write back the value already stored at their index.
            src = torch.where(
                valid.unsqueeze(1),
                src,
                self.cumulated_repetition_penalties.gather(dim=1, index=index),
            )
        self.cumulated_repetition_penalties.scatter_(dim=1, index=index, src=src)

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        apply_scaling_penalties(logits, self.cumulated_repetition_penalties)
        return logits

    def get_scaling_penalties(self) -> torch.Tensor:
        return self.cumulated_repetition_penalties

    def _filter(self, keep_indices: torch.Tensor):
        self.repetition_penalties = self.repetition_penalties[keep_indices]
        self.cumulated_repetition_penalties = self.cumulated_repetition_penalties[
            keep_indices
        ]

    def _merge(self, their: "BatchedRepetitionPenalizer"):
        self.repetition_penalties = torch.cat(
            [self.repetition_penalties, their.repetition_penalties], dim=0
        )
        self.cumulated_repetition_penalties = torch.cat(
            [self.cumulated_repetition_penalties, their.cumulated_repetition_penalties],
            dim=0,
        )

    def _teardown(self) -> None:
        for name in ("repetition_penalties", "cumulated_repetition_penalties"):
            if hasattr(self, name):
                delattr(self, name)
