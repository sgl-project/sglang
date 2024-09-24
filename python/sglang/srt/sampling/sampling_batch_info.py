from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List

import torch

import sglang.srt.sampling.penaltylib as penaltylib

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclasses.dataclass
class SamplingBatchInfo:
    # Basic Info
    vocab_size: int

    # Batched sampling params
    temperatures: torch.Tensor = None
    top_ps: torch.Tensor = None
    top_ks: torch.Tensor = None
    min_ps: torch.Tensor = None

    # Dispatch in CUDA graph
    need_min_p_sampling: bool = False

    # Bias Tensors
    logit_bias: torch.Tensor = None
    vocab_mask: torch.Tensor = None

    # Penalizer
    penalizer_orchestrator: penaltylib.BatchedPenalizerOrchestrator = None
    linear_penalties: torch.Tensor = None
    scaling_penalties: torch.Tensor = None

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
        reqs = batch.reqs
        ret = cls(vocab_size=vocab_size)

        with torch.device("cuda"):
            ret.temperatures = torch.tensor(
                [r.sampling_params.temperature for r in reqs],
                dtype=torch.float,
            ).view(-1, 1)
            ret.top_ps = torch.tensor(
                [r.sampling_params.top_p for r in reqs], dtype=torch.float
            )
            ret.top_ks = torch.tensor(
                [r.sampling_params.top_k for r in reqs], dtype=torch.int
            )
            ret.min_ps = torch.tensor(
                [r.sampling_params.min_p for r in reqs], dtype=torch.float
            )

        ret.need_min_p_sampling = any(r.sampling_params.min_p > 0 for r in reqs)

        # Each penalizers will do nothing if they evaluate themselves as not required by looking at
        # the sampling_params of the requests (See {_is_required()} of each penalizers). So this
        # should not add hefty computation overhead other than simple checks.
        #
        # While we choose not to even create the class instances if they are not required, this
        # could add additional complexity to the {ScheduleBatch} class, especially we need to
        # handle {filter_batch()} and {merge()} cases as well.
        ret.penalizer_orchestrator = penaltylib.BatchedPenalizerOrchestrator(
            vocab_size=vocab_size,
            batch=batch,
            device="cuda",
            Penalizers={
                penaltylib.BatchedFrequencyPenalizer,
                penaltylib.BatchedMinNewTokensPenalizer,
                penaltylib.BatchedPresencePenalizer,
                penaltylib.BatchedRepetitionPenalizer,
            },
        )

        # Handle logit bias but only allocate when needed
        ret.logit_bias = None

        return ret

    def __len__(self):
        return len(self.temperatures)

    def update_penalties(self):
        self.scaling_penalties = None
        self.linear_penalties = None

        for penalizer in self.penalizer_orchestrator.penalizers.values():
            if isinstance(penalizer, penaltylib.BatchedRepetitionPenalizer):
                if penalizer.is_prepared():
                    self.scaling_penalties = penalizer.cumulated_repetition_penalties
            else:
                if penalizer.is_prepared():
                    if self.linear_penalties is None:
                        bs = self.penalizer_orchestrator.batch.batch_size()
                        self.linear_penalties = torch.zeros(
                            (bs, self.vocab_size),
                            dtype=torch.float32,
                            device="cuda",
                        )
                    self.linear_penalties = penalizer.apply(self.linear_penalties)

    def update_regex_vocab_mask(self, batch: ScheduleBatch):
        has_regex = any(req.regex_fsm is not None for req in batch.reqs)

        # Reset the vocab mask
        self.vocab_mask = None

        if has_regex:
            self.vocab_mask = torch.zeros(
                batch.batch_size(), self.vocab_size, dtype=torch.bool, device="cuda"
            )
            for i, req in enumerate(batch.reqs):
                if req.regex_fsm is not None:
                    self.vocab_mask[i].fill_(1)
                    self.vocab_mask[i][
                        req.regex_fsm.get_next_instruction(req.regex_fsm_state).tokens
                    ] = 0

    def filter(self, unfinished_indices: List[int], new_indices: torch.Tensor):
        self.penalizer_orchestrator.filter(unfinished_indices, new_indices)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
            "logit_bias",
        ]:
            self_val = getattr(self, item, None)
            if self_val is not None:  # logit_bias can be None
                setattr(self, item, self_val[new_indices])

    @staticmethod
    def merge_bias_tensor(
        lhs: torch.Tensor, rhs: torch.Tensor, bs1: int, bs2: int, default: int = 0
    ):
        # bias tensor can be None
        if lhs is not None or rhs is not None:
            shape, dtype = None, None
            if lhs is not None:
                shape, dtype = lhs.shape[1:], lhs.dtype
            else:
                shape, dtype = rhs.shape[1:], rhs.dtype
            with torch.dtype(dtype):
                if lhs is None:
                    lhs = torch.empty((bs1, *shape), device="cuda").fill_(default)
                if rhs is None:
                    rhs = torch.empty((bs2, *shape), device="cuda").fill_(default)
            return torch.cat([lhs, rhs])

        return None

    def merge(self, other: "SamplingBatchInfo"):
        self.penalizer_orchestrator.merge(other.penalizer_orchestrator)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
        ]:
            self_val = getattr(self, item, None)
            other_val = getattr(other, item, None)
            setattr(self, item, torch.concat([self_val, other_val]))

        self.logit_bias = SamplingBatchInfo.merge_bias_tensor(
            self.logit_bias, other.logit_bias, len(self), len(other)
        )
