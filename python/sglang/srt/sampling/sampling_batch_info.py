from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List

import torch

import sglang.srt.sampling.penaltylib as penaltylib
from sglang.srt.constrained import RegexGuide

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclasses.dataclass
class SamplingBatchInfo:
    # Batched sampling params
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor

    # Dispatch in CUDA graph
    need_min_p_sampling: bool

    # Bias Tensors
    vocab_size: int
    logit_bias: torch.Tensor = None
    vocab_mask: torch.Tensor = None

    # FSM states
    regex_fsms: List[RegexGuide] = None
    regex_fsm_states: List[int] = None

    # Penalizer
    penalizer_orchestrator: penaltylib.BatchedPenalizerOrchestrator = None
    linear_penalties: torch.Tensor = None
    scaling_penalties: torch.Tensor = None

    # Device
    device: str = "cuda"

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
        reqs = batch.reqs
        with batch.input_ids.device:
            temperatures = torch.tensor(
                [r.sampling_params.temperature for r in reqs],
                dtype=torch.float,
            ).view(-1, 1)
            top_ps = torch.tensor(
                [r.sampling_params.top_p for r in reqs], dtype=torch.float
            )
            top_ks = torch.tensor(
                [r.sampling_params.top_k for r in reqs], dtype=torch.int
            )
            min_ps = torch.tensor(
                [r.sampling_params.min_p for r in reqs], dtype=torch.float
            )

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
            vocab_size=vocab_size,
            device=batch.input_ids.device,
        )
        # TODO (lianmin): `need_min_p_sampling` needs to be updated in filter and merge.

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
            device=batch.input_ids.device,
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
                            device=self.device,
                        )
                    self.linear_penalties = penalizer.apply(self.linear_penalties)

    def update_regex_vocab_mask(self):
        has_regex = self.regex_fsms and any(regex_fsm for regex_fsm in self.regex_fsms)

        # Reset the vocab mask
        self.vocab_mask = None

        if has_regex:
            self.vocab_mask = torch.zeros(
                len(self.temperatures),
                self.vocab_size,
                dtype=torch.bool,
                device=self.device,
            )
            for i, regex_fsm in enumerate(self.regex_fsms):
                if regex_fsm is not None:
                    self.vocab_mask[i].fill_(1)
                    self.vocab_mask[i][
                        regex_fsm.get_next_instruction(self.regex_fsm_states[i]).tokens
                    ] = 0

    def filter_batch(self, unfinished_indices: List[int], new_indices: torch.Tensor):
        self.penalizer_orchestrator.filter(unfinished_indices, new_indices)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
            "logit_bias",
        ]:
            value = getattr(self, item, None)
            if value is not None:  # logit_bias can be None
                setattr(self, item, value[new_indices])

    @staticmethod
    def merge_bias_tensor(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        bs1: int,
        bs2: int,
        device: str,
        default: int = 0,
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
                    lhs = torch.empty((bs1, *shape), device=device).fill_(default)
                if rhs is None:
                    rhs = torch.empty((bs2, *shape), device=device).fill_(default)
            return torch.cat([lhs, rhs])

        return None

    def merge_batch(self, other: "SamplingBatchInfo"):
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
            self.logit_bias, other.logit_bias, len(self), len(other), self.device
        )
