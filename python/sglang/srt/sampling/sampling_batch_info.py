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

    def has_bias(self):
        return (
            self.logit_bias is not None
            or self.vocab_mask is not None
            or self.linear_penalties is not None
            or self.scaling_penalties is not None
        )

    @classmethod
    def dummy_one(cls, max_bs: int, vocab_size: int):
        ret = cls(vocab_size=vocab_size)
        ret.temperatures = torch.ones((max_bs, 1), dtype=torch.float, device="cuda")
        ret.top_ps = torch.ones((max_bs,), dtype=torch.float, device="cuda")
        ret.top_ks = torch.ones((max_bs,), dtype=torch.int, device="cuda")
        ret.min_ps = torch.zeros((max_bs,), dtype=torch.float, device="cuda")
        return ret

    def __getitem__(self, key):
        if isinstance(key, slice):
            # NOTE: We do not use cuda graph when there is bias tensors
            assert not self.has_bias()
            return SamplingBatchInfo(
                vocab_size=self.vocab_size,
                temperatures=self.temperatures[key],
                top_ps=self.top_ps[key],
                top_ks=self.top_ks[key],
                min_ps=self.min_ps[key],
                need_min_p_sampling=self.need_min_p_sampling,
            )
        else:
            raise NotImplementedError

    def inplace_assign(self, bs: int, other: SamplingBatchInfo):
        # NOTE: We do not use cuda graph when there is bias tensors
        assert not self.has_bias()

        self.vocab_size = other.vocab_size
        self.need_min_p_sampling = other.need_min_p_sampling

        self.temperatures[:bs] = other.temperatures
        self.top_ps[:bs] = other.top_ps
        self.top_ks[:bs] = other.top_ks
        self.min_ps[:bs] = other.min_ps

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
        device = "cuda"
        reqs = batch.reqs
        ret = cls(vocab_size=vocab_size)

        ret.temperatures = torch.tensor(
            [r.sampling_params.temperature for r in reqs],
            dtype=torch.float,
            device=device,
        ).view(-1, 1)
        ret.top_ps = torch.tensor(
            [r.sampling_params.top_p for r in reqs], dtype=torch.float, device=device
        )
        ret.top_ks = torch.tensor(
            [r.sampling_params.top_k for r in reqs], dtype=torch.int, device=device
        )
        ret.min_ps = torch.tensor(
            [r.sampling_params.min_p for r in reqs], dtype=torch.float, device=device
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
            device=device,
            Penalizers={
                penaltylib.BatchedFrequencyPenalizer,
                penaltylib.BatchedMinNewTokensPenalizer,
                penaltylib.BatchedPresencePenalizer,
                penaltylib.BatchedRepetitionPenalizer,
            },
        )

        # Handle logit bias but only allocate when needed
        ret.logit_bias = None

        ret.update_regex_vocab_mask(batch)

        return ret

    def prepare_penalties(self):
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
        bs, reqs = batch.batch_size(), batch.reqs
        device = "cuda"
        has_regex = any(req.regex_fsm is not None for req in reqs)

        # Reset the vocab mask
        self.vocab_mask = None

        if has_regex:
            for i, req in enumerate(reqs):
                if req.regex_fsm is not None:
                    if self.vocab_mask is None:
                        self.vocab_mask = torch.zeros(
                            bs, self.vocab_size, dtype=torch.bool, device=device
                        )
                    self.vocab_mask[i][
                        req.regex_fsm.get_next_instruction(req.regex_fsm_state).tokens
                    ] = 1

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

        # logit_bias can be None
        if self.logit_bias is not None or other.logit_bias is not None:
            vocab_size = (
                self.logit_bias.shape[1]
                if self.logit_bias is not None
                else other.logit_bias.shape[1]
            )
            if self.logit_bias is None:
                self.logit_bias = torch.zeros(
                    (len(self.reqs), vocab_size), dtype=torch.float32, device="cuda"
                )
            if other.logit_bias is None:
                other.logit_bias = torch.zeros(
                    (len(other.reqs), vocab_size), dtype=torch.float32, device="cuda"
                )
            self.logit_bias = torch.concat([self.logit_bias, other.logit_bias])
