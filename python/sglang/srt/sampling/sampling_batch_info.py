from __future__ import annotations

import dataclasses
import logging
import threading
from typing import TYPE_CHECKING, Callable, List, Optional

import torch

import sglang.srt.sampling.penaltylib as penaltylib

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


@dataclasses.dataclass
class SamplingBatchInfo:
    # Batched sampling params
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor

    # All requests use greedy sampling
    is_all_greedy: bool

    # Dispatch in CUDA graph
    need_min_p_sampling: bool

    # Bias Tensors
    vocab_size: int
    grammars: Optional[List] = None
    sampling_info_done: Optional[threading.Event] = None
    logit_bias: torch.Tensor = None
    vocab_mask: Optional[torch.Tensor] = None
    apply_mask: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None

    # Penalizer
    penalizer_orchestrator: Optional[penaltylib.BatchedPenalizerOrchestrator] = None
    linear_penalties: Optional[torch.Tensor] = None
    scaling_penalties: Optional[torch.Tensor] = None

    # Device
    device: str = "cuda"

    @classmethod
    def from_schedule_batch(
        cls, batch: ScheduleBatch, vocab_size: int, enable_overlap_schedule: bool
    ):
        reqs = batch.reqs
        device = batch.device
        temperatures = (
            torch.tensor(
                [r.sampling_params.temperature for r in reqs],
                dtype=torch.float,
            )
            .view(-1, 1)
            .to(device, non_blocking=True)
        )
        top_ps = torch.tensor(
            [r.sampling_params.top_p for r in reqs], dtype=torch.float
        ).to(device, non_blocking=True)
        top_ks = torch.tensor(
            [r.sampling_params.top_k for r in reqs], dtype=torch.int32
        ).to(device, non_blocking=True)
        min_ps = torch.tensor(
            [r.sampling_params.min_p for r in reqs], dtype=torch.float
        ).to(device, non_blocking=True)

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
            vocab_size=vocab_size,
            device=device,
        )
        # TODO (lianmin): `need_min_p_sampling` needs to be updated in filter and merge.

        if enable_overlap_schedule:
            # TODO (lianmin): Some penalizers such as frequency and presence depend on model outputs,
            # so it is kind of tricky to make it work with overlap scheduler.
            # It requires correcly updating the penalty logits before the sampling and syncing the events.
            # We will support them later.
            penalizers = {
                penaltylib.BatchedMinNewTokensPenalizer,
            }
            if (
                any(req.sampling_params.frequency_penalty != 0.0 for req in reqs)
                or any(req.sampling_params.presence_penalty != 0.0 for req in reqs)
                or any(req.sampling_params.repetition_penalty != 1.0 for req in reqs)
            ):
                logger.warning(
                    "frequency_penalty, presence_penalty, and repetition_penalty are not supported "
                    "when using the default overlap scheduler. They will be ignored. "
                    "Please add `--disable-overlap` when launching the server if you need these features. "
                    "The speed will be slower in that case."
                )
        else:
            penalizers = {
                penaltylib.BatchedFrequencyPenalizer,
                penaltylib.BatchedMinNewTokensPenalizer,
                penaltylib.BatchedPresencePenalizer,
                penaltylib.BatchedRepetitionPenalizer,
            }

        # Each penalizers will do nothing if they evaluate themselves as not required by looking at
        # the sampling_params of the requests (See {_is_required()} of each penalizers). So this
        # should not add hefty computation overhead other than simple checks.
        #
        # While we choose not to even create the class instances if they are not required, this
        # could add additional complexity to the {ScheduleBatch} class, especially we need to
        # handle {filter_batch()} and {merge_batch()} cases as well.
        ret.penalizer_orchestrator = penaltylib.BatchedPenalizerOrchestrator(
            vocab_size=vocab_size,
            batch=batch,
            device=batch.device,
            Penalizers=penalizers,
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
            if not penalizer.is_prepared():
                continue

            if isinstance(penalizer, penaltylib.BatchedRepetitionPenalizer):
                self.scaling_penalties = penalizer.cumulated_repetition_penalties
            else:
                if self.linear_penalties is None:
                    bs = self.penalizer_orchestrator.batch.batch_size()
                    self.linear_penalties = torch.zeros(
                        (bs, self.vocab_size),
                        dtype=torch.float32,
                        device=self.device,
                    )
                self.linear_penalties = penalizer.apply(self.linear_penalties)

    def update_regex_vocab_mask(self):
        if not self.grammars:
            self.vocab_mask = None
            self.apply_mask = None
            return

        # find a grammar from the list
        first_grammar = next(grammar for grammar in self.grammars if grammar)

        # maybe we can reuse the existing mask?
        self.vocab_mask = first_grammar.allocate_vocab_mask(
            vocab_size=self.vocab_size,
            batch_size=len(self.temperatures),
            device=self.device,
        )
        self.apply_mask = first_grammar.apply_vocab_mask  # force to use static method

        # Apply the mask
        for i, grammar in enumerate(self.grammars):
            if grammar and not grammar.finished:
                grammar.fill_vocab_mask(self.vocab_mask, i)

        # Move the mask to the device if needed
        self.vocab_mask = first_grammar.move_vocab_mask(self.vocab_mask, self.device)

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

        self.is_all_greedy = self.is_all_greedy and other.is_all_greedy
        self.logit_bias = SamplingBatchInfo.merge_bias_tensor(
            self.logit_bias, other.logit_bias, len(self), len(other), self.device
        )
