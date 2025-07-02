from __future__ import annotations

import dataclasses
import logging
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

import sglang.srt.sampling.penaltylib as penaltylib
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.utils import merge_bias_tensor

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SamplingBatchInfo:
    # Basic batched sampling params
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor

    # Whether all requests use greedy sampling
    is_all_greedy: bool

    # Whether any requests use top_p sampling
    need_top_p_sampling: bool

    # Whether any requests use top_k sampling
    need_top_k_sampling: bool

    # Whether any request needs min_p sampling
    need_min_p_sampling: bool

    # Masking tensors for grammar-guided structured outputs
    vocab_size: int
    grammars: Optional[List] = None
    vocab_mask: Optional[torch.Tensor] = None
    apply_mask_func: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None

    # An event used for overlap schedule
    sampling_info_done: Optional[threading.Event] = None

    # Penalizer
    penalizer_orchestrator: Optional[penaltylib.BatchedPenalizerOrchestrator] = None
    linear_penalty: torch.Tensor = None

    # Whether any request has custom logit processor
    has_custom_logit_processor: bool = False
    # Custom parameters
    custom_params: Optional[List[Optional[Dict[str, Any]]]] = None
    # Custom logit processor
    custom_logit_processor: Optional[
        Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]
    ] = None

    # Device
    device: str = "cuda"

    # Handle logit bias
    logit_bias: Optional[torch.Tensor] = None

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
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

        logit_bias = None
        if any(r.sampling_params.logit_bias is not None for r in reqs):
            logit_bias = torch.zeros(len(reqs), vocab_size, device=device)
            for i, r in enumerate(reqs):
                if r.sampling_params.logit_bias is not None:
                    for key, value in r.sampling_params.logit_bias.items():
                        logit_bias[i, int(key)] = value

        # Check if any request has custom logit processor
        has_custom_logit_processor = (
            batch.enable_custom_logit_processor  # check the flag first.
            and any(r.custom_logit_processor for r in reqs)  # then check the requests.
        )

        if has_custom_logit_processor:
            # Merge the same type of custom logit processors together
            processor_dict = {}
            for i, r in enumerate(reqs):
                if r.custom_logit_processor is None:
                    continue
                processor_str = r.custom_logit_processor
                if processor_str not in processor_dict:
                    processor_dict[processor_str] = []
                processor_dict[processor_str].append(i)

            merged_custom_logit_processor = {
                hash(processor_str): (
                    # The deserialized custom logit processor object
                    CustomLogitProcessor.from_str(processor_str),
                    # The mask tensor for the requests that use this custom logit processor
                    torch.zeros(len(reqs), dtype=torch.bool)
                    .scatter_(0, torch.tensor(true_indices), True)
                    .to(device, non_blocking=True),
                )
                for processor_str, true_indices in processor_dict.items()
            }
            custom_params = [r.sampling_params.custom_params for r in reqs]
        else:
            merged_custom_logit_processor = None
            custom_params = None

        # Each penalizers will do nothing if they evaluate themselves as not required by looking at
        # the sampling_params of the requests (See {_is_required()} of each penalizers). So this
        # should not add hefty computation overhead other than simple checks.
        #
        # While we can choose not to even create the class instances if they are not required, this
        # could add additional complexity to the {ScheduleBatch} class, especially we need to
        # handle {filter_batch()} and {merge_batch()} cases as well.
        penalizer_orchestrator = penaltylib.BatchedPenalizerOrchestrator(
            vocab_size=vocab_size,
            batch=batch,
            penalizers={
                penaltylib.BatchedFrequencyPenalizer,
                penaltylib.BatchedMinNewTokensPenalizer,
                penaltylib.BatchedPresencePenalizer,
            },
        )

        ret = cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
            need_top_p_sampling=any(r.sampling_params.top_p != 1.0 for r in reqs),
            need_top_k_sampling=any(r.sampling_params.top_k != TOP_K_ALL for r in reqs),
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
            vocab_size=vocab_size,
            penalizer_orchestrator=penalizer_orchestrator,
            has_custom_logit_processor=has_custom_logit_processor,
            custom_params=custom_params,
            custom_logit_processor=merged_custom_logit_processor,
            device=device,
            logit_bias=logit_bias,
        )
        return ret

    def __len__(self):
        return len(self.temperatures)

    def update_regex_vocab_mask(self):
        if not self.grammars:
            self.vocab_mask = None
            self.apply_mask_func = None
            return

        # Find a grammar from the list
        first_grammar = next(grammar for grammar in self.grammars if grammar)

        # TODO(lianmin): Maybe we can reuse the existing mask?
        self.vocab_mask = first_grammar.allocate_vocab_mask(
            vocab_size=self.vocab_size,
            batch_size=len(self.temperatures),
            device=self.device,
        )
        self.apply_mask_func = (
            first_grammar.apply_vocab_mask
        )  # force to use static method

        # Apply the mask
        for i, grammar in enumerate(self.grammars):
            if grammar and not grammar.finished and not grammar.is_terminated():
                grammar.fill_vocab_mask(self.vocab_mask, i)

        # Move the mask to the device if needed
        self.vocab_mask = first_grammar.move_vocab_mask(self.vocab_mask, self.device)

    def update_penalties(self):
        if self.penalizer_orchestrator.is_required:
            self.linear_penalty = torch.zeros(
                (len(self.temperatures), self.vocab_size),
                dtype=torch.float32,
                device=self.temperatures.device,
            )
            self.penalizer_orchestrator.apply(self.linear_penalty)
        else:
            self.linear_penalty = None

    def apply_logits_bias(self, logits: torch.Tensor):
        if self.linear_penalty is not None:
            # Used in the overlap mode
            logits.add_(self.linear_penalty)

        if self.penalizer_orchestrator and self.penalizer_orchestrator.is_required:
            # Used in the non-overlap mode
            self.penalizer_orchestrator.apply(logits)

        if self.vocab_mask is not None:
            self.apply_mask_func(logits=logits, vocab_mask=self.vocab_mask)

        if self.logit_bias is not None:
            logits.add_(self.logit_bias)

    def filter_batch(self, keep_indices: List[int], keep_indices_device: torch.Tensor):
        self.penalizer_orchestrator.filter(keep_indices_device)

        if self.has_custom_logit_processor:
            self._filter_batch_custom_logit_processor(keep_indices, keep_indices_device)

        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
        ]:
            value = getattr(self, item, None)
            setattr(self, item, value[keep_indices_device])

        if self.logit_bias is not None:
            self.logit_bias = self.logit_bias[keep_indices_device]

    def _filter_batch_custom_logit_processor(
        self, keep_indices: List[int], keep_indices_device: torch.Tensor
    ):
        """Filter the custom logit processor and custom params"""
        self.custom_logit_processor = {
            k: (p, mask[keep_indices_device])
            for k, (p, mask) in self.custom_logit_processor.items()
            if torch.any(
                mask[keep_indices_device]
            )  # ignore the custom logit processor whose mask is all False
        }
        self.custom_params = [self.custom_params[i] for i in keep_indices]

        # If the custom logit processor is an empty dict, set the flag to False,
        # and set the custom logit processor and custom params to None.
        if len(self.custom_logit_processor) == 0:
            self.custom_logit_processor = None
            self.custom_params = None
            self.has_custom_logit_processor = False

    @staticmethod
    def merge_custom_logit_processor(
        lhs: Optional[Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]],
        rhs: Optional[Dict[int, Tuple[CustomLogitProcessor, torch.Tensor]]],
        bs1: int,
        bs2: int,
        device: str,
    ):
        if lhs is None and rhs is None:
            return None
        lhs, rhs = lhs or {}, rhs or {}

        keys = set(lhs.keys()).union(set(rhs.keys()))
        merged_dict = {}

        for k in keys:
            # Get the logit processor object
            processor = lhs[k][0] if k in lhs else rhs[k][0]
            # Get and merge the mask tensors from the two dicts
            left_mask = (
                lhs[k][1]
                if k in lhs
                else torch.zeros(bs1, dtype=torch.bool, device=device)
            )
            right_mask = (
                rhs[k][1]
                if k in rhs
                else torch.zeros(bs2, dtype=torch.bool, device=device)
            )
            merged_dict[k] = (processor, torch.cat([left_mask, right_mask]))

            assert merged_dict[k][1].shape[0] == bs1 + bs2, (
                f"The batch size of merged mask ({merged_dict[k][1].shape[0]}) does not match "
                f"the sum of the batch sizes of the two masks ({bs1 + bs2})"
                f"\n{left_mask=}\n{right_mask=}\n{bs1=}\n{bs2=}"
                f"\n{lhs=}\n{rhs=}"
            )

        return merged_dict

    def merge_batch(self, other: "SamplingBatchInfo"):
        self.penalizer_orchestrator.merge(other.penalizer_orchestrator)

        # Merge the custom logit processors and custom params lists
        if self.has_custom_logit_processor or other.has_custom_logit_processor:
            # Merge the custom logit processors
            self.custom_logit_processor = (
                SamplingBatchInfo.merge_custom_logit_processor(
                    self.custom_logit_processor,
                    other.custom_logit_processor,
                    len(self),
                    len(other),
                    self.device,
                )
            )
            # Merge the custom params lists
            self.custom_params = self.custom_params or [None] * len(self)
            other.custom_params = other.custom_params or [None] * len(other)
            self.custom_params.extend(other.custom_params)

            # Set the flag to True if any of the two has custom logit processor
            self.has_custom_logit_processor = True

        # Note: because the __len()__ operator is defined on the temperatures tensor,
        # please make sure any merge operation with len(self) or len(other) is done before
        # the merge operation of the temperatures tensor below.
        for item in [
            "temperatures",
            "top_ps",
            "top_ks",
            "min_ps",
        ]:
            self_val = getattr(self, item, None)
            other_val = getattr(other, item, None)
            setattr(self, item, torch.cat([self_val, other_val]))

        self.is_all_greedy &= other.is_all_greedy
        self.need_top_p_sampling |= other.need_top_p_sampling
        self.need_top_k_sampling |= other.need_top_k_sampling
        self.need_min_p_sampling |= other.need_min_p_sampling

        # Merge logit bias
        self.logit_bias = merge_bias_tensor(
            self.logit_bias, other.logit_bias, len(self), len(other), self.device, 0.0
        )
