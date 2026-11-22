from __future__ import annotations

import dataclasses
import logging
import operator
import random
from functools import cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarObject,
    GrammarMask,
    GrammarRow,
)
from sglang.srt.environ import envs
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule, get_spec
from sglang.srt.sampling.custom_logit_processor import (
    UOCR_PAGE_GUARD_PROCESSOR,
    CustomLogitProcessor,
)
from sglang.srt.sampling.logit_bias import build_logit_bias
from sglang.srt.sampling.penalties import StepPenalties
from sglang.srt.sampling.watermarking.interface import WatermarkRows
from sglang.srt.utils.device import async_h2d, is_pin_memory_available
from sglang.srt.utils.log_utils import print_warning_once

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.sampling.sampler_params import SamplerParams


logger = logging.getLogger(__name__)


def _rows_with_seed(
    reqs: List[Req], *, enable_deterministic: bool
) -> Optional[List[bool]]:
    """Which rows sample with their request's own seed, outside deterministic mode.

    None when no row does, in deterministic mode (which seeds every row), and under
    speculative decoding, whose draft graphs have seed buffers only in deterministic
    mode. A greedy row never draws, so its seed does not count.
    """
    if enable_deterministic:
        return None
    rows = [
        r.sampling_params.sampling_seed is not None and r.sampling_params.top_k > 1
        for r in reqs
    ]
    if not any(rows):
        return None
    if get_spec().speculative_algorithm is not None:
        print_warning_once(
            "sampling_seed is ignored with speculative decoding unless "
            "--enable-deterministic-inference is set."
        )
        return None
    return rows


# TP ranks sample independently, so their filler seeds must match. Every rank builds
# the same batches, but other code (e.g. tracing) may draw from the global `random`
# on one rank and not another, which would put the ranks' sequences out of step.
_FILLER_SEED_RNG = random.Random(0)  # Any constant works; it must match across ranks.


def _filler_seed() -> int:
    # The seeded samplers hash each seed as uint64; any int64 works.
    return _FILLER_SEED_RNG.getrandbits(63)


def _filler_seeds(n: int, device: str) -> torch.Tensor:
    return torch.tensor([_filler_seed() for _ in range(n)], dtype=torch.int64).to(
        device, non_blocking=True
    )


@dataclasses.dataclass
class ProcessorEntry:
    """A custom logit processor and the batch rows it applies to."""

    processor: CustomLogitProcessor
    rows: List[int]
    indices: torch.Tensor


@dataclasses.dataclass
class SamplingBatchInfo:
    vocab_size: int

    # The SamplerParams fields, set by the scheduler at each launch.
    temperatures: Optional[torch.Tensor] = None
    top_ps: Optional[torch.Tensor] = None
    top_ks: Optional[torch.Tensor] = None
    min_ps: Optional[torch.Tensor] = None
    is_all_greedy: bool = False
    is_any_greedy: bool = False
    need_top_p_sampling: bool = False
    need_top_k_sampling: bool = False
    need_min_p_sampling: bool = False
    npu_top_k_top_p_eligible: bool = False
    max_top_k: Optional[int] = None

    # Masking tensors for grammar-guided structured outputs
    grammars: Optional[List[Optional[BaseGrammarObject]]] = None
    rids_int: Optional[torch.Tensor] = None
    bootstrap_room_ids_int: Optional[torch.Tensor] = None
    grammar_mask: Optional[GrammarMask] = None

    # Set by the scheduler at each launch.
    step_penalties: Optional[StepPenalties] = None

    # Set by the scheduler at each launch (set_plugin_state).
    has_custom_logit_processor: bool = False
    custom_params: Optional[List[Optional[Dict[str, Any]]]] = None
    custom_logit_processor: Optional[Dict[int, ProcessorEntry]] = None

    # Per-row seeds, or None to sample unseeded; deterministic mode seeds every row.
    sampling_seed: Optional[torch.Tensor] = None
    # Outside deterministic mode, which rows sample with their own seed; None if none do.
    rows_with_seed: Optional[List[bool]] = None

    # CPU request flags and their derived device indices. The flags make batch
    # filtering cheap; the indices keep sampler work limited to opted-in rows.
    return_sampling_masks: Optional[List[bool]] = None
    return_sampling_support_logprobs: Optional[List[bool]] = None
    sampling_mask_batch_indices: Optional[torch.Tensor] = None
    sampling_support_logprobs_capture_indices: Optional[torch.Tensor] = None
    # Per-row top_k of mask requests (1 elsewhere); bounds the capture width.
    sampling_mask_top_ks: Optional[List[int]] = None

    # Per-row watermark policy, None unless some row is a watermark candidate;
    # set by the scheduler at each launch.
    watermark: Optional[WatermarkRows] = None

    # Device
    device: str = "cuda"

    # Set by the scheduler at each launch (set_plugin_state).
    logit_bias: Optional[torch.Tensor] = None

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
        enable_deterministic = get_exec().deterministic.enable_deterministic_inference

        reqs = batch.reqs
        device = batch.device
        _pin = is_pin_memory_available(device)
        rows_with_seed = _rows_with_seed(
            reqs, enable_deterministic=enable_deterministic
        )
        sampling_seed = None
        if enable_deterministic or rows_with_seed is not None:
            seeds = []
            for r in reqs:
                seed = r.sampling_params.sampling_seed
                if seed is None:
                    # A deterministic run defaults to 42; otherwise the row stays random.
                    seed = 42 if enable_deterministic else _filler_seed()
                seeds.append(seed)
            sampling_seed = torch.tensor(seeds, dtype=torch.int64, pin_memory=_pin).to(
                device, non_blocking=True
            )

        return_sampling_masks = [r.return_sampling_mask for r in reqs]
        return_sampling_support_logprobs = [
            r.return_sampling_mask and r.sampling_logprobs_mode == "support"
            for r in reqs
        ]
        sampling_mask_batch_indices = cls._make_sampling_mask_batch_indices(
            return_sampling_masks, device
        )
        sampling_support_logprobs_capture_indices = (
            cls._make_sampling_support_logprobs_capture_indices(
                return_sampling_masks,
                return_sampling_support_logprobs,
                device,
            )
        )
        sampling_mask_top_ks = [
            r.sampling_params.top_k if r.return_sampling_mask else 1 for r in reqs
        ]

        ret = cls(
            sampling_seed=sampling_seed,
            rows_with_seed=rows_with_seed,
            vocab_size=vocab_size,
            device=device,
            return_sampling_masks=return_sampling_masks,
            return_sampling_support_logprobs=return_sampling_support_logprobs,
            sampling_mask_batch_indices=sampling_mask_batch_indices,
            sampling_support_logprobs_capture_indices=(
                sampling_support_logprobs_capture_indices
            ),
            sampling_mask_top_ks=sampling_mask_top_ks,
        )
        ret.adjusted_from_schedule_batch(batch, vocab_size)
        return ret

    # placeholder for override
    def adjusted_from_schedule_batch(self, batch: ScheduleBatch, vocab_size: int):
        pass

    # placeholder for override
    def adjusted_merge_batch(self, other: SamplingBatchInfo):
        pass

    # placeholder for override
    def adjusted_filter_batch(
        self, keep_indices: List[int], keep_indices_device: torch.Tensor
    ):
        pass

    @staticmethod
    def _make_sampling_mask_batch_indices(
        return_sampling_masks: List[bool], device: str
    ) -> Optional[torch.Tensor]:
        """Build device row indices for requests that return sampling masks."""
        indices = [
            i for i, should_return in enumerate(return_sampling_masks) if should_return
        ]
        if not indices:
            return None
        return async_h2d(indices, dtype=torch.long, device=device)

    @staticmethod
    def _make_sampling_support_logprobs_capture_indices(
        return_sampling_masks: List[bool],
        return_sampling_support_logprobs: Optional[List[bool]],
        device: str,
    ) -> Optional[torch.Tensor]:
        """Map support-mode requests to rows in the compact mask capture."""
        if return_sampling_support_logprobs is None:
            return None
        capture_index = -1
        indices = []
        for return_mask, return_support in zip(
            return_sampling_masks,
            return_sampling_support_logprobs,
            strict=True,
        ):
            if return_mask:
                capture_index += 1
            if return_support:
                indices.append(capture_index)
        if not indices:
            return None
        return async_h2d(indices, dtype=torch.long, device=device)

    def set_sampler_params(self, params: SamplerParams) -> None:
        for name, value in zip(params._fields, params):
            setattr(self, name, value)

    def set_plugin_state(self, reqs: List[Req]) -> None:
        """Build logit_bias and the custom logit processors from the step's requests."""
        self.logit_bias = build_logit_bias(
            reqs, vocab_size=self.vocab_size, device=self.device
        )

        # Group the step's requests by custom logit processor.
        enable_custom_processor = get_exec().features.enable_custom_logit_processor
        self.has_custom_logit_processor = any(
            r.custom_logit_processor
            and (
                enable_custom_processor
                or r.custom_logit_processor == UOCR_PAGE_GUARD_PROCESSOR
            )
            for r in reqs
        )
        if not self.has_custom_logit_processor:
            self.custom_params = None
            self.custom_logit_processor = None
            return
        rows_by_processor: Dict[str, List[int]] = {}
        for i, r in enumerate(reqs):
            if r.custom_logit_processor is not None:
                rows_by_processor.setdefault(r.custom_logit_processor, []).append(i)
        self.custom_params = [r.sampling_params.custom_params for r in reqs]
        self.custom_logit_processor = {
            hash(processor_str): ProcessorEntry(
                processor=CustomLogitProcessor.from_str(processor_str),
                rows=rows,
                indices=async_h2d(rows, dtype=torch.long, device=self.device),
            )
            for processor_str, rows in rows_by_processor.items()
        }

    def update_regex_vocab_mask(self):
        if not self.grammars:
            self.grammar_mask = None
            return

        # Find a grammar from the list
        first_grammar = next(grammar for grammar in self.grammars if grammar)

        vocab_mask = first_grammar.allocate_vocab_mask(
            vocab_size=self.vocab_size,
            batch_size=len(self.grammars),
            device=self.device,
        )

        # Rows omitted here (finished / terminated / non-grammar requests) retain
        # the freshly allocated buffer's unconstrained value.
        entries = [
            GrammarRow(row=row, grammar=grammar)
            for row, grammar in enumerate(self.grammars)
            if grammar and not grammar.finished and not grammar.is_terminated()
        ]
        first_grammar.fill_vocab_mask_batched(entries, vocab_mask)

        # Move the mask to the device if needed
        vocab_mask = first_grammar.move_vocab_mask(vocab_mask, self.device)
        self.grammar_mask = GrammarMask(first_grammar, vocab_mask)

    def filter_batch(self, keep_indices: List[int], keep_indices_device: torch.Tensor):
        if self.rows_with_seed is not None:
            self.rows_with_seed = [self.rows_with_seed[i] for i in keep_indices]
            if not any(self.rows_with_seed):
                # The last seeded row left; go back to the unseeded samplers.
                self.rows_with_seed = None
                self.sampling_seed = None

        if self.sampling_seed is not None:
            self.sampling_seed = self.sampling_seed[keep_indices_device]

        if self.return_sampling_masks is not None:
            self.return_sampling_masks = [
                self.return_sampling_masks[i] for i in keep_indices
            ]
            if self.sampling_mask_top_ks is not None:
                self.sampling_mask_top_ks = [
                    self.sampling_mask_top_ks[i] for i in keep_indices
                ]
            if self.return_sampling_support_logprobs is not None:
                self.return_sampling_support_logprobs = [
                    self.return_sampling_support_logprobs[i] for i in keep_indices
                ]
            if self.sampling_mask_batch_indices is not None:
                self.sampling_mask_batch_indices = (
                    self._make_sampling_mask_batch_indices(
                        self.return_sampling_masks, self.device
                    )
                )
                self.sampling_support_logprobs_capture_indices = (
                    self._make_sampling_support_logprobs_capture_indices(
                        self.return_sampling_masks,
                        self.return_sampling_support_logprobs,
                        self.device,
                    )
                )

        self.adjusted_filter_batch(keep_indices, keep_indices_device)

    def merge_batch(self, other: SamplingBatchInfo, *, self_len: int, other_len: int):
        if (
            self.return_sampling_masks is not None
            or other.return_sampling_masks is not None
        ):
            if other.sampling_mask_batch_indices is not None:
                other_sampling_mask_batch_indices = (
                    other.sampling_mask_batch_indices + self_len
                )
                self.sampling_mask_batch_indices = (
                    other_sampling_mask_batch_indices
                    if self.sampling_mask_batch_indices is None
                    else torch.cat(
                        [
                            self.sampling_mask_batch_indices,
                            other_sampling_mask_batch_indices,
                        ]
                    )
                )
            self.return_sampling_masks = (
                self.return_sampling_masks or [False] * self_len
            ) + (other.return_sampling_masks or [False] * other_len)
            self.return_sampling_support_logprobs = (
                self.return_sampling_support_logprobs or [False] * self_len
            ) + (other.return_sampling_support_logprobs or [False] * other_len)
            self.sampling_support_logprobs_capture_indices = (
                self._make_sampling_support_logprobs_capture_indices(
                    self.return_sampling_masks,
                    self.return_sampling_support_logprobs,
                    self.device,
                )
            )
            self.sampling_mask_top_ks = (
                self.sampling_mask_top_ks or [1] * self_len
            ) + (other.sampling_mask_top_ks or [1] * other_len)

        if self.rows_with_seed is not None or other.rows_with_seed is not None:
            self.rows_with_seed = (self.rows_with_seed or [False] * self_len) + (
                other.rows_with_seed or [False] * other_len
            )
            # Rows of an unseeded side sample with random seeds, as in from_schedule_batch.
            if self.sampling_seed is None:
                self.sampling_seed = _filler_seeds(self_len, self.device)
            if other.sampling_seed is None:
                other.sampling_seed = _filler_seeds(other_len, self.device)

        if self.sampling_seed is not None and other.sampling_seed is not None:
            self.sampling_seed = torch.cat([self.sampling_seed, other.sampling_seed])

        self.adjusted_merge_batch(other)

    def copy_for_forward(self):
        if _fast_copy_enabled() and type(self) is SamplingBatchInfo:
            new = object.__new__(SamplingBatchInfo)
            new.__dict__.update(zip(_FIELD_NAMES, _FIELD_GETTER(self.__dict__)))
        else:
            new = dataclasses.replace(self)
        return new


def adjusts_logits(info: SamplingBatchInfo, *, has_grammar: bool) -> bool:
    """Whether sampling edits the logits before choosing tokens. has_grammar comes
    from the batch: verify builds its grammar mask per step, outside the sampling info.
    """
    return (
        has_grammar
        or info.step_penalties is not None
        or info.logit_bias is not None
        or info.has_custom_logit_processor
        or info.watermark is not None
    )


def need_truncated_sampling(info: SamplingBatchInfo) -> bool:
    """Whether some row truncates its distribution with top-k, top-p, or min-p."""
    return (
        info.need_top_k_sampling or info.need_top_p_sampling or info.need_min_p_sampling
    )


@cache
def _fast_copy_enabled() -> bool:
    # Preserve the original EAGLE runtime policy without the resource-pool dependency.
    return (
        current_platform.is_cuda()
        and envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.get()
        and (get_spec().speculative_algorithm or "").upper() == "EAGLE"
        and (get_spec().speculative_eagle_topk or 1) == 1
        and not get_schedule().disable_overlap_schedule
        and get_parallel().moe_ep_size == 1
    )


_FIELD_NAMES = tuple(f.name for f in dataclasses.fields(SamplingBatchInfo))
_FIELD_GETTER = operator.itemgetter(*_FIELD_NAMES)
