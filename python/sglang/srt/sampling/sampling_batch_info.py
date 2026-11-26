from __future__ import annotations

import dataclasses
import logging
import operator
from functools import cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
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
from sglang.srt.utils.device import async_h2d
from sglang.srt.utils.log_utils import print_warning_once

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampler_params import SamplerParams


logger = logging.getLogger(__name__)


def _any_row_with_seed(reqs: List[Req]) -> bool:
    """Whether some row samples with its request's own seed, when deterministic
    mode is off (it seeds every row).

    False under speculative decoding, whose draft graphs have seed buffers only in
    deterministic mode. A greedy row never draws, so its seed does not count.
    """
    if not any(
        r.sampling_params.sampling_seed is not None and r.sampling_params.top_k > 1
        for r in reqs
    ):
        return False
    if get_spec().speculative_algorithm is not None:
        print_warning_once(
            "sampling_seed is ignored with speculative decoding unless "
            "--enable-deterministic-inference is set."
        )
        return False
    return True


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

    # Hashed request ids for the KV canary's token oracle; set in ForwardBatch.
    rids_int: Optional[torch.Tensor] = None
    bootstrap_room_ids_int: Optional[torch.Tensor] = None

    # Set by the scheduler at each launch.
    step_penalties: Optional[StepPenalties] = None

    # Set by the scheduler at each launch (set_plugin_state).
    has_custom_logit_processor: bool = False
    custom_params: Optional[List[Optional[Dict[str, Any]]]] = None
    custom_logit_processor: Optional[Dict[int, ProcessorEntry]] = None

    # Set by the scheduler at each launch (set_seeds_and_mask_rows).
    # Per-row seeds, or None to sample unseeded; deterministic mode seeds every row.
    sampling_seed: Optional[torch.Tensor] = None
    # Rows of the requests that return sampling masks, and each support-mode row's
    # index among them; None when no row has one. They limit capture to those rows.
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
    # The step's grammar objects, None when no request has one (see grammar_mask.py).
    grammars: Optional[List[Optional[BaseGrammarObject]]] = None

    def set_sampler_params(self, params: SamplerParams) -> None:
        for name, value in zip(params._fields, params):
            setattr(self, name, value)

    def set_plugin_state(self, reqs: List[Req]) -> None:
        """Build logit_bias, the grammars, and the custom logit processors from the
        step's requests."""
        self.logit_bias = build_logit_bias(
            reqs, vocab_size=self.vocab_size, device=self.device
        )
        grammars = [r.grammar for r in reqs]
        self.grammars = grammars if any(grammars) else None

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

    def set_seeds_and_mask_rows(self, reqs: List[Req]) -> None:
        """Build the per-row seeds and the sampling-mask rows from the step's requests."""
        enable_deterministic = get_exec().deterministic.enable_deterministic_inference
        seeded = enable_deterministic or _any_row_with_seed(reqs)
        seeds = []
        if seeded:
            for r in reqs:
                seed = r.sampling_params.sampling_seed
                if seed is None:
                    # A deterministic run defaults to 42; otherwise the row stays random.
                    seed = 42 if enable_deterministic else r.filler_seed
                seeds.append(seed)
        mask_rows = [i for i, r in enumerate(reqs) if r.return_sampling_mask]
        capture_rows = [
            j
            for j, i in enumerate(mask_rows)
            if reqs[i].sampling_logprobs_mode == "support"
        ]

        rows = seeds + mask_rows + capture_rows
        device_rows = None
        if rows:
            # One copy for the three int64 lists.
            device_rows = async_h2d(rows, dtype=torch.int64, device=self.device)
        elif seeded:
            # A deterministic idle batch still seeds, with no rows.
            device_rows = torch.empty(0, dtype=torch.int64, device=self.device)
        num_seeds, num_masks = len(seeds), len(mask_rows)
        self.sampling_seed = device_rows[:num_seeds] if seeded else None
        self.sampling_mask_batch_indices = (
            device_rows[num_seeds : num_seeds + num_masks] if mask_rows else None
        )
        self.sampling_support_logprobs_capture_indices = (
            device_rows[num_seeds + num_masks :] if capture_rows else None
        )
        self.sampling_mask_top_ks = (
            [r.sampling_params.top_k if r.return_sampling_mask else 1 for r in reqs]
            if mask_rows
            else None
        )

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
