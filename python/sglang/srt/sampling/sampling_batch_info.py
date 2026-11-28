from __future__ import annotations

import dataclasses
import logging
import operator
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.runtime_context import get_exec
from sglang.srt.sampling.custom_logit_processor import (
    UOCR_PAGE_GUARD_PROCESSOR,
    CustomLogitProcessor,
)
from sglang.srt.sampling.logit_bias import build_logit_bias
from sglang.srt.sampling.penalties import StepPenalties
from sglang.srt.sampling.watermarking.interface import WatermarkRows
from sglang.srt.utils.device import async_h2d

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params_pool import SamplingRows


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProcessorEntry:
    """A custom logit processor and the batch rows it applies to."""

    processor: CustomLogitProcessor
    rows: List[int]
    indices: torch.Tensor


@dataclasses.dataclass
class SamplingBatchInfo:
    vocab_size: int

    # The sampler params, set by the scheduler at each launch (set_rows).
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

    # Set by the scheduler at each launch (set_rows).
    step_penalties: Optional[StepPenalties] = None

    # Set by the scheduler at each launch (set_step_state).
    has_custom_logit_processor: bool = False
    custom_params: Optional[List[Optional[Dict[str, Any]]]] = None
    custom_logit_processor: Optional[Dict[int, ProcessorEntry]] = None

    # Set by the scheduler at each launch (set_rows).
    # Per-row seeds, or None to sample unseeded; deterministic mode seeds every row.
    sampling_seed: Optional[torch.Tensor] = None

    # Set by the scheduler at each launch (set_step_state).
    # Rows of the requests that return sampling masks, and each support-mode row's
    # index among them; None when no row has one. They limit capture to those rows.
    sampling_mask_batch_indices: Optional[torch.Tensor] = None
    sampling_support_logprobs_capture_indices: Optional[torch.Tensor] = None
    # Per-row top_k of mask requests (1 elsewhere); bounds the capture width.
    sampling_mask_top_ks: Optional[List[int]] = None

    # Per-row watermark policy, None unless some row is a watermark candidate;
    # set by the scheduler at each launch (set_rows).
    watermark: Optional[WatermarkRows] = None

    # Device
    device: str = "cuda"

    # Set by the scheduler at each launch (set_step_state).
    logit_bias: Optional[torch.Tensor] = None
    # The step's grammar objects, None when no request has one (see grammar_mask.py).
    grammars: Optional[List[Optional[BaseGrammarObject]]] = None

    def set_rows(self, rows: SamplingRows) -> None:
        """Set the step's state from its rows of `SamplingParamsPool`."""
        for name, value in zip(rows._fields, rows):
            setattr(self, name, value)

    def set_step_state(self, reqs: List[Req]) -> None:
        """Build the stages' step state from the step's requests: logit_bias, the
        grammars, the custom logit processors, and the sampling-mask rows."""
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
        if self.has_custom_logit_processor:
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
        else:
            self.custom_params = None
            self.custom_logit_processor = None

        mask_rows = [i for i, r in enumerate(reqs) if r.return_sampling_mask]
        capture_rows = [
            j
            for j, i in enumerate(mask_rows)
            if reqs[i].sampling_logprobs_mode == "support"
        ]
        if mask_rows:
            # One copy for both int64 lists.
            device_rows = async_h2d(
                mask_rows + capture_rows, dtype=torch.int64, device=self.device
            )
            self.sampling_mask_batch_indices = device_rows[: len(mask_rows)]
            self.sampling_support_logprobs_capture_indices = (
                device_rows[len(mask_rows) :] if capture_rows else None
            )
        else:
            self.sampling_mask_batch_indices = None
            self.sampling_support_logprobs_capture_indices = None
        self.sampling_mask_top_ks = (
            [r.sampling_params.top_k if r.return_sampling_mask else 1 for r in reqs]
            if mask_rows
            else None
        )

    def copy_for_forward(self):
        if type(self) is SamplingBatchInfo:
            # The same fields as dataclasses.replace (no __post_init__, no init=False
            # field), without a keyword __init__ call per forward.
            new = object.__new__(SamplingBatchInfo)
            new.__dict__.update(zip(_FIELD_NAMES, _FIELD_GETTER(self.__dict__)))
        else:
            new = dataclasses.replace(self)
        return new


def need_truncated_sampling(info: SamplingBatchInfo) -> bool:
    """Whether some row truncates its distribution with top-k, top-p, or min-p."""
    return (
        info.need_top_k_sampling or info.need_top_p_sampling or info.need_min_p_sampling
    )


_FIELD_NAMES = tuple(f.name for f in dataclasses.fields(SamplingBatchInfo))
_FIELD_GETTER = operator.itemgetter(*_FIELD_NAMES)
