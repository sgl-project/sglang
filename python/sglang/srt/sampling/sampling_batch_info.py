"""Sampling metadata of one step, built from the step's requests.

Invariants:
- `build` reads only each request's params, `req.grammar`, and whether it
  holds a penalty row, and has no side effects, so building twice gives equal
  infos.
- A different set of requests gets a new info, never a filtered or merged one.
- A forward fills `grammar_mask`, `rids_int`, and `bootstrap_room_ids_int` on
  its own copy (`forward_sampling_info`); the step's info keeps its built
  values, except that the MLX worker sets `grammars` on it.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.constrained.base_grammar_backend import (
    BaseGrammarObject,
    GrammarMask,
    GrammarRow,
)
from sglang.srt.runtime_context import get_exec
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.sampling.penalties import (
    PenaltyStore,
    StepPenalties,
    apply_scaling_penalties,
)
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.sampling.watermarking.core import build_watermark_batch_config
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_observer import SamplingObserver


logger = logging.getLogger(__name__)


def _rows_to_device_indices(
    rows: List[int], device: str, pin: Optional[bool] = None
) -> torch.Tensor:
    """Move batch row numbers to the device without blocking on a pageable copy."""
    if pin is None:
        pin = is_pin_memory_available(device)
    return torch.tensor(rows, dtype=torch.long, pin_memory=pin).to(
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
    # Basic batched sampling params
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor

    # Whether all requests use greedy sampling
    is_all_greedy: bool

    is_any_greedy: bool

    # Whether any requests use top_p sampling
    need_top_p_sampling: bool

    # Whether any requests use top_k sampling
    need_top_k_sampling: bool

    # Whether any request needs min_p sampling
    need_min_p_sampling: bool

    # Masking tensors for grammar-guided structured outputs
    vocab_size: int
    grammars: Optional[List[Optional[BaseGrammarObject]]] = None
    rids_int: Optional[torch.Tensor] = None
    bootstrap_room_ids_int: Optional[torch.Tensor] = None
    grammar_mask: Optional[GrammarMask] = None

    # Penalized requests and their slots; None when no request uses one.
    penalties: Optional[StepPenalties] = None

    # Whether any request has custom logit processor
    has_custom_logit_processor: bool = False
    # Custom parameters
    custom_params: Optional[List[Optional[Dict[str, Any]]]] = None
    # Custom logit processor
    custom_logit_processor: Optional[Dict[int, ProcessorEntry]] = None

    # Used for deterministic sampling
    sampling_seed: Optional[torch.Tensor] = None

    # CPU request flags and their derived device indices. The indices keep
    # sampler work limited to opted-in rows.
    return_sampling_masks: Optional[List[bool]] = None
    return_sampling_support_logprobs: Optional[List[bool]] = None
    sampling_mask_batch_indices: Optional[torch.Tensor] = None
    sampling_support_logprobs_capture_indices: Optional[torch.Tensor] = None
    # Per-row top_k of mask requests (1 elsewhere); bounds the capture width.
    sampling_mask_top_ks: Optional[List[int]] = None

    watermark_keys: Optional[torch.Tensor] = None
    watermark_context_windows: Optional[torch.Tensor] = None
    watermark_enabled: Optional[torch.Tensor] = None
    watermark_candidates_host: Optional[List[bool]] = None
    has_watermark_candidates: bool = False

    # Device
    device: str = "cuda"

    # Handle logit bias
    logit_bias: Optional[torch.Tensor] = None

    # Host-side eligibility for torch_npu.npu_top_k_top_p. Keeping this off the
    # device avoids a scalar synchronization in the per-token sampling path.
    npu_top_k_top_p_eligible: bool = False

    # Host-side upper bound on every row's top_k, or None when some row samples
    # from the whole vocabulary.
    max_top_k: Optional[int] = None

    @classmethod
    def build(
        cls,
        reqs: Sequence,
        vocab_size: int,
        device: str,
        penalty_store: Optional[PenaltyStore] = None,
        req_pool_indices: Optional[torch.Tensor] = None,
    ) -> SamplingBatchInfo:
        enable_deterministic = get_exec().deterministic.enable_deterministic_inference

        _pin = is_pin_memory_available(device)
        temperatures = (
            torch.tensor(
                [r.sampling_params.temperature for r in reqs],
                dtype=torch.float,
                pin_memory=_pin,
            )
            .to(device, non_blocking=True)
            .view(-1, 1)
        )
        top_ps = torch.tensor(
            [r.sampling_params.top_p for r in reqs],
            dtype=torch.float,
            pin_memory=_pin,
        ).to(device, non_blocking=True)
        top_ks = torch.tensor(
            [r.sampling_params.top_k for r in reqs],
            dtype=torch.int32,
            pin_memory=_pin,
        ).to(device, non_blocking=True)
        min_ps = torch.tensor(
            [r.sampling_params.min_p for r in reqs],
            dtype=torch.float,
            pin_memory=_pin,
        ).to(device, non_blocking=True)
        sampling_seed = (
            torch.tensor(
                [
                    (
                        r.sampling_params.sampling_seed
                        if r.sampling_params.sampling_seed is not None
                        else 42
                    )
                    for r in reqs
                ],
                dtype=torch.int64,
                pin_memory=_pin,
            ).to(device, non_blocking=True)
            if enable_deterministic
            else None
        )

        logit_bias = None
        if any(r.sampling_params.logit_bias is not None for r in reqs):
            logit_bias = torch.zeros(len(reqs), vocab_size, device=device)
            rows, cols, vals = [], [], []
            for i, r in enumerate(reqs):
                if r.sampling_params.logit_bias is not None:
                    # Dedup on int(key) first ("1" and "01" collide): duplicate
                    # indices make the index_put below nondeterministic on CUDA.
                    row_bias = {
                        int(key): value
                        for key, value in r.sampling_params.logit_bias.items()
                    }
                    for token_id, value in row_bias.items():
                        rows.append(i)
                        cols.append(token_id)
                        vals.append(value)
            if rows:
                logit_bias[
                    _rows_to_device_indices(rows, device, _pin),
                    _rows_to_device_indices(cols, device, _pin),
                ] = torch.tensor(vals, dtype=logit_bias.dtype, pin_memory=_pin).to(
                    device, non_blocking=True
                )

        # Check if any request has custom logit processor
        has_custom_logit_processor = (
            get_exec().features.enable_custom_logit_processor
            and any(r.custom_logit_processor for r in reqs)  # check the flag first.
        )  # then check the requests.
        return_sampling_masks = [r.return_sampling_mask for r in reqs]
        return_sampling_support_logprobs = [
            r.return_sampling_mask and r.sampling_logprobs_mode == "support"
            for r in reqs
        ]
        sampling_mask_batch_indices = cls._make_sampling_mask_batch_indices(
            return_sampling_masks, device
        )
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        features = get_exec().features
        if isinstance(batch, ScheduleBatch) and features.enable_watermark:
            watermark_config = build_watermark_batch_config(
                reqs,
                default_key=features.watermark_key,
                default_context_window=features.watermark_context_window,
                default_enabled=features.watermark_default_enabled,
                enforce_all=features.watermark_enforce_all,
                device=device,
            )
            watermark_keys = watermark_config.keys
            watermark_context_windows = watermark_config.context_windows
            watermark_enabled = watermark_config.enabled
            watermark_candidates_host = watermark_config.candidates_host
            has_watermark_candidates = watermark_config.has_candidates
        else:
            watermark_keys = None
            watermark_context_windows = None
            watermark_enabled = None
            watermark_candidates_host = None
            has_watermark_candidates = False
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
                hash(processor_str): ProcessorEntry(
                    processor=CustomLogitProcessor.from_str(processor_str),
                    rows=rows,
                    indices=_rows_to_device_indices(rows, device, _pin),
                )
                for processor_str, rows in processor_dict.items()
            }
            custom_params = [r.sampling_params.custom_params for r in reqs]
        else:
            merged_custom_logit_processor = None
            custom_params = None

        return cls(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            sampling_seed=sampling_seed,
            is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
            is_any_greedy=any(r.sampling_params.top_k <= 1 for r in reqs),
            need_top_p_sampling=any(r.sampling_params.top_p != 1.0 for r in reqs),
            need_top_k_sampling=any(r.sampling_params.top_k != TOP_K_ALL for r in reqs),
            need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
            npu_top_k_top_p_eligible=all(
                1 <= r.sampling_params.top_k <= 1024 for r in reqs
            ),
            max_top_k=(
                max(r.sampling_params.top_k for r in reqs)
                if reqs and all(1 <= r.sampling_params.top_k < TOP_K_ALL for r in reqs)
                else None
            ),
            vocab_size=vocab_size,
            grammars=(
                [r.grammar for r in reqs] if any(r.grammar for r in reqs) else None
            ),
            penalties=StepPenalties.build(
                reqs, req_pool_indices, penalty_store, device
            ),
            has_custom_logit_processor=has_custom_logit_processor,
            custom_params=custom_params,
            custom_logit_processor=merged_custom_logit_processor,
            device=device,
            logit_bias=logit_bias,
            return_sampling_masks=return_sampling_masks,
            return_sampling_support_logprobs=return_sampling_support_logprobs,
            sampling_mask_batch_indices=sampling_mask_batch_indices,
            watermark_keys=watermark_keys,
            watermark_context_windows=watermark_context_windows,
            watermark_enabled=watermark_enabled,
            watermark_candidates_host=watermark_candidates_host,
            has_watermark_candidates=has_watermark_candidates,
            sampling_support_logprobs_capture_indices=(
                sampling_support_logprobs_capture_indices
            ),
            sampling_mask_top_ks=sampling_mask_top_ks,
        )

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
        return _rows_to_device_indices(indices, device)

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
        return _rows_to_device_indices(indices, device)

    def __len__(self):
        return len(self.temperatures)

    def update_regex_vocab_mask(self):
        if not self.grammars:
            self.grammar_mask = None
            return

        # Find a grammar from the list
        first_grammar = next(grammar for grammar in self.grammars if grammar)

        vocab_mask = first_grammar.allocate_vocab_mask(
            vocab_size=self.vocab_size,
            batch_size=len(self.temperatures),
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

    def logit_penalties(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (additive, scaling) penalties of this step, each None if unused."""
        if self.penalties is None:
            return None, None
        return self.penalties.logit_penalties()

    def commit_sampled_tokens(
        self, token_ids: torch.Tensor, accept_mask: Optional[torch.Tensor] = None
    ) -> None:
        """Write this step's output tokens into the penalty history."""
        if self.penalties is not None:
            self.penalties.commit(token_ids, accept_mask)

    def _apply_pre_grammar_logits_transforms(self, logits: torch.Tensor) -> None:
        additive, scaling = self.logit_penalties()
        if additive is not None:
            logits.add_(additive)
        if scaling is not None:
            apply_scaling_penalties(logits, scaling)

    def _apply_post_grammar_logits_transforms(self, logits: torch.Tensor) -> None:
        if self.logit_bias is not None:
            logits.add_(self.logit_bias)

    def apply_logits_bias(self, logits: torch.Tensor):
        self._apply_pre_grammar_logits_transforms(logits)

        if self.grammar_mask is not None:
            self.grammar_mask.apply(logits)

        self._apply_post_grammar_logits_transforms(logits)

    def apply_logits_bias_with_observer(
        self,
        logits: torch.Tensor,
        observer: SamplingObserver,
    ) -> Any:
        self._apply_pre_grammar_logits_transforms(logits)
        observer_state = observer.before_grammar(logits, self)

        if self.grammar_mask is not None:
            self.grammar_mask.apply(logits)

        self._apply_post_grammar_logits_transforms(logits)

        return observer_state
