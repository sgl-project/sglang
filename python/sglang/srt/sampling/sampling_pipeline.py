"""The order of one sampling step, written in one place:

    penalty -> grammar mask -> logit_bias -> watermark -> custom logit processor
    -> sanitize -> sampler (logprobs, TP token sync) -> commit

A `SamplingObserver` sees the logits between the penalty and the grammar mask.
Each stage reads its step state from `SamplingBatchInfo`, which the scheduler
builds at each launch from the step's requests. The grammar mask, logit_bias,
and the custom logit processor are plugins: stateless `(logits, sampling_info)`
functions; speculative verify calls the last two with several rows per request
(`rows_per_request`) and builds its own grammar mask. Penalty and watermark are
adapters over their existing code.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch

from sglang.srt.sampling.custom_logit_processor import apply_custom_logit_processor
from sglang.srt.sampling.grammar_mask import apply_grammar_mask
from sglang.srt.sampling.logit_bias import apply_logit_bias
from sglang.srt.sampling.penalties import apply_scaling_penalties
from sglang.srt.utils.async_probe import maybe_detect_nan, sanitize_nan_logits

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.sampler import Sampler
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.sampling.sampling_observer import SamplingObserver
    from sglang.srt.sampling.watermarking.interface import WatermarkEngine


class SamplingPipeline(msgspec.Struct, frozen=True):
    sampler: Sampler
    watermark: Optional[WatermarkEngine]

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        *,
        observer: Optional[SamplingObserver],
        graph_token_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Sample the step's next tokens and commit them; logprobs and the
        observer's output land in `logits_output`.

        `graph_token_ids`: tokens a CUDA graph already sampled from these
        logits; they replace the stages after the watermark.
        """
        sampling_info = forward_batch.sampling_info
        if observer is not None and not observer.is_active(sampling_info):
            observer = None
        observer_state = _adjust_before_watermark(
            logits_output.next_token_logits, sampling_info, observer=observer
        )

        watermark = (
            self.watermark.bind_sample(forward_batch)
            if self.watermark is not None
            else None
        )
        # The info the sampler reads; the watermark may replace it.
        sampler_info = sampling_info
        logprob_logits = None
        next_token_ids = None
        if watermark is not None:
            step = watermark.apply(
                logits_output.next_token_logits,
                keep_logits=forward_batch.return_logprob,
            )
            sampler_info = step.sampling_info
            logprob_logits = step.logprob_logits
            next_token_ids = step.token_ids
        if next_token_ids is None:
            next_token_ids = graph_token_ids

        if next_token_ids is None:
            maybe_detect_nan(
                logits_output.next_token_logits, "sampler: next_token_logits"
            )
            _adjust_after_watermark(logits_output.next_token_logits, sampler_info)
            if logprob_logits is not None:
                _adjust_after_watermark(logprob_logits, sampler_info)
            next_token_ids = self.sampler(
                logits_output,
                sampler_info,
                forward_batch.return_logprob,
                forward_batch.top_logprobs_nums,
                forward_batch.token_ids_logprobs,
                # For prefill, we only use the position of the last token.
                (
                    forward_batch.positions
                    if forward_batch.forward_mode.is_decode()
                    else forward_batch.seq_lens - 1
                ),
                **(
                    {"logprob_logits": logprob_logits}
                    if logprob_logits is not None
                    else {}
                ),
            )
        if observer_state is not None:
            logits_output.auxiliary_device_output = observer.after_sample(
                observer_state, next_token_ids
            )

        if watermark is not None:
            watermark.commit(next_token_ids)
        if sampling_info.step_penalties is not None:
            sampling_info.step_penalties.commit(next_token_ids)
        return next_token_ids

    def compute_logprobs_only(
        self, logits_output: LogitsProcessorOutput, forward_batch: ForwardBatch
    ) -> None:
        """The requested tokens' logprobs, without sampling or the watermark."""
        if not forward_batch.token_ids_logprobs:
            return
        sampling_info = forward_batch.sampling_info
        _adjust_before_watermark(
            logits_output.next_token_logits, sampling_info, observer=None
        )
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            preprocess_fn=partial(_adjust_after_watermark, sampling_info=sampling_info),
        )


def _adjust_before_watermark(
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    *,
    observer: Optional[SamplingObserver],
) -> Any:
    """Penalty, grammar mask, then logit_bias; returns the observer's state."""
    if sampling_info.step_penalties is not None:
        additive, scaling = sampling_info.step_penalties.logit_penalties()
        if additive is not None:
            logits.add_(additive)
        if scaling is not None:
            apply_scaling_penalties(logits, scaling)
    observer_state = (
        observer.before_grammar(logits, sampling_info) if observer is not None else None
    )
    apply_grammar_mask(logits, sampling_info)
    apply_logit_bias(logits, sampling_info)
    return observer_state


def _adjust_after_watermark(
    logits: torch.Tensor, sampling_info: SamplingBatchInfo
) -> torch.Tensor:
    """The custom logit processors, then non-finite sanitizing, in place."""
    apply_custom_logit_processor(logits, sampling_info)
    sanitize_nan_logits(logits, "sampler: next_token_logits")
    return logits
