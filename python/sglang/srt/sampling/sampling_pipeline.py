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
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import msgspec
import torch

from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.sampling.custom_logit_processor import apply_custom_logit_processor
from sglang.srt.sampling.grammar_mask import apply_grammar_mask
from sglang.srt.sampling.logit_bias import apply_logit_bias
from sglang.srt.sampling.sampling_batch_info import need_truncated_sampling
from sglang.srt.utils.async_probe import maybe_detect_nan, sanitize_nan_logits

if TYPE_CHECKING:
    from sglang.srt.constrained.base_grammar_backend import GrammarMask
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.sampler import Sampler
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.sampling.sampling_observer import SamplingObserver
    from sglang.srt.sampling.verify_accept import VerifyAccept
    from sglang.srt.sampling.watermarking.interface import (
        VerifyWatermark,
        WatermarkEngine,
    )
    from sglang.srt.speculative.spec_utils import GrammarTree


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

    def bind_verify(
        self, batch: ScheduleBatch, *, watermark_full_mask: bool
    ) -> VerifyStep:
        """Bind a target verify's sampling state: its route and its watermark.
        Call it before the target forward launches, so the route can pick the
        forward's own fast paths.

        `watermark_full_mask`: the verify's custom mask has the FULL_MASK layout,
        prefix columns included.
        """
        info = batch.sampling_info
        watermark = None
        if self.watermark is not None:
            watermark = self.watermark.bind_verify(
                info.watermark, full_mask=watermark_full_mask
            )
        return VerifyStep(
            batch=batch,
            route=VerifyRoute.derive(info, has_grammar=batch.has_grammar),
            watermark=watermark,
            watermark_full_mask=watermark_full_mask,
        )

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
        sampling_info.step_penalties.apply(logits)
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


class VerifyRoute(msgspec.Struct, frozen=True):
    """The sampling facts that pick a target verify's fast paths."""

    # Some stage edits the logits before the accept reads them.
    adjusts_logits: bool
    # Some row truncates its distribution with top-k, top-p, or min-p.
    need_truncated_sampling: bool

    @classmethod
    def derive(
        cls, info: Optional[SamplingBatchInfo], *, has_grammar: bool
    ) -> VerifyRoute:
        """`has_grammar` comes from the batch: verify builds its grammar mask over
        the draft tree, not from the sampling info."""
        # Without sampling info, only a grammar edits the logits.
        if info is None:
            return cls(adjusts_logits=has_grammar, need_truncated_sampling=False)
        return cls(
            adjusts_logits=(
                has_grammar
                or info.step_penalties is not None
                or info.logit_bias is not None
                or info.has_custom_logit_processor
                or info.watermark is not None
            ),
            need_truncated_sampling=need_truncated_sampling(info),
        )


class VerifyStep(msgspec.Struct):
    """One target verify's sampling state, from `SamplingPipeline.bind_verify`:
    the route its fast paths read, the bound watermark, and the grammar mask,
    built after the target forward launches."""

    batch: ScheduleBatch
    route: VerifyRoute
    watermark: Optional[VerifyWatermark]
    watermark_full_mask: bool
    grammar_mask: Optional[GrammarMask] = None

    def build_grammar_mask(
        self,
        tree: Optional[GrammarTree],
        *,
        logits_output: LogitsProcessorOutput,
        barrier: Optional[Callable[[], None]],
    ) -> None:
        """Build the grammar mask over the verify tree. Call it after the target
        verify launches, so the mask's host work overlaps the forward."""
        if tree is None:
            return
        # Lazy: spec_utils imports the scheduler stack.
        from sglang.srt.speculative.spec_utils import build_grammar_vocab_mask

        self.grammar_mask = build_grammar_vocab_mask(
            reqs=self.batch.reqs,
            tree=tree,
            sampling_info=self.batch.sampling_info,
            device=logits_output.next_token_logits.device,
            barrier=barrier,
        )

    def adjust(
        self,
        logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        *,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, SamplingBatchInfo]:
        """Run the stages over a chain verify's logits (`draft_tokens` is
        [bs, width]; `positions` too, read by the watermark). Returns the logits
        and the sampling info to accept with: the watermark may replace both,
        and leaves `logits` unwatermarked when the batch returns logprobs."""
        batch = self.batch
        info = batch.sampling_info
        adjust_verify_logits(
            logits,
            sampling_info=info,
            grammar_mask=self.grammar_mask,
            draft_tokens=draft_tokens,
            has_custom_logit_processor=info.has_custom_logit_processor,
        )
        if self.watermark is not None:
            logits, info = self.watermark.apply(
                logits,
                sampling_info=info,
                req_pool_indices=batch.req_pool_indices,
                draft_tokens=draft_tokens,
                positions=positions,
                draft_token_num=draft_tokens.shape[1],
                custom_mask=None,
                keep_logits=batch.return_logprob,
            )
        return logits, info

    def commit(
        self, accept: VerifyAccept, *, logits_output: LogitsProcessorOutput
    ) -> None:
        """Commit `accept` to each stage's per-request state, then compute the
        logprobs of its tokens when the batch returns them."""
        batch = self.batch
        info = batch.sampling_info
        if info.step_penalties is not None:
            info.step_penalties.commit_accept(accept)
        if self.watermark is not None:
            self.watermark.commit(accept)
        if not batch.return_logprob or batch.forward_mode.is_idle():
            return
        if accept.accept_index is not None:
            compute_spec_logprobs(
                batch, logits_output, accept.tokens, accept_index=accept.accept_index
            )
        else:
            compute_spec_logprobs(
                batch,
                logits_output,
                accept.tokens.reshape(-1),
                chain_stride=accept.tokens.shape[1],
            )


def adjust_verify_logits(
    logits: torch.Tensor,
    *,
    sampling_info: SamplingBatchInfo,
    grammar_mask: Optional[GrammarMask],
    draft_tokens: torch.Tensor,
    next_child: Optional[torch.Tensor] = None,
    next_sibling: Optional[torch.Tensor] = None,
    has_custom_logit_processor: bool,
) -> None:
    """Penalty, grammar mask, logit_bias, then custom logit processor over a
    verify's logits, in place. Row ``i * width + j`` of ``logits`` is request
    i's draft token j (``draft_tokens`` is [bs, width]): the penalty follows the
    draft chain, or the tree ``next_child`` / ``next_sibling`` give, and
    logit_bias and the custom logit processor get ``width`` rows per request."""
    width = draft_tokens.shape[1]
    if sampling_info.step_penalties is not None:
        sampling_info.step_penalties.apply_verify(
            logits, draft_tokens, next_child=next_child, next_sibling=next_sibling
        )
    if grammar_mask is not None:
        grammar_mask.apply(logits)
    apply_logit_bias(logits, sampling_info, rows_per_request=width)
    if has_custom_logit_processor:
        apply_custom_logit_processor(logits, sampling_info, rows_per_request=width)
