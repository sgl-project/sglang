"""The stage order of one sampling step, written once for decode and for every
speculative target verify:

    penalty -> grammar mask -> logit_bias -> watermark -> custom logit processor
    -> sanitize (decode only) -> sampler or accept core -> commit

Two entries run it:

- decode: `SamplingPipeline.sample`, which binds, applies, samples, and commits
  in one call;
- verify: `SamplingPipeline.plan_verify` returns a `VerifyStep` before the
  target forward launches; its route picks the verify's fast paths. After the
  launch the verify worker calls `build_grammar_mask`, the stages
  (`VerifyStep.adjust`, inside `eagle_sample` for a tree), its accept core, and
  `commit`.

Each stage reads its step state from `SamplingBatchInfo`, which the scheduler
builds at each launch (`set_rows`, `set_step_state`). A verify differs only in what each
stage gets: several rows per request, the penalty along the draft tree or
chain, and a grammar mask built over the draft tree. Penalty and watermark keep
per-request state, so each commits the sampled tokens or the verify's
`VerifyAccept`. A `SamplingObserver` sees the decode logits between the penalty
and the grammar mask.
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
    from sglang.srt.speculative.eagle_info import EagleVerifyInput
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
            _adjust_after_watermark(
                logits_output.next_token_logits,
                sampler_info,
                logprob_logits=logprob_logits,
            )
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

    def plan_verify(
        self, batch: ScheduleBatch, *, watermark_full_mask: bool
    ) -> VerifyStep:
        """Plan a target verify's sampling: derive its route and bind its watermark.
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
            route=VerifyRoute.derive(
                info,
                has_grammar=batch.has_grammar,
                has_watermark=info is not None and info.watermark is not None,
            ),
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
    observer: Optional[SamplingObserver] = None,
    draft_tokens: Optional[torch.Tensor] = None,
    next_child: Optional[torch.Tensor] = None,
    next_sibling: Optional[torch.Tensor] = None,
    grammar_mask: Optional[GrammarMask] = None,
) -> Any:
    """Penalty, grammar mask, then logit_bias, in place; returns the observer's
    state.

    Without ``draft_tokens``, one row per request, and the grammar mask is built
    here from the step's grammars. A verify passes ``draft_tokens`` [bs, width]
    (row ``i * width + j`` is request i's draft token j) and the mask it built
    over the draft tree: the penalty follows the draft chain, or the tree
    ``next_child`` / ``next_sibling`` give, and logit_bias gets ``width`` rows
    per request.
    """
    penalties = sampling_info.step_penalties
    if penalties is not None:
        if draft_tokens is None:
            penalties.apply(logits)
        else:
            penalties.apply_verify(
                logits, draft_tokens, next_child=next_child, next_sibling=next_sibling
            )
    observer_state = (
        observer.before_grammar(logits, sampling_info) if observer is not None else None
    )
    if draft_tokens is None:
        apply_grammar_mask(logits, sampling_info)
    elif grammar_mask is not None:
        grammar_mask.apply(logits)
    rows_per_request = 1 if draft_tokens is None else draft_tokens.shape[1]
    apply_logit_bias(logits, sampling_info, rows_per_request=rows_per_request)
    return observer_state


def _adjust_after_watermark(
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    *,
    logprob_logits: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    _custom_logit_processor_stage(logits, sampling_info, logprob_logits=logprob_logits)
    sanitize_nan_logits(logits, "sampler: next_token_logits")
    if logprob_logits is not None:
        sanitize_nan_logits(logprob_logits, "sampler: next_token_logits")
    return logits


def _custom_logit_processor_stage(
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    *,
    logprob_logits: Optional[torch.Tensor],
    rows_per_request: int = 1,
) -> None:
    """The custom logit processor stage, in place, on the logits the sampler or
    accept core reads and on `logprob_logits`, the unwatermarked copy the
    watermark keeps for the logprobs."""
    apply_custom_logit_processor(
        logits, sampling_info, rows_per_request=rows_per_request
    )
    if logprob_logits is not None:
        apply_custom_logit_processor(
            logprob_logits, sampling_info, rows_per_request=rows_per_request
        )


class VerifyRoute(msgspec.Struct, frozen=True):
    """The sampling facts that pick a target verify's fast paths."""

    # Some stage edits the logits before the accept reads them.
    adjusts_logits: bool
    # Some row truncates its distribution with top-k, top-p, or min-p.
    need_truncated_sampling: bool

    @classmethod
    def derive(
        cls,
        info: Optional[SamplingBatchInfo],
        *,
        has_grammar: bool,
        has_watermark: bool,
    ) -> VerifyRoute:
        """`has_grammar` and `has_watermark` come from the caller: verify builds
        its grammar mask over the draft tree, and the provider's sampling view
        carries neither stage's state."""
        if info is None:
            return cls(adjusts_logits=has_grammar, need_truncated_sampling=False)
        return cls(
            adjusts_logits=(
                has_grammar
                or has_watermark
                or info.step_penalties is not None
                or info.logit_bias is not None
                or info.has_custom_logit_processor
            ),
            need_truncated_sampling=need_truncated_sampling(info),
        )


class VerifyStep(msgspec.Struct):
    """One target verify's sampling state, from `SamplingPipeline.plan_verify`:
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
        tree: Optional[EagleVerifyInput] = None,
    ) -> Tuple[torch.Tensor, SamplingBatchInfo]:
        """The stages over a verify's logits, in sampling's order (see
        `_adjust_before_watermark` for the rows); returns the logits and the
        sampling info to accept with. The watermark may replace both, and when
        it keeps the unwatermarked logits for the logprobs, the custom logit
        processor runs on both, as in sampling. `draft_tokens` is
        [bs, width]. A chain verify passes `positions` [bs, width], read by the
        watermark; a tree verify passes its verify input as `tree` instead: the
        penalty walks its `retrieve_next_token` / `retrieve_next_sibling`, and
        the watermark reads its flat draft tokens, positions, and custom mask."""
        batch = self.batch
        sampling_info = batch.sampling_info
        _adjust_before_watermark(
            logits,
            sampling_info,
            draft_tokens=draft_tokens,
            next_child=None if tree is None else tree.retrieve_next_token,
            next_sibling=None if tree is None else tree.retrieve_next_sibling,
            grammar_mask=self.grammar_mask,
        )
        # Read before the watermark, which may replace the sampling info.
        has_custom_logit_processor = sampling_info.has_custom_logit_processor
        verify_logits = logits
        if self.watermark is not None:
            if tree is None:
                geometry = dict(
                    draft_tokens=draft_tokens, positions=positions, custom_mask=None
                )
            else:
                geometry = dict(
                    draft_tokens=tree.draft_token,
                    positions=tree.positions,
                    custom_mask=tree.custom_mask,
                )
            verify_logits, sampling_info = self.watermark.apply(
                logits,
                sampling_info=sampling_info,
                req_pool_indices=batch.req_pool_indices,
                draft_token_num=draft_tokens.shape[1],
                # compute_spec_logprobs reads the unwatermarked logits after verify.
                keep_logits=batch.return_logprob,
                **geometry,
            )
        if has_custom_logit_processor:
            _custom_logit_processor_stage(
                verify_logits,
                sampling_info,
                logprob_logits=None if verify_logits is logits else logits,
                rows_per_request=draft_tokens.shape[1],
            )
        return verify_logits, sampling_info

    def commit(
        self, accept: VerifyAccept, *, logits_output: LogitsProcessorOutput
    ) -> None:
        batch = self.batch
        info = batch.sampling_info
        if info.step_penalties is not None:
            info.step_penalties.commit(
                accept.accept_tokens(), accept_mask=accept.accept_mask()
            )
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
