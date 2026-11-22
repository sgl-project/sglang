"""The steps every speculative target verify runs around its accept core."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import msgspec

from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.sampling.sampling_batch_info import (
    adjusts_logits,
    need_truncated_sampling,
)
from sglang.srt.sampling.verify_accept import VerifyAccept, commit_verify_accept
from sglang.srt.speculative.spec_utils import GrammarTree, build_grammar_vocab_mask

if TYPE_CHECKING:
    import torch

    from sglang.srt.constrained.base_grammar_backend import GrammarMask
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.sampling.watermarking.interface import (
        VerifyWatermark,
        WatermarkEngine,
    )


def adjust_verify_logits(
    logits: torch.Tensor,
    *,
    sampling_info: SamplingBatchInfo,
    grammar_mask: Optional[GrammarMask],
    draft_tokens: torch.Tensor,
    next_child: Optional[torch.Tensor] = None,
    next_sibling: Optional[torch.Tensor] = None,
    custom_logit_processor: bool,
) -> None:
    """Apply the sampler's logit adjustments to the verify logits in place, in
    decoding's order: penalties, grammar mask, logit bias, custom logit
    processor. The watermark, if any, applies after.

    ``draft_tokens`` [bs, width]: each request's verify tokens, a tree's nodes
    linked by ``next_child`` / ``next_sibling``, or a chain without them. Row
    ``i * width + j`` of ``logits`` is request i's token j, so a custom
    processor sees ``width`` rows per request, each with that request's params.
    ``custom_logit_processor``: run the batch's custom logit processors; a
    provider's sampling view carries none.
    """
    width = draft_tokens.shape[1]
    if sampling_info.step_penalties is not None:
        sampling_info.step_penalties.apply_verify(
            logits, draft_tokens, next_child=next_child, next_sibling=next_sibling
        )
    if grammar_mask is not None:
        grammar_mask.apply(logits)
    if sampling_info.logit_bias is not None:
        # In the bias's own dtype, so the sum rounds once to the logits dtype.
        logits.view(-1, width, logits.shape[-1]).add_(
            sampling_info.logit_bias[:, None, :]
        )
    if custom_logit_processor:
        apply_custom_logit_processor(logits, sampling_info, num_tokens_in_batch=width)


class VerifyRoute(msgspec.Struct, frozen=True):
    """The sampling facts that pick a target verify's fast paths, derived once per
    verify step before the target forward launches."""

    adjusts_logits: bool
    need_truncated_sampling: bool

    @classmethod
    def derive(
        cls, info: Optional[SamplingBatchInfo], *, has_grammar: bool
    ) -> VerifyRoute:
        # Without sampling info, only a grammar edits the logits.
        if info is None:
            return cls(adjusts_logits=has_grammar, need_truncated_sampling=False)
        return cls(
            adjusts_logits=adjusts_logits(info, has_grammar=has_grammar),
            need_truncated_sampling=need_truncated_sampling(info),
        )


class TargetVerify(msgspec.Struct, frozen=True):
    """What a target verify's accept core reads besides the logits: the grammar
    mask and the bound watermark. ``commit`` takes what the core accepted."""

    grammar_mask: Optional[GrammarMask]
    watermark: Optional[VerifyWatermark]
    # The verify's custom mask has the FULL_MASK layout, prefix columns included.
    watermark_full_mask: bool

    @classmethod
    def prepare(
        cls,
        batch: ScheduleBatch,
        *,
        logits_output: LogitsProcessorOutput,
        grammar_tree: Optional[GrammarTree],
        grammar_barrier: Optional[Callable[[], None]],
        watermark_engine: Optional[WatermarkEngine],
        watermark_full_mask: bool,
    ) -> TargetVerify:
        """Build the grammar mask over ``grammar_tree`` and bind the watermark.

        Call it after the target verify launch, so the mask's host work overlaps
        the forward."""
        grammar_mask = None
        if grammar_tree is not None:
            grammar_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                tree=grammar_tree,
                sampling_info=batch.sampling_info,
                device=logits_output.next_token_logits.device,
                barrier=grammar_barrier,
            )
        watermark = None
        if watermark_engine is not None:
            watermark = watermark_engine.bind_verify(
                batch.sampling_info.watermark, full_mask=watermark_full_mask
            )
        return cls(
            grammar_mask=grammar_mask,
            watermark=watermark,
            watermark_full_mask=watermark_full_mask,
        )

    def commit(
        self,
        batch: ScheduleBatch,
        *,
        logits_output: LogitsProcessorOutput,
        accept: VerifyAccept,
    ) -> None:
        """Commit ``accept`` to the per-request sampling state, then compute the
        logprobs of its tokens when the batch returns them."""
        commit_verify_accept(batch.sampling_info, accept, watermark=self.watermark)
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
