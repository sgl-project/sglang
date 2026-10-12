from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    draft_markov_block,
    select_draft_hidden_without_anchor,
    step_logits_width,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync

logger = logging.getLogger(__name__)


class DsparkDraftSampler:
    """The eager path's markov block, run as the draft cuda graph's tail."""

    def __init__(
        self,
        *,
        model,
        gamma,
        max_bs,
        device,
        noise: DraftNoise,
        tp_sync: SpecTpSync,
        greedy_draft: bool,
        confidence_fn=None,
        out=None,
    ):
        self.model = model
        self.markov_head = model.markov_head
        self.gamma = int(gamma)
        self.sample_from_anchor = bool(model.sample_from_anchor)
        self.query_token_num = self.gamma if self.sample_from_anchor else self.gamma + 1
        self.max_bs = int(max_bs)
        if out is not None:
            assert out.shape == (self.max_bs * self.gamma,)
            assert out.dtype == torch.int64
            self.out = out
        else:
            self.out = torch.empty(
                (self.max_bs * self.gamma,), dtype=torch.int64, device=device
            )
        self.confidence_fn = confidence_fn
        self.confidence_out = (
            torch.empty((self.max_bs, self.gamma), dtype=torch.float32, device=device)
            if confidence_fn is not None
            else None
        )
        self.greedy_draft = greedy_draft
        self._noise = noise
        self._tp_sync = tp_sync
        # Staged before each replay; step_temperatures is 0 on argmax rows.
        self.temperatures = torch.ones(
            (self.max_bs,), dtype=torch.float32, device=device
        )
        self.greedy_mask = torch.ones((self.max_bs,), dtype=torch.bool, device=device)
        self.step_temperatures = torch.zeros(
            (self.max_bs,), dtype=torch.float32, device=device
        )
        self._zero = torch.zeros((), dtype=torch.float32, device=device)
        # [max_bs, gamma, vocab] step logits for the sampling accept; a greedy
        # draft keeps none.
        self.step_logits = (
            None
            if greedy_draft
            else torch.empty(
                (self.max_bs, self.gamma, _step_logits_width(model)),
                dtype=_step_logits_dtype(model),
                device=device,
            )
        )

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Must run before the draft graph replay that reads these buffers."""
        if sampling_info is None:
            self.temperatures[:bs].fill_(1.0)
            self.greedy_mask[:bs].fill_(True)
            self.step_temperatures[:bs].zero_()
            return
        torch.clamp(
            sampling_info.temperatures.view(-1)[:bs].to(torch.float32),
            min=1e-5,
            out=self.temperatures[:bs],
        )
        torch.le(sampling_info.top_ks.view(-1)[:bs], 1, out=self.greedy_mask[:bs])
        torch.where(
            self.greedy_mask[:bs],
            self._zero,
            self.temperatures[:bs],
            out=self.step_temperatures[:bs],
        )

    def __call__(self, hidden_states, input_ids):
        bs = hidden_states.shape[0] // self.query_token_num
        if self.sample_from_anchor:
            model_hidden = hidden_states
            sample_hidden = hidden_states.view(bs, self.gamma, -1)
        else:
            model_hidden, sample_hidden = select_draft_hidden_without_anchor(
                hidden_states,
                bs=bs,
                gamma=self.gamma,
            )
        base_logits, confidence_tap = self.model.compute_base_logits(model_hidden)
        base_logits = base_logits.view(bs, self.gamma, -1)
        anchor = input_ids.view(bs, self.query_token_num)[:, 0]

        draft_tokens, _ = draft_markov_block(
            markov_head=self.markov_head,
            base_logits=base_logits,
            anchor_tokens=anchor,
            draft_hidden=sample_hidden,
            step_temperatures=self.step_temperatures[:bs],
            noise=self._noise,
            tp_sync=self._tp_sync,
            logits_out=None if self.greedy_draft else self.step_logits[:bs],
            greedy_draft=self.greedy_draft,
        )
        self.out[: draft_tokens.numel()].copy_(draft_tokens.reshape(-1))
        if self.confidence_out is not None:
            confidence = self.confidence_fn(
                draft_hidden=sample_hidden,
                anchor_tokens=anchor,
                draft_tokens=draft_tokens,
                confidence_tap=confidence_tap,
            )
            self.confidence_out[:bs].copy_(confidence)


def _step_logits_width(model) -> int:
    # Unsharded base logits span the cropped vocab.
    return step_logits_width(model.markov_head, int(model.lm_head.org_vocab_size))


def _step_logits_dtype(model) -> torch.dtype:
    # A quantized head's packed weight has no logits dtype; it emits the draft's.
    weight = model.lm_head.weight
    if weight.is_floating_point():
        base_dtype = weight.dtype
    else:
        base_dtype = next(model.markov_head.parameters()).dtype
    return model.markov_head.step_logits_dtype(base_dtype)


def maybe_build_draft_sampler(
    *,
    draft_model,
    gamma: int,
    max_bs: int,
    device,
    tp_rank: int,
    noise: DraftNoise,
    greedy_draft: bool,
    tp_sync: SpecTpSync,
    confidence_fn=None,
    out=None,
) -> Optional[DsparkDraftSampler]:
    """Build the graph-folded draft sampler, or None (reason logged) when the
    proposal must stay eager."""

    def _eager(reason):
        if tp_rank == 0:
            logger.info("DSpark draft proposal kept eager (reason=%s).", reason)
        return None

    if gamma <= 0:
        return _eager("gamma<=0")
    if draft_model.markov_head is None:
        return _eager("no markov head")
    if tp_rank == 0:
        logger.info(
            "DSpark draft proposal (%s) folded into the draft cuda graph.",
            "greedy draft" if greedy_draft else "sampling",
        )
    return DsparkDraftSampler(
        model=draft_model,
        gamma=gamma,
        max_bs=max_bs,
        device=device,
        noise=noise,
        tp_sync=tp_sync,
        greedy_draft=greedy_draft,
        confidence_fn=confidence_fn,
        out=out,
    )
