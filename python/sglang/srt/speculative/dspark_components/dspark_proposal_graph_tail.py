from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    DraftParams,
    DraftProposalTensors,
    resolve_draft_params,
    run_draft_proposal,
    step_logits_width,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync

logger = logging.getLogger(__name__)


class DraftProposalGraphTail:
    """``run_draft_proposal`` run as the draft cuda graph's tail, over static
    buffers the eager proposal reads after a replay."""

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
        out=None,
    ):
        self.model = model
        self.gamma = int(gamma)
        self.query_token_num = (
            self.gamma if model.sample_from_anchor else self.gamma + 1
        )
        self.max_bs = int(max_bs)
        if out is not None:
            assert out.shape == (self.max_bs * self.gamma,)
            assert out.dtype == torch.int64
            self.out = out
        else:
            self.out = torch.empty(
                (self.max_bs * self.gamma,), dtype=torch.int64, device=device
            )
        confidence_head = model.confidence_head
        if confidence_head is None:
            self.confidence_out = self.confidence_raw_out = None
        else:
            self.confidence_out = torch.empty(
                (self.max_bs, self.gamma), dtype=torch.float32, device=device
            )
            self.confidence_raw_out = torch.empty(
                (self.max_bs, self.gamma),
                dtype=confidence_head.proj.weight.dtype,
                device=device,
            )
        self.greedy_draft = greedy_draft
        self._noise = noise
        self._tp_sync = tp_sync
        # Staged before each replay.
        self.params = DraftParams(
            temperatures=torch.ones((self.max_bs,), dtype=torch.float32, device=device),
            greedy_mask=torch.ones((self.max_bs,), dtype=torch.bool, device=device),
            step_temperatures=torch.zeros(
                (self.max_bs,), dtype=torch.float32, device=device
            ),
        )
        # [max_bs, gamma, vocab] step logits for the sampling accept; a greedy
        # draft keeps none.
        self.step_logits = (
            None
            if greedy_draft
            else torch.empty(
                (self.max_bs, self.gamma, _step_logits_width(model)),
                dtype=model.markov_head.step_logits_dtype(model.base_logits_dtype),
                device=device,
            )
        )

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Must run before the draft graph replay that reads these buffers."""
        resolve_draft_params(sampling_info=sampling_info, bs=bs, out=self.params)

    def __call__(self, hidden_states, input_ids):
        bs = hidden_states.shape[0] // self.query_token_num
        run_draft_proposal(
            model=self.model,
            hidden_states=hidden_states,
            anchor_tokens=input_ids.view(bs, self.query_token_num)[:, 0],
            bs=bs,
            gamma=self.gamma,
            step_temperatures=self.params.step_temperatures[:bs],
            noise=self._noise,
            tp_sync=self._tp_sync,
            greedy_draft=self.greedy_draft,
            out=DraftProposalTensors(
                self.out, self.step_logits, self.confidence_out, self.confidence_raw_out
            ),
        )


def _step_logits_width(model) -> int:
    # Unsharded base logits span the cropped vocab.
    return step_logits_width(model.markov_head, int(model.lm_head.org_vocab_size))


def maybe_build_proposal_graph_tail(
    *,
    draft_model,
    gamma: int,
    max_bs: int,
    device,
    tp_rank: int,
    noise: DraftNoise,
    greedy_draft: bool,
    tp_sync: SpecTpSync,
    out=None,
) -> Optional[DraftProposalGraphTail]:
    """Build the proposal's graph tail, or None (reason logged) when the
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
    return DraftProposalGraphTail(
        model=draft_model,
        gamma=gamma,
        max_bs=max_bs,
        device=device,
        noise=noise,
        tp_sync=tp_sync,
        greedy_draft=greedy_draft,
        out=out,
    )
