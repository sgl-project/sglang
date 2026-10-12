from __future__ import annotations

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    DraftParams,
    DraftProposalTensors,
    resolve_draft_params,
    run_draft_proposal,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync


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
        else:
            out = torch.empty(
                (self.max_bs * self.gamma,), dtype=torch.int64, device=device
            )
        confidence_head = model.confidence_head
        confidence = confidence_raw = None
        if confidence_head is not None:
            confidence = torch.empty(
                (self.max_bs, self.gamma), dtype=torch.float32, device=device
            )
            confidence_raw = torch.empty(
                (self.max_bs, self.gamma),
                dtype=confidence_head.proj.weight.dtype,
                device=device,
            )
        # The [max_bs, gamma, vocab] step logits are sized on the first call; a
        # greedy draft keeps none.
        self.buffers = DraftProposalTensors(
            out.view(self.max_bs, self.gamma), None, confidence, confidence_raw
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

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Must run before the draft graph replay that reads these buffers."""
        resolve_draft_params(sampling_info=sampling_info, bs=bs, out=self.params)

    def views(self, bs: int) -> tuple[DraftProposalTensors, DraftParams]:
        """The last replay's proposal and the params staged for it."""

        def head(t):
            return None if t is None else t[:bs]

        return (
            DraftProposalTensors(*map(head, self.buffers)),
            DraftParams(*map(head, self.params)),
        )

    def __call__(self, hidden_states, input_ids):
        bs = hidden_states.shape[0] // self.query_token_num
        buffers = self.buffers
        first_call = buffers.step_logits is None and not self.greedy_draft
        if first_call:
            # The runner's warmup calls the tail eagerly before capture.
            assert not torch.cuda.is_current_stream_capturing()
        proposal = run_draft_proposal(
            model=self.model,
            hidden_states=hidden_states,
            anchor_tokens=input_ids.view(bs, self.query_token_num)[:, 0],
            bs=bs,
            gamma=self.gamma,
            step_temperatures=self.params.step_temperatures[:bs],
            noise=self._noise,
            tp_sync=self._tp_sync,
            greedy_draft=self.greedy_draft,
            step_logits_out=(
                None if buffers.step_logits is None else buffers.step_logits[:bs]
            ),
        )
        buffers.draft_tokens[:bs].copy_(proposal.draft_tokens)
        if buffers.confidence is not None:
            buffers.confidence[:bs].copy_(proposal.confidence)
            buffers.confidence_raw[:bs].copy_(proposal.confidence_raw)
        if first_call:
            # Take the shape and dtype of the step logits the proposal allocated.
            step_logits = proposal.step_logits
            self.buffers = buffers._replace(
                step_logits=step_logits.new_empty((self.max_bs, *step_logits.shape[1:]))
            )
            self.buffers.step_logits[:bs].copy_(step_logits)
