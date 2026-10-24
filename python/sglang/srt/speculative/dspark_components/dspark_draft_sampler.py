from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_dp_tp_group
from sglang.srt.megakernels.runtime import megakernel_agree, megakernel_draft_sampler
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    DraftParams,
    draft_markov_block,
    resolve_draft_params,
    select_draft_hidden_without_anchor,
    step_logits_width,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.srt.utils import is_xpu

logger = logging.getLogger(__name__)

_is_xpu = is_xpu()


class DsparkDraftSampler:
    """The eager path's markov block, run as the draft cuda graph's tail."""

    verify_ids = None

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
        megakernel_sampler: Optional[Callable] = None,
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
        # [max_bs, gamma + 1] verify ids ([anchor | draft tokens]) written by the
        # graph tail, so the step skips its eager concat (SGLANG_OPT_DSV41_DSPARK_MID_FOLD).
        self.verify_ids: Optional[torch.Tensor] = None
        if envs.SGLANG_OPT_DSV41_DSPARK_MID_FOLD.get():
            self.verify_ids = torch.empty(
                (self.max_bs, self.gamma + 1), dtype=torch.int64, device=device
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
        # [max_bs, gamma, vocab] step logits for the sampling accept, sized on
        # the first call; a greedy draft keeps none.
        self.step_logits: Optional[torch.Tensor] = None
        self._megakernel_steps = None
        if megakernel_sampler is not None:
            self._megakernel_steps = self._build_megakernel_steps(megakernel_sampler)

    def _build_megakernel_steps(self, megakernel_sampler: Callable):
        if not self.greedy_draft:
            # The megakernel leaves temperature-0 rows' logits unwritten; a mixed accept
            # still reads them.
            self.step_logits = torch.zeros(
                (self.max_bs, self.gamma, _step_logits_width(self.model)),
                dtype=self.markov_head.step_logits_dtype(torch.bfloat16),
                device=self.out.device,
            )
        steps = megakernel_sampler(
            gamma=self.gamma,
            max_bs=self.max_bs,
            greedy_draft=self.greedy_draft,
            noise=self._noise,
            draft_tokens=self.out,
            step_logits=self.step_logits,
            step_temperatures=self.params.step_temperatures,
        )
        # A rank on SGLang's block would wait in its per-step token sync (this group),
        # which the megakernel's step skips.
        served = tuple(bs for bs in range(1, self.max_bs + 1) if steps(bs) is not None)
        megakernel_agree(
            served, what="draft sampler batch sizes", group=get_dp_tp_group()
        )
        return steps

    def _step_logits(self, base_logits: torch.Tensor) -> torch.Tensor:
        if self.step_logits is None:
            # The runner's warmup calls the tail eagerly before capture.
            # torch.cuda's capture query raises on XPU builds.
            capturing = (
                torch.xpu.is_current_stream_capturing()
                if _is_xpu
                else torch.cuda.is_current_stream_capturing()
            )
            assert not capturing
            self.step_logits = torch.empty(
                (self.max_bs, self.gamma, _step_logits_width(self.model)),
                dtype=self.markov_head.step_logits_dtype(base_logits.dtype),
                device=base_logits.device,
            )
        return self.step_logits

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Must run before the draft graph replay that reads these buffers."""
        resolve_draft_params(sampling_info=sampling_info, bs=bs, out=self.params)

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
        anchor = input_ids.view(bs, self.query_token_num)[:, 0]
        step = None if self._megakernel_steps is None else self._megakernel_steps(bs)
        if step is not None:
            draft_tokens, confidence_features = step(model_hidden, anchor)
        else:
            draft_tokens, confidence_features = self._markov_block(
                model_hidden, sample_hidden, anchor, bs
            )
        if self.confidence_out is not None:
            confidence, confidence_raw = self.model.compute_confidence(
                anchor_tokens=anchor,
                draft_tokens=draft_tokens,
                confidence_features=confidence_features,
            )
            self.confidence_out[:bs].copy_(confidence)
            self.confidence_raw_out[:bs].copy_(confidence_raw)
        if self.verify_ids is not None:
            torch.cat(
                [anchor.view(bs, 1), self.out[: bs * self.gamma].view(bs, self.gamma)],
                dim=1,
                out=self.verify_ids[:bs],
            )

    def _markov_block(self, model_hidden, sample_hidden, anchor, bs: int):
        base_logits, confidence_features = self.model.compute_base_logits(model_hidden)
        base_logits = base_logits.view(bs, self.gamma, -1)
        draft_tokens, _ = draft_markov_block(
            markov_head=self.markov_head,
            base_logits=base_logits,
            anchor_tokens=anchor,
            draft_hidden=sample_hidden,
            step_temperatures=self.params.step_temperatures[:bs],
            noise=self._noise,
            tp_sync=self._tp_sync,
            logits_out=None
            if self.greedy_draft
            else self._step_logits(base_logits)[:bs],
            greedy_draft=self.greedy_draft,
        )
        self.out[: draft_tokens.numel()].copy_(draft_tokens.reshape(-1))
        return draft_tokens, confidence_features


def _step_logits_width(model) -> int:
    # Unsharded base logits span the cropped vocab.
    return step_logits_width(model.markov_head, int(model.lm_head.org_vocab_size))


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
    megakernel_sampler = megakernel_draft_sampler(draft_model)
    if tp_rank == 0:
        logger.info(
            "DSpark draft proposal (%s) folded into the draft cuda graph%s.",
            "greedy draft" if greedy_draft else "sampling",
            "; the NVIDIA megakernel may run it" if megakernel_sampler else "",
        )
    return DsparkDraftSampler(
        model=draft_model,
        gamma=gamma,
        max_bs=max_bs,
        device=device,
        noise=noise,
        tp_sync=tp_sync,
        greedy_draft=greedy_draft,
        out=out,
        megakernel_sampler=megakernel_sampler,
    )
