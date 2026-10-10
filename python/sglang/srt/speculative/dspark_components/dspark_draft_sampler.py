from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import DsparkFoldedSampling, envs
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    draft_markov_block,
    select_draft_hidden_without_anchor,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite

logger = logging.getLogger(__name__)

# Free GPU memory the draft graph capture needs left over, in GB.
DRAFT_CAPTURE_MIN_FREE_GB = 1.0


class DsparkDraftSampler:
    """The draft proposal folded into the draft cuda graph as its tail: the
    eager path's markov block, run over the graph's hidden states. Without
    folded sampling every row takes the argmax, so the tail drafts
    greedily and keeps no step logits."""

    def __init__(
        self,
        *,
        model,
        gamma,
        max_bs,
        device,
        noise: DraftNoise,
        tp_sync: SpecTpSync,
        confidence_fn=None,
        out=None,
        folded_sampling: bool = True,
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
        self.folded_sampling = folded_sampling
        self._noise = noise
        self._tp_sync = tp_sync
        self._site = (
            SpecTpSyncSite.DSPARK_GRAPH_SAMPLE
            if folded_sampling
            else SpecTpSyncSite.DSPARK_GRAPH_GREEDY
        )
        # The requests' temperatures and greedy rows, staged before each replay.
        self.temperatures = torch.ones(
            (self.max_bs,), dtype=torch.float32, device=device
        )
        self.greedy_mask = torch.ones((self.max_bs,), dtype=torch.bool, device=device)
        # [max_bs, gamma, local vocab] step logits the sampling accept reads.
        self.step_logits = (
            torch.empty(
                (self.max_bs, self.gamma, _local_vocab_width(model)),
                dtype=_step_logits_dtype(model),
                device=device,
            )
            if folded_sampling
            else None
        )

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Host-side refresh of the static sampling params; must run before the
        draft graph replay that consumes them."""
        if not self.folded_sampling:
            return
        if sampling_info is None:
            self.temperatures[:bs].fill_(1.0)
            self.greedy_mask[:bs].fill_(True)
            return
        torch.clamp(
            sampling_info.temperatures.view(-1)[:bs].to(torch.float32),
            min=1e-5,
            out=self.temperatures[:bs],
        )
        torch.le(sampling_info.top_ks.view(-1)[:bs], 1, out=self.greedy_mask[:bs])

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

        logits_out = None if self.step_logits is None else self.step_logits[:bs]
        step_temperatures = self.temperatures[:bs]
        if self.folded_sampling:
            step_temperatures = step_temperatures.masked_fill(
                self.greedy_mask[:bs], 0.0
            )
        draft_tokens, _ = draft_markov_block(
            markov_head=self.markov_head,
            base_logits=base_logits,
            anchor_tokens=anchor,
            draft_hidden=sample_hidden,
            step_temperatures=step_temperatures,
            noise=self._noise,
            tp_sync=self._tp_sync,
            site=self._site,
            logits_out=logits_out,
            keep_logits=self.folded_sampling,
            # Without folded sampling every row takes the argmax and no
            # temperature is read.
            has_sampled_rows=None if self.folded_sampling else False,
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


def _local_vocab_width(model) -> int:
    return model.markov_head.local_vocab_width(model.lm_head)


def _step_logits_dtype(model) -> torch.dtype:
    # A quantized lm_head's packed weight carries no logits dtype; its kernel
    # emits the activation dtype, the draft's own parameter dtype.
    weight = model.lm_head.weight
    if weight.is_floating_point():
        base_dtype = weight.dtype
    else:
        base_dtype = next(model.markov_head.parameters()).dtype
    return model.markov_head.step_logits_dtype(base_dtype)


def _resolve_folded_sampling(
    *, model, gamma, max_bs, device, tp_rank, available_memory_gb: float
) -> bool:
    """The step-logits buffer is baked into the captured draft graph, so AUTO
    must decide before capture from a free-memory probe. ``available_memory_gb``
    is the group minimum, so every rank folds identically."""
    mode = envs.SGLANG_DSPARK_FOLDED_SAMPLING.get()
    if mode == DsparkFoldedSampling.OFF:
        return False
    if mode == DsparkFoldedSampling.FORCE:
        return True
    itemsize = _step_logits_dtype(model).itemsize
    need_gb = max_bs * gamma * _local_vocab_width(model) * itemsize / (1 << 30)
    if available_memory_gb - need_gb >= DRAFT_CAPTURE_MIN_FREE_GB:
        return True
    if tp_rank == 0:
        logger.warning(
            "DSpark folded sampling disabled: its step-logits buffer needs %.2f GB "
            "but only %.2f GB GPU memory is free; sampling batches will take "
            "the eager proposal path. Set SGLANG_DSPARK_FOLDED_SAMPLING=%d "
            "to force.",
            need_gb,
            available_memory_gb,
            int(DsparkFoldedSampling.FORCE),
        )
    return False


def maybe_build_draft_sampler(
    *,
    draft_model,
    gamma: int,
    max_bs: int,
    device,
    tp_rank: int,
    noise: DraftNoise,
    tp_sync: SpecTpSync,
    available_memory_gb: float,
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
    folded_sampling = _resolve_folded_sampling(
        model=draft_model,
        gamma=gamma,
        max_bs=max_bs,
        device=device,
        tp_rank=tp_rank,
        available_memory_gb=available_memory_gb,
    )
    if tp_rank == 0:
        logger.info(
            "DSpark draft proposal (%s) folded into the draft cuda graph.",
            "greedy + sampling" if folded_sampling else "greedy only",
        )
    return DsparkDraftSampler(
        model=draft_model,
        gamma=gamma,
        max_bs=max_bs,
        device=device,
        noise=noise,
        tp_sync=tp_sync,
        confidence_fn=confidence_fn,
        out=out,
        folded_sampling=folded_sampling,
    )
