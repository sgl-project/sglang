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
from sglang.srt.speculative.spec_tp_sync import SpecTpSync

logger = logging.getLogger(__name__)

# Free GPU memory the draft graph capture needs left over, in GB.
DRAFT_CAPTURE_MIN_FREE_GB = 1.0


class DsparkDraftSampler:
    """The draft proposal folded into the draft cuda graph as its tail: the
    eager path's markov block, run over the graph's hidden states. Without
    folded sampling every row stays at temperature 0, so the tail drafts
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
        # Per-row draft temperature, 0 for argmax rows; staged before each replay.
        self.temperatures = torch.zeros(
            (self.max_bs,), dtype=torch.float32, device=device
        )
        # [max_bs, gamma, local vocab] step logits, allocated by the first warmup.
        self.step_logits: Optional[torch.Tensor] = None

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Host-side refresh of the static temperatures; must run before the
        draft graph replay that consumes them."""
        if not self.folded_sampling:
            return
        if sampling_info is None:
            self.temperatures[:bs].zero_()
            return
        temperatures = (
            sampling_info.temperatures.view(-1)[:bs].to(torch.float32).clamp_min(1e-5)
        )
        greedy = (sampling_info.top_ks <= 1).view(-1)[:bs]
        self.temperatures[:bs].copy_(temperatures.masked_fill(greedy, 0.0))

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

        logits_out = None
        if self.folded_sampling:
            if self.step_logits is None:
                self.step_logits = self._new_step_logits(
                    base_logits, anchor, sample_hidden
                )
            logits_out = self.step_logits[:bs]
        draft_tokens, _ = draft_markov_block(
            markov_head=self.markov_head,
            base_logits=base_logits,
            anchor_tokens=anchor,
            draft_hidden=sample_hidden,
            step_temperatures=self.temperatures[:bs],
            noise=self._noise,
            tp_sync=self._tp_sync,
            logits_out=logits_out,
            keep_logits=self.folded_sampling,
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

    def _new_step_logits(self, base_logits, anchor, sample_hidden) -> torch.Tensor:
        # Replays read this buffer after the graph ends, so it must come from a
        # warmup run: graph-pool memory is reused once a capture finishes.
        assert not (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        bias, _ = self.markov_head.step_bias(
            anchor.long(),
            hidden_states=sample_hidden[:, 0],
            state=self.markov_head.init_state(sample_hidden),
        )
        return torch.empty(
            (self.max_bs, self.gamma, base_logits.shape[-1]),
            dtype=self.markov_head.step_logits_dtype(base_logits.dtype, bias.dtype),
            device=base_logits.device,
        )


def _resolve_folded_sampling(
    *, model, gamma, max_bs, device, tp_rank, available_memory_gb: float
) -> bool:
    """The sampling buffers are baked into the captured draft graph, so AUTO
    must decide before capture from a free-memory probe. ``available_memory_gb``
    is the group minimum, so every rank folds identically."""
    mode = envs.SGLANG_DSPARK_FOLDED_SAMPLING.get()
    if mode == DsparkFoldedSampling.OFF:
        return False
    if mode == DsparkFoldedSampling.FORCE:
        return True
    # Upper bound: FP32 step logits over the whole vocab; a sharded head keeps
    # only its slice.
    vocab = int(model.lm_head.org_vocab_size)
    need_gb = max_bs * gamma * vocab * 4 / (1 << 30)
    if available_memory_gb - need_gb >= DRAFT_CAPTURE_MIN_FREE_GB:
        return True
    if tp_rank == 0:
        logger.warning(
            "DSpark folded sampling disabled: its static buffers need %.2f GB "
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
