from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    SampleStepTokens,
)
from sglang.srt.environ import DsparkFoldedSampling, envs
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dspark_components.dspark_draft import (
    select_draft_hidden_without_anchor,
)
from sglang.srt.utils import get_available_gpu_memory

logger = logging.getLogger(__name__)

# Same free-memory floor init_cuda_graphs requires before draft capture.
_CAPTURE_HEADROOM_GB = 1.0


def _base_logits_dtype(model) -> torch.dtype:
    """Dtype of the block logits; a quantized head's packed `weight` carries no
    logits dtype, its kernel emits the activation (draft param) dtype instead."""
    weight = model.lm_head.weight
    if weight.is_floating_point():
        return weight.dtype
    return next(model.markov_head.parameters()).dtype


def greedy_step_sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
    del step_idx
    return torch.argmax(step_logits, dim=-1)


class DsparkDraftSampler:
    """Draft proposal head folded into the draft graph as a tail hook; with
    folded_sampling it also Gumbel-samples non-greedy rows in-graph."""

    def __init__(
        self,
        *,
        model,
        gamma,
        max_bs,
        device,
        confidence_fn=None,
        out=None,
        folded_sampling: bool = True,
    ):
        self.model = model
        self.markov_head = model.markov_head
        self.gamma = int(gamma)
        self.sample_from_anchor = bool(model.sample_from_anchor)
        self.query_token_num = self.gamma if self.sample_from_anchor else self.gamma + 1
        max_bs = int(max_bs)
        # Resolved once: this sampler runs inside cuda-graph capture, so the
        # branch below is baked into the captured graph anyway.
        self._fused_greedy = envs.SGLANG_DSPARK_OPT_FUSED_GREEDY_MARKOV.get()
        if out is not None:
            assert out.shape == (max_bs * self.gamma,) and out.dtype == torch.int64
            self.out = out
        else:
            self.out = torch.empty(
                (max_bs * self.gamma,), dtype=torch.int64, device=device
            )
        self.confidence_fn = confidence_fn
        self.confidence_out = (
            torch.empty((max_bs, self.gamma), dtype=torch.float32, device=device)
            if confidence_fn is not None
            else None
        )
        self.folded_sampling = folded_sampling
        self.temperatures = None
        self.greedy_mask = None
        self.exp_noise = None
        self.corrected_out = None
        if folded_sampling:
            vocab = int(model.lm_head.org_vocab_size)
            self.temperatures = torch.ones(
                (max_bs,), dtype=torch.float32, device=device
            )
            self.greedy_mask = torch.ones((max_bs,), dtype=torch.bool, device=device)
            self.exp_noise = torch.empty(
                (max_bs, vocab), dtype=torch.float32, device=device
            )
            self.corrected_out = torch.empty(
                (max_bs * self.gamma, vocab),
                dtype=_base_logits_dtype(model),
                device=device,
            )

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Host-side refresh of the static sampling params; must run before
        the draft graph replay that consumes them."""
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
        self.greedy_mask[:bs].copy_((sampling_info.top_ks <= 1).view(-1)[:bs])

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

        # Fused greedy fast path: only valid for the greedy (non-sampling) fold.
        # Gated/RNN subclasses return None (hidden-state-dependent bias); fall
        # through to the block sampler below.
        draft_tokens = None
        if (
            not self.folded_sampling
            and self._fused_greedy
            and isinstance(self.markov_head, VanillaMarkov)
        ):
            draft_tokens = self.markov_head.sample_block_greedy_fused(
                base_logits, first_prev_tokens=anchor
            )

        if draft_tokens is None:
            if self.folded_sampling:

                def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
                    del step_idx
                    # In-graph philox noise: each replay advances the generator
                    # and redraws.
                    noise = self.exp_noise[:bs].exponential_()
                    return SampleStepTokens.execute(
                        step_logits=step_logits,
                        temperatures=self.temperatures[:bs],
                        greedy_mask=self.greedy_mask[:bs],
                        exp_noise=noise,
                    )

            else:
                sampler = greedy_step_sampler

            draft_tokens, corrected_logits = self.markov_head.sample_block(
                base_logits,
                first_prev_tokens=anchor,
                hidden_states=hidden_states.view(bs, self.gamma, -1),
                sampler=sampler,
                collect_corrected=self.folded_sampling,
            )
            if self.folded_sampling:
                self.corrected_out[: bs * self.gamma].copy_(
                    corrected_logits.reshape(bs * self.gamma, -1)
                )

        else:
            sampler = greedy_step_sampler

        draft_tokens, corrected_logits = self.markov_head.sample_block(
            base_logits,
            first_prev_tokens=anchor,
            hidden_states=sample_hidden,
            sampler=sampler,
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


def _resolve_folded_sampling(*, model, gamma, max_bs, device, tp_rank) -> bool:
    """The sampling buffers are baked into the captured draft graph, so AUTO
    must decide before capture from a free-memory probe."""
    mode = envs.SGLANG_DSPARK_FOLDED_SAMPLING.get()
    if mode == DsparkFoldedSampling.OFF:
        return False
    if mode == DsparkFoldedSampling.FORCE:
        return True
    vocab = int(model.lm_head.org_vocab_size)
    noise_bytes = max_bs * vocab * 4
    logits_bytes = max_bs * gamma * vocab * _base_logits_dtype(model).itemsize
    need_gb = (noise_bytes + logits_bytes) / (1 << 30)
    available_gb = get_available_gpu_memory(
        device, torch.get_device_module().current_device()
    )
    if available_gb - need_gb >= _CAPTURE_HEADROOM_GB:
        return True
    if tp_rank == 0:
        logger.warning(
            "DSpark folded sampling disabled: its static buffers need %.2f GB "
            "but only %.2f GB GPU memory is free; sampling batches will take "
            "the eager proposal path. Set SGLANG_DSPARK_FOLDED_SAMPLING=%d "
            "to force.",
            need_gb,
            available_gb,
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
    if not hasattr(draft_model, "compute_base_logits"):
        return _eager("no compute_base_logits")
    if getattr(draft_model, "markov_head", None) is None:
        return _eager("no markov head")
    folded_sampling = _resolve_folded_sampling(
        model=draft_model, gamma=gamma, max_bs=max_bs, device=device, tp_rank=tp_rank
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
        confidence_fn=confidence_fn,
        out=out,
        folded_sampling=folded_sampling,
    )
