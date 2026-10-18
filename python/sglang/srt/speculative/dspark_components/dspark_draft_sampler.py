from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dspark_components.dspark_config import (
    resolve_markov_candidate_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    DraftParams,
    draft_markov_block,
    resolve_draft_params,
    select_draft_hidden_without_anchor,
    step_logits_width,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync

logger = logging.getLogger(__name__)


def _target_vocab_size(model) -> int:
    return int(getattr(model, "target_vocab_size", model.lm_head.org_vocab_size))


def initialize_markov_candidate_sampler(
    *,
    model,
    draft_hf_config,
    gamma: int,
    capacity: Optional[int],
    tp_size: int,
    markov_topk: Optional[int] = None,
    markov_bias_topk: Optional[int] = None,
):
    """Resolve the path once after loading weights, before sizing the KV pool.

    The proposal cache is owned by physical sampler slots and shared by eager
    and all graph tiers. A verifier consumes it on the same stream before the
    next proposal can clear it. Weight reloads rebuild the table in place;
    configuration/layout changes require worker restart and graph recapture.
    """
    vocab = int(getattr(model, "draft_vocab_size", model.lm_head.org_vocab_size))
    config = resolve_markov_candidate_config(
        draft_hf_config,
        markov_topk=markov_topk,
        markov_bias_topk=markov_bias_topk,
        draft_vocab_size=vocab,
    )
    model.markov_candidate_sampler = None
    head = model.markov_head
    reason = None
    if config.effective_topk == 0:
        reason = "K=0 selects the existing full-vocabulary path"
    elif type(head) is not VanillaMarkov:
        reason = "candidate walk requires an ordinary vanilla Markov head"
    else:
        from sglang.kernels.ops.speculative.dspark.dspark_markov_topk import (
            MAX_STEPS,
            CandidateCapacityError,
            MarkovCandidateSampler,
            candidate_support_reason,
        )

        w1 = getattr(head.markov_w1, "weight", None)
        w2 = getattr(head.markov_w2, "weight", None)
        if w1 is None or w2 is None:
            reason = "Markov weights are not directly readable"
        else:
            reason = candidate_support_reason(
                w1,
                w2,
                topk=config.effective_topk,
                bias_topk=config.effective_bias_topk,
                tp_size=tp_size,
            )
        if reason is None and not 0 < gamma <= MAX_STEPS:
            reason = f"candidate gamma exceeds the bounded range 1..{MAX_STEPS}"
        if reason is None and (capacity is None or capacity <= 0):
            reason = (
                "set --max-running-requests to bound FP32 proposal storage "
                "before KV capacity planning"
            )
        if reason is None:
            try:
                model.markov_candidate_sampler = MarkovCandidateSampler(
                    w1,
                    w2,
                    alpha=float(getattr(model, "logit_scale", 1.0)),
                    topk=config.effective_topk,
                    bias_topk=config.effective_bias_topk,
                    target_vocab_size=_target_vocab_size(model),
                    gamma=gamma,
                    capacity=capacity,
                    d2t_offset=getattr(model, "draft_id_to_target_id", None),
                    logits_dtype=_base_logits_dtype(model),
                )
            except CandidateCapacityError as exc:
                # Only a pre-allocation capability check may choose a memory
                # fallback. Loading, mapping, OOM and compile errors propagate.
                reason = str(exc)
    enabled = model.markov_candidate_sampler is not None
    logger.info(
        "DSpark Markov requested K/M=%d/%d effective K/M=%d/%d "
        "head=%s Vt=%d Vd=%d R=%s gamma=%d TP=%d path=%s reason=%s",
        config.requested_topk,
        config.requested_bias_topk,
        config.effective_topk if enabled else 0,
        config.effective_bias_topk if enabled else 0,
        type(head).__name__,
        _target_vocab_size(model),
        vocab,
        getattr(head, "markov_rank", "unknown"),
        gamma,
        tp_size,
        "candidate-triton" if enabled else "full-vocabulary",
        reason or "supported",
    )
    return model.markov_candidate_sampler


def _base_logits_dtype(model) -> torch.dtype:
    """Dtype of the block logits; a quantized head's packed `weight` carries no
    logits dtype, its kernel emits the activation (draft param) dtype instead."""
    weight = model.lm_head.weight
    if weight.is_floating_point():
        return weight.dtype
    return next(model.markov_head.parameters()).dtype


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
        self.candidate_sampler = getattr(model, "markov_candidate_sampler", None)
        self.num_valid = None
        if self.candidate_sampler is not None:
            if self.max_bs > self.candidate_sampler.capacity:
                raise ValueError("DSpark graph exceeds the candidate cache capacity")
            self.num_valid = torch.full(
                (), self.max_bs, dtype=torch.int32, device=device
            )
            if not self.greedy_draft:
                self.step_logits = self.candidate_sampler.corrected_logits

    def _step_logits(self, base_logits: torch.Tensor) -> torch.Tensor:
        if self.step_logits is None:
            # The runner's warmup calls the tail eagerly before capture.
            assert not torch.cuda.is_current_stream_capturing()
            self.step_logits = torch.empty(
                (self.max_bs, self.gamma, _step_logits_width(self.model)),
                dtype=self.markov_head.step_logits_dtype(base_logits.dtype),
                device=base_logits.device,
            )
        return self.step_logits

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Must run before the draft graph replay that reads these buffers."""
        resolve_draft_params(sampling_info=sampling_info, bs=bs, out=self.params)
        if self.num_valid is not None:
            self.num_valid.fill_(bs)

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
        base_logits, confidence_features = self.model.compute_base_logits(model_hidden)
        base_logits = base_logits.view(bs, self.gamma, -1)
        anchor = input_ids.view(bs, self.query_token_num)[:, 0]

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
            candidate_sampler=self.candidate_sampler,
            num_valid=self.num_valid,
        )
        self.out[: draft_tokens.numel()].copy_(draft_tokens.reshape(-1))
        if self.confidence_out is not None:
            confidence, confidence_raw = self.model.compute_confidence(
                anchor_tokens=anchor,
                draft_tokens=draft_tokens,
                confidence_features=confidence_features,
            )
            self.confidence_out[:bs].copy_(confidence)
            self.confidence_raw_out[:bs].copy_(confidence_raw)


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
        out=out,
    )
