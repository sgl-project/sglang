from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    SampleStepTokens,
)
from sglang.srt.environ import DsparkFoldedSampling, envs
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.dspark_components.dspark_config import (
    resolve_markov_candidate_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    select_draft_hidden_without_anchor,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite

logger = logging.getLogger(__name__)

# Same free-memory floor init_cuda_graphs requires before draft capture.
_CAPTURE_HEADROOM_GB = 1.0


def dspark_effective_graph_capacity(
    *,
    max_running_requests: Optional[int],
    attn_dp_size: int,
    decode_graph_max_bs: Optional[int],
) -> Optional[int]:
    """Tight upper bound on the batch size a DSpark graph can reach.

    ``get_batch_sizes_to_capture`` filters the decode-bs list by
    ``req_to_token_pool.size``, which derives from ``max_running_requests``.
    No graph replay can exceed that bound, so buffers sized by the
    unclamped decode-bs maximum waste memory. This function computes the
    capped value: the graph's max-bs ceiling limited by the request bound.

    Returns None when neither bound is available (both are None).
    """
    if max_running_requests is not None:
        per_rank = max(1, max_running_requests // attn_dp_size)
    else:
        per_rank = None
    if decode_graph_max_bs is None:
        return per_rank
    if per_rank is None:
        return decode_graph_max_bs
    capped_graph = min(decode_graph_max_bs, max_running_requests)
    return max(per_rank, capped_graph)


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
        max_bs = int(max_bs)
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
        self.candidate_sampler = getattr(model, "markov_candidate_sampler", None)
        self._tp_sync = tp_sync
        self.temperatures = None
        self.greedy_mask = None
        self.exp_noise = None
        self.corrected_out = None
        self.num_valid = None
        self.anchors = None
        if self.candidate_sampler is not None:
            if max_bs > self.candidate_sampler.capacity:
                raise ValueError("DSpark graph exceeds the candidate cache capacity")
            self.temperatures = torch.ones(max_bs, dtype=torch.float32, device=device)
            self.greedy_mask = torch.ones(max_bs, dtype=torch.bool, device=device)
            self.num_valid = torch.full((), max_bs, dtype=torch.int32, device=device)
            self.anchors = torch.empty(max_bs, dtype=torch.int64, device=device)
            # This is a view of the sampler's authoritative FP32 cache. No
            # legacy full-vocabulary logits or exponential-noise buffer exists.
            self.corrected_out = self.candidate_sampler.corrected_logits.view(
                -1, _target_vocab_size(model)
            )
        elif folded_sampling:
            vocab = _target_vocab_size(model)
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
        if self.num_valid is not None:
            self.num_valid.fill_(bs)
        if not self.folded_sampling:
            if self.candidate_sampler is not None:
                # The proposer consumes this graph tail only for all-greedy
                # batches. Sampling batches run the candidate sampler eagerly.
                self.temperatures[:bs].fill_(1.0)
                self.greedy_mask[:bs].fill_(True)
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

        if self.candidate_sampler is not None:
            self.anchors[:bs].copy_(anchor)
            result = self.candidate_sampler.sample(
                base_logits,
                self.anchors[:bs],
                self.temperatures[:bs],
                self.greedy_mask[:bs],
                num_valid=self.num_valid,
            )
            self.out[: bs * self.gamma].copy_(result.tokens.reshape(-1))
            if self.confidence_out is not None:
                confidence = self.confidence_fn(
                    draft_hidden=sample_hidden,
                    anchor_tokens=result.prev_tokens[:, 0],
                    draft_tokens=result.tokens,
                    confidence_tap=confidence_tap,
                )
                self.confidence_out[:bs].copy_(confidence)
            return

        # Fused greedy fast path: only valid for the greedy (non-sampling) fold.
        # Gated/RNN subclasses return None (hidden-state-dependent bias); fall
        # through to the block sampler below.
        draft_tokens = None
        fused_greedy = getattr(self.markov_head, "supports_sharded_greedy", False) or (
            envs.SGLANG_DSPARK_OPT_FUSED_GREEDY_MARKOV.get()
            and isinstance(self.markov_head, VanillaMarkov)
        )
        if not self.folded_sampling and fused_greedy:
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
                    return self._tp_sync.sync(
                        SpecTpSyncSite.DSPARK_GRAPH_SAMPLE,
                        SampleStepTokens.execute(
                            step_logits=step_logits,
                            temperatures=self.temperatures[:bs],
                            greedy_mask=self.greedy_mask[:bs],
                            exp_noise=noise,
                        ),
                    )

            else:

                def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
                    return self._tp_sync.sync(
                        SpecTpSyncSite.DSPARK_GRAPH_GREEDY,
                        greedy_step_sampler(step_logits, step_idx),
                    )

            draft_tokens, corrected_logits = self.markov_head.sample_block(
                base_logits,
                first_prev_tokens=anchor,
                hidden_states=sample_hidden,
                sampler=sampler,
                collect_corrected=self.folded_sampling,
            )
            if self.folded_sampling:
                self.corrected_out[: bs * self.gamma].copy_(
                    corrected_logits.reshape(bs * self.gamma, -1)
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
      
    # The V4.1 TP head reduces compact argmax summaries in the greedy graph.
    if getattr(model.markov_head, "supports_sharded_greedy", False):
        return False
    vocab = _target_vocab_size(model)
    candidate = getattr(model, "markov_candidate_sampler", None)
    if candidate is not None:
        # Table, FP32 proposal cache, candidate IDs and RNG storage were
        # allocated before KV planning; the free-memory probe already excludes
        # them. Account here only for graph staging and Top-K outputs/scratch.
        noise_bytes = max_bs * (4 + 1 + 8) + 4
        logits_bytes = (
            max_bs * gamma * candidate.topk * (_base_logits_dtype(model).itemsize + 8)
        )
    else:
        noise_bytes = max_bs * vocab * 4
        logits_bytes = max_bs * gamma * vocab * _base_logits_dtype(model).itemsize
          
    need_gb = (noise_bytes + logits_bytes) / (1 << 30)
    if available_memory_gb - need_gb >= _CAPTURE_HEADROOM_GB:
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
    if not hasattr(draft_model, "compute_base_logits"):
        return _eager("no compute_base_logits")
    if getattr(draft_model, "markov_head", None) is None:
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
        tp_sync=tp_sync,
        confidence_fn=confidence_fn,
        out=out,
        folded_sampling=folded_sampling,
    )
