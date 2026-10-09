from __future__ import annotations

import logging
import time
from typing import Optional

import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    SampleStepTokens,
)
from sglang.kernels.ops.speculative.dspark.markov_walk import (
    MARKOV_RANK,
    MAX_BS,
    MAX_STEPS,
    MarkovWalker,
)
from sglang.srt.environ import DsparkFoldedSampling, envs
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.runtime_context import get_disagg
from sglang.srt.speculative.dspark_components.dspark_block_accept_estimator import (
    block_accept_estimate_enabled,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    select_draft_hidden_without_anchor,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.srt.utils import is_cuda

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
        self._tp_sync = tp_sync
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
        self._markov_walker: Optional[MarkovWalker] = None
        self._zero_temperature = None
        # Rows below this may still be non-greedy from the last sampling batch.
        self._sampling_rows_hi = 0

    def attach_int8_markov_walker(self, walker: MarkovWalker) -> None:
        """Route every graph bucket the walker supports through it; must run
        before capture, the branch in __call__ is fixed per captured bucket."""
        assert walker.gamma == self.gamma, (walker.gamma, self.gamma)
        walker.warmup(corrected_out=self.corrected_out)
        if self.folded_sampling:
            # The kernel skips greedy rows, and the mixed-batch verifier
            # softmaxes every row: they must hold finite values, not torch.empty.
            self.corrected_out.zero_()
            self._zero_temperature = self.temperatures.new_zeros(())
        self.markov_head._derived_weight_cache_error = (
            "Online weight updates are not supported with "
            "SGLANG_DSPARK_OPT_INT8_MARKOV_WALK=1: the draft walks an int8 copy "
            "of the markov head made at startup. Restart without it to update "
            "weights online."
        )
        self._markov_walker = walker

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Host-side refresh of the static sampling params; must run before
        the draft graph replay that consumes them."""
        if not self.folded_sampling:
            return
        if self._markov_walker is not None:
            self._clear_stale_sampling_rows(bs=bs, sampling_info=sampling_info)
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

    def _clear_stale_sampling_rows(self, *, bs: int, sampling_info) -> None:
        # The graph walks the padded bucket but staging writes [:bs]; a stale
        # non-greedy pad row would make the int8 kernel sample it for nothing.
        if self._sampling_rows_hi > bs:
            self.greedy_mask[bs : self._sampling_rows_hi].fill_(True)
        all_greedy = sampling_info is None or sampling_info.is_all_greedy
        self._sampling_rows_hi = 0 if all_greedy else bs

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
        # bs is the padded graph bucket, a Python int: fixed per captured graph.
        if self._markov_walker is not None and self._markov_walker.supports(bs):
            draft_tokens = self._walk_int8(base_logits=base_logits, anchor=anchor)
        else:
            draft_tokens = self._walk_default(
                base_logits=base_logits, anchor=anchor, sample_hidden=sample_hidden
            )
        if self.confidence_out is not None:
            confidence = self.confidence_fn(
                draft_hidden=sample_hidden,
                anchor_tokens=anchor,
                draft_tokens=draft_tokens,
                confidence_tap=confidence_tap,
            )
            self.confidence_out[:bs].copy_(confidence)

    def _walk_int8(self, *, base_logits, anchor) -> torch.Tensor:
        bs = base_logits.shape[0]
        n = bs * self.gamma
        temps, corrected = None, None
        if self.folded_sampling:
            # Kernel greedy is T <= 0; sglang marks greedy rows by top_k <= 1.
            temps = self._markov_walker.temps_buf[:bs]
            torch.where(
                self.greedy_mask[:bs],
                self._zero_temperature,
                self.temperatures[:bs],
                out=temps,
            )
            corrected = self.corrected_out[:n]
        return self._markov_walker.walk(
            base_logits=base_logits,
            anchor=anchor,
            temps=temps,
            tokens_out=self.out[:n],
            corrected_out=corrected,
        )

    def _walk_default(self, *, base_logits, anchor, sample_hidden) -> torch.Tensor:
        bs = base_logits.shape[0]
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
        return draft_tokens


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
    vocab = int(model.lm_head.org_vocab_size)
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
    tp_size: int,
    tp_sync: SpecTpSync,
    available_memory_gb: float,
    seed: int,
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
    walk_started = time.perf_counter()
    markov_walker = None
    if envs.SGLANG_DSPARK_OPT_INT8_MARKOV_WALK.get():
        markov_walker = _maybe_build_int8_markov_walker(
            draft_model=draft_model,
            gamma=gamma,
            max_bs=max_bs,
            device=device,
            tp_rank=tp_rank,
            tp_size=tp_size,
            seed=seed,
        )
    if markov_walker is not None:
        available_memory_gb -= markov_walker.memory_bytes() / (1 << 30)
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
    draft_sampler = DsparkDraftSampler(
        model=draft_model,
        gamma=gamma,
        max_bs=max_bs,
        device=device,
        tp_sync=tp_sync,
        confidence_fn=confidence_fn,
        out=out,
        folded_sampling=folded_sampling,
    )
    if markov_walker is not None:
        _attach_int8_markov_walker(
            draft_sampler=draft_sampler,
            walker=markov_walker,
            tp_rank=tp_rank,
            started=walk_started,
        )
    return draft_sampler


def log_int8_markov_walk_off(*, tp_rank: int, reason: str) -> None:
    if tp_rank == 0:
        logger.warning(
            "SGLANG_DSPARK_OPT_INT8_MARKOV_WALK ignored, falling back to the default "
            "markov walk (reason=%s).",
            reason,
        )


def _int8_markov_walk_unsupported_reason(
    *, draft_model, gamma: int, tp_size: int, device
) -> Optional[str]:
    """None when the int8 kernels compute this drafter's default walk, up to the
    int8 weights and bf16 rounding."""
    if tp_size > 1:
        return f"tp_size={tp_size}, the kernels run on a single rank"
    if get_disagg().enable_pdmux:
        return "pdmux partitions the SMs the cooperative grid needs"
    if not is_cuda() or torch.cuda.get_device_capability(device) != (9, 0):
        return "needs sm_90"
    if block_accept_estimate_enabled():
        return "the block-accept estimator reads greedy rows' corrected logits"
    head = draft_model.markov_head
    # Exact type: subclasses quantize W2 or feed hidden states into the bias.
    if type(head) is not VanillaMarkov:
        return f"markov head is {type(head).__name__}, not VanillaMarkov"
    if head.markov_rank != MARKOV_RANK:
        return f"markov_rank={head.markov_rank}, the kernels need {MARKOV_RANK}"
    lm_head = draft_model.lm_head
    if not isinstance(lm_head, VocabParallelEmbedding):
        return f"lm_head is {type(lm_head).__name__}"
    vocab = int(lm_head.org_vocab_size)
    if head.vocab_size != vocab:
        return f"markov vocab {head.vocab_size} != lm_head vocab {vocab}"
    if lm_head.num_embeddings_padded != vocab:
        # The kernels read base logits with row stride V, not a cropped slice.
        return f"lm_head vocab is padded to {lm_head.num_embeddings_padded}"
    logits_dtype = _base_logits_dtype(draft_model)
    if logits_dtype != torch.bfloat16:
        return f"base logits are {logits_dtype}, the kernels take bf16"
    if not 1 <= gamma <= MAX_STEPS:
        return f"gamma={gamma} outside [1, {MAX_STEPS}]"
    return None


def _maybe_build_int8_markov_walker(
    *,
    draft_model,
    gamma: int,
    max_bs: int,
    device,
    tp_rank: int,
    tp_size: int,
    seed: int,
) -> Optional[MarkovWalker]:
    reason = _int8_markov_walk_unsupported_reason(
        draft_model=draft_model, gamma=gamma, tp_size=tp_size, device=device
    )
    if reason is not None:
        log_int8_markov_walk_off(tp_rank=tp_rank, reason=reason)
        return None
    head = draft_model.markov_head
    try:
        return MarkovWalker(
            w1=head.markov_w1.weight.detach(),
            w2=head.markov_w2.weight.detach(),
            gamma=gamma,
            max_bs=min(max_bs, MAX_BS),
            device=device,
            seed=seed,
        )
    # A JIT build failure, V % 8 != 0 or a V too large for the SM count.
    except Exception as e:
        log_int8_markov_walk_off(tp_rank=tp_rank, reason=f"{type(e).__name__}: {e}")
        return None


def _attach_int8_markov_walker(
    *, draft_sampler: DsparkDraftSampler, walker: MarkovWalker, tp_rank: int, started
) -> None:
    try:
        draft_sampler.attach_int8_markov_walker(walker)
    except Exception as e:
        reason = f"warmup failed: {type(e).__name__}: {e}"
        log_int8_markov_walk_off(tp_rank=tp_rank, reason=reason)
        return
    if tp_rank == 0:
        logger.info(
            "DSpark int8 markov walk on: kernels %s for bs <= %d (larger graph "
            "buckets keep the default walk), gamma=%d, %.0f MiB resident, ready "
            "in %.1f s.",
            "wgmma" if walker.weights.big_vocab else "single/small_batch/wgmma",
            walker.max_bs,
            walker.gamma,
            walker.memory_bytes() / (1 << 20),
            time.perf_counter() - started,
        )
