from __future__ import annotations

import logging
import time
from typing import Optional

import torch

from sglang.kernels.ops.speculative.dspark.markov_walk import MAX_BS, MarkovWalker
from sglang.srt.environ import envs
from sglang.srt.kernel_plugins.nvidia.binding import megakernel_draft_sampler
from sglang.srt.kernel_plugins.nvidia.dspark_binding import (
    bind_dspark_draft_strategy,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.speculative.dspark_components.dspark_block_accept_estimator import (
    block_accept_estimate_enabled,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftNoise,
    DraftParams,
    draft_markov_block,
    resolve_draft_params,
    select_draft_hidden_without_anchor,
    step_logits_width,
)
from sglang.srt.speculative.online_acceptance_estimator import (
    OnlineAcceptanceEstimator,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
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
        sharded_accept: bool = False,
        out=None,
        acceptance_estimator: Optional[OnlineAcceptanceEstimator] = None,
        markov_walker: Optional[MarkovWalker] = None,
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
        self.acceptance_estimator = acceptance_estimator
        self.acceptance_features_out: Optional[torch.Tensor] = None
        if acceptance_estimator is not None:
            self.acceptance_features_out = torch.empty(
                (self.max_bs, self.gamma), dtype=torch.float32, device=device
            )
            self.confidence_out = torch.empty_like(self.acceptance_features_out)
            self.confidence_raw_out = None
        elif confidence_head is None:
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
        self.sharded_accept = sharded_accept
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
        # [max_bs, gamma, vocab] step logits for sampling accept. The stock path
        # sizes it on first use; a bound specialized path may allocate it now.
        bound_strategy = bind_dspark_draft_strategy(
            draft_model=self.model,
            gamma=self.gamma,
            max_bs=self.max_bs,
            greedy_draft=self.greedy_draft,
            noise=self._noise,
            draft_tokens=self.out,
            step_temperatures=self.params.step_temperatures,
            # Temperature-0 rows may be left unwritten, while mixed accept still
            # reads the entire buffer.
            allocate_step_logits=lambda: torch.zeros(
                (
                    self.max_bs,
                    self.gamma,
                    _step_logits_width(self.model, self.sharded_accept),
                ),
                dtype=self.markov_head.step_logits_dtype(torch.bfloat16),
                device=self.out.device,
            ),
            specialize=acceptance_estimator is None,
        )
        self._draft_strategy = bound_strategy.strategy
        self.step_logits = bound_strategy.step_logits
        self._markov_walker = markov_walker
        # With the int8 walker: step temperatures are 0 from this row on.
        self._staged_bs = 0
        if markov_walker is not None:
            self._init_markov_walker(markov_walker)

    def _init_markov_walker(self, walker: MarkovWalker) -> None:
        assert walker.gamma == self.gamma, (walker.gamma, self.gamma)
        if not self.greedy_draft:
            # The kernel writes bf16 logits and leaves temperature-0 rows' unwritten; a
            # mixed accept still reads them.
            self.step_logits = torch.zeros(
                (self.max_bs, self.gamma, walker.vocab),
                dtype=torch.bfloat16,
                device=self.out.device,
            )
        self.markov_head._derived_weight_cache_error = (
            "Online weight updates are not supported with "
            "SGLANG_DSPARK_OPT_INT8_MARKOV_WALK: the draft walks an int8 copy of the "
            "markov head made at startup."
        )

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
            width = _step_logits_width(self.model, self.sharded_accept)
            self.step_logits = torch.empty(
                (self.max_bs, self.gamma, width),
                dtype=self.markov_head.step_logits_dtype(base_logits.dtype),
                device=base_logits.device,
            )
        return self.step_logits

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Must run before the draft graph replay that reads these buffers."""
        resolve_draft_params(sampling_info=sampling_info, bs=bs, out=self.params)
        if self._markov_walker is not None:
            # The graph walks the padded bucket but staging writes [:bs]; the int8
            # kernel would sample a stale T > 0 padding row for nothing.
            if self._staged_bs > bs:
                self.params.step_temperatures[bs : self._staged_bs].zero_()
            self._staged_bs = bs

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
        draft_tokens, confidence_features = self._draft_strategy.run(
            model_hidden=model_hidden,
            sample_hidden=sample_hidden,
            anchor_tokens=anchor,
            batch_size=bs,
            stock=self._markov_block,
        )
        if self.acceptance_estimator is not None:
            self.acceptance_estimator.predict(
                logits=self.step_logits[:bs],
                temperatures=self.params.step_temperatures[:bs],
                features_out=self.acceptance_features_out[:bs],
                predictions_out=self.confidence_out[:bs],
            )
        elif self.confidence_out is not None:
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
        # bs is the padded graph bucket, a Python int: fixed per captured graph.
        if self._markov_walker is not None and self._markov_walker.supports(bs):
            draft_tokens = self._markov_walker.walk(
                base_logits=base_logits,
                anchor=anchor,
                temps=None if self.greedy_draft else self.params.step_temperatures[:bs],
                tokens_out=self.out[: bs * self.gamma],
                corrected_out=None if self.greedy_draft else self.step_logits[:bs],
            )
            # Every rank walks all the steps; rank 0's tokens are kept, as after each
            # step of the default walk.
            draft_tokens = self._tp_sync.sync(SpecTpSyncSite.DSPARK_DRAFT, draft_tokens)
            return draft_tokens, confidence_features
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
            sharded_accept=self.sharded_accept,
        )
        self.out[: draft_tokens.numel()].copy_(draft_tokens.reshape(-1))
        return draft_tokens, confidence_features


def _step_logits_width(model, sharded_accept: bool) -> int:
    if sharded_accept:
        return int(model.lm_head.num_embeddings_per_partition)
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
    sharded_accept: bool = False,
    out=None,
    acceptance_estimator: Optional[OnlineAcceptanceEstimator] = None,
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
    markov_walker = None
    if envs.SGLANG_DSPARK_OPT_INT8_MARKOV_WALK.get():
        markov_walker = _build_int8_markov_walker(
            draft_model=draft_model,
            gamma=gamma,
            max_bs=min(max_bs, MAX_BS),
            device=device,
            seed=noise.key,
            megakernel=megakernel_draft_sampler(draft_model) is not None,
            online_estimator=acceptance_estimator is not None,
            tp_rank=tp_rank,
        )
    sampler = DsparkDraftSampler(
        model=draft_model,
        gamma=gamma,
        max_bs=max_bs,
        device=device,
        noise=noise,
        tp_sync=tp_sync,
        greedy_draft=greedy_draft,
        sharded_accept=sharded_accept,
        out=out,
        acceptance_estimator=acceptance_estimator,
        markov_walker=markov_walker,
    )
    if tp_rank == 0:
        logger.info(
            "DSpark draft proposal (%s) folded into the draft cuda graph%s.",
            "greedy draft" if greedy_draft else "sampling",
            sampler._draft_strategy.log_suffix,
        )
    return sampler


def _int8_markov_walk_unsupported_reason(
    *, draft_model, megakernel: bool, online_estimator: bool
) -> Optional[str]:
    """None when the int8 kernel computes this drafter's default walk, up to the
    int8 weights and bf16 rounding; MarkovWalker checks the GPU, rank and gamma."""
    if megakernel:
        return "the NVIDIA megakernel plugin runs the draft sampler"
    if block_accept_estimate_enabled():
        return "the block-accept estimator reads greedy rows' step logits"
    if online_estimator:
        return "the online acceptance estimator reads greedy rows' step logits"
    head = draft_model.markov_head
    if head.vocab_shard is not None:
        return "markov_w2 is vocab-sharded over the TP ranks"
    # The fused greedy step's test: the bias is exactly W2 W1[prev].
    if head.greedy_w2_weight() is None:
        return f"the {type(head).__name__} bias is not the markov_w2 GEMV"
    lm_head = draft_model.lm_head
    if not isinstance(lm_head, VocabParallelEmbedding):
        return f"lm_head is {type(lm_head).__name__}"
    vocab = int(lm_head.org_vocab_size)
    if head.vocab_size != vocab:
        return f"markov vocab {head.vocab_size} != lm_head vocab {vocab}"
    if lm_head.num_embeddings_padded != vocab:
        # The kernel reads base logits with row stride V, not a cropped slice.
        return f"lm_head vocab is padded to {lm_head.num_embeddings_padded}"
    if lm_head.weight.dtype != torch.bfloat16:
        return f"lm_head is {lm_head.weight.dtype}, the kernel reads bf16 base logits"
    return None


def _build_int8_markov_walker(
    *,
    draft_model,
    gamma: int,
    max_bs: int,
    device,
    seed: int,
    megakernel: bool,
    online_estimator: bool,
    tp_rank: int,
) -> MarkovWalker:
    reason = _int8_markov_walk_unsupported_reason(
        draft_model=draft_model,
        megakernel=megakernel,
        online_estimator=online_estimator,
    )
    if reason is not None:
        raise ValueError(
            f"SGLANG_DSPARK_OPT_INT8_MARKOV_WALK cannot run this setup: {reason}."
        )
    tic = time.perf_counter()
    head = draft_model.markov_head
    walker = MarkovWalker(
        w1=head.markov_w1.weight.detach(),
        w2=head.greedy_w2_weight().detach(),
        gamma=gamma,
        max_bs=max_bs,
        device=device,
        seed=seed,
    )
    walker.warmup()
    if tp_rank == 0:
        logger.info(
            "Build DSpark int8 markov walk end. max_bs=%d, elapsed=%.2f s, "
            "mem usage=%.2f GB.",
            walker.max_bs,
            time.perf_counter() - tic,
            walker.memory_bytes() / (1 << 30),
        )
    return walker
