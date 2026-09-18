"""Serving support for the LiLiCorr candidate-lattice reranker on DFLASH drafts.

Head geometry, the candidate lattice, the eager draft seam and the CUDA-graph-folded
draft sampler. The head itself is `sglang.srt.models.lilicorr`.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.speculative.lilicorr import (
    MAX_FUSED_CANDIDATE_TOPK,
    lilicorr_topk_lse,
)
from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.runtime_context import get_exec
from sglang.srt.speculative.dflash_utils import _get_dflash_config
from sglang.srt.speculative.dspark_components.dspark_draft import resolve_greedy_mask

logger = logging.getLogger(__name__)


def _env_bool(name: str, message: str) -> bool:
    raw = os.environ.get(name, "0").strip().lower()
    if raw not in ("0", "1", "off", "on", "false", "true"):
        raise ValueError(f"{name}={raw!r} is not a boolean. {message}")
    return raw in ("1", "on", "true")


# Off is byte-identical to the greedy commit: same kernel, same argmax, no proposal. On
# samples each slot from softmax(scores / T) and publishes the row it drew from, so verify
# pays sum_c min(p, q) rather than p(argmax).
# Read at import, not per call: `select` is torch.compiled and replayed from a captured
# CUDA graph, so an os.environ read in the body is a graph break.
SAMPLING_ENABLED = _env_bool(
    "LILICORR_SAMPLING",
    "Refusing rather than defaulting: a typo would report the greedy number under the "
    "sampled arm's name.",
)
logger.info(
    "LiLiCorr draft commit: %s",
    "sampled (T>0 aware)" if SAMPLING_ENABLED else "greedy",
)

# Forgetting LILICORR_SAMPLING entirely is silent and serves the greedy commit under a
# sampled arm's name; set this on any run whose name claims to be sampled.
if (
    _env_bool(
        "SGLANG_LILICORR_REQUIRE_SAMPLING",
        "Refusing rather than defaulting: a gate that silently reads as off is not a gate.",
    )
    and not SAMPLING_ENABLED
):
    raise RuntimeError(
        "SGLANG_LILICORR_REQUIRE_SAMPLING is set but LILICORR_SAMPLING is not enabled. "
        "This process would serve the greedy commit under a sampled arm's name."
    )


# ===== Head geometry =====


class LiLiCorrConfig(msgspec.Struct, frozen=True):
    candidate_topk: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    factor_dim: int
    vector_eps: float
    logit_scale: float

    def resolve_hidden_size(self, *, model_hidden_size: int) -> int:
        # 0 means "as wide as the draft"; the exporter records it for a head with no
        # token_proj.
        return int(self.hidden_size) if self.hidden_size else int(model_hidden_size)


def _parse_lilicorr_config(dflash_cfg: dict) -> Optional[LiLiCorrConfig]:
    # Absence and an explicit lilicorr_enabled: false both mean "no head".
    if not any(key.startswith("lilicorr_") for key in dflash_cfg):
        return None
    enabled = dflash_cfg.get("lilicorr_enabled")
    if enabled is not None and not bool(enabled):
        return None

    def required(key: str, cast, *, positive: bool = True):
        # No field may be defaulted: most change a tensor shape and would be caught at
        # weight load, but logit_scale and vector_eps would not.
        full_key = f"lilicorr_{key}"
        if full_key not in dflash_cfg:
            raise ValueError(
                f"DFLASH dflash_config.{full_key} is required to rebuild the LiLiCorr "
                "head. The checkpoint does not carry it, so the head this would "
                "construct is not the head that was trained."
            )
        try:
            value = cast(dflash_cfg[full_key])
        except Exception as e:
            raise ValueError(
                f"Invalid dflash_config.{full_key}={dflash_cfg[full_key]!r}."
            ) from e
        if positive and value <= 0:
            raise ValueError(f"dflash_config.{full_key} must be positive, got {value}.")
        return value

    candidate_topk = required("candidate_topk", int)
    if candidate_topk & (candidate_topk - 1):
        raise ValueError(
            f"dflash_config.lilicorr_candidate_topk must be a power of two, got "
            f"{candidate_topk}. The tiled candidate top-k selects its tiles inside a "
            "single Triton lane group, and tl.arange requires a power-of-two extent."
        )
    if candidate_topk > MAX_FUSED_CANDIDATE_TOPK:
        # A wider pool serves correctly but falls off the fused greedy commit onto the
        # torch path, which is roughly three kernel launches per slot.
        raise ValueError(
            f"dflash_config.lilicorr_candidate_topk={candidate_topk} exceeds the fused "
            f"greedy commit's width of {MAX_FUSED_CANDIDATE_TOPK}, which is one Triton "
            "lane group. A wider head would serve on the reference path."
        )

    return LiLiCorrConfig(
        candidate_topk=candidate_topk,
        hidden_size=required("hidden_size", int, positive=False),
        num_layers=required("num_layers", int),
        num_heads=required("num_heads", int),
        mlp_ratio=required("mlp_ratio", float),
        factor_dim=required("factor_dim", int),
        vector_eps=required("vector_eps", float),
        logit_scale=required("logit_scale", float),
    )


def parse_lilicorr_draft_config(*, draft_hf_config: Any) -> LiLiCorrConfig:
    config = _parse_lilicorr_config(_get_dflash_config(draft_hf_config))
    if config is None:
        raise ValueError(
            "LiLiCorr requires the lilicorr_* geometry fields in dflash_config. "
            'A checkpoint declaring architectures=["LiLiCorrDraftModel"] without them '
            "cannot be rebuilt into the head that was trained."
        )
    return config


# ===== The candidate lattice =====


def resolve_vocab_shard(lm_head) -> Tuple[int, int]:
    """(num_org, org_vocab_start) for this rank's slice of the target head."""
    if not isinstance(lm_head, VocabParallelEmbedding):
        return int(lm_head.weight.shape[0]), 0
    shard = lm_head.shard_indices
    if int(shard.num_added_elements) != 0:
        raise NotImplementedError(
            "LiLiCorr's candidate head does not support added vocabulary: the added rows "
            "sit past the padded base shard, so a single contiguous top-k would skip them."
        )
    return int(shard.num_org_elements), int(shard.org_vocab_start_index)


def target_input_embeddings(target_model):
    # Deliberately not the worker's _resolve_dflash_embedding_module, which returns the
    # draft's own table for Nemotron-3.5 drafts: the head must embed candidate ids with
    # the table it was trained against.
    embed = target_model.get_input_embeddings()
    if embed is None:
        raise RuntimeError("DFLASH target model exposes no input embeddings.")
    return embed


def lilicorr_candidates(
    *,
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    num_org: int,
    org_vocab_start: int,
    topk: int,
    logits_out: Optional[torch.Tensor] = None,
    tp_group=None,
    chunk_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row top-k over the target LM head, as normalized log-probs and global ids.

    Returns (log_probs [N, topk] fp32, tokens [N, topk] int64), equivalent to
    log_softmax(logits).topk(topk). The head scores log-probs normalized over the full
    vocabulary, so the log-partition is part of the contract. tp_group is required
    when the head is vocabulary-sharded.
    """
    topk = int(topk)
    if topk > int(num_org):
        raise ValueError(
            f"LiLiCorr candidate topk={topk} exceeds this rank's vocabulary slice "
            f"({num_org} rows), so the lattice cannot be filled."
        )
    tp_size = 1 if tp_group is None else int(tp_group.world_size)
    num_rows = int(hidden_states.shape[0])

    def one_span(rows: torch.Tensor, logits_buf: Optional[torch.Tensor]):
        if rows.dtype != weight.dtype:
            rows = rows.to(weight.dtype)
        if logits_buf is None:
            logits = torch.matmul(rows, weight[:num_org].T)
        else:
            logits = logits_buf
            torch.matmul(rows, weight[:num_org].T, out=logits)
        vals, tokens, lse = lilicorr_topk_lse(logits, topk)
        tokens = tokens + org_vocab_start
        if tp_size > 1:
            vals, tokens, lse = _combine_across_ranks(
                vals=vals, tokens=tokens, lse=lse, topk=topk, tp_group=tp_group
            )
        return vals - lse.unsqueeze(-1), tokens

    # The folded path supplies a preallocated buffer so no large allocation lands in a
    # CUDA graph's private pool; it is one span by construction and returns directly.
    if logits_out is not None:
        return one_span(hidden_states, logits_out)

    device = hidden_states.device
    out_vals = torch.empty((num_rows, topk), dtype=torch.float32, device=device)
    out_tokens = torch.empty((num_rows, topk), dtype=torch.int64, device=device)
    for start in range(0, num_rows, int(chunk_size)):
        end = min(num_rows, start + int(chunk_size))
        vals, tokens = one_span(hidden_states[start:end], None)
        out_vals[start:end] = vals
        out_tokens[start:end] = tokens
    return out_vals, out_tokens


def _combine_across_ranks(
    *,
    vals: torch.Tensor,
    tokens: torch.Tensor,
    lse: torch.Tensor,
    topk: int,
    tp_group,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Three per-step collectives fused into one all-gather by packing
    # [vals | tokens as fp32 | lse] per row; ids below 2**24 are exact in fp32.
    tp_size = int(tp_group.world_size)
    rows = int(vals.shape[0])
    width = 2 * topk + 1
    packed = torch.empty((rows, width), dtype=torch.float32, device=vals.device)
    packed[:, :topk] = vals.float()
    packed[:, topk : 2 * topk] = tokens.to(torch.float32)
    packed[:, 2 * topk] = lse.float()

    gathered = torch.empty(
        tp_size * rows * width, dtype=torch.float32, device=vals.device
    )
    tp_group.all_gather_into_tensor(gathered, packed.contiguous().view(-1))
    gathered = gathered.view(tp_size, rows, width)

    all_vals = gathered[:, :, :topk].permute(1, 0, 2).reshape(rows, tp_size * topk)
    all_tokens = (
        gathered[:, :, topk : 2 * topk]
        .permute(1, 0, 2)
        .reshape(rows, tp_size * topk)
        .round()
        .to(torch.int64)
    )
    top_vals, top_idx = torch.topk(all_vals, topk, dim=-1)
    return (
        top_vals,
        torch.gather(all_tokens, 1, top_idx),
        torch.logsumexp(gathered[:, :, 2 * topk], dim=0),
    )


def per_request_last_row(
    *,
    num_rows: int,
    extend_lens: Optional[torch.Tensor],
    commit_lens: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Index of each request's last committed row, or None if not recoverable.

    The two callers hand over different layouts: verify passes commit_lens against a
    buffer padded to [bs, block_size] and flattened, so requests sit at a constant
    stride; prefill/extend passes extend_lens against a packed request-major buffer,
    so requests are ragged and back to back.
    """
    if commit_lens is not None:
        bs = int(commit_lens.shape[0])
        if bs == 0 or num_rows % bs != 0:
            return None
        stride = num_rows // bs
        base = torch.arange(bs, device=commit_lens.device, dtype=torch.int64) * stride
        return (base + commit_lens.to(torch.int64) - 1).clamp_min(0)
    if extend_lens is None or extend_lens.numel() == 0:
        return None
    lens = extend_lens.to(torch.int64).flatten()
    if int(lens.sum()) != int(num_rows):
        return None
    return (torch.cumsum(lens, dim=0) - 1).clamp_min(0)


def publish_anchor(
    *,
    draft_sampler,
    ctx_hidden: torch.Tensor,
    extend_lens: Optional[torch.Tensor] = None,
    commit_lens: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Each request's last committed context row, as the head's anchor.

    ctx_hidden is the fc-projected target context the caller already computed for the
    KV write. Unrecoverable boundaries return None, which the head scores as "no anchor";
    fabricating one would be a silent acceptance regression.
    """
    ends = per_request_last_row(
        num_rows=int(ctx_hidden.shape[0]),
        extend_lens=extend_lens,
        commit_lens=commit_lens,
    )
    anchor = None if ends is None else ctx_hidden.index_select(0, ends)
    if draft_sampler is not None:
        draft_sampler.set_anchor(anchor, 0 if anchor is None else int(anchor.shape[0]))
    return anchor


# ===== The eager draft seam =====


def propose_lilicorr_block(
    *,
    head,
    draft_hidden: torch.Tensor,
    lm_head,
    embed_tokens,
    anchor: Optional[torch.Tensor],
    sampling_info=None,
    sampling_enabled: bool = SAMPLING_ENABLED,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Reranked draft block in place of the per-slot greedy argmax.

    Serves the steps the draft CUDA graph cannot: prefill, extend, batches past the
    captured buckets, and tp>1. draft_hidden is [bs, block_size, hidden], slot 0
    being the anchor position. Returns ``(tokens [bs, block_size - 1], candidate_tokens,
    q_rows); the latter two are None unless sampling_enabled``.

    sampling_enabled is the worker's device gate rather than the module constant: this
    path publishes into the selector's accept kernel, which does not run everywhere
    LILICORR_SAMPLING can be set.
    """
    bs, block_size, hidden_size = draft_hidden.shape
    slots = block_size - 1
    pass_hidden = draft_hidden[:, 1:, :]
    tp_group = get_tp_group()
    num_org, org_vocab_start = resolve_vocab_shard(lm_head)

    log_probs, candidate_tokens = lilicorr_candidates(
        hidden_states=pass_hidden.reshape(bs * slots, hidden_size),
        weight=lm_head.weight,
        num_org=num_org,
        org_vocab_start=org_vocab_start,
        topk=int(head.candidate_topk),
        tp_group=tp_group if int(tp_group.world_size) > 1 else None,
    )
    candidate_tokens = candidate_tokens.view(bs, slots, int(head.candidate_topk))

    feat = int(head.context_proj.in_features)
    if anchor is None or int(anchor.shape[0]) != bs:
        # A row count that disagrees with the batch means the batch changed size between
        # the context append that wrote the anchor and this read, so the whole batch
        # scores as invalid rather than against another request's row.
        anchor_hidden = torch.zeros(
            (bs, feat), device=draft_hidden.device, dtype=draft_hidden.dtype
        )
        anchor_valid = torch.zeros((bs,), dtype=torch.bool, device=draft_hidden.device)
    else:
        anchor_hidden = anchor.view(bs, feat)
        anchor_valid = torch.ones((bs,), dtype=torch.bool, device=draft_hidden.device)

    # The embedding lookup happens outside the head because on the target model it may be
    # a TP-sharded collective.
    common = dict(
        token_embeddings=embed_tokens(candidate_tokens).detach(),
        candidate_tokens=candidate_tokens,
        candidate_log_probs=log_probs.view(bs, slots, int(head.candidate_topk)),
        pass_hidden=pass_hidden,
        anchor_hidden=anchor_hidden,
        anchor_valid=anchor_valid,
    )
    if not sampling_enabled:
        return head.select(**common).to(torch.long), None, None

    device = draft_hidden.device
    temperatures = (
        torch.ones(bs, dtype=torch.float32, device=device)
        if sampling_info is None
        # Clamped like DSpark and the DFlash2 selector so a greedy row, whose temperature
        # may be exactly 0, cannot divide by zero before greedy_mask discards its pick.
        else sampling_info.temperatures.view(-1)[:bs].float().clamp_min(1e-5)
    )
    selected, q_rows = head.select_with_proposal(
        uniforms=torch.rand(bs, slots, dtype=torch.float32, device=device),
        temperatures=temperatures,
        greedy_mask=resolve_greedy_mask(
            bs=bs, sampling_info=sampling_info, device=device
        ),
        **common,
    )
    return selected.to(torch.long), candidate_tokens, q_rows


# ===== The graph-folded draft sampler =====


class _RecompileLimitWatcher(logging.Handler):
    # Past a recompile limit dynamo permanently runs the original function and only logs a
    # warning, so the run would be a mixture of compiled and eager buckets.

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.hits: list[str] = []

    def emit(self, record) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        if "recompile_limit" in message or "cache_size_limit" in message:
            self.hits.append(message.strip().splitlines()[0][:200])


def _pin_inductor_to_eager_numerics() -> None:
    # Fused intermediates stay in fp32 where eager rounds to bf16 between ops, and split
    # reductions are a different summation order; both move an argmax over near-ties.
    import torch._inductor.config as inductor_config

    inductor_config.emulate_precision_casts = True
    inductor_config.split_reductions = False


class LiLiCorrDraftSampler:
    """LiLiCorr select, run inside the draft CUDA graph.

    Follows the DFLASH draft sampler contract: __call__(hidden_states, input_ids)
    writes the drafted tokens for block positions 1.. into self.out. The worker
    registers it on draft_model_runner.capture_tail_hooks, so it executes immediately
    after the draft forward and the worker reads out after the replay.
    """

    def __init__(
        self,
        *,
        head,
        embed_tokens,
        weight: torch.Tensor,
        block_size: int,
        num_org: int,
        org_vocab_start: int,
        max_bs: int,
        anchor_features: int,
        sampling_enabled: bool,
    ) -> None:
        self.head = head
        # Device-gated by the worker, as the selector's sampler is: the sampled accept
        # path needs chain_speculative_sampling_triton, which does not run on NPU.
        self.sampling_enabled = bool(sampling_enabled)
        self.embed_tokens = embed_tokens
        self.weight = weight
        self.block_size = int(block_size)
        self.slots = self.block_size - 1
        self.num_org = int(num_org)
        self.org_vocab_start = int(org_vocab_start)
        self.topk = int(head.candidate_topk)
        self.max_bs = int(max_bs)

        device, dtype = weight.device, weight.dtype
        max_rows = self.max_bs * self.slots

        # Read by the worker after the replay.
        self.out = torch.empty((max_rows,), dtype=torch.int64, device=device)
        # Static, so the in-graph GEMM does not put a large allocation in each bucket's
        # private CUDA-graph pool.
        self.logits = torch.empty((max_rows, self.num_org), dtype=dtype, device=device)
        # Written by the worker before each replay.
        self.anchor = torch.zeros(
            (self.max_bs, int(anchor_features)), dtype=dtype, device=device
        )
        self.anchor_valid = torch.zeros((self.max_bs,), dtype=torch.bool, device=device)
        self.token_table = head.build_token_table(embed_tokens)

        # Sampling state: temperatures and greedy_mask written by the worker before the
        # replay, uniforms drawn inside it, q_out and candidate_out read after it.
        # Allocated unconditionally at ~90 KiB so both paths share one object graph.
        self.temperatures = torch.ones(
            (self.max_bs,), dtype=torch.float32, device=device
        )
        self.greedy_mask = torch.ones((self.max_bs,), dtype=torch.bool, device=device)
        self.uniforms = torch.empty(
            (self.max_bs, self.slots), dtype=torch.float32, device=device
        )
        self.q_out = torch.empty(
            (self.max_bs, self.slots, self.topk), dtype=torch.float32, device=device
        )
        # Verify needs the ids q is indexed against, and the candidate tensor is an
        # intermediate inside the compiled region, so it is copied to a fixed address.
        self.candidate_out = torch.empty(
            (self.max_bs, self.slots, self.topk), dtype=torch.int64, device=device
        )

        self._warmed_bs: set[int] = set()
        self._prewarming = False
        _pin_inductor_to_eager_numerics()
        self._select = torch.compile(
            head.select_with_proposal if sampling_enabled else head.select,
            # The head's GEMMs already go to cuBLAS; the gap being closed is pointwise
            # fusion, which default mode does.
            mode="default",
            # The greedy commit is a Triton kernel, hence a graph break.
            fullgraph=False,
            # One static shape per bucket. Measured on an H100 at the served geometry:
            # dynamic=True costs +6.9% head time at bs=1 and +13.8% at bs=30.
            dynamic=False,
        )
        self._watcher = _RecompileLimitWatcher()
        logging.getLogger("torch._dynamo").addHandler(self._watcher)

    def set_anchor(self, rows: Optional[torch.Tensor], bs: int) -> None:
        """Publish this step's anchor into the buffer the graph reads.

        Rows the caller did not supply are zeroed and marked invalid rather than left
        stale, because the graph runs at the padded bucket batch size; the head multiplies
        the projected anchor by its validity flag.

        Row i is the same request across steps only while the batch composition is
        unchanged, so filter/merge can hand a row a neighbour's anchor. That feeds the
        drafter and not verify, so it costs acceptance, never correctness. Carrying the
        anchor on DFlashDraftInputV2 instead so filter/merge reorder it measured -4.67%
        acceptance at concurrency 32; measure before trying it again.
        """
        count = int(bs)
        if rows is None or int(rows.shape[0]) != count or count > self.max_bs:
            count = 0
        if count:
            self.anchor[:count].copy_(rows)
            self.anchor_valid[:count].fill_(True)
        if count < self.max_bs:
            self.anchor[count:].zero_()
            self.anchor_valid[count:].fill_(False)

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Refresh the static sampling params; must run before the replay that reads them.

        Rows past bs are left alone: they already score a zeroed anchor and the worker
        discards them with out[: bs * slots].
        """
        if not self.sampling_enabled:
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
        self.greedy_mask[:bs].copy_(
            resolve_greedy_mask(
                bs=bs, sampling_info=sampling_info, device=self.greedy_mask.device
            )
        )

    def _raise_if_recompile_limit_hit(self, where: str) -> None:
        if not self._watcher.hits:
            return
        hits, self._watcher.hits = list(self._watcher.hits), []
        raise RuntimeError(
            f"dynamo hit a recompile limit during {where}, so some capture buckets run "
            "eager inside a graph labelled 'compiled' and any throughput delta is a "
            f"mixture rather than a fusion result: {hits[:3]}."
        )

    def prewarm_compile(self, bs_list, hidden_dtype: torch.dtype) -> None:
        """Trace and compile the body at every bucket, before any capture.

        Compiling under capture is illegal, so inductor has to do its codegen here. The
        synthetic inputs are junk; only their shape matters, and self.out is
        overwritten by the first real replay.
        """
        buckets = sorted({int(b) for b in bs_list if 0 < int(b) <= self.max_bs})
        if not buckets:
            raise RuntimeError(
                "LiLiCorr compile prewarm has no buckets to warm, so capture would raise. "
                f"Resolved max_bs={self.max_bs} against bs_list={sorted(bs_list)}."
            )

        # Guards key on strides and alignment as well as shape, so the variant count is
        # not predictable: size the limit so it cannot bind and let the watcher report if
        # it somehow still does.
        needed = max(256, 16 * len(buckets))
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit, needed
        )
        torch._dynamo.config.accumulated_recompile_limit = max(
            torch._dynamo.config.accumulated_recompile_limit, 8 * needed
        )

        hidden_size = int(self.weight.shape[1])
        self._prewarming = True
        try:
            with torch.no_grad():
                for bs in buckets:
                    self(
                        torch.zeros(
                            (bs * self.block_size, hidden_size),
                            dtype=hidden_dtype,
                            device=self.weight.device,
                        )
                    )
                    self._warmed_bs.add(bs)
        finally:
            self._prewarming = False
        self._raise_if_recompile_limit_hit("prewarm")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info(
            "LiLiCorr in-graph select compiled and prewarmed for %d buckets %s (dtype=%s).",
            len(buckets),
            buckets,
            hidden_dtype,
        )

    def __call__(self, hidden_states: torch.Tensor, input_ids=None) -> None:
        del input_ids  # the lattice is scored from hidden states and the anchor
        bs = hidden_states.shape[0] // self.block_size
        rows = bs * self.slots

        # max_bs sizes every static buffer here, so a larger replay would return short
        # slices and corrupt the drafts for the largest batches.
        if bs > self.max_bs:
            raise RuntimeError(
                f"LiLiCorrDraftSampler was built for max_bs={self.max_bs} but the draft "
                f"graph replayed at bs={bs}; its static buffers are undersized."
            )
        if not self._prewarming:
            if bs not in self._warmed_bs and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"the draft graph is capturing bs={bs}, which the compile prewarm "
                    f"never covered (warmed: {sorted(self._warmed_bs)}). Compiling under "
                    "capture is illegal, so fix the bucket list rather than widening the "
                    "prewarm."
                )
            self._raise_if_recompile_limit_hit("cuda-graph capture or decode")

        hidden_size = hidden_states.shape[-1]
        pass_hidden = hidden_states.view(bs, self.block_size, hidden_size)[:, 1:, :]
        log_probs, candidate_tokens = lilicorr_candidates(
            hidden_states=pass_hidden.reshape(rows, hidden_size),
            weight=self.weight,
            num_org=self.num_org,
            org_vocab_start=self.org_vocab_start,
            topk=self.topk,
            logits_out=self.logits[:rows],
        )
        candidate_tokens = candidate_tokens.view(bs, self.slots, self.topk)

        pre_projected = self.token_table is not None
        token_embeddings = (
            self.token_table[candidate_tokens]
            if pre_projected
            else self.embed_tokens(candidate_tokens).detach()
        )
        common = dict(
            token_embeddings=token_embeddings,
            candidate_tokens=candidate_tokens,
            candidate_log_probs=log_probs.view(bs, self.slots, self.topk),
            pass_hidden=pass_hidden,
            anchor_hidden=self.anchor[:bs],
            anchor_valid=self.anchor_valid[:bs],
            already_projected=pre_projected,
        )
        if self.sampling_enabled:
            selected, q_rows = self._select(
                # In-graph philox draw: each replay advances the generator and redraws.
                # Drawn here because an RNG op in the compiled body is a graph break.
                uniforms=self.uniforms[:bs].uniform_(),
                temperatures=self.temperatures[:bs],
                greedy_mask=self.greedy_mask[:bs],
                **common,
            )
            self.q_out[:bs].copy_(q_rows)
            self.candidate_out[:bs].copy_(candidate_tokens)
        else:
            selected = self._select(**common)
        self.out[:rows].copy_(selected.reshape(-1).to(torch.int64))


def draft_graph_batch_sizes() -> list[int]:
    # Every bucket, not just the max: the folded head is captured once per bucket and the
    # compile prewarm has to cover all of them.
    return sorted(
        {int(bs) for bs in get_exec().graph.cuda_graph_config.decode.bs if bs > 0}
    )


def build_lilicorr_draft_sampler(
    *,
    head,
    draft_model,
    embed_tokens,
    lm_head,
    block_size: int,
    sampling_enabled: bool = SAMPLING_ENABLED,
) -> Optional[LiLiCorrDraftSampler]:
    """Build the graph-folded LiLiCorr sampler, or None to keep the head eager.

    Returning None is not a neutral choice: the eager head costs a large fraction of
    throughput and drops the anchor for the whole batch whenever the batch size changes,
    so each refusal is logged with its reason.
    """

    def eager(reason: str) -> None:
        logger.warning(
            "LiLiCorr head kept eager (reason=%s). This is a bring-up path, not a serving "
            "configuration: expect a large throughput regression and numbers that are not "
            "comparable to any published row.",
            reason,
        )
        return None

    tp_group = get_tp_group()
    if int(tp_group.world_size) != 1:
        # tp>1 needs the packed all-gather inside the graph. Legal but unwritten.
        return eager("tp>1")
    batch_sizes = draft_graph_batch_sizes()
    if not batch_sizes:
        return eager("no draft graph batch sizes to size the static buffers from")
    max_bs = batch_sizes[-1]

    num_org, org_vocab_start = resolve_vocab_shard(lm_head)
    device, dtype = lm_head.weight.device, lm_head.weight.dtype

    # The cached attention bias and fused edge heads must exist before capture.
    # load_weights already built them; this is idempotent.
    head.materialize_inference_buffers(device, dtype)

    sampler = LiLiCorrDraftSampler(
        head=head,
        embed_tokens=embed_tokens,
        weight=lm_head.weight,
        block_size=int(block_size),
        num_org=num_org,
        org_vocab_start=org_vocab_start,
        max_bs=int(max_bs),
        anchor_features=int(draft_model.fc.out_features),
        sampling_enabled=sampling_enabled,
    )
    logger.info(
        "LiLiCorr select folded into the draft cuda graph: max_bs=%d block_size=%d K=%d "
        "anchor_features=%d, logits buffer %.1f MiB.",
        max_bs,
        int(block_size),
        sampler.topk,
        int(draft_model.fc.out_features),
        sampler.logits.numel() * sampler.logits.element_size() / 2**20,
    )
    # Prewarm with the dtype the draft forward will hand us: a prewarm at the wrong dtype
    # compiles a graph the capture then misses.
    try:
        hidden_dtype = next(draft_model.parameters()).dtype
    except StopIteration:
        hidden_dtype = dtype
    sampler.prewarm_compile(batch_sizes, hidden_dtype)
    return sampler
