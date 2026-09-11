"""The LiLiCorr head as a graph-folded draft sampler.

DFLASH already runs its draft head inside the draft CUDA graph: ``init_cuda_graphs``
registers a sampler on ``draft_model_runner.capture_tail_hooks``, so the head
executes immediately after the draft forward and writes its tokens into a static
buffer the worker reads after the replay. The head is a long tail of small
kernels, so what it costs is host dispatch rather than arithmetic, and riding the
same hook removes the dispatch instead of making the kernels cheaper. What runs
inside the graph is the candidate vocab GEMM, the top-k plus log-partition, the
candidate embedding gather and the correlator ``select``; all of it is
static-shape, host-sync-free and collective-free at tp=1.

Two things the graph changes about the head:

1. The anchor has to live at a fixed address, so the worker copies each step's
   anchor into ``self.anchor`` before the replay. The write ordering is unchanged
   from the eager path.
2. The batch is padded to the graph's bucket, so rows past the live batch score
   stale anchors and produce garbage drafts. They are discarded -- the worker
   slices ``out[: bs_real * slots]`` -- and the lattice attention does not cross
   requests, so a padded row cannot affect a live one.

The body is also compiled, not merely batched: the graph removes the host cost of
the launches but changes neither their number nor their memory traffic, and most
of them are pointwise ops each making its own HBM round trip. Compiling under
capture is illegal, so every bucket the draft graph will capture is warmed first;
an unwarmed bucket reaching capture raises rather than silently serving an eager
body under a "compiled" label.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.runtime_context import get_exec

# Shared with the worker's own staging rather than reimplemented: "greedy" must mean
# the same predicate on both sides or a row could be sampled here and accepted as
# deterministic there. The worker already imports this module, so it costs nothing.
from sglang.srt.speculative.dspark_components.dspark_draft import resolve_greedy_mask
from sglang.srt.speculative.lilicorr_components.lilicorr_candidates import (
    lilicorr_candidates,
    resolve_vocab_shard,
)
from sglang.srt.speculative.lilicorr_components.lilicorr_config import SAMPLING_ENABLED

logger = logging.getLogger(__name__)


class _RecompileLimitWatcher(logging.Handler):
    # Past a recompile limit, dynamo permanently runs the original function for
    # every further shape and only logs a warning, so the run would report a
    # mixture of compiled and eager buckets. The limit is raised by construction
    # below; this catches the warning if any code object hits one anyway.

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
    # Two inductor defaults change the function for a bf16 body whose argmax sits
    # over near-ties: fused intermediates stay in fp32 where eager rounds to bf16
    # between ops, and split reductions are a different summation order. Both move
    # the selected path.
    import torch._inductor.config as inductor_config

    inductor_config.emulate_precision_casts = True
    inductor_config.split_reductions = False


class LiLiCorrDraftSampler:
    """LiLiCorr select, run inside the draft CUDA graph.

    Follows the DFLASH draft sampler contract: ``__call__(hidden_states,
    input_ids)`` writes the drafted tokens for block positions 1.. into
    ``self.out``.
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
        # Device-gated by the worker, exactly as the selector's sampler is: the
        # sampled accept path needs `chain_speculative_sampling_triton`, which does
        # not run on NPU. Off here means the argmax commit and no published
        # proposal, which is what the existing verify fallback is correct for.
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
        # Static, so the in-graph GEMM does not put a large allocation in each
        # bucket's private CUDA-graph pool.
        self.logits = torch.empty((max_rows, self.num_org), dtype=dtype, device=device)
        # Written by the worker before each replay.
        self.anchor = torch.zeros(
            (self.max_bs, int(anchor_features)), dtype=dtype, device=device
        )
        self.anchor_valid = torch.zeros((self.max_bs,), dtype=torch.bool, device=device)
        self.token_table = head.build_token_table(embed_tokens)

        # Sampling state. Written by the worker before each replay (temperatures,
        # greedy_mask), drawn inside it (uniforms), read after it (q_out,
        # candidate_out). Sized by max_bs like every other buffer here, and allocated
        # unconditionally at ~90 KiB total so the greedy and sampled paths differ only
        # in whether they are consumed, not in the object graph.
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
        # The verify needs the ids q is indexed against, and the candidate tensor the
        # body produces is an intermediate inside the compiled region, so it is copied
        # to a fixed address rather than referenced.
        self.candidate_out = torch.empty(
            (self.max_bs, self.slots, self.topk), dtype=torch.int64, device=device
        )

        self._warmed_bs: set[int] = set()
        self._prewarming = False
        _pin_inductor_to_eager_numerics()
        self._select = torch.compile(
            # Exactly one of the two is compiled per process: the mode is a module
            # constant, so there is no bucket that could end up with the other body.
            head.select_with_proposal if sampling_enabled else head.select,
            # "default", not max-autotune: the head's GEMMs already go to cuBLAS
            # and the gap being closed is pointwise fusion, which default mode
            # does.
            mode="default",
            # The greedy commit is a Triton kernel, hence a graph break, so
            # fullgraph=True would refuse to compile at all.
            fullgraph=False,
            # One static shape per bucket, each warmed and captured separately. A
            # single symbolic graph measured slower at small batch, which is the
            # regime this method leads in.
            dynamic=False,
        )
        self._watcher = _RecompileLimitWatcher()
        logging.getLogger("torch._dynamo").addHandler(self._watcher)

    def set_anchor(self, rows: Optional[torch.Tensor], bs: int) -> None:
        """Publish this step's anchor into the buffer the graph reads.

        Rows the caller did not supply are zeroed and marked invalid rather than
        left stale, because the graph runs at the padded bucket batch size. The
        head multiplies the projected anchor by its validity flag, so an invalid
        row scores exactly as "no anchor".

        Row i at step t+1 is the same request as row i at step t only while the
        batch composition is unchanged; filter/merge can change it at constant
        batch size, so a row can be handed a neighbour's anchor. The anchor feeds
        the drafter and not verify, so that costs acceptance, never correctness,
        and it is bounded: reordering cannot happen at concurrency 1 and happens
        constantly at 32, where gsm8k reads 7.5573 and 7.5669 respectively.

        Carrying it on DFlashDraftInputV2 instead, so filter/merge reorder it and
        the row always matches the request, measured -4.67% at concurrency 32
        (7.2131 against 7.5669). Publishing later is not the cost; the carrier is.
        Measure before trying it again.
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
        """Host-side refresh of the static sampling params; must run before the draft
        graph replay that consumes them.

        Rows past ``bs`` are left alone deliberately: the graph replays at the padded
        bucket size and those rows already score a zeroed anchor, so their drafts are
        discarded by the worker's ``out[: bs * slots]`` slice whatever they sample.
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
            f"dynamo hit a recompile limit during {where}, so some capture "
            "buckets run eager inside a graph labelled 'compiled' and any "
            f"throughput delta is a mixture rather than a fusion result: {hits[:3]}."
        )

    def prewarm_compile(self, bs_list, hidden_dtype: torch.dtype) -> None:
        """Trace and compile the body at every bucket, before any capture.

        Runs the real ``__call__`` on synthetic hidden states once per bucket, so
        inductor does its tracing and codegen outside the capture region. The
        inputs are junk; the point is the shape, and ``self.out`` is overwritten
        by the first real replay.
        """
        buckets = sorted({int(b) for b in bs_list if 0 < int(b) <= self.max_bs})
        if not buckets:
            raise RuntimeError(
                "LiLiCorr compile prewarm has no buckets to warm, so capture "
                f"would raise. Resolved max_bs={self.max_bs} against bs_list="
                f"{sorted(bs_list)}."
            )

        # Guards key on strides and alignment as well as shape, so the true
        # variant count is not predictable: size the limit so it cannot be the
        # binding constraint and let the watcher report if it somehow still is. A
        # cache entry is cheap; a silently-eager bucket is not.
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
            "LiLiCorr in-graph select compiled and prewarmed for %d buckets %s "
            "(dtype=%s).",
            len(buckets),
            buckets,
            hidden_dtype,
        )

    def __call__(self, hidden_states: torch.Tensor, input_ids=None) -> None:
        del input_ids  # the lattice is scored from hidden states and the anchor
        bs = hidden_states.shape[0] // self.block_size
        rows = bs * self.slots

        # max_bs sizes every static buffer here, so a larger replay would silently
        # return short slices and corrupt the drafts for the largest batches.
        if bs > self.max_bs:
            raise RuntimeError(
                f"LiLiCorrDraftSampler was built for max_bs={self.max_bs} but the "
                f"draft graph replayed at bs={bs}; its static buffers are "
                "undersized."
            )
        if not self._prewarming:
            if bs not in self._warmed_bs and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"the draft graph is capturing bs={bs}, which the compile "
                    f"prewarm never covered (warmed: {sorted(self._warmed_bs)}). "
                    "Compiling under capture is illegal, so fix the bucket list "
                    "rather than widening the prewarm."
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
                # Drawn here rather than inside `_select` because an RNG op in the
                # compiled body is a graph break; passing the tensor keeps it pure.
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
    """Every batch size the draft decode graph is captured for, ascending.

    The list and not just its max, because a graph-folded head is captured once
    per bucket and the compile prewarm has to cover every one of them.
    """
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

    Returning None is not a neutral choice: the eager head costs a large fraction
    of throughput and drops the anchor for the whole batch whenever the batch size
    changes, so each refusal is logged with its reason.
    """

    def eager(reason: str) -> None:
        logger.warning(
            "LiLiCorr head kept eager (reason=%s). This is a bring-up path, not a "
            "serving configuration: expect a large throughput regression and "
            "numbers that are not comparable to any published row.",
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
    # load_weights already built them; this is idempotent and covers a head that
    # reached the worker by another route.
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
        "LiLiCorr select folded into the draft cuda graph: max_bs=%d block_size=%d "
        "K=%d anchor_features=%d, logits buffer %.1f MiB.",
        max_bs,
        int(block_size),
        sampler.topk,
        int(draft_model.fc.out_features),
        sampler.logits.numel() * sampler.logits.element_size() / 2**20,
    )
    # Prewarm with the dtype the draft forward will actually hand us: a prewarm at
    # the wrong dtype compiles a graph the capture then misses.
    try:
        hidden_dtype = next(draft_model.parameters()).dtype
    except StopIteration:
        hidden_dtype = dtype
    sampler.prewarm_compile(batch_sizes, hidden_dtype)
    return sampler
