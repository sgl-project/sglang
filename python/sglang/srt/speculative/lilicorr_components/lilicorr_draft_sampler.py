"""The LiLiCorr head as a graph-folded draft sampler.

DFLASH already runs its draft head inside the draft CUDA graph:
``init_cuda_graphs`` registers a sampler on ``draft_model_runner
.capture_tail_hooks``, so the head executes immediately after the draft forward
and writes its tokens into a static buffer the worker reads after the replay.
The head is a long tail of small kernels, so what it costs is host dispatch
rather than arithmetic, and riding the same hook removes the dispatch instead of
making the kernels cheaper.

What runs inside the graph: candidate vocab GEMM, top-k plus log-partition,
candidate embedding gather, the correlator ``select`` (lattice transformer plus
the greedy walk). All of it is static-shape, host-sync-free, and collective-free
at tp=1, which is what capture requires.

Two things the graph changes about the head, both handled here:

1. **The anchor has to live at a fixed address.** A captured graph reads whatever
   is at the address it captured, so the worker copies each step's anchor into
   ``self.anchor`` before the replay. The write ordering is unchanged from the
   eager path: the context append runs after verify at step t and the draft
   forward consumes it at step t+1.
2. **The batch is padded to the graph's bucket.** ``bs`` here is the bucket size,
   so rows past the live batch score stale anchors and produce garbage drafts.
   They are discarded, because the worker slices ``out[: bs_real * slots]``, and
   every row is scored independently -- the lattice attention does not cross
   requests -- so a padded row cannot affect a live one.

Compiling the body, not just batching its launches: the graph removes the host
cost of the head's kernel launches but changes neither their number nor their
memory traffic, and most of them are pointwise or reduction ops each making its
own HBM round trip. Inductor's pointwise fusion is aimed at exactly that gap, and
a fixed-shape, sync-free, collective-free body is its best case.

Compiling under capture is illegal -- tracing and autotuning both launch kernels
and synchronize -- so every bucket the draft graph will capture is warmed before
capture starts. An unwarmed bucket reaching capture raises rather than silently
serving an eager body under a "compiled" label.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.runtime_context import get_exec
from sglang.srt.speculative.lilicorr_components.lilicorr_candidates import (
    lilicorr_candidates,
    resolve_vocab_shard,
    target_input_embeddings,
)

logger = logging.getLogger(__name__)


class _RecompileLimitWatcher(logging.Handler):
    """Turn dynamo's recompile-limit warning into something observable.

    With ``dynamic=False`` there is one graph per capture bucket, so against a
    default recompile limit dynamo compiles the first few shapes, logs one
    warning, and then permanently runs the *original* function for every larger
    bucket -- including the largest, which is the bucket a high-concurrency run
    actually replays. Nothing raises and every call still succeeds, so the run
    reports a mixture of compiled and eager buckets, which is unusable in either
    direction. The limit is raised by construction below; this catches the
    warning in case any other code object hits it anyway.
    """

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
    """Without these, inductor changes the function.

    Two of its defaults do that to a bf16 body whose argmax sits over near-ties:
    fused intermediates stay in fp32 where eager rounds to bf16 between ops, and
    reductions get split, which is a different summation order. Both move the
    selected path.
    """
    import torch._inductor.config as inductor_config

    for attr, want in (("emulate_precision_casts", True), ("split_reductions", False)):
        if getattr(inductor_config, attr, None) is not None:
            setattr(inductor_config, attr, want)


class LiLiCorrDraftSampler:
    """LiLiCorr select, run inside the draft CUDA graph.

    Follows the DFLASH draft sampler contract: ``__call__(hidden_states,
    input_ids)`` writes the drafted ids for block positions 1.. into ``self.out``.
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
    ) -> None:
        self.head = head
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
        # Static logits buffer, so the in-graph GEMM does not put a large
        # allocation in each bucket's private CUDA-graph pool.
        self.logits = torch.empty((max_rows, self.num_org), dtype=dtype, device=device)
        # The anchor, written by the worker before each replay.
        self.anchor = torch.zeros(
            (self.max_bs, int(anchor_features)), dtype=dtype, device=device
        )
        self.anchor_valid = torch.zeros((self.max_bs,), dtype=torch.bool, device=device)
        self.token_table = head.build_token_table(embed_tokens)

        self._warmed_bs: set[int] = set()
        self._prewarming = False
        _pin_inductor_to_eager_numerics()
        self._select = torch.compile(
            head.select,
            # "default", not max-autotune: the head's GEMMs already go to cuBLAS
            # and the gap being closed is pointwise fusion, which default mode
            # does. Autotune would benchmark templates once per bucket at startup
            # for the part that is not the problem.
            mode="default",
            # The greedy commit is a Triton kernel, which is a graph break, so
            # fullgraph=True would refuse to compile at all.
            fullgraph=False,
            # One static shape per bucket, each warmed and captured separately.
            # Dynamic shapes produce a single symbolic graph whose guards
            # re-evaluate per bucket and whose kernels are sized for none of them;
            # measured, that costs head time at small batch, which is the regime
            # this method leads in.
            dynamic=False,
        )
        self._watcher = _RecompileLimitWatcher()
        logging.getLogger("torch._dynamo").addHandler(self._watcher)

    def set_anchor(self, rows: Optional[torch.Tensor], bs: int) -> None:
        """Publish this step's anchor into the buffer the graph reads.

        Rows the caller did not supply are zeroed and marked invalid rather than
        left stale, which matters because the graph runs at the padded bucket
        batch size and because a request can join between the context append and
        the next draft forward. The head multiplies the projected anchor by its
        validity flag, so an invalid row scores exactly as "no anchor".

        Row i at step t+1 is the same request as row i at step t only while the
        batch composition is unchanged: if a request in the middle finishes, the
        survivors shift down and a row can be handed a neighbour's anchor. That
        costs acceptance on the affected step, never correctness -- verify still
        checks every drafted token against the target.
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

        # One graph per bucket against a default limit of 8, so dynamo would hit
        # the limit and then run the original function for every remaining shape,
        # silently, since those calls still succeed. Guards also key on strides
        # and alignment and on the other callers of the same code object, so the
        # true variant count is not something to predict: size the limit so it
        # cannot be the binding constraint and let the watcher report if it
        # somehow still is. A cache entry is cheap; a silently-eager bucket inside
        # a graph labelled "compiled" is not.
        needed = max(256, 16 * len(buckets))
        for attr, want in (
            ("recompile_limit", needed),
            ("cache_size_limit", needed),
            ("accumulated_recompile_limit", 8 * needed),
            ("accumulated_cache_size_limit", 8 * needed),
        ):
            current = getattr(torch._dynamo.config, attr, None)
            if current is not None and int(current) < want:
                setattr(torch._dynamo.config, attr, want)

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

        # max_bs sizes every static buffer here. If the graph ever replays at a
        # larger bucket, the slices below silently return short tensors and
        # corrupt the drafts for the largest batches. This runs at capture time,
        # once per bucket, so it costs nothing in steady state.
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
            # Catches a limit blown by an earlier bucket's capture, which is
            # where it actually happens.
            self._raise_if_recompile_limit_hit("cuda-graph capture or decode")

        hidden_size = hidden_states.shape[-1]
        pass_hidden = hidden_states.view(bs, self.block_size, hidden_size)[:, 1:, :]
        log_probs, ids = lilicorr_candidates(
            hidden_states=pass_hidden.reshape(rows, hidden_size),
            weight=self.weight,
            num_org=self.num_org,
            org_vocab_start=self.org_vocab_start,
            topk=self.topk,
            logits_out=self.logits[:rows],
        )
        ids = ids.view(bs, self.slots, self.topk)

        pre_projected = self.token_table is not None
        token_embeddings = (
            self.token_table[ids] if pre_projected else self.embed_tokens(ids).detach()
        )
        selected = self._select(
            token_embeddings=token_embeddings,
            candidate_token_ids=ids,
            candidate_log_probs=log_probs.view(bs, self.slots, self.topk),
            pass_hidden=pass_hidden,
            anchor_hidden=self.anchor[:bs],
            anchor_valid=self.anchor_valid[:bs],
            already_projected=pre_projected,
        )
        self.out[:rows].copy_(selected.reshape(-1).to(torch.int64))


def draft_graph_batch_sizes() -> list[int]:
    """Every batch size the draft decode graph is captured for, ascending.

    The list and not just its max, because a graph-folded head is captured once
    per bucket and the compile prewarm has to cover every one of them before
    capture starts.
    """
    return sorted(
        {int(bs) for bs in get_exec().graph.cuda_graph_config.decode.bs if bs > 0}
    )


def build_lilicorr_draft_sampler(*, worker, lm_head) -> Optional[LiLiCorrDraftSampler]:
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

    head = worker.lilicorr
    draft_model = worker.draft_model
    num_org, org_vocab_start = resolve_vocab_shard(lm_head)
    device, dtype = lm_head.weight.device, lm_head.weight.dtype

    # The cached attention bias and fused edge heads must exist before capture.
    # load_weights already built them; this is idempotent and covers a head that
    # reached the worker by another route.
    head.materialize_inference_buffers(device, dtype)

    sampler = LiLiCorrDraftSampler(
        head=head,
        embed_tokens=target_input_embeddings(worker),
        weight=lm_head.weight,
        block_size=int(worker.block_size),
        num_org=num_org,
        org_vocab_start=org_vocab_start,
        max_bs=int(max_bs),
        anchor_features=int(draft_model.fc.out_features),
    )
    logger.info(
        "LiLiCorr select folded into the draft cuda graph: max_bs=%d block_size=%d "
        "K=%d anchor_features=%d, logits buffer %.1f MiB.",
        max_bs,
        int(worker.block_size),
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
