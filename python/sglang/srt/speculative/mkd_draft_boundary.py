"""Single-token greedy MTP draft step boundary in two launches instead of seven.

Today: shard argmax pack, all-gather, select, next step's fc slice GEMV, all-gather of the
fp32 partials, rank-order sum. Here: one launch does the shard argmax and a multicast push of
each rank's candidate row together with its hidden-half fc partial (computed beside the
lm_head), selects the winner on every rank, and a second launch finishes the fc from the
exchanged data. Every value equals the split-fc path bit for bit: same argmax keys and ties,
same per-slice dot arithmetic, same rank-order fp32 sum and bf16 rounding.

The memo variant ships, per candidate, the embedding-half partials already summed in rank
order (built at load time from the checkpoint weights with the same kernel arithmetic and
checked against the split-fc kernels on every hot row), so the embedding-half GEMV leaves
the critical path.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.distributed as dist

from sglang.kernels.ops.speculative.draft_fc_partial import (
    draft_fc_partial,
    draft_fc_partial_rows,
)
from sglang.kernels.ops.speculative.draft_vocab_argmax import (
    NAN_KEY_SPLIT_ARGMAX,
    NAN_KEY_TORCH_ARGMAX,
)
from sglang.kernels.ops.speculative.mkd_draft_fc import (
    mkd_fc_memo,
    mkd_fc_sum,
    mkd_hidden_partial,
)
from sglang.kernels.ops.speculative.mkd_draft_head import mkd_draft_head
from sglang.kernels.ops.speculative.mkd_draft_xchg import (
    mkd_xchg,
    mkd_xchg_words,
    mkd_xchg_zero,
)
from sglang.kernels.ops.speculative.mkd_prefetch import mkd_l2_prefetch
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# Memo rows built per launch at load time.
_MEMO_BLOCK = 4096
# Exchange calls kept in the diagnostic trace ring.
_TRACE_SLOTS = 64


# Sources of the draft-step JIT kernels; load_jit keys its build cache on their contents.
_JIT_SOURCES = (
    "speculative/mkd_draft_xchg.cuh",
    "speculative/mkd_draft_fc.cuh",
    "speculative/mkd_prefetch.cuh",
    "speculative/draft_moe_fused.cuh",
    "speculative/mkd_draft_head.cuh",
)
# CTAs of mkd_draft_head per SM; one per SM measured fastest (tunable).
_HEAD_CTAS_PER_SM = 1
# Draft calls between checks that every rank takes the same extend-fc fill decision (arbitrary).
_FILL_AGREE_EVERY = 512


def _jit_source_digests() -> str:
    import hashlib
    import os

    from sglang.kernels.jit.utils.compile.spec import resolve_sources

    return " ".join(
        f"{os.path.basename(path)}={hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]}"
        for path in resolve_sources(list(_JIT_SOURCES))
    )


def _clear_negative_zero(rows: torch.Tensor) -> torch.Tensor:
    # The exchanged embedding row has -0.0 cleared (vocab_argmax_pack); memo from the same row.
    bits = rows.view(torch.int16)
    return torch.where(bits == -32768, torch.zeros_like(bits), bits).view(
        torch.bfloat16
    )


def _unsupported_reason(model, exchange) -> Optional[str]:
    if model.fc_split_slice is None or model._fc_gemv_rows is None:
        return "needs SGLANG_Q35MF_DRAFT_FC_SPLIT and SGLANG_Q35MF_DRAFT_FC_GEMV"
    world = get_parallel().tp_size
    if (
        exchange.token_map.numel() == 0
        and exchange.local_vocab * world != model.config.vocab_size
    ):
        return "full-vocabulary shards with padding rows"
    if world not in (4, 8):
        return f"tp_size {world}"
    if model.config.hidden_size != 4096:
        return f"hidden size {model.config.hidden_size}"
    return None


class MkdDraftBoundary:
    """Exchange buffers, the optional memo table and the per-step launches."""

    def __init__(self, model, exchange, hot_ids: torch.Tensor) -> None:
        from flashinfer.comm.mnnvl import McastGPUBuffer

        from sglang.srt.distributed.parallel_state import get_tp_group
        from sglang.srt.layers import flashinfer_comm_fusion

        self.model = model
        parallel = get_parallel()
        self.world, self.rank = parallel.tp_size, parallel.tp_rank
        hidden = model.config.hidden_size
        self.hidden = hidden
        use_hidden, start, width = model.fc_split_slice
        self.is_hidden_rank = use_hidden
        self.slice_start, self.width = start, width
        self.n_embed = hidden // width
        dev = model.fc.weight.device
        self.memo = envs.SGLANG_Q35MF_MKD_TTABLE.get()
        rows = exchange.local_vocab
        self.hot_ids = hot_ids.to(dev)
        if exchange.token_map.numel() > 0:
            if not torch.equal(
                self.hot_ids[self.rank * rows : (self.rank + 1) * rows],
                exchange.token_map,
            ):
                raise RuntimeError(
                    "Draft step boundary: hot shard is not the rank-th slice of the hot list"
                )
        elif exchange.vocab_start != self.rank * rows:
            raise RuntimeError(
                "Draft step boundary: vocabulary shard is not the rank-th equal slice"
            )
        self.cand = self._build_memo(exchange) if self.memo else exchange.embed_weight

        tp = get_tp_group()
        backend = flashinfer_comm_fusion._TorchDistBackend(
            device_group=tp.device_group, cpu_group=tp.cpu_group
        )
        words = mkd_xchg_words(hidden, self.world, self.memo)
        self.buf = McastGPUBuffer(words * 8, self.world, self.rank, dev, backend)
        self.mc = self.buf.get_multicast_ptr()
        self.uc = self.buf.get_unicast_ptr(self.rank)
        mkd_xchg_zero(hidden, self.world, self.memo, self.uc)
        torch.cuda.synchronize(dev)
        dist.barrier(group=tp.cpu_group)

        self.epoch_done = torch.tensor([1, 0], dtype=torch.int32, device=dev)
        self.own_partial = torch.zeros(1, hidden, dtype=torch.float32, device=dev)
        # [CTA counter, epoch flag] of the side-stream hidden partial (mkd_hidden_partial).
        self.partial_sync = torch.zeros(2, dtype=torch.int32, device=dev)
        # Per-CTA shard-argmax partials for large (full-vocabulary) shards.
        self.scan_part = torch.zeros(3 * 16, dtype=torch.int64, device=dev)
        self.out_partials = torch.zeros(
            self.world - self.n_embed, hidden, dtype=torch.float32, device=dev
        )
        self.out_cand = torch.empty(
            1, hidden, dtype=torch.float32 if self.memo else torch.bfloat16, device=dev
        )
        self.fc_out = torch.empty(1, hidden, dtype=torch.bfloat16, device=dev)
        # Memo mode: the exchange writes the next step's fc output here directly.
        # Zeros: graph-capture warmups read it before any exchange has written it.
        self.fc_buf = (
            torch.zeros(1, hidden, dtype=torch.bfloat16, device=dev)
            if self.memo
            else None
        )
        # The draft extend's exchange only needs the token; its payload lands here unused.
        self.ext_cand = torch.empty_like(self.out_cand)
        self.ext_partials = torch.empty_like(self.out_partials)
        self.side_stream = torch.cuda.Stream(device=dev)
        self._install_l2_prefetch(envs.SGLANG_Q35MF_MKD_L2PF_MB.get())
        self.trace = None
        if envs.SGLANG_Q35MF_MKD_TRACE.get():
            self._start_trace_reader()
        self._install_head(envs.SGLANG_Q35MF_MKD_HEAD.get(), dev)
        self.qkv_pf, self.qkv_pf_bytes = self._qkv_prefetch_target(
            envs.SGLANG_Q35MF_MKD_L2PF_QKV_MB.get()
        )
        self.qkv_stream = (
            torch.cuda.Stream(device=dev) if self.qkv_pf is not None else None
        )
        self.qkv_pending = False
        # extend_fc: the draft extend's exchange writes fc_buf for the next draft's first step.
        # input_seq counts calls that produce a draft input (draft extends after prefill or
        # decode); fc_buf_key is the input_seq whose extend wrote fc_buf. Both follow the same
        # SPMD control flow on every rank, so every rank makes the same fill decision.
        self.extend_fc = self.memo and envs.SGLANG_Q35MF_MKD_EXTEND_FC.get()
        self.input_seq = 0
        self.fc_buf_key: Optional[int] = None
        self._fill_checks = 0
        # armed: this draft step ends in the exchange; pending: the next step's fc reads it;
        # last_step: no draft step follows, so the exchange carries the token alone.
        self.armed = False
        self.pending = False
        self.step = 0
        self.last_step = False
        self.partial_pending = False
        self.pf_pending = False
        logger.info(
            "MKD draft boundary active (tp=%d, %s candidates, %.1f MB exchange buffer); JIT sources %s.",
            self.world,
            "memo" if self.memo else "embedding",
            words * 8 / 2**20,
            _jit_source_digests(),
        )

    @classmethod
    def maybe_create(cls, model, exchange, hot) -> Optional[MkdDraftBoundary]:
        reason = _unsupported_reason(model, exchange)
        if reason is not None:
            logger.warning(
                "MKD draft boundary disabled (%s); using the stock path.", reason
            )
            return None
        if hot is not None:
            return cls(model, exchange, hot.token_ids)
        # Full vocabulary: candidate position p (rank-major) is vocabulary id p.
        ids = torch.arange(
            exchange.local_vocab * get_parallel().tp_size, dtype=torch.int64
        )
        return cls(model, exchange, ids)

    # ---------------------------------------------------------------- memo table

    def _build_memo(self, exchange) -> torch.Tensor:
        model = self.model
        emb = exchange.embed_weight
        rows = emb.shape[0]
        memo = torch.empty(rows, self.hidden, dtype=torch.float32, device=emb.device)
        norm = model.pre_fc_norm_embedding
        for i in range(0, rows, _MEMO_BLOCK):
            block = _clear_negative_zero(emb[i : i + _MEMO_BLOCK].clone())
            mkd_fc_memo(
                block,
                model.fc.weight,
                norm.weight.data,
                norm.variance_epsilon,
                self.world,
                memo[i : i + _MEMO_BLOCK],
            )
        bad = self._check_memo(emb, memo)
        bad_t = torch.tensor([bad, rows], dtype=torch.int64, device=emb.device)
        dist.all_reduce(bad_t, group=get_parallel().tp_group.device_group)
        if int(bad_t[0]) != 0:
            raise RuntimeError(
                f"Draft fc memo table differs from the split fc on {int(bad_t[0])} of {int(bad_t[1])} rows"
            )
        logger.info(
            "MKD memo table: %d rows on this rank (%.1f MB), equal to the split fc on all %d rows.",
            rows,
            memo.numel() * 4 / 2**20,
            int(bad_t[1]),
        )
        return memo

    def _check_memo(self, emb: torch.Tensor, memo: torch.Tensor) -> int:
        """Rows whose memo differs from the split-fc kernels (every row, all slices)."""
        model = self.model
        norm = model.pre_fc_norm_embedding
        sms = torch.cuda.get_device_properties(emb.device).multi_processor_count
        width = self.width
        gemv_rows = draft_fc_partial_rows(self.hidden, width, sms)
        shards = [
            model.fc.weight[:, s * width : (s + 1) * width].contiguous()
            for s in range(self.n_embed)
        ]
        part = torch.empty(1, self.hidden, dtype=torch.float32, device=emb.device)
        bad = torch.zeros((), dtype=torch.int64, device=emb.device)
        for v in range(emb.shape[0]):
            x = _clear_negative_zero(emb[v : v + 1].clone())
            acc = None
            for s in range(self.n_embed):
                draft_fc_partial(
                    x,
                    norm.weight.data,
                    norm.variance_epsilon,
                    shards[s],
                    s * width,
                    part,
                    gemv_rows,
                )
                acc = part.clone() if acc is None else acc + part
            bad += (acc.view(torch.int32) != memo[v : v + 1].view(torch.int32)).any()
        return int(bad)

    # ---------------------------------------------------------------- L2 prefetch

    def _install_l2_prefetch(self, mb: int) -> None:
        self.pf_bytes = mb * 2**20
        self.pf_stream = None
        if self.pf_bytes <= 0:
            return
        from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

        moe = None
        for module in self.model.modules():
            if (
                isinstance(module, Qwen2MoeSparseMoeBlock)
                and module._single_token_draft_moe is not None
            ):
                moe = module._single_token_draft_moe
        if moe is None:
            logger.warning("MKD L2 prefetch disabled (no single-token draft MoE).")
            self.pf_bytes = 0
            return
        self.pf_stream = torch.cuda.Stream(device=self.model.fc.weight.device)
        moe.after_hook = self._prefetch_lm_head
        logger.info(
            "MKD L2 prefetch: %d MB of the draft lm_head after the draft MoE.", mb
        )

    def _qkv_prefetch_target(self, mb: int):
        if mb <= 0:
            return None, 0
        from sglang.srt.layers.linear import QKVParallelLinear

        for module in self.model.modules():
            if isinstance(module, QKVParallelLinear):
                logger.info(
                    "MKD L2 prefetch: %d MB of the draft qkv weight beside the exchange.",
                    mb,
                )
                return module.weight, mb * 2**20
        logger.warning("MKD qkv L2 prefetch disabled (no qkv projection found).")
        return None, 0

    def _prefetch_lm_head(self) -> None:
        if not self.armed:
            return
        self.pf_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.pf_stream):
            mkd_l2_prefetch(self.model.lm_head.weight, self.pf_bytes)
        self.pf_pending = True

    # ---------------------------------------------------------------- draft lm_head

    def _install_head(self, enabled: bool, dev: torch.device) -> None:
        self.head_part = None
        self.head_used = False
        self.head_stats = None
        if not enabled:
            return
        head = self.model.lm_head.weight
        if (
            head.dtype != torch.bfloat16
            or head.dim() != 2
            or head.shape[1] != self.hidden
            or head.shape[0] > 65536
        ):
            logger.warning(
                "MKD draft head disabled (lm_head %s %s).",
                head.dtype,
                tuple(head.shape),
            )
            return
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.head_part = torch.zeros(
            _HEAD_CTAS_PER_SM * sms, dtype=torch.int32, device=dev
        )
        self.model.logits_processor.mkd_head = self
        if envs.SGLANG_Q35MF_MKD_HEAD_CHECK.get():
            self._start_head_check(dev)
        logger.info(
            "MKD draft head active: %d x %d shard, %d CTAs.",
            head.shape[0],
            head.shape[1],
            self.head_part.numel(),
        )

    def head(self, hidden_states: torch.Tensor, lm_head) -> torch.Tensor:
        """The one-row draft logits [1, N] bf16 (the exchange then reads the per-CTA maxima)."""
        x = hidden_states.to(torch.bfloat16).contiguous()
        logits = torch.empty(
            1, lm_head.weight.shape[0], dtype=torch.bfloat16, device=x.device
        )
        # Alternate the row walk per draft step so the next step first re-reads the rows read last;
        # measured faster in A/B.
        mkd_draft_head(
            x, lm_head.weight, logits, self.head_part, reverse=self.step % 2 == 1
        )
        self.head_used = True
        if self.head_stats is not None:
            self._check_head(x, lm_head.weight, logits)
        return logits

    def _start_head_check(self, dev: torch.device) -> None:
        """Diagnostic: steps, argmax differences vs the served GEMM, and the largest served-logit
        gap between the two picks, copied to pinned memory each step and logged by a thread."""
        import threading
        import time

        self.head_stats = torch.zeros(3, dtype=torch.float32, device=dev)
        self.head_stats_host = torch.zeros(3, dtype=torch.float32, pin_memory=True)
        host = self.head_stats_host.numpy()

        def run() -> None:
            while True:
                time.sleep(20)
                logger.info(
                    "MKD draft head check rank %d: %d steps, %d argmax differences, max served-logit gap %.5f.",
                    self.rank,
                    int(host[0]),
                    int(host[1]),
                    float(host[2]),
                )

        threading.Thread(target=run, daemon=True, name="draft-head-check").start()

    def _check_head(
        self, x: torch.Tensor, weight: torch.Tensor, logits: torch.Tensor
    ) -> None:
        ref = torch.matmul(x, weight.t()).float()
        mine = logits.float()
        pick_ref = ref.argmax(dim=-1)
        pick_mine = mine.argmax(dim=-1)
        diff = (pick_ref != pick_mine).float()
        gap = (
            ref.gather(-1, pick_ref.view(1, 1)) - ref.gather(-1, pick_mine.view(1, 1))
        ).view(-1) * diff
        st = self.head_stats
        st[0:1] += 1.0
        st[1:2] += diff
        torch.maximum(st[2:3], gap, out=st[2:3])
        self.head_stats_host.copy_(st, non_blocking=True)

    # ---------------------------------------------------------------- per step

    def hidden_partial(self, hidden_states: torch.Tensor) -> None:
        """This rank's hidden-slice fc partial for the next draft step, on a side stream;
        the exchange waits on its flag, not on a stream join."""
        if not self.is_hidden_rank or self.last_step:
            return
        model = self.model
        norm = model.pre_fc_norm_hidden
        self.side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.side_stream):
            mkd_hidden_partial(
                hidden_states,
                model._fc_shard,
                norm.weight.data,
                norm.variance_epsilon,
                self.slice_start,
                self.own_partial,
                self.epoch_done,
                self.partial_sync,
                self.world,
            )
        self.partial_pending = True

    def _join_side_streams(self) -> None:
        # Joins only end graph capture cleanly: the exchange syncs on flags, and the L2
        # prefetch is a hint; a join before the exchange would cost its early launch.
        if self.partial_pending:
            torch.cuda.current_stream().wait_stream(self.side_stream)
            self.partial_pending = False
        if self.pf_stream is not None and self.pf_pending:
            torch.cuda.current_stream().wait_stream(self.pf_stream)
            self.pf_pending = False
        if self.qkv_pending:
            torch.cuda.current_stream().wait_stream(self.qkv_stream)
            self.qkv_pending = False

    def _prefetch_qkv(self) -> None:
        # HBM idles during the latency-bound exchange; the next step's qkv GEMV reads from L2.
        self.qkv_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.qkv_stream):
            mkd_l2_prefetch(self.qkv_pf, self.qkv_pf_bytes)
        self.qkv_pending = True

    def exchange(
        self,
        local_logits: torch.Tensor,
        topk_index: torch.Tensor,
        topk_p: torch.Tensor,
        draft_tokens: Optional[torch.Tensor],
        positions: torch.Tensor,
        draft_column: int,
        mrope_positions: Optional[torch.Tensor],
    ) -> None:
        partial_flag = (
            self.partial_sync[1:]
            if self.partial_pending and not self.last_step
            else None
        )
        if self.qkv_pf is not None and not self.last_step:
            self._prefetch_qkv()
        mkd_xchg(
            local_logits,
            self.cand,
            self.hot_ids,
            self.own_partial,
            self.epoch_done,
            self.out_cand,
            self.out_partials,
            topk_index,
            topk_p,
            draft_tokens,
            positions,
            mrope_positions,
            self.mc,
            self.uc,
            self.rank,
            self.world,
            draft_column,
            NAN_KEY_SPLIT_ARGMAX,
            trace=self.trace,
            fc_out=self.fc_buf,
            token_only=self.last_step,
            partial_flag=partial_flag,
            scan_part=self.scan_part,
            head_part=self.head_part if self.head_used else None,
        )
        self.head_used = False
        self.pending = not self.last_step
        if self.last_step:
            self._join_side_streams()

    def _start_trace_reader(self) -> None:
        """Diagnostic: the kernel stamps a pinned host ring; a thread logs phase times, no CUDA calls."""
        import threading
        import time

        import numpy as np

        self.trace = torch.zeros(
            _TRACE_SLOTS, 16, 9, dtype=torch.int64, pin_memory=True
        )
        ring = self.trace.numpy()
        names = ("scan", "sent", "recv", "end")

        def run() -> None:
            while True:
                time.sleep(20)
                t = ring.astype(np.float64)
                ok = (t[:, :, 0] > 0).all(axis=1) & (t[:, :, 8] >= t[:, :, 1]).all(
                    axis=1
                )
                t = t[ok]
                if len(t) == 0:
                    continue
                w0 = t[:, :, 1].min(axis=1)
                pre = (w0 - t[:, :, 0].min(axis=1)) / 1e3
                # Latest CTA at each phase, from the earliest CTA's PDL wait release.
                rel = (t[:, :, [5, 6, 7, 8]].max(axis=1) - w0[:, None]) / 1e3
                q = np.percentile(
                    np.concatenate([pre[:, None], rel], axis=1), [10, 50, 90], axis=0
                )
                logger.info(
                    "MKD exchange trace rank %d, %d calls, us p10/p50/p90: pre_wait %s; %s",
                    self.rank,
                    len(t),
                    "/".join(f"{x:.2f}" for x in q[:, 0]),
                    "; ".join(
                        f"{n} {'/'.join(f'{x:.2f}' for x in q[:, i + 1])}"
                        for i, n in enumerate(names)
                    ),
                )

        threading.Thread(target=run, daemon=True, name="draft-xchg-trace").start()

    def exchange_extend(
        self, local_logits: torch.Tensor, hidden_states: Optional[torch.Tensor]
    ):
        """Draft-extend token (torch.argmax NaN rule) through the same one-launch exchange;
        with extend_fc it also writes the next draft's first fc into fc_buf."""
        topk_index = torch.empty((1, 1), dtype=torch.int64, device=local_logits.device)
        topk_p = torch.empty((1, 1), dtype=torch.float32, device=local_logits.device)
        payload = (
            self.extend_fc and hidden_states is not None and hidden_states.shape[0] == 1
        )
        if payload and self.is_hidden_rank:
            model = self.model
            norm = model.pre_fc_norm_hidden
            draft_fc_partial(
                hidden_states.contiguous(),
                norm.weight.data,
                norm.variance_epsilon,
                model._fc_shard,
                self.slice_start,
                self.own_partial,
                model._fc_gemv_rows,
            )
        mkd_xchg(
            local_logits,
            self.cand,
            self.hot_ids,
            self.own_partial,
            self.epoch_done,
            self.ext_cand,
            self.ext_partials,
            topk_index,
            topk_p,
            None,
            None,
            None,
            self.mc,
            self.uc,
            self.rank,
            self.world,
            0,
            NAN_KEY_TORCH_ARGMAX,
            trace=self.trace,
            fc_out=self.fc_buf if payload else None,
            token_only=not payload,
            scan_part=self.scan_part,
        )
        self.fc_buf_key = self.input_seq if payload else None
        return topk_p, topk_index

    def note_draft_input(self) -> None:
        """A draft extend (after prefill or decode) is about to produce the next draft input."""
        self.input_seq += 1

    def take_fc_buf(self) -> bool:
        """True when fc_buf holds the first fc of the coming draft: its extend produced the
        current draft input. Consumes it (the draft's own exchanges overwrite fc_buf)."""
        holds = self.fc_buf_key is not None and self.fc_buf_key == self.input_seq
        self.fc_buf_key = None
        self._fill_checks += 1
        if self._fill_checks % _FILL_AGREE_EVERY == 1:
            self._check_ranks_agree(holds)
        return holds

    def _check_ranks_agree(self, holds: bool) -> None:
        # The decision feeds eager collectives (fill_fc); confirm the SPMD premise now and then.
        t = torch.tensor([self.input_seq, int(holds)], dtype=torch.int64)
        lo, hi = t.clone(), t.clone()
        group = get_parallel().tp_group.cpu_group
        dist.all_reduce(lo, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX, group=group)
        if not torch.equal(lo, hi):
            raise RuntimeError(
                f"Draft extend fc: ranks disagree on the fill decision ({lo.tolist()} vs {hi.tolist()})"
            )

    def fill_fc(
        self, token_ids: torch.Tensor, hidden_states: torch.Tensor, forward_batch
    ) -> None:
        """Stock fc of a draft's first step into fc_buf, when no extend exchange produced it."""
        model = self.model
        self.pending = False
        embeds = model.model.embed_tokens(token_ids.flatten())
        self.fc_buf.copy_(model._split_fc(embeds, hidden_states, forward_batch))

    def fc(self) -> torch.Tensor:
        """The draft fc output [1, H] of the step after an exchange."""
        self.pending = False
        if self.fc_buf is not None:
            return self.fc_buf
        model = self.model
        norm = model.pre_fc_norm_embedding
        out = torch.empty_like(self.fc_out)
        mkd_fc_sum(
            self.out_cand,
            self.out_cand if self.memo else None,
            self.out_partials,
            model.fc.weight,
            norm.weight.data,
            norm.variance_epsilon,
            out,
            self.world,
        )
        return out
