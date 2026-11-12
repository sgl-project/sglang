# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""DFLASH decode aux stage split around the accept (SGLANG_OPT_DFLASH_SHADOW_CTX, default off).

The decode aux stage turns the verify's aux hidden rows into draft context KV. With the served context fusion
(fc shard, context kv_a GEMM, context store norm fold) it is one CUDA graph replayed on the aux side stream after
the accept. Only its last kernel, the context KV store, reads the accept's output (commit_lens: which rows are
committed). Here the same kernels on the same static inputs are captured as three graphs:

  A1  this rank's fc column block (cuBLASLt GEMM; no collective);
  A2  the fc-shard multimem all-gather, the draft hidden norm and the context kv_a GEMM;
  B   the context KV store (RoPE, kv_a norm fold, FP8 cast, scatter of the committed rows).

A1 and A2 read only the verify's aux hidden rows and weights, so they run beside the accept; B runs after it.

Collective order. A2 holds the stage's only spinning grid (the gatherer waits on its peers). It waits for an event
recorded on the worker stream right after the accept's TP sync, the accept's last collective, so on every rank at
most one collective is in flight and every rank issues them in one order: the verify's collectives, the accept
sync, the gatherer, then the draft's collectives after the aux join. Mode 1 launches A1 and A2 together after that
event; mode 2 launches A1 at verify end and A2 after the event.

Graphs are captured in the round that captures the stock aux graph for the same shapes (the same round on every
rank), from the same static inputs, so the extra eager warm-up gather happens on every rank or on none.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from sglang.kernels.ops.memory.small_copy import try_small_copy
from sglang.kernels.ops.speculative.dflash_context import (
    dflash_ctx_kv_store_norm,
    lt_matmul,
)
from sglang.srt.environ import envs
from sglang.srt.speculative.dflash_context_fusion import MAX_FUSED_ROWS

logger = logging.getLogger(__name__)


def split_row_cap(knob: int, fc_max_rows: int) -> int:
    """Largest aux batch (rows) the split takes: SGLANG_OPT_DFLASH_SHADOW_CTX_MAX_ROWS (0 = MAX_FUSED_ROWS, the original
    decode cap), never above the fc shard's cap (the gatherer's capacity; A2 needs the shard's gather)."""
    if knob < 0:
        raise ValueError(f"SGLANG_OPT_DFLASH_SHADOW_CTX_MAX_ROWS={knob}: needs >= 0")
    return min(MAX_FUSED_ROWS if knob == 0 else knob, fc_max_rows)


class _Split:
    """The three graphs of one aux key and the tensors they share."""

    def __init__(self, static, a1, a2, b, part, latents, buffers) -> None:
        self.static = static  # the stock aux graph's static inputs: (hidden, cache_loc, positions, commit_lens)
        self.a1, self.a2, self.b = a1, a2, b
        self.part = part
        self.latents = latents
        self.buffers = buffers


class ShadowContext:
    def __init__(self, worker) -> None:
        self.w = worker
        self.base = int(envs.SGLANG_OPT_DFLASH_SHADOW_CTX.get())
        self.check_n = int(envs.SGLANG_TEST_DFLASH_SHADOW_CTX_CHECK.get())
        self.check_k = 0
        self.max_rows_knob = int(envs.SGLANG_OPT_DFLASH_SHADOW_CTX_MAX_ROWS.get())
        split_row_cap(self.max_rows_knob, MAX_FUSED_ROWS)  # rejects a negative knob at startup
        self._why: Optional[str] = None
        self._splits: Dict[tuple, Optional[_Split]] = {}
        self._sync_ev: Optional[torch.cuda.Event] = None
        # This round, set by begin_round: [split, mode, stream, A2 launched, check].
        self._cur: Optional[list] = None
        self.ct = {"split": 0, "stock": 0}

    # ---- engagement -------------------------------------------------------------------------------------

    def refusal(self) -> str:
        """Why the split may not run in this process ('' = it may); decided once."""
        if self._why is not None:
            return self._why
        w = self.w
        cf = w._context_fusion
        why = ""
        if not w._use_aux_graph:
            why = "no aux CUDA graph"
        elif not w._aux_overlap:
            why = "the aux graph does not run on a side stream"
        elif cf is None or not (cf.fc_shard and cf.ctx_gemm and cf.ctx_store_norm):
            why = "context fusion is not fc shard + context GEMM + store norm fold"
        elif cf._check_n:
            why = "an aux-graph test hook is on"
        self._why = why
        if why:
            logger.warning("DFLASH shadow ctx REFUSED (%s): the stock aux graph runs", why)
        else:
            logger.info("DFLASH shadow ctx ENGAGED (base mode %d, check first %d split rounds)", self.base, self.check_n)
            if self.max_rows_knob:
                logger.info("DFLASH shadow ctx: split row cap %d (SGLANG_OPT_DFLASH_SHADOW_CTX_MAX_ROWS=%d, fc shard cap %d)",
                            split_row_cap(self.max_rows_knob, cf.fc_max_rows), self.max_rows_knob, cf.fc_max_rows)  # fmt: skip
        return why

    def _stream(self) -> torch.cuda.Stream:
        w = self.w
        if w._aux_stream is None:
            w._aux_stream = torch.cuda.Stream(device=w.device)
            w._aux_done = torch.cuda.Event()
        if w._aux_high_prio:
            if w._aux_stream_hi is None:
                w._aux_stream_hi = torch.cuda.Stream(device=w.device, priority=-100)
                logger.info("DFLASH aux high-priority stream created (priority %d; default stream priority 0)",
                            w._aux_stream_hi.priority)  # fmt: skip
            return w._aux_stream_hi
        return w._aux_stream

    # ---- capture ----------------------------------------------------------------------------------------

    def capture(self, key: tuple, static: tuple, shape2d: tuple, stream: torch.cuda.Stream) -> None:
        """Capture A1 / A2 / B for this aux key, right after the stock aux graph of the same key was captured on
        `stream` (its warm-up pinned the algorithms and probed the store). The warm-up rewrites this round's committed
        rows with the values the stock stage writes (same kernels, same inputs). A failure turns the split off for the
        process (every rank runs the same code on the same shapes, so it fails on every rank alike)."""
        if key in self._splits or self.refusal():
            return
        try:
            self._capture(key, static, shape2d, stream)
        except Exception as e:  # noqa: BLE001
            self._why = f"capture failed: {e!r}"[:300]
            self._splits[key] = None
            logger.warning("DFLASH shadow ctx REFUSED (%s): the stock aux graph runs", self._why)

    def _capture(self, key: tuple, static: tuple, shape2d: tuple, stream: torch.cuda.Stream) -> None:
        w = self.w
        cf = w._context_fusion
        hidden, loc, pos, cl = static
        rows = int(hidden.shape[0])
        weight = w.draft_model.fc.weight
        fc_algo = cf._fc_algo.get(rows)
        ctx_algo = cf._ctx_algo.get(rows)
        why = ""
        cap = split_row_cap(self.max_rows_knob, cf.fc_max_rows)
        if rows > cap or fc_algo is None or ctx_algo is None or cf._gatherer is None:
            why = "no pinned fc / context GEMM algorithm or no gatherer at these rows"
            if self.max_rows_knob and rows > cap:
                why = f"above the split's row cap {cap} (SGLANG_OPT_DFLASH_SHADOW_CTX_MAX_ROWS; fc shard cap {cf.fc_max_rows})"
        elif not (cf._store_ok and cf._norm_ok):
            why = "context store norm fold not engaged"
        if why:
            # Rank-consistent: the fc pin is TP-agreed; the others come from TP-replicated probes.
            logger.warning("DFLASH shadow ctx: rows %d stay on the stock aux graph (%s)", rows, why)
            self._splits[key] = None
            return
        from sglang.srt.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
        n = weight.shape[0] // tp.world_size
        lo = tp.rank_in_group * n
        w_blk = weight[lo : lo + n]
        ws = cf._ws(weight.device)
        pool = cf.pool
        buffers = []
        for a in cf.attns:
            buf = pool.kv_buffer[a.attn_mqa.layer_id - pool.start_layer]
            buffers.append(buf.view(buf.shape[0], -1))
        norms = [a.kv_a_layernorm for a in cf.attns]
        cos_sin = [a.rotary_emb.cos_sin_cache for a in cf.attns]
        loc2d = loc.view(shape2d)

        def a1():
            return lt_matmul(hidden, w_blk, fc_algo, ws)

        def a2(part):
            return cf._latents(w.draft_model.hidden_norm(cf._gatherer(part)))

        def b(latents):
            # The dtype casts the stock stage makes on its inputs (_append_target_hidden_to_draft_kv_by_loc).
            p = pos if pos.dtype == torch.int64 else pos.to(torch.int64)
            l2 = loc2d if loc2d.dtype == torch.int64 else loc2d.to(torch.int64)
            c = cl if cl.dtype == torch.int32 else cl.to(torch.int32)
            dflash_ctx_kv_store_norm(latents, [m.weight for m in norms], [m.variance_epsilon for m in norms], buffers,
                                     cos_sin, p, l2, c)  # fmt: skip

        current = torch.cuda.current_stream(hidden.device)
        stream.wait_stream(current)
        with torch.inference_mode():
            with torch.cuda.stream(stream):
                b(a2(a1()))
            current.wait_stream(stream)
            g1, g2, g3 = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(g1, stream=stream):
                part = a1()
            with torch.cuda.graph(g2, stream=stream):
                latents = a2(part)
            with torch.cuda.graph(g3, stream=stream):
                b(latents)
            current.wait_stream(stream)
        self._splits[key] = _Split(static, g1, g2, g3, part, latents, buffers)
        logger.info("DFLASH shadow ctx graphs captured rows=%d (A1 fc block %d cols, A2 gather + norm + kv_a, B store)",
                    rows, n)  # fmt: skip

    # ---- per round --------------------------------------------------------------------------------------

    def begin_round(self, mode: int, hidden: Optional[torch.Tensor], bs: int, block: int, batch) -> None:
        """At verify end. Mode 0, an ineligible round or no captured split: the stock aux graph runs after the accept."""
        self._cur = None
        if mode <= 0 or hidden is None or self.refusal():
            return
        if batch.return_logprob or batch.has_grammar:
            self.ct["stock"] += 1
            return
        rows = bs * block
        if rows == 0 or hidden.numel() % rows != 0:
            self.ct["stock"] += 1
            return
        h = hidden.reshape(rows, -1)
        split = self._splits.get(((rows, int(h.shape[1])), (bs, block)))
        if split is None:
            self.ct["stock"] += 1
            return
        w = self.w
        stream = self._stream()
        current = torch.cuda.current_stream(h.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            split.static[0].copy_(h)
            if mode == 2:
                split.a1.replay()
        h.record_stream(stream)
        w._aux_done.record(stream)
        w._aux_pending = True
        self._cur = [split, mode, stream, False, self.check_k < self.check_n]
        self.ct["split"] += 1

    def after_sync(self) -> None:
        """On the worker stream right after the accept's TP sync: A2 (and A1 in mode 1) may start."""
        cur = self._cur
        if cur is None or cur[3]:
            return
        split, mode, stream = cur[:3]
        if self._sync_ev is None:
            self._sync_ev = torch.cuda.Event()
        self._sync_ev.record(torch.cuda.current_stream(self.w.device))
        stream.wait_event(self._sync_ev)
        with torch.cuda.stream(stream):
            if mode == 1:
                split.a1.replay()
            split.a2.replay()
        self.w._aux_done.record(stream)
        cur[3] = True

    def finish(self, target_hidden, cache_loc, cache_loc_2d, positions, commit_lens) -> bool:
        """After the accept, at the stock aux site: B for a split round (True), else False (stock aux graph)."""
        cur = self._cur
        if cur is None:
            return False
        if not cur[3]:
            self.after_sync()  # no accept sync ran: this point is after the accept, so still after its collectives
        self._cur = None
        split, mode, stream, _, check = cur
        w = self.w
        current = torch.cuda.current_stream(target_hidden.device)
        snap = None
        if check:
            torch.cuda.synchronize()
            snap = self._rejected_rows(split, cache_loc_2d, commit_lens)
            torch.cuda.synchronize()
        static = split.static
        inputs = (cache_loc, positions, commit_lens)
        if not (w._seam_small_copy and try_small_copy(list(static[1:]), list(inputs))):
            for dst, src in zip(static[1:], inputs):
                dst.copy_(src)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            split.b.replay()
        w._aux_done.record(stream)
        w._aux_pending = True
        if check:
            self.check_k += 1
            self._check(split, target_hidden, cache_loc_2d, positions, commit_lens, snap)
        return True

    # ---- CHECK (test hook) ------------------------------------------------------------------------------

    @staticmethod
    def _rows(cache_loc_2d, commit_lens):
        block = int(cache_loc_2d.shape[1])
        slots = cache_loc_2d.reshape(-1).long()
        r = torch.arange(slots.numel(), device=slots.device)
        valid = (r % block) < commit_lens.long()[r // block]
        live = slots != 0
        return slots, valid & live, (~valid) & live

    def _rejected_rows(self, split, cache_loc_2d, commit_lens) -> List[torch.Tensor]:
        slots, _, rej = self._rows(cache_loc_2d, commit_lens)
        idx = slots[rej]
        return [buf[idx].clone() for buf in split.buffers]

    def _check(self, split, target_hidden, cache_loc_2d, positions, commit_lens, snap) -> None:
        """Committed rows: the stored bytes == the stock stage computed eagerly on this round's inputs into scratch rows
        (no collective: the gathered fc output is F.linear over the full fc weight, the result the fc shard pin checks
        bitwise). Rejected rows: unchanged since before B. Latents: A2's output == the eager latents."""
        torch.cuda.synchronize()
        w = self.w
        cf = w._context_fusion
        bs, block = int(cache_loc_2d.shape[0]), int(cache_loc_2d.shape[1])
        rows = bs * block
        h = target_hidden.reshape(rows, -1)
        with torch.inference_mode():
            lat_ref = cf._latents(w.draft_model.hidden_norm(F.linear(h, w.draft_model.fc.weight)))
            scratch = [torch.zeros(rows + 1, buf.shape[1], dtype=buf.dtype, device=buf.device) for buf in split.buffers]
            loc_s = torch.arange(1, rows + 1, dtype=torch.int64, device=h.device).view(bs, block)
            norms = [a.kv_a_layernorm for a in cf.attns]
            dflash_ctx_kv_store_norm(lat_ref, [m.weight for m in norms], [m.variance_epsilon for m in norms], scratch,
                                     [a.rotary_emb.cos_sin_cache for a in cf.attns], positions.long(), loc_s,
                                     commit_lens.to(torch.int32))  # fmt: skip
        torch.cuda.synchronize()
        slots, com, rej = self._rows(cache_loc_2d, commit_lens)
        ci = com.nonzero().squeeze(1)
        bad_c = 0
        for buf, sc in zip(split.buffers, scratch):
            bad_c += int((buf[slots[ci]] != sc[ci + 1]).any(dim=1).sum().item())
        bad_r = 0
        for buf, before in zip(split.buffers, snap):
            bad_r += int((buf[slots[rej]] != before).any(dim=1).sum().item())
        bad_l = sum(int((a != b).sum().item()) for a, b in zip(split.latents, lat_ref))
        tag = "OK" if bad_c == 0 and bad_r == 0 and bad_l == 0 else "MISMATCH"
        logger.info("DFLASH shadow ctx CHECK %d (env first %d): %s (committed rows bad %d of %d x %d layers; rejected rows "
                    "changed %d of %d; latent values differing %d)", self.check_k, self.check_n, tag, bad_c,
                    int(ci.numel()), len(split.buffers), bad_r, int(rej.sum().item()), bad_l)  # fmt: skip
