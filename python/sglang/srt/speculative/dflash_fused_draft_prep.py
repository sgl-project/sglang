# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""DFLASH draft block prep as one launch.

The eager draft block prep (DFlashWorkerV2: prepare block, noise embedding, compact window, compact draft
req->token rebuild; the draft graph runner's fill_from + input_embeds copy; TritonAttnBackend's target-verify
metadata) is ~17 tiny launches per round. ``_dflash_fused_draft_prep_kernel`` writes the same values in one
launch: program (b, 0) the integer outputs of request b, program (b, 1 + j) its embedding row j.

This revision is CHECK ONLY (SGLANG_OPT_DFLASH_FUSED_DRAFT_PREP_CHECK=N): for the first N eligible graph rounds the
kernel writes SHADOW tensors, the served eager prep runs unchanged, and after the draft forward every shadow is
compared bitwise with the buffer the eager prep wrote ("fused draft prep check k/N: OK" or the first differing
buffer). The served outputs never come from the kernel here.

ON form (SGLANG_OPT_DFLASH_FUSED_DRAFT_PREP, DFlashFusedDraftPrepOn): the kernel writes the SERVED buffers (worker block ids /
positions / verify cache locs, the compact window lengths, the draft req->token rows by request, the draft graph runner's static
inputs incl. input_embeds, the Triton backend's target-verify buffers) and the worker skips the eager ops; the runner then skips
fill_from and the backend's out-graph prep for that replay (DecodeCudaGraphRunner._take_dflash_fused_prep), or falls back to them
(still correct: they rewrite the same values from the kernel-written worker buffers).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_EMBED_CHUNK = 1024
_ROW_CHUNK = 256


@triton.jit
def _dflash_fused_draft_prep_kernel(
    bonus_ptr, prefix_ptr, rpi_ptr, t_r2t_ptr, embed_ptr,
    bi_ptr, pos_ptr, cl_ptr, dlen_ptr, sstart_ptr, r2t_out_ptr,
    s_ids_ptr, s_rpi_ptr, s_seq_ptr, s_ocl_ptr, s_pos_ptr, s_emb_ptr,
    qo_ptr, kvp_ptr, kvi_ptr, wkvp_ptr, wkvi_ptr, woff_ptr,
    bs, t_r2t_stride, t_r2t_width, r2t_out_stride, mask_token_id,
    H: tl.constexpr, B: tl.constexpr, BP: tl.constexpr, WIN: tl.constexpr, PAGE: tl.constexpr, SWIN: tl.constexpr,
    BSMAX: tl.constexpr, ROWS_BY_REQ: tl.constexpr,
):  # fmt: skip
    b = tl.program_id(0)
    e = tl.program_id(1)
    bonus = tl.load(bonus_ptr + b).to(tl.int64)
    if e > 0:
        # Noise embedding row e - 1 of request b (F.embedding of the block ids) -> the static input_embeds.
        e_j = e - 1
        e_tok = tl.where(e_j == 0, bonus, mask_token_id.to(tl.int64))
        e_offs = tl.arange(0, 1024).to(tl.int64)
        e_row = (b * B + e_j).to(tl.int64)
        for e_c in range(0, H, 1024):
            e_m = e_c + e_offs < H
            e_v = tl.load(embed_ptr + e_tok * H + e_c + e_offs, mask=e_m)
            tl.store(s_emb_ptr + e_row * H + e_c + e_offs, e_v, mask=e_m)
    else:
        prefix = tl.load(prefix_ptr + b).to(tl.int64)
        req = tl.load(rpi_ptr + b).to(tl.int64)
        # _prepare_dflash_draft_block_contig_kernel: BP = next power of 2 of B (tl.arange needs one), lanes >= B masked
        cols = tl.arange(0, BP)
        row_mask = cols < B
        pos = prefix + cols
        cl = tl.load(t_r2t_ptr + req * t_r2t_stride + pos, mask=row_mask & (pos < t_r2t_width), other=0).to(tl.int64)
        ids = tl.where(cols == 0, bonus, tl.zeros((BP,), tl.int64) + mask_token_id.to(tl.int64))
        tl.store(bi_ptr + b * B + cols, ids, mask=row_mask)
        tl.store(pos_ptr + b * B + cols, pos, mask=row_mask)
        tl.store(cl_ptr + b * B + cols, cl, mask=row_mask)
        # fill_from's static fields (input_ids, positions, out_cache_loc, req_pool_indices, seq_lens)
        tl.store(s_ids_ptr + b * B + cols, ids, mask=row_mask)
        tl.store(s_pos_ptr + b * B + cols, pos, mask=row_mask)
        tl.store(s_ocl_ptr + b * B + cols, cl, mask=row_mask)
        tl.store(s_rpi_ptr + b, req)
        # _compact_draft_window_kernel (PAGED)
        vis = tl.minimum(prefix.to(tl.int32), WIN)
        start = prefix - vis.to(tl.int64)
        dlen = (prefix - (start - start % PAGE)).to(tl.int32)
        sstart = prefix - dlen.to(tl.int64)
        tl.store(dlen_ptr + b, dlen)
        tl.store(sstart_ptr + b, sstart)
        tl.store(s_seq_ptr + b, dlen)
        # TritonAttnBackend._update_target_verify_buffers: qo_indptr, kv_indptr (+ cumsum over earlier rows),
        # window lens min(seq, SWIN), window indptr, window offsets seq - wlen.
        ob = tl.arange(0, BSMAX)
        mb = ob < b
        po = tl.load(prefix_ptr + ob, mask=mb, other=0).to(tl.int64)
        vo = tl.minimum(po.to(tl.int32), WIN)
        so = po - vo.to(tl.int64)
        do = (po - (so - so % PAGE)).to(tl.int32)
        do = tl.where(mb, do, 0)
        wo = tl.where(mb, tl.minimum(do, SWIN), 0)
        kv_base = tl.sum(do, axis=0)
        w_base = tl.sum(wo, axis=0)
        wlen = tl.minimum(dlen, SWIN)
        wstart = dlen - wlen
        tl.store(qo_ptr + b + 1, (b + 1) * B)
        tl.store(kvp_ptr + b + 1, kv_base + dlen)
        tl.store(wkvp_ptr + b + 1, w_base + wlen)
        if b == 0:
            tl.store(qo_ptr, 0)
            tl.store(kvp_ptr, 0)
            tl.store(wkvp_ptr, 0)
        tl.store(woff_ptr + b, wstart)
        # rebuild_compact_draft_req_to_token: [0, dlen) the committed suffix window, then the block slots;
        # create_flashinfer_kv_indices_triton over that row: kv indices = row[0, dlen), window = row[wstart, dlen).
        if ROWS_BY_REQ:
            dst = r2t_out_ptr + req * r2t_out_stride
        else:
            dst = r2t_out_ptr + b.to(tl.int64) * r2t_out_stride
        src = t_r2t_ptr + req * t_r2t_stride + sstart
        total = dlen.to(tl.int64) + B
        offs = tl.arange(0, 256).to(tl.int64)
        for i in range(0, tl.cdiv(total, 256)):
            col = offs + i * 256
            in_p = col < dlen
            in_b = (col >= dlen) & (col < total)
            v = tl.load(src + col, mask=in_p, other=0)
            bpos = prefix + (col - dlen)
            vb = tl.load(t_r2t_ptr + req * t_r2t_stride + bpos, mask=in_b & (bpos < t_r2t_width), other=0)
            val = tl.where(in_p, v, vb)
            tl.store(dst + col, val, mask=in_p | in_b)
            tl.store(kvi_ptr + kv_base + col, v, mask=in_p)
            in_w = in_p & (col >= wstart)
            tl.store(wkvi_ptr + w_base + (col - wstart), v, mask=in_w)


class DFlashFusedDraftPrep:
    """CHECK-mode driver: shadow launch, bitwise compare."""

    def __init__(self, worker, check_rounds: int) -> None:
        self.worker = worker
        self.check_rounds = int(check_rounds)
        self.checked = 0
        self.ok = 0
        self.skipped: Dict[str, int] = {}
        self.disabled = False
        self._shadow: Optional[Dict[str, torch.Tensor]] = None
        self._bs = 0
        self._l2p: Optional[List[torch.Tensor]] = None
        self._l2p_snap: Optional[List[torch.Tensor]] = None
        self._l2p_replays = -1

    # ---------------------------------------------------------------- eligibility
    @staticmethod
    def unsupported_reason(worker) -> Optional[str]:
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
        from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind

        if not worker.use_compact_draft_cache:
            return "no compact draft cache"
        if not getattr(worker, "_fused_compact_window", False):
            return "SGLANG_OPT_DFLASH_COMPACT_WINDOW_FUSED off"
        if int(worker.page_size) <= 1:
            return "page_size <= 1"
        if not (worker._use_triton_prepare_block and worker._use_triton_compact_rebuild):
            return "eager (non-Triton) prepare / rebuild"
        if getattr(worker, "_full_embed_gpu", None) is None and DFlashFusedDraftPrep._embedding_module(worker) is None:
            return "no replicated or gatherable draft embedding"
        if float(worker._noise_embed_scale) != 1.0:
            return "noise_embed_scale != 1"
        runner = getattr(worker.draft_model_runner, "decode_cuda_graph_runner", None)
        if runner is None:
            return "no draft decode graph runner"
        be = runner.attn_backend
        if type(be) is not TritonAttnBackend:
            return f"draft backend {type(be).__name__}"
        if not be.sliding_window_size or be.sliding_window_size <= 0:
            return "no draft sliding window"
        if getattr(be, "use_sliding_window_kv_pool", False) or getattr(be, "dcp_size", 1) != 1:
            return "SWA KV pool or DCP"
        if not be._trim_verify_metadata:
            return "SGLANG_OPT_TRITON_VERIFY_METADATA_TRIM off"
        tr = be.runner_ids
        if tr._reads_translated(IdSpaceKind.FULL) or tr._reads_translated(tr.window_kv_kind):
            return "draft translator reads are translated"
        return None

    @staticmethod
    def _embedding_module(worker):
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
        from sglang.srt.speculative import dflash_worker_v2 as wv2

        emb = wv2.unwrap_lora_layer(
            wv2._resolve_dflash_embedding_module(worker.draft_model, worker.target_worker.model_runner.model)
        )
        return emb if isinstance(emb, VocabParallelEmbedding) else None

    def _embed_table(self) -> torch.Tensor:
        """The served replicated table (attention DP), else a gathered copy for the check only (one TP all-gather,
        issued by every rank at the same round: every rank checks the same rounds)."""
        w = self.worker
        if w._full_embed_gpu is not None:
            return w._full_embed_gpu
        if getattr(self, "_gathered_embed", None) is None:
            from sglang.srt.speculative.dflash_dp import gather_draft_embedding

            self._gathered_embed = gather_draft_embedding(self._embedding_module(w))
            logger.info("DFLASH fused draft prep check: gathered a draft embedding copy %s for the check (no replicated table here)",
                        tuple(self._gathered_embed.shape))
        return self._gathered_embed

    # ---------------------------------------------------------------- shadow launch
    def wants_round(self) -> bool:
        return not self.disabled and self.checked < self.check_rounds

    def _alloc(self, bs: int) -> Dict[str, torch.Tensor]:
        w = self.worker
        runner = w.draft_model_runner.decode_cuda_graph_runner
        be = runner.attn_backend
        buf = runner.buffers
        B = int(w.block_size)
        dev = w.device
        H = int(self._embed_table().shape[1])
        win = int(w.draft_window_size)
        maxrow = win + int(w.page_size) + B
        like = lambda t, n: torch.full((n,) + tuple(t.shape[1:]), -7, dtype=t.dtype, device=dev)  # noqa: E731
        sh = {
            "block_ids": torch.full((bs, B), -7, dtype=w._draft_block_ids_buf.dtype, device=dev),
            "positions": torch.full((bs, B), -7, dtype=w._draft_block_positions_buf.dtype, device=dev),
            "cache_loc": torch.full((bs, B), -7, dtype=w._draft_verify_out_cache_loc_buf.dtype, device=dev),
            "dlen": torch.full((bs,), -7, dtype=torch.int32, device=dev),
            "sstart": torch.full((bs,), -7, dtype=torch.int64, device=dev),
            "r2t_rows": torch.full(
                (bs, maxrow), -7, dtype=w.draft_model_runner.req_to_token_pool.req_to_token.dtype, device=dev
            ),
            "s_input_ids": like(buf.input_ids, bs * B),
            "s_req_pool_indices": like(buf.req_pool_indices, bs),
            "s_seq_lens": like(buf.seq_lens, bs),
            "s_out_cache_loc": like(buf.out_cache_loc, bs * B),
            "s_positions": like(buf.positions, bs * B),
            "s_input_embeds": torch.zeros((bs * B, H), dtype=buf.input_embeds.dtype, device=dev),
            "qo_indptr": like(be.qo_indptr, bs + 1),
            "kv_indptr": like(be.kv_indptr, bs + 1),
            "kv_indices": like(be.cuda_graph_kv_indices, bs * maxrow),
            "window_kv_indptr": like(be.window_kv_indptr, bs + 1),
            "window_kv_indices": like(be.cuda_graph_window_kv_indices, bs * maxrow),
            "window_kv_offsets": like(be.cuda_graph_window_kv_offsets, bs),
        }
        return sh

    def run_shadow(self, *, bs: int, bonus: torch.Tensor, prefix_lens: torch.Tensor, req_pool_indices: torch.Tensor):
        w = self.worker
        runner = w.draft_model_runner.decode_cuda_graph_runner
        be = runner.attn_backend
        if self._l2p_replays < 0:
            self._l2p_begin()
        if self._shadow is None or self._bs != bs:
            self._shadow = self._alloc(bs)
            self._bs = bs
        else:
            for t in self._shadow.values():
                if t.dtype == torch.bfloat16:
                    t.zero_()
                else:
                    t.fill_(-7)
        sh = self._shadow
        t_r2t = w.model_runner.req_to_token_pool.req_to_token
        B = int(w.block_size)
        embed = self._embed_table()
        H = int(embed.shape[1])
        _dflash_fused_draft_prep_kernel[(bs, 1 + B)](
            bonus, prefix_lens, req_pool_indices, t_r2t, embed,
            sh["block_ids"], sh["positions"], sh["cache_loc"], sh["dlen"], sh["sstart"], sh["r2t_rows"],
            sh["s_input_ids"], sh["s_req_pool_indices"], sh["s_seq_lens"], sh["s_out_cache_loc"],
            sh["s_positions"], sh["s_input_embeds"],
            sh["qo_indptr"], sh["kv_indptr"], sh["kv_indices"], sh["window_kv_indptr"],
            sh["window_kv_indices"], sh["window_kv_offsets"],
            bs, t_r2t.stride(0), t_r2t.shape[1], sh["r2t_rows"].stride(0), int(w._mask_token_id),
            H=H, B=B, BP=triton.next_power_of_2(B), WIN=int(w.draft_window_size), PAGE=int(w.page_size), SWIN=int(be.sliding_window_size),
            BSMAX=max(16, triton.next_power_of_2(bs)), ROWS_BY_REQ=False,
        )  # fmt: skip

    # ---------------------------------------------------------------- compare
    def compare(self, *, bs: int, ran_graph: bool, req_pool_indices: torch.Tensor, input_embeds_src=None) -> None:
        w = self.worker
        runner = w.draft_model_runner.decode_cuda_graph_runner
        if not ran_graph:
            self.skipped["eager draft"] = self.skipped.get("eager draft", 0) + 1
            return
        self._l2p_after_replay()
        if getattr(runner, "raw_bs", bs) != getattr(runner, "bs", bs) or runner.raw_bs != bs:
            self.skipped["padded graph bs"] = self.skipped.get("padded graph bs", 0) + 1
            return
        be = runner._replay_attn_backend()
        buf = runner.buffers
        sh = self._shadow
        B = int(w.block_size)
        dlen = sh["dlen"].cpu()
        rpi = req_pool_indices.view(-1)[:bs].cpu()
        kv_total = int(dlen.to(torch.int64).sum())
        swin = int(be.sliding_window_size)
        w_total = int(torch.clamp(dlen.to(torch.int64), max=swin).sum())
        d_r2t = w.draft_model_runner.req_to_token_pool.req_to_token
        pairs: List[Tuple[str, torch.Tensor, torch.Tensor]] = [
            ("worker block_ids", sh["block_ids"], w._draft_block_ids_buf[:bs]),
            ("worker positions", sh["positions"], w._draft_block_positions_buf[:bs]),
            ("worker verify cache loc", sh["cache_loc"], w._draft_verify_out_cache_loc_buf[:bs]),
            ("static input_ids", sh["s_input_ids"], buf.input_ids[: bs * B]),
            ("static req_pool_indices", sh["s_req_pool_indices"], buf.req_pool_indices[:bs]),
            ("static seq_lens", sh["s_seq_lens"], buf.seq_lens[:bs]),
            ("static out_cache_loc", sh["s_out_cache_loc"], buf.out_cache_loc[: bs * B]),
            ("static positions", sh["s_positions"], buf.positions[: bs * B]),
            # The draft graph rewrites its static input_embeds in place (the first layer's residual stream), so after the
            # replay the eager fill's value survives only in its source: the batch's input_embeds (copied exactly by fill).
            ("static input_embeds (eager source)", sh["s_input_embeds"],
             input_embeds_src[: bs * B] if input_embeds_src is not None else buf.input_embeds[: bs * B]),
            ("qo_indptr", sh["qo_indptr"], be.qo_indptr[: bs + 1]),
            ("kv_indptr", sh["kv_indptr"], be.kv_indptr[: bs + 1]),
            ("kv_indices", sh["kv_indices"][:kv_total], be.cuda_graph_kv_indices[:kv_total]),
            ("window_kv_indptr", sh["window_kv_indptr"], be.window_kv_indptr[: bs + 1]),
            ("window_kv_indices", sh["window_kv_indices"][:w_total], be.cuda_graph_window_kv_indices[:w_total]),
            ("window_kv_offsets", sh["window_kv_offsets"], be.cuda_graph_window_kv_offsets[:bs]),
        ]
        for b in range(bs):
            n = int(dlen[b]) + B
            pairs.append((f"draft req_to_token row {b}", sh["r2t_rows"][b, :n], d_r2t[int(rpi[b]), :n]))
        bad = None
        for name, s, r in pairs:
            if s.dtype != r.dtype or s.shape != r.shape:
                bad = f"{name}: dtype/shape {s.dtype}{tuple(s.shape)} vs served {r.dtype}{tuple(r.shape)}"
                break
            if not torch.equal(s, r):
                diff = (s != r).nonzero()
                first = diff[0].tolist() if diff.numel() else []
                bad = (
                    f"{name}: {int((s != r).sum())} of {s.numel()} differ, first at {first}: "
                    f"fused {s[tuple(first)].item() if first else '?'} served {r[tuple(first)].item() if first else '?'}"
                )
                break
        self.checked += 1
        if bad is None:
            self.ok += 1
            if self.checked in (1, 2, 10, 50) or self.checked % 100 == 0 or self.checked == self.check_rounds:
                logger.info(
                    "DFLASH fused draft prep check %d/%d: OK (bs %d, draft lens %s, %d buffers bitwise; skipped %s)",
                    self.checked, self.check_rounds, bs, dlen.tolist(), len(pairs), self.skipped,
                )  # fmt: skip
        else:
            logger.warning(
                "DFLASH fused draft prep check %d/%d: MISMATCH (bs %d, draft lens %s, prefix rows %s) %s; "
                "check disabled for this process", self.checked, self.check_rounds, bs, dlen.tolist(), rpi.tolist(), bad,
            )  # fmt: skip
            self.disabled = True

    # ---------------------------------------------------------------- draft L2 prefetch check
    def _l2p_begin(self) -> None:
        """Snapshot every tensor the served draft L2 prefetch table covers (when that prefetch is installed)."""
        self._l2p_replays = 0
        from sglang.srt.speculative.dflash_draft_splitk import DraftL2Prefetch

        for hook in getattr(self.worker.draft_model_runner, "capture_head_hooks", []):
            owner = getattr(hook, "__self__", None)
            if isinstance(owner, DraftL2Prefetch):
                self._l2p = list(owner.tensors)
                self._l2p_snap = [t.detach().clone() for t in self._l2p]
                logger.info("DFLASH draft L2 prefetch check: snapshot of %d prefetched tensors (%d bytes)", len(self._l2p),
                            sum(t.numel() * t.element_size() for t in self._l2p))
                return

    def _l2p_after_replay(self) -> None:
        if self._l2p is None:
            return
        self._l2p_replays += 1
        if self._l2p_replays < 3:
            return
        changed = []
        for i, (t, s) in enumerate(zip(self._l2p, self._l2p_snap)):
            a = t.detach().contiguous().view(-1).view(torch.uint8)
            b = s.contiguous().view(-1).view(torch.uint8)
            if not torch.equal(a, b):
                changed.append(f"{i}:{t.dtype}{tuple(t.shape)}:{int((a != b).sum())}B")
        logger.info("DFLASH draft L2 prefetch check: %d prefetched tensors %s over %d draft graph replays%s", len(self._l2p),
                    "bitwise unchanged" if not changed else "CHANGED", self._l2p_replays,
                    "" if not changed else " " + " ".join(changed[:8]))
        self._l2p, self._l2p_snap = None, None


class FusedPrepOut:
    __slots__ = ("dlen", "sstart", "input_embeds")

    def __init__(self, dlen: torch.Tensor, sstart: torch.Tensor, input_embeds: torch.Tensor) -> None:
        self.dlen = dlen
        self.sstart = sstart
        self.input_embeds = input_embeds


class DFlashFusedDraftPrepOn:
    """ON-form driver (served): one launch writes every buffer the eager draft block prep writes."""

    def __init__(self, worker) -> None:
        self.worker = worker
        self._dlen: Optional[torch.Tensor] = None
        self._sstart: Optional[torch.Tensor] = None
        self._gathered_embed: Optional[torch.Tensor] = None
        self.on_rounds = 0
        self.off_reasons: Dict[str, int] = {}

    @staticmethod
    def unsupported_reason(worker) -> Optional[str]:
        why = DFlashFusedDraftPrep.unsupported_reason(worker)
        if why:
            return why
        if (
            getattr(worker, "_full_embed_gpu", None) is None
            and not DFlashFusedDraftPrepOn.replicated_embedding(worker)
        ):
            return "no replicated draft embedding table (attention DP's, or an unsharded draft embedding)"
        if getattr(worker, "_is_domino", False):
            return "domino draft"
        return None

    @staticmethod
    def replicated_embedding(worker) -> bool:
        """An unsharded, unquantized draft embedding (e.g. SGLANG_ENABLE_EMBED_REPLICATION=1): gather_draft_embedding returns a
        zero-copy view of its weight (dflash_dp.py, tp_size 1 branch), with no collective and no extra memory; the eager prep reads
        the same table through the module."""
        from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod

        emb = DFlashFusedDraftPrep._embedding_module(worker)
        return (
            emb is not None
            and emb.tp_size == 1
            and emb.num_embeddings == emb.org_vocab_size
            and isinstance(emb.quant_method, UnquantizedEmbeddingMethod)
            and emb.output_dtype is None
        )

    def _table(self) -> torch.Tensor:
        w = self.worker
        if w._full_embed_gpu is not None:
            return w._full_embed_gpu
        if self._gathered_embed is None:
            from sglang.srt.speculative.dflash_dp import gather_draft_embedding

            emb = DFlashFusedDraftPrep._embedding_module(w)
            self._gathered_embed = gather_draft_embedding(emb)
            if self.replicated_embedding(w):
                same = self._gathered_embed.data_ptr() == emb.weight.data_ptr()
                logger.info("DFLASH fused draft prep ON: table = the replicated draft embedding %s (zero-copy view: %s)",
                            tuple(self._gathered_embed.shape), same)  # fmt: skip
            else:
                logger.info("DFLASH fused draft prep ON: TEST table = a gathered draft embedding copy %s (no replicated table)",
                            tuple(self._gathered_embed.shape))  # fmt: skip
        return self._gathered_embed

    def round_reason(self, bs: int, batch) -> Optional[str]:
        """None when this round can take the fused prep."""
        runner = self.worker.draft_model_runner.decode_cuda_graph_runner
        # As DecodeCudaGraphRunner.can_run: the batch's DP graph vote counts only when the runner needs MLP sync
        # (reading batch.can_run_decode_cuda_graph unconditionally gives False without DP, so no round would take the fused prep).
        if getattr(runner, "require_mlp_sync", False) and not batch.can_run_decode_cuda_graph:
            return "no decode graph this round (DP vote)"
        caps = getattr(runner, "capture_bs", None)
        if not caps or bs not in caps:
            return "bs not a captured graph bs"
        return None

    def note_off(self, reason: str) -> None:
        n = self.off_reasons.get(reason, 0)
        self.off_reasons[reason] = n + 1
        if n == 0:
            logger.info("DFLASH fused draft prep ON: eager prep this round (%s); further rounds with this reason are counted", reason)

    def run_on(self, *, bs: int, bonus: torch.Tensor, prefix_lens: torch.Tensor, req_pool_indices: torch.Tensor) -> FusedPrepOut:
        w = self.worker
        runner = w.draft_model_runner.decode_cuda_graph_runner
        be = runner._replay_attn_backend()
        buf = runner.buffers
        B = int(w.block_size)
        if self._dlen is None or self._dlen.shape[0] < bs:
            cap = max(bs, int(w._draft_block_ids_buf.shape[0]))
            self._dlen = torch.empty((cap,), dtype=torch.int32, device=w.device)
            self._sstart = torch.empty((cap,), dtype=torch.int64, device=w.device)
        dlen, sstart = self._dlen[:bs], self._sstart[:bs]
        t_r2t = w.model_runner.req_to_token_pool.req_to_token
        d_r2t = w.draft_model_runner.req_to_token_pool.req_to_token
        embed = self._table()
        _dflash_fused_draft_prep_kernel[(bs, 1 + B)](
            bonus, prefix_lens, req_pool_indices, t_r2t, embed,
            w._draft_block_ids_buf, w._draft_block_positions_buf, w._draft_verify_out_cache_loc_buf, dlen, sstart, d_r2t,
            buf.input_ids, buf.req_pool_indices, buf.seq_lens, buf.out_cache_loc, buf.positions, buf.input_embeds,
            be.qo_indptr, be.kv_indptr, be.cuda_graph_kv_indices, be.window_kv_indptr,
            be.cuda_graph_window_kv_indices, be.cuda_graph_window_kv_offsets,
            bs, t_r2t.stride(0), t_r2t.shape[1], d_r2t.stride(0), int(w._mask_token_id),
            H=int(embed.shape[1]), B=B, BP=triton.next_power_of_2(B), WIN=int(w.draft_window_size), PAGE=int(w.page_size),
            SWIN=int(be.sliding_window_size), BSMAX=max(16, triton.next_power_of_2(bs)), ROWS_BY_REQ=True,
        )  # fmt: skip
        self.on_rounds += 1
        if self.on_rounds in (1, 10) or self.on_rounds % 200 == 0:
            logger.info("DFLASH fused draft prep ON: %d fused rounds (bs %d), eager rounds by reason %s, runner skips %d, "
                        "runner fallbacks %s", self.on_rounds, bs, self.off_reasons, getattr(runner, "dflash_fused_prep_skips", -1),
                        getattr(runner, "dflash_fused_prep_fallbacks", None))  # fmt: skip
        return FusedPrepOut(dlen, sstart, buf.input_embeds[: bs * B])
