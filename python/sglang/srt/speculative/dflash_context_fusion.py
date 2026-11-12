# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Default-off fusions of the DFlash target-context stage (draft fc, MLA context-KV writes), each bitwise
equal to the stock path it replaces.

- fc shard (SGLANG_OPT_DFLASH_FC_SHARD): each TP rank computes its column block of the replicated fc with the
  cuBLASLt algorithm F.linear picks for the full fc, then a multimem all-gather rebuilds the rows.
- context GEMM (SGLANG_OPT_DFLASH_CTX_KV_GEMM): the draft layers' kv_a projections of the context rows as one
  GEMM over the concatenated weights, with the algorithm F.linear picks for one layer.
- context store (SGLANG_OPT_DFLASH_CTX_KV_STORE): the k_pe RoPE, FP8 casts and scatter of every layer in one
  kernel (dflash_ctx_kv_store).
- context store norm (SGLANG_OPT_DFLASH_CTX_KV_STORE_NORM, with the store): the layers' kv_a RMSNorms folded into that
  kernel (dflash_ctx_kv_store_norm, flashinfer's H = 512 BF16 order), accepted only after a bitwise probe against the
  served norm modules on this device.

A pinned algorithm is accepted only after it reproduces F.linear bitwise on probe inputs, once per row count.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.kernels.ops.speculative.dflash_context import (
    TORCH_CUBLAS_WORKSPACE_BYTES,
    dflash_ctx_kv_store,
    dflash_ctx_kv_store_norm,
    lt_heuristics,
    lt_matmul,
)
from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, MLATokenToKVPool

logger = logging.getLogger(__name__)

# Decode-sized context batches only; larger (prefill) batches take the stock path.
MAX_FUSED_ROWS = 64
_NOPE, _ROPE = 512, 64


def _probe_inputs(rows: int, k: int, device: torch.device) -> List[torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(20261030)
    x = torch.randn(rows, k, device=device, generator=g)
    # Pairs of opposite-sign terms: partial sums cancel, so a different reduction order shows.
    c = x.abs()
    c[:, 1::2] = -c[:, 0::2]
    c[:, ::7] *= 1.0001
    big = x.clone()
    big[::2] *= 1000.0
    return [t.to(torch.bfloat16) for t in (x, c, big)]


def pin_lt_algo(
    rows: int,
    ref_weight: torch.Tensor,
    variant_weight: torch.Tensor,
    expected: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    workspace: torch.Tensor,
) -> Optional[torch.Tensor]:
    """The first cuBLASLt algorithm whose output equals F.linear(x, ref_weight) bitwise and whose output on
    variant_weight equals expected(x, F.linear(x, ref_weight)) bitwise, on every probe input; else None."""
    xs = _probe_inputs(rows, ref_weight.shape[1], ref_weight.device)
    refs = [F.linear(x, ref_weight) for x in xs]
    for algo in lt_heuristics(rows, ref_weight):
        if int(algo[8]) > workspace.numel():
            continue
        try:
            same = all(
                torch.equal(lt_matmul(x, ref_weight, algo, workspace), r)
                for x, r in zip(xs, refs)
            ) and all(
                torch.equal(lt_matmul(x, variant_weight, algo, workspace), expected(x, r))
                for x, r in zip(xs, refs)
            )
        except RuntimeError:
            continue
        if same:
            return algo
    return None


def context_store_supported(pool, attns: Sequence) -> bool:
    """Whether stock writes these layers' context rows through the plain FP8 MLA scatter the kernel mirrors."""
    from sglang.srt.runtime_context import get_parallel

    rotaries = [a.rotary_emb for a in attns]
    return (
        type(pool) is MLATokenToKVPool
        and pool.dtype == torch.float8_e4m3fn
        and pool.store_dtype == torch.uint8
        and not pool.use_dsa
        and not pool.dsa_kv_cache_store_fp8
        and not pool.fp8_cast_on_write
        and pool.kv_lora_rank == _NOPE
        and pool.qk_rope_head_dim == _ROPE
        and get_parallel().attn_dcp_size == 1
        and all(
            not r.use_fallback_kernel
            and not r.is_neox_style
            and r.head_size == _ROPE
            and r.rotary_dim == _ROPE
            and r.cos_sin_cache.dtype == torch.float32
            for r in rotaries
        )
        and all(a.kv_lora_rank == _NOPE and a.qk_rope_head_dim == _ROPE for a in attns)
        and len(attns) <= 8
    )


def _norm_modules_supported(attns: Sequence) -> bool:
    """Whether every kv_a_layernorm is a plain BF16 RMSNorm(512) (no residual, cast or quant fusion)."""
    from sglang.srt.layers.layernorm import RMSNorm

    for a in attns:
        n = getattr(a, "kv_a_layernorm", None)
        if (
            type(n) is not RMSNorm
            or not n.has_weight
            or n.variance_size_override is not None
            or n.cast_x_before_out_mul
            or n.override_orig_dtype is not None
            or n._quant_linear is not None
        ):
            return False
        w = n.weight
        if w.dtype != torch.bfloat16 or tuple(w.shape) != (_NOPE,) or not w.is_contiguous():
            return False
    return True


class DFlashContextFusion:
    def __init__(self, draft_model) -> None:
        from sglang.srt.models.dflash import DFlashDraftModel, DFlashMLAAttention

        self.draft_model = draft_model
        self.attns = [layer.self_attn for layer in draft_model.layers]
        mla = len(self.attns) > 0 and all(
            type(a) is DFlashMLAAttention and a.q_lora_rank is not None for a in self.attns
        )
        # Context rows reach every layer unchanged only through the base hook.
        mla = mla and (
            type(draft_model).prepare_context_hidden_for_kv
            is DFlashDraftModel.prepare_context_hidden_for_kv
        )
        self.fc_shard = (
            envs.SGLANG_OPT_DFLASH_FC_SHARD.get()
            and not draft_model.quantizable
            and type(draft_model).project_target_hidden
            is DFlashDraftModel.project_target_hidden
        )
        # Largest context batch (rows) the fc shard / context GEMM take; larger (prefill) batches keep the stock path.
        self.max_rows = max(MAX_FUSED_ROWS, int(envs.SGLANG_OPT_DFLASH_CTX_FUSED_MAX_ROWS.get()))
        # The fc shard's own cap (<= max_rows; 0 = max_rows): above it the stock replicated fc runs while the context GEMM
        # still takes rows up to max_rows. The two halves are independent (the context GEMM reads the fc output either way).
        fc_cap = int(envs.SGLANG_OPT_DFLASH_FC_SHARD_MAX_ROWS.get())
        self.fc_max_rows = self.max_rows if fc_cap <= 0 else min(self.max_rows, fc_cap)
        self.ctx_gemm = envs.SGLANG_OPT_DFLASH_CTX_KV_GEMM.get() and mla
        self.ctx_store = envs.SGLANG_OPT_DFLASH_CTX_KV_STORE.get() and mla
        self.ctx_store_norm = (
            self.ctx_store and envs.SGLANG_OPT_DFLASH_CTX_KV_STORE_NORM.get() and _norm_modules_supported(self.attns)
        )
        if envs.SGLANG_OPT_DFLASH_CTX_KV_STORE_NORM.get() and not self.ctx_store_norm:
            logger.warning("DFLASH context store norm fold unsupported here (needs CTX_KV_STORE + plain BF16 RMSNorm(512)).")
        self._norm_ok: Optional[bool] = None
        self._check_n = envs.SGLANG_TEST_DFLASH_CTX_KV_STORE_NORM_CHECK.get()
        self._check_k = 0
        self._check_refs = None
        self.pool = None
        self._store_ok = False
        self._workspace: Optional[torch.Tensor] = None
        self._fc_algo: Dict[int, Optional[torch.Tensor]] = {}
        self._ctx_algo: Dict[int, Optional[torch.Tensor]] = {}
        self._ctx_weight: Optional[torch.Tensor] = None
        self._gatherer = None

    @property
    def enabled(self) -> bool:
        return self.fc_shard or self.ctx_gemm or self.ctx_store

    def writes_mla(self, pool) -> bool:
        """Whether write_context_kv replaces the per-layer MLA context writes into this pool."""
        if pool is not self.pool:
            self.pool = pool
            self._store_ok = self.ctx_store and context_store_supported(pool, self.attns)
            if self.ctx_store and not self._store_ok:
                logger.warning("DFLASH context store fusion unsupported here; stock writes kept.")
        return self.ctx_gemm or self._store_ok

    def _ws(self, device: torch.device) -> torch.Tensor:
        if self._workspace is None:
            self._workspace = torch.empty(
                TORCH_CUBLAS_WORKSPACE_BYTES, dtype=torch.uint8, device=device
            )
        return self._workspace

    # ---- fc shard -------------------------------------------------------------------------------------

    def _fc_pin(self, rows: int, weight: torch.Tensor) -> Optional[torch.Tensor]:
        if rows in self._fc_algo:
            return self._fc_algo[rows]
        if torch.cuda.is_current_stream_capturing():
            return None
        from sglang.srt.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
        n = weight.shape[0] // tp.world_size
        lo = tp.rank_in_group * n
        algo = None
        if weight.shape[0] % tp.world_size == 0 and n % 8 == 0:
            algo = pin_lt_algo(
                rows,
                weight,
                weight[lo : lo + n],
                lambda x, r: r[:, lo : lo + n],
                self._ws(weight.device),
            )
        # Every rank must take the same path: the gather is a collective.
        ok = torch.tensor([int(algo is not None)], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=tp.cpu_group)
        if not int(ok.item()):
            algo = None
            logger.warning("DFLASH fc shard: no bitwise algorithm at %d rows; stock fc kept.", rows)
        if algo is not None and self._gatherer is None:
            from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
                MultimemAllGatherer,
            )

            # The fc-shard gatherer skips its entry barrier, as the target logits gatherer does (logits_processor.py).
            self._gatherer = MultimemAllGatherer(max_tokens=self.fc_max_rows, skip_entry_sync=True)
        self._fc_algo[rows] = algo
        return algo

    def project_fc(self, target_hidden: torch.Tensor) -> Optional[torch.Tensor]:
        """The fc output rows, sharded and gathered; None when the stock fc must run."""
        rows = target_hidden.shape[0]
        if not self.fc_shard or rows == 0 or rows > self.fc_max_rows:
            return None
        weight = self.draft_model.fc.weight
        algo = self._fc_pin(rows, weight)
        if algo is None:
            return None
        from sglang.srt.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
        n = weight.shape[0] // tp.world_size
        lo = tp.rank_in_group * n
        part = lt_matmul(target_hidden, weight[lo : lo + n], algo, self._ws(weight.device))
        return self._gatherer(part)

    # ---- context KV -----------------------------------------------------------------------------------

    def _kv_a_weight(self, attn) -> torch.Tensor:
        return attn.fused_qkv_a_proj_with_mqa.weight[attn.q_lora_rank :]

    def _ctx_pin(self, rows: int) -> Optional[torch.Tensor]:
        if rows in self._ctx_algo:
            return self._ctx_algo[rows]
        if torch.cuda.is_current_stream_capturing():
            return None
        views = [self._kv_a_weight(a) for a in self.attns]
        if self._ctx_weight is None:
            self._ctx_weight = torch.cat(views, 0).contiguous()
        algo = pin_lt_algo(
            rows,
            views[0],
            self._ctx_weight,
            lambda x, r: torch.cat([F.linear(x, w) for w in views], 1),
            self._ws(views[0].device),
        )
        if algo is None:
            logger.warning("DFLASH context GEMM: no bitwise algorithm at %d rows; stock kept.", rows)
        self._ctx_algo[rows] = algo
        return algo

    def _latents(self, ctx_hidden: torch.Tensor) -> List[torch.Tensor]:
        rows = ctx_hidden.shape[0]
        if self.ctx_gemm and rows <= self.max_rows:
            algo = self._ctx_pin(rows)
            if algo is not None:
                out = lt_matmul(ctx_hidden, self._ctx_weight, algo, self._ws(ctx_hidden.device))
                width = _NOPE + _ROPE
                return [out[:, i * width : (i + 1) * width] for i in range(len(self.attns))]
        return [F.linear(ctx_hidden, self._kv_a_weight(a)) for a in self.attns]

    # ---- context store norm fold ------------------------------------------------------------------------

    def _norm_probe(self, device: torch.device):
        """Bitwise: the fold's BF16 rows == the served kv_a_layernorm modules' outputs on probe latents (no KV writes:
        every loc is 0)."""
        rows, layers = MAX_FUSED_ROWS, len(self.attns)
        g = torch.Generator(device=device).manual_seed(20261101)
        norms = [a.kv_a_layernorm for a in self.attns]
        bufs = [self.pool.kv_buffer[a.attn_mqa.layer_id - self.pool.start_layer] for a in self.attns]
        bufs = [b.view(b.shape[0], -1) for b in bufs]
        zeros = torch.zeros(rows, dtype=torch.int64, device=device)
        bad = 0
        for trial in range(16):
            scale = torch.exp2(torch.randint(-12, 12, (rows, 1), generator=g, device=device).float())
            lat = (torch.randn(rows, layers * (_NOPE + _ROPE), generator=g, device=device) * scale).to(torch.bfloat16)
            lats = [lat[:, i * (_NOPE + _ROPE) : (i + 1) * (_NOPE + _ROPE)] for i in range(layers)]
            ref = [n(lt[..., :_NOPE]) for n, lt in zip(norms, lats)]
            dbg = torch.empty(layers, rows, _NOPE, dtype=torch.bfloat16, device=device)
            dflash_ctx_kv_store_norm(
                lats, [n.weight for n in norms], [n.variance_epsilon for n in norms], bufs,
                [a.rotary_emb.cos_sin_cache for a in self.attns], zeros, zeros, dbg_out=dbg,
            )  # fmt: skip
            bad += sum(int((dbg[i] != ref[i]).sum().item()) for i in range(layers))
        return bad == 0, bad

    def _use_norm_fold(self, device: torch.device) -> bool:
        want = self.ctx_store_norm
        if not want:
            return False
        if self._norm_ok is None:
            if torch.cuda.is_current_stream_capturing():
                return False
            from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

            if is_batch_invariant_mode_enabled():
                self._norm_ok = False
                logger.warning("DFLASH context store norm fold off: batch-invariant mode.")
                return False
            ok, bad = self._norm_probe(device)
            if ok:
                logger.info("DFLASH context store norm fold ON (bitwise probe: 0 differing values)")
            else:
                logger.warning("DFLASH context store norm fold OFF: probe differs in %d values; stock norms kept.", bad)
            self._norm_ok = ok
        return self._norm_ok

    def take_check_refs(self):
        """Test hook: the tensors the last captured fold wrote from (for check_after_replay), then cleared."""
        refs, self._check_refs = self._check_refs, None
        return refs

    def check_after_replay(self, refs) -> None:
        """Test hook SGLANG_TEST_DFLASH_CTX_KV_STORE_NORM_CHECK: after a replay of a graph whose capture took the fold, compare the
        stored k_nope FP8 bytes of every committed row with the served kv_a_layernorm + torch's FP8 cast of the same latent."""
        if not self._check_n or self._check_k >= self._check_n or refs is None:
            return
        lats, loc, commit_lens, bufs = refs
        rows = lats[0].shape[0]
        dev = lats[0].device
        if commit_lens is not None:
            width = loc.shape[1]
            r = torch.arange(rows, device=dev)
            valid = (r % width) < commit_lens.long()[r // width]
        else:
            valid = torch.ones(rows, dtype=torch.bool, device=dev)
        slots = loc.reshape(-1)
        valid &= slots != 0
        idx = valid.nonzero().squeeze(1)
        self._check_k += 1
        k = self._check_k
        bad_rows, first = 0, None
        for li, (a, lt, buf) in enumerate(zip(self.attns, lats, bufs)):
            exp = a.kv_a_layernorm(lt[..., :_NOPE]).to(torch.float8_e4m3fn).view(torch.uint8)[idx]
            act = buf[slots[idx]][:, :_NOPE]
            diff = (exp != act).any(dim=1)
            n = int(diff.sum().item())
            if n and first is None:
                j = int(diff.nonzero()[0, 0].item())
                col = int((exp[j] != act[j]).nonzero()[0, 0].item())
                first = f"layer {li} row {int(idx[j].item())} slot {int(slots[idx[j]].item())} col {col} exp {int(exp[j, col].item())} got {int(act[j, col].item())}"
            bad_rows += n
        if bad_rows:
            logger.info("DFLASH ctx norm check %d/%d: MISMATCH rows %d of %d committed x %d layers; first %s", k, self._check_n,
                        bad_rows, int(idx.numel()), len(self.attns), first)  # fmt: skip
        elif k in (1, 2, 10, 50) or k % 100 == 0 or k == self._check_n:
            logger.info("DFLASH ctx norm check %d/%d: OK (%d committed rows x %d layers)", k, self._check_n, int(idx.numel()), len(self.attns))

    def write_context_kv(
        self,
        ctx_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """Every MLA draft layer's context rows, as DFlashMLAAttention.write_context_kv writes them."""
        latents = self._latents(ctx_hidden)
        if self._store_ok and ctx_hidden.shape[0] > 0 and self._use_norm_fold(ctx_hidden.device):
            pool = self.pool
            buffers = []
            for a in self.attns:
                buf = pool.kv_buffer[a.attn_mqa.layer_id - pool.start_layer]
                buffers.append(buf.view(buf.shape[0], -1))
            loc = cache_loc_2d if cache_loc_2d is not None else cache_loc
            cl = commit_lens if cache_loc_2d is not None else None
            dflash_ctx_kv_store_norm(
                latents,
                [a.kv_a_layernorm.weight for a in self.attns],
                [a.kv_a_layernorm.variance_epsilon for a in self.attns],
                buffers,
                [a.rotary_emb.cos_sin_cache for a in self.attns],
                positions,
                loc,
                cl,
            )
            if self._check_n and torch.cuda.is_current_stream_capturing():
                self._check_refs = (latents, loc, cl, buffers)
            return
        k_nope = [a.kv_a_layernorm(lt[..., :_NOPE]) for a, lt in zip(self.attns, latents)]
        if self._store_ok and ctx_hidden.shape[0] > 0:
            pool = self.pool
            buffers = []
            for a in self.attns:
                buf = pool.kv_buffer[a.attn_mqa.layer_id - pool.start_layer]
                buffers.append(buf.view(buf.shape[0], -1))
            dflash_ctx_kv_store(
                k_nope,
                latents,
                buffers,
                [a.rotary_emb.cos_sin_cache for a in self.attns],
                positions,
                cache_loc_2d if cache_loc_2d is not None else cache_loc,
                commit_lens if cache_loc_2d is not None else None,
            )
            return
        if cache_loc_2d is not None:
            offsets = torch.arange(cache_loc_2d.shape[1], device=cache_loc_2d.device)
            loc = torch.where(offsets < commit_lens[:, None], cache_loc_2d, 0).reshape(-1)
        else:
            loc = cache_loc
        for a, lt, kn in zip(self.attns, latents, k_nope):
            k_pe = lt[..., _NOPE:].unsqueeze(1)
            _, k_pe = a.rotary_emb(positions, torch.empty_like(k_pe), k_pe)
            self.pool.set_mla_kv_buffer(
                a.attn_mqa, KVWriteLoc(loc, physical=True), kn.unsqueeze(1), k_pe
            )
