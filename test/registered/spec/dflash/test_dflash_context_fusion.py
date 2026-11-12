# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""DFlash context-stage fusions (speculative/dflash_context_fusion.py) are bitwise equal to the stock path:
the per-layer MLA context-KV writes, the replicated draft fc, and the worker hooks that route to them."""

import functools
import os
import sys
import types
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

HIDDEN, Q_LORA, NOPE, ROPE, LAYERS = 7168, 1536, 512, 64, 5
POOL_SLOTS, MAX_POS = 2048, 8192
DEV = "cuda"


def _rope():
    from sglang.srt.layers.rotary_embedding.factory import get_rope_wrapper

    return get_rope_wrapper(
        ROPE,
        rotary_dim=ROPE,
        max_position=MAX_POS,
        base=50000,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
        is_neox_style=False,
        device=DEV,
    )


def _attn(layer_id: int, gen: torch.Generator):
    """A DFlashMLAAttention carrying only what the context-KV write reads."""
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.dflash import DFlashMLAAttention

    a = DFlashMLAAttention.__new__(DFlashMLAAttention)
    nn.Module.__init__(a)
    a.q_lora_rank, a.kv_lora_rank, a.qk_rope_head_dim = Q_LORA, NOPE, ROPE
    w = torch.randn(Q_LORA + NOPE + ROPE, HIDDEN, device=DEV, generator=gen) * 0.02
    a.fused_qkv_a_proj_with_mqa = types.SimpleNamespace(weight=w.to(torch.bfloat16))
    a.kv_a_layernorm = RMSNorm(NOPE, eps=1e-5).to(DEV, torch.bfloat16)
    with torch.no_grad():
        a.kv_a_layernorm.weight.copy_(1 + 0.1 * torch.randn(NOPE, device=DEV, generator=gen))
    a.rotary_emb = _rope().to(DEV)
    assert a.rotary_emb.cos_sin_cache.is_cuda
    a.attn_mqa = types.SimpleNamespace(layer_id=layer_id)
    return a


def _draft_model(gen: torch.Generator, fc_in: int = 6 * HIDDEN):
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.dflash import DFlashDraftModel

    m = DFlashDraftModel.__new__(DFlashDraftModel)
    nn.Module.__init__(m)
    m.layers = [types.SimpleNamespace(self_attn=_attn(i, gen)) for i in range(LAYERS)]
    m.quantizable = False
    m.fc = nn.Linear(fc_in, HIDDEN, bias=False, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        m.fc.weight.copy_(torch.randn(HIDDEN, fc_in, device=DEV, generator=gen) * 0.02)
    m.hidden_norm = RMSNorm(HIDDEN, eps=1e-5).to(DEV, torch.bfloat16)
    m.fc_tp_columns = None  # replicated fc and no context fc gather: the constructor's single-rank values
    m._context_fc_gatherer = None
    return m


def _pool():
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    return MLATokenToKVPool(
        size=POOL_SLOTS,
        page_size=64,
        dtype=torch.float8_e4m3fn,
        kv_lora_rank=NOPE,
        qk_rope_head_dim=ROPE,
        layer_num=LAYERS,
        device=DEV,
        enable_memory_saver=False,
    )


def _ctx_hidden(rows: int, gen: torch.Generator, kind: str) -> torch.Tensor:
    x = torch.randn(rows, HIDDEN, device=DEV, generator=gen)
    if kind == "large":
        # Context K values beyond the FP8 E4M3 range (448; torch turns >= 480 into NaN).
        x = x * 40.0
    elif kind == "cancel":
        x = x.abs()
        x[:, 1::2] = -x[:, 0::2]
    return x.to(torch.bfloat16)


def _stock_write(model, pool, ctx, positions, loc_2d, commit_lens):
    offsets = torch.arange(loc_2d.shape[1], device=DEV)
    loc = torch.where(offsets < commit_lens[:, None], loc_2d, 0).reshape(-1)
    for layer in model.layers:
        layer.self_attn.write_context_kv(
            pool=pool, positions=positions, hidden_states=ctx, cache_loc=loc
        )


def _snapshot(pool):
    return [buf.clone() for buf in pool.kv_buffer]


def _cases():
    """(rows, bs, commit_lens) with the verify block of 8 rows per request."""
    return [(8, 1, [8]), (8, 1, [1]), (8, 1, [0]), (8, 1, [5]), (16, 2, [3, 8]), (4, 1, [4])]


class TestDFlashContextFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    @classmethod
    def tearDownClass(cls):
        reset_context()

    def _fusion(self, model, *, fc=False, gemm=False, store=False):
        from sglang.srt.environ import envs
        from sglang.srt.speculative.dflash_context_fusion import DFlashContextFusion

        with envs.SGLANG_OPT_DFLASH_FC_SHARD.override(fc), envs.SGLANG_OPT_DFLASH_CTX_KV_GEMM.override(
            gemm
        ), envs.SGLANG_OPT_DFLASH_CTX_KV_STORE.override(store):
            return DFlashContextFusion(model)

    def _run_case(self, model, fusion, rows, bs, commit, kind, seed):
        gen = torch.Generator(device=DEV).manual_seed(seed)
        ctx = _ctx_hidden(rows, gen, kind)
        width = rows // bs
        positions = torch.randint(0, MAX_POS, (rows,), device=DEV, generator=gen)
        # Distinct nonzero slots; slot 0 is the reserved padding slot.
        loc_2d = (torch.randperm(POOL_SLOTS - 1, device=DEV, generator=gen)[:rows] + 1).view(bs, width)
        commit_lens = torch.tensor(commit, dtype=torch.int32, device=DEV)
        ref_pool, got_pool = _pool(), _pool()
        for p in (ref_pool, got_pool):
            for buf in p.kv_buffer:
                buf.copy_(torch.randint(0, 256, buf.shape, dtype=torch.uint8, device=DEV, generator=gen))
        got_pool.kv_buffer = [b.clone() for b in ref_pool.kv_buffer]
        before = [b.clone() for b in ref_pool.kv_buffer]
        _stock_write(model, ref_pool, ctx, positions, loc_2d, commit_lens)
        # The reference itself must have written exactly the committed rows of every layer.
        changed = sum(int((a != b).any(dim=-1).sum()) for a, b in zip(before, ref_pool.kv_buffer))
        self.assertEqual(changed, LAYERS * sum(min(c, rows // bs) for c in commit))
        assert fusion.writes_mla(got_pool)
        fusion.write_context_kv(ctx, positions, loc_2d.reshape(-1), loc_2d, commit_lens)
        torch.cuda.synchronize()
        bad = []
        for l, (r, g) in enumerate(zip(ref_pool.kv_buffer, got_pool.kv_buffer)):
            nd = int((r != g).sum())
            if nd:
                bad.append(f"layer {l} rows {rows} commit {commit} kind {kind}: {nd} bytes differ")
        return bad

    def _matrix(self, **flags):
        gen = torch.Generator(device=DEV).manual_seed(7)
        model = _draft_model(gen)
        fusion = self._fusion(model, **flags)
        seed, bad = 100, []
        for rows, bs, commit in _cases():
            for kind in ("rand", "large", "cancel"):
                seed += 1
                bad += self._run_case(model, fusion, rows, bs, commit, kind, seed)
        self.assertEqual(bad, [], f"{len(bad)} mismatching layer cases")
        return fusion

    def test_context_store_bitwise(self):
        """Fused RoPE + FP8 store == per-layer stock writes, pool bytes incl. the padding slot 0."""
        fusion = self._matrix(store=True)
        self.assertTrue(fusion.ctx_store)

    def test_context_gemm_bitwise(self):
        """One pinned GEMM over the concatenated kv_a weights == per-layer F.linear (stock tail)."""
        fusion = self._matrix(gemm=True)
        self.assertTrue(fusion.ctx_gemm)
        self.assertTrue(all(fusion._ctx_algo[r] is not None for r in (4, 8, 16)))

    def test_context_gemm_and_store_bitwise(self):
        fusion = self._matrix(gemm=True, store=True)
        self.assertTrue(fusion.ctx_gemm and fusion.ctx_store)

    def test_fp8_store_exhaustive(self):
        """Every BF16 bit pattern stored by the kernel == torch's BF16 -> float8_e4m3fn cast."""
        from sglang.kernels.ops.speculative.dflash_context import dflash_ctx_kv_store

        pats = torch.arange(65536, dtype=torch.int32, device=DEV).to(torch.int16).view(torch.bfloat16)
        rows = 65536 // NOPE
        k_nope = pats.view(rows, NOPE).contiguous()
        latent = torch.zeros(rows, NOPE + ROPE, dtype=torch.bfloat16, device=DEV)
        buf = torch.zeros(rows + 1, NOPE + ROPE, dtype=torch.uint8, device=DEV)
        cos_sin = torch.zeros(4, ROPE, dtype=torch.float32, device=DEV)
        cos_sin[:, : ROPE // 2] = 1.0
        positions = torch.zeros(rows, dtype=torch.int64, device=DEV)
        loc = torch.arange(1, rows + 1, dtype=torch.int64, device=DEV)
        dflash_ctx_kv_store([k_nope], [latent], [buf], [cos_sin], positions, loc)
        ref = k_nope.to(torch.float8_e4m3fn).view(torch.uint8)
        got = buf[1:, :NOPE]
        bad = (got != ref).nonzero()
        self.assertEqual(bad.shape[0], 0, f"{bad.shape[0]} patterns differ, first {bad[:4].tolist()}")

    def test_fc_shard_columns_bitwise(self):
        """The pinned algorithm on each TP16 / TP8 column block == the replicated F.linear's columns,
        on fresh inputs the pin search never saw."""
        from sglang.srt.speculative.dflash_context_fusion import pin_lt_algo

        gen = torch.Generator(device=DEV).manual_seed(11)
        w = (torch.randn(HIDDEN, 6 * HIDDEN, device=DEV, generator=gen) * 0.02).to(torch.bfloat16)
        ws = torch.empty(32 << 20, dtype=torch.uint8, device=DEV)
        from sglang.kernels.ops.speculative.dflash_context import lt_matmul

        for rows in (1, 8):
            for tp in (16, 8):
                n = HIDDEN // tp
                for r in (0, tp // 2 + 1, tp - 1):
                    lo = r * n
                    algo = pin_lt_algo(rows, w, w[lo : lo + n], lambda x, ref: ref[:, lo : lo + n], ws)
                    self.assertIsNotNone(algo, f"rows {rows} tp {tp} rank {r}: no bitwise algorithm")
                    for kind in ("rand", "large", "cancel"):
                        x = _ctx_hidden(rows, gen, kind).repeat(1, 6)
                        got = lt_matmul(x, w[lo : lo + n], algo, ws)
                        nd = int((got != F.linear(x, w)[:, lo : lo + n]).sum())
                        self.assertEqual(nd, 0, f"rows {rows} tp {tp} rank {r} {kind}: {nd} cells differ")

    def test_worker_hook_routes_context_write(self):
        """DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc with the fusion == without it (pool bytes),
        through the prefix-valid (cache_loc_2d) and the plain (cache_loc) entries."""
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        gen = torch.Generator(device=DEV).manual_seed(5)
        model = _draft_model(gen, fc_in=2 * HIDDEN)
        fusion = self._fusion(model, gemm=True, store=True)

        def worker(pool, fusion_obj):
            w = types.SimpleNamespace(
                model_runner=types.SimpleNamespace(device=torch.device(DEV)),
                draft_model=model,
                draft_model_runner=types.SimpleNamespace(token_to_kv_pool=pool),
                draft_parallel_layout=types.SimpleNamespace(tp_group=None, disable_cp=False),
                lilicorr=None,
                _use_fused_kv_materialize=False,
                _fused_kv_helper=None,
                _context_fusion=fusion_obj,
            )
            w._project_target_hidden = functools.partial(DFlashWorkerV2._project_target_hidden, w)
            w._append_target_hidden_sequential = functools.partial(
                DFlashWorkerV2._append_target_hidden_sequential, w
            )
            return w

        for two_d in (True, False):
            rows = 8
            target = _ctx_hidden(rows, gen, "rand").repeat(1, 2)
            positions = torch.randint(0, MAX_POS, (rows,), device=DEV, generator=gen)
            loc = torch.randperm(POOL_SLOTS - 1, device=DEV, generator=gen)[:rows] + 1
            commit = torch.tensor([5], dtype=torch.int32, device=DEV)
            ref_pool, got_pool = _pool(), _pool()
            for b in ref_pool.kv_buffer:
                b.zero_()
            got_pool.kv_buffer = [b.clone() for b in ref_pool.kv_buffer]
            for pool, fobj in ((ref_pool, None), (got_pool, fusion)):
                kw = dict(target_hidden=target, cache_loc=loc, positions=positions)
                if two_d:
                    kw.update(cache_loc_2d=loc.view(1, rows), commit_lens=commit)
                DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc(worker(pool, fobj), **kw)
            torch.cuda.synchronize()
            for l, (r, g) in enumerate(zip(ref_pool.kv_buffer, got_pool.kv_buffer)):
                self.assertEqual(int((r != g).sum()), 0, f"two_d {two_d} layer {l}")
            written = sum(int((b != 0).any(dim=-1).sum()) for b in got_pool.kv_buffer)
            self.assertEqual(written, LAYERS * (5 if two_d else rows))

    def test_compact_window_bitwise(self):
        """compact_draft_window == the stock lens (dtype too) and suffix start at window / page edges, and
        _rebuild_compact_draft_cache writes the same draft req->token rows with the fused suffix start."""
        from sglang.kernels.ops.speculative.dflash_context import compact_draft_window
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        gen = torch.Generator(device=DEV).manual_seed(9)
        bad = []
        for window in (1024, 4096):
            for page in (1, 16, 64):
                for dtype in (torch.int64, torch.int32):
                    edge = [0, 1, 2, page - 1, page, window - 1, window, window + 1,
                            window + page - 1, window + page, 2 * window + 3, 200000]
                    rnd = torch.randint(0, 3 * window, (12,), generator=gen, device=DEV).tolist()
                    seq = torch.tensor(edge + rnd, dtype=dtype, device=DEV)
                    w = types.SimpleNamespace(draft_window_size=window, page_size=page, device=torch.device(DEV))
                    ref = DFlashWorkerV2._compute_compact_draft_seq_lens(w, seq)
                    ref_start = seq.to(torch.int64) - ref.to(torch.int64)
                    got, start = compact_draft_window(seq, window, page)
                    if ref.dtype != got.dtype or not torch.equal(ref, got) or not torch.equal(ref_start, start):
                        bad.append(f"window {window} page {page} {dtype}: lens {(ref != got).sum()} start {(ref_start != start).sum()}")
        self.assertEqual(bad, [])

        # The rebuild with the fused suffix start writes what the stock one writes.
        bs, block, ctx = 3, 8, 6000
        prefix = torch.tensor([5, 1500, 4100], dtype=torch.int64, device=DEV)
        target_r2t = torch.randint(1, 1 << 20, (4, ctx), dtype=torch.int32, device=DEV, generator=gen)
        req = torch.tensor([2, 0, 3], dtype=torch.int64, device=DEV)
        loc2d = torch.randint(1, 1 << 20, (bs, block), dtype=torch.int64, device=DEV, generator=gen)
        outs = []
        for fused in (False, True):
            draft_r2t = torch.zeros(4, ctx, dtype=torch.int32, device=DEV)
            w = types.SimpleNamespace(
                draft_window_size=1024, page_size=64, device=torch.device(DEV), _use_triton_compact_rebuild=True,
                draft_model_runner=types.SimpleNamespace(req_to_token_pool=types.SimpleNamespace(req_to_token=draft_r2t)),
                model_runner=types.SimpleNamespace(req_to_token_pool=types.SimpleNamespace(req_to_token=target_r2t)),
            )
            if fused:
                lens, start = compact_draft_window(prefix, 1024, 64)
            else:
                lens, start = DFlashWorkerV2._compute_compact_draft_seq_lens(w, prefix), None
            DFlashWorkerV2._rebuild_compact_draft_cache(
                w, req_pool_indices=req, prefix_lens=prefix, draft_prefix_lens=lens,
                verify_out_cache_loc_2d=loc2d, bs=bs, block_size=block, suffix_start=start,
            )
            outs.append(draft_r2t)
        self.assertTrue(bool((outs[0] != 0).any()))
        self.assertTrue(torch.equal(outs[0], outs[1]))


if __name__ == "__main__":
    unittest.main()
