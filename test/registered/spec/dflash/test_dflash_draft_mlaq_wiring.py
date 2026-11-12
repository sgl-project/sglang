# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_DRAFT_MLAQ wiring in DFlashWorkerV2: DraftMlaQuery writes the draft KV pool, which ModelRunner creates only in
alloc_memory_pool. The test runs the real DFlashWorkerV2.__init__ (through its draft-module install block) and the real
DFlashWorkerV2.alloc_memory_pool in the scheduler's order (init_memory_pools, then init_all_cuda_graphs) on a real ModelRunner
instance, real DFlashMLAAttention layers and a real FP8 MLATokenToKVPool. It checks that __init__ survives without a pool, that
alloc_memory_pool arms each MLA layer on that layer's rows of the draft pool (not another pool), that the armed forward stores the
same KV bytes as the stock rotary + set_kv_buffer path, and that a graph captured after alloc_memory_pool compiles nothing."""

import contextlib
import types
import unittest
from unittest import mock

import torch

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")

DEV = "cuda"
HEADS, EPS, SLOTS, PAGE, LAYERS = 6, 1e-6, 4096, 64, 5
MLA_LAYERS = (0, 2)  # draft layer ids that carry DFlashMLAAttention
ROWS = 8


class _InitReachedSelector(Exception):
    """Raised by the fake draft model when __init__ reads candidate_selector, the first line after the install block."""


def _rope():
    from sglang.srt.layers.rotary_embedding.factory import get_rope_wrapper

    rope_scaling = {"rope_type": "deepseek_yarn", "factor": 32.0, "original_max_position_embeddings": 4096,
                    "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0}  # fmt: skip
    return get_rope_wrapper(64, rotary_dim=64, max_position=8192, base=50000, rope_scaling=rope_scaling,
                            is_neox_style=False, device=DEV).to(DEV)  # fmt: skip


def _pool(gen):
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    pool = MLATokenToKVPool(size=SLOTS, page_size=PAGE, dtype=torch.float8_e4m3fn, kv_lora_rank=512, qk_rope_head_dim=64,
                            layer_num=LAYERS, device=DEV, enable_memory_saver=False, start_layer=0)  # fmt: skip
    for buf in pool.kv_buffer:
        buf.copy_(torch.randint(0, 256, buf.shape, dtype=torch.uint8, device=DEV, generator=gen))
    return pool


def _attn(layer_id, rope, gen):
    """A DFlashMLAAttention with the TP16 rank-0 MLA shapes of the K3.1 drafter and random weights."""
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.models.dflash import DFlashMLAAttention

    a = DFlashMLAAttention.__new__(DFlashMLAAttention)
    torch.nn.Module.__init__(a)
    qa, kva = RMSNorm(1536, eps=EPS).to(DEV), RMSNorm(512, eps=EPS).to(DEV)
    qa.weight = torch.nn.Parameter((1 + 0.1 * torch.randn(1536, generator=gen, device=DEV)).bfloat16(), requires_grad=False)
    kva.weight = torch.nn.Parameter((1 + 0.1 * torch.randn(512, generator=gen, device=DEV)).bfloat16(), requires_grad=False)
    a.q_a_layernorm, a.kv_a_layernorm, a.rotary_emb = qa, kva, rope
    a.q_b_proj = types.SimpleNamespace(weight=(0.02 * torch.randn(HEADS * 192, 1536, generator=gen, device=DEV)).bfloat16())
    w_uk = (0.05 * torch.randn(HEADS, 128, 512, generator=gen, device=DEV)).bfloat16()
    for k, v in dict(q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128, dcp_size=1,
                     num_local_heads=HEADS, layer_id=layer_id, w_kc=w_uk.transpose(1, 2).contiguous().transpose(1, 2),
                     current_attention_backend="triton", _draft_mlaq=None, _kv_cache_saved_early=False,
                     _fused_kv_write_rope_rows=None, fused_kv_write=False, fused_kv_write_pdl=False,
                     _q8kv8_qprep_overlap_pending=False).items():  # fmt: skip
        object.__setattr__(a, k, v)
    a.attn_mqa = types.SimpleNamespace(layer_id=layer_id, k_scale=None)
    return a


class _FakeDraftModel:
    def __init__(self, attns):
        self.layers = [types.SimpleNamespace(self_attn=a) for a in attns]

    @property
    def candidate_selector(self):
        raise _InitReachedSelector()


class _DraftTpWorker:
    """Stands in for the draft TpModelWorker: alloc_memory_pool gives the real ModelRunner its token_to_kv_pool, the one attribute
    ModelRunner.alloc_memory_pool sets that DraftMlaQuery reads (model_runner.py: self.token_to_kv_pool = result.token_to_kv_pool)."""

    def __init__(self, runner, make_pool):
        self.runner, self.make_pool = runner, make_pool

    def alloc_memory_pool(self, memory_pool_config=None, req_to_token_pool=None, token_to_kv_pool_allocator=None):
        self.runner.token_to_kv_pool = self.make_pool()


class TestDraftMlaqWiring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    @classmethod
    def tearDownClass(cls):
        reset_context()

    def setUp(self):
        import os

        # The draft KV ring (on by default) reads a model config this stub runner does not have; this test covers the
        # MLA-query pool wiring, so it runs with the ring off.
        self.env = mock.patch.dict(os.environ, {"SGLANG_OPT_DFLASH_DRAFT_MLAQ": "1", "SGLANG_DFLASH_DRAFT_KV_RING": "0"})
        self.env.start()
        for name in ("SGLANG_OPT_DFLASH_DRAFT_SPLITK", "SGLANG_OPT_DFLASH_DRAFT_AR_SITE", "SGLANG_OPT_DFLASH_DRAFT_EARLY_TRIGGER"):
            os.environ.pop(name, None)

    def tearDown(self):
        self.env.stop()

    def _boot(self):
        """Real DFlashWorkerV2.__init__ up to the selector read, with the distributed / config lookups of its prelude stubbed."""
        import sglang.srt.speculative.dflash_worker_v2 as wv2
        from sglang.srt.model_executor.model_runner import ModelRunner

        gen = torch.Generator(device=DEV).manual_seed(20261105)
        rope = _rope()
        attns = [_attn(i, rope, gen) for i in MLA_LAYERS]
        runner = ModelRunner.__new__(ModelRunner)  # a real ModelRunner before alloc_memory_pool: no token_to_kv_pool yet
        draft_worker = _DraftTpWorker(runner, lambda: _pool(gen))
        target_pool = _pool(gen)  # a second plain FP8 pool: arming on it instead of the draft pool must be caught
        target_runner = types.SimpleNamespace(spec_aux_config=types.SimpleNamespace(draft_model_config=None), spec_algorithm=None,
                                              model_config=None, token_to_kv_pool=target_pool)  # fmt: skip
        target = types.SimpleNamespace(model_runner=target_runner, device=DEV, random_seed=0)
        bundle = types.SimpleNamespace(draft_worker=draft_worker, draft_model_runner=runner, draft_model=_FakeDraftModel(attns))
        stubs = dict(
            get_schedule=lambda: types.SimpleNamespace(page_size=PAGE),
            get_spec=lambda: types.SimpleNamespace(speculative_draft_window_size=None),
            get_dp_tp_group=lambda: None,
            SpecTpSync=lambda group: None,
            check_simulate_acc_token_mode=lambda: None,
            model_draft_parallel_spec=lambda *a, **k: None,
            resolve_draft_parallel_layout=lambda *a, **k: None,
            get_parallel=lambda: types.SimpleNamespace(tp_rank=0, pp_group=types.SimpleNamespace(is_last_rank=True)),
            get_pp_stage_load_group=lambda: None,
            draft_pp_context=contextlib.nullcontext,
            draft_tp_context=lambda layout=None: contextlib.nullcontext(),
            build_draft_tp_worker=lambda **kw: bundle,
        )
        worker = wv2.DFlashWorkerV2.__new__(wv2.DFlashWorkerV2)
        with contextlib.ExitStack() as stack:
            for name, fn in stubs.items():
                stack.enter_context(mock.patch.object(wv2, name, fn))
            with self.assertRaises(_InitReachedSelector):  # anything else (the pre-fix AttributeError) fails the test
                worker.__init__(server_args=None, gpu_id=0, nccl_port=0, target_worker=target)
            armed_in_init = [a._draft_mlaq is not None for a in attns]
            # Scheduler.init_memory_pools -> DFlashWorkerV2.alloc_memory_pool (before init_all_cuda_graphs)
            worker.alloc_memory_pool(memory_pool_config=None, req_to_token_pool=None, token_to_kv_pool_allocator=None)
        return worker, runner, attns, target_pool, armed_in_init

    def test_alloc_memory_pool_arms_the_draft_pool_and_writes_it_like_stock(self):
        import sglang.srt.models.dflash as dflash
        import sglang.srt.speculative.dflash_draft_mlaq as dm
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc
        from sgl_kernel import rmsnorm

        worker, runner, attns, target_pool, armed_in_init = self._boot()
        pool = runner.token_to_kv_pool
        self.assertEqual(armed_in_init, [False] * len(attns), "nothing can be armed before the draft pool exists")
        for a in attns:
            self.assertIsNotNone(a._draft_mlaq, f"layer {a.layer_id} not armed after alloc_memory_pool")
            rows = a._draft_mlaq.kv_rows
            self.assertEqual(rows.data_ptr(), pool.kv_buffer[a.layer_id].data_ptr(), f"layer {a.layer_id}: not the draft pool's rows")
            self.assertEqual(tuple(rows.shape), tuple(pool.kv_buffer[a.layer_id].shape))
            self.assertNotEqual(rows.data_ptr(), target_pool.kv_buffer[a.layer_id].data_ptr())

        # The armed forward (forward_absorb_prepare) stores the stock path's KV bytes in the draft pool, and nothing else.
        gen = torch.Generator(device=DEV).manual_seed(7)
        for a in attns:
            lid = a.layer_id
            qkv = (torch.randn(ROWS, 2112, generator=gen, device=DEV) * 2).bfloat16()
            pos = torch.randint(0, 8192, (ROWS,), generator=gen, device=DEV)
            loc = torch.randperm(SLOTS - 1, generator=gen, device=DEV)[:ROWS] + 1
            before = pool.kv_buffer[lid].clone()
            other = target_pool.kv_buffer[lid].clone()
            stock = _pool(torch.Generator(device=DEV).manual_seed(1))
            stock.kv_buffer[lid].copy_(before)
            k_nope = rmsnorm(qkv[:, 1536:2048].contiguous(), a.kv_a_layernorm.weight, EPS).unsqueeze(1)
            q_pe = torch.zeros(ROWS, HEADS, 64, dtype=torch.bfloat16, device=DEV)
            _, k_pe = a.rotary_emb(pos, q_pe, qkv[:, 2048:2112].unsqueeze(1).contiguous())
            stock.set_kv_buffer(types.SimpleNamespace(layer_id=lid), KVWriteLoc(loc), torch.cat([k_nope, k_pe], -1), k_nope)
            calls = a._draft_mlaq.calls
            with mock.patch.object(dflash, "get_attn_tp_context", lambda: types.SimpleNamespace(fetch_qkv_latent=lambda: qkv)):
                a.forward_absorb_prepare(pos, None, types.SimpleNamespace(out_cache_loc=loc), None)
            torch.cuda.synchronize()
            self.assertEqual(a._draft_mlaq.calls, calls + 1, "the forward did not take DraftMlaQuery")
            got, want = pool.kv_buffer[lid], stock.kv_buffer[lid]
            diff = int((got != want).sum())
            self.assertEqual(diff, 0, f"layer {lid}: {diff} KV bytes differ from the stock write")
            self.assertTrue(torch.equal(target_pool.kv_buffer[lid], other), f"layer {lid}: the other pool was written")

        # init_cuda_graphs comes after alloc_memory_pool: a capture then compiles nothing and replays the eager store.
        a = attns[0]
        lid = a.layer_id
        qkv = (torch.randn(ROWS, 2112, generator=gen, device=DEV) * 2).bfloat16()
        pos = torch.randint(0, 8192, (ROWS,), generator=gen, device=DEV)
        loc = torch.randperm(SLOTS - 1, generator=gen, device=DEV)[:ROWS] + 1
        start = pool.kv_buffer[lid].clone()
        a._draft_mlaq.run(qkv, pos, loc)
        torch.cuda.synchronize()
        eager = pool.kv_buffer[lid].clone()
        pool.kv_buffer[lid].copy_(start)
        forms = len(dm._COMPILED)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            a._draft_mlaq.run(qkv, pos, loc)
        self.assertEqual(len(dm._COMPILED), forms, "DraftMlaQuery compiled during capture")
        pool.kv_buffer[lid].copy_(start)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(pool.kv_buffer[lid], eager), "graph replay store differs from eager")


if __name__ == "__main__":
    unittest.main(verbosity=2)
