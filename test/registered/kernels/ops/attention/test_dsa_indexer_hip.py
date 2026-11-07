"""Kimi-K3.1's ROCm DSA indexer hooks (dsa_indexer_hip) against their plain counterparts.

- Store under DCP: a rank's index-K equals a DCP1 store of only the tokens it owns
  (slot % N == rank) at row slot // N, as with the CUDA fused store.
- Target verify: the shared-index-K logits equal the per-row aiter call on every
  row's keys, layer after layer and forward after forward.
- NoPE prologue: quantized queries, head weights and stored keys match the generic path.
- FP4 index-K: AITER's row-group logits read the keys and queries our kernels write.
"""

import contextlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.layers.attention.dsa.dsa_indexer as dsa_indexer
import sglang.srt.layers.attention.dsa.dsa_indexer_hip as dsa_indexer_hip
import sglang.srt.layers.attention.dsa.indexer_prologue_hip as prologue_hip
from sglang.kernels.ops.attention.dsa import aiter_paged_mqa_logits
from sglang.kernels.ops.attention.dsa.fp4_index_hip import (
    fp4_paged_mqa_logits,
    fp4_ragged_mqa_logits,
    gather_fp4_index_k_rows,
    indexer_q_fp4_head_gate,
    stage_fp4_index_k_rows,
    store_fp4_index_k,
)
from sglang.kernels.ops.attention.dsa.head_gate import indexer_q_quant_head_gate
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.srt.layers.dcp import layout as dcp_layout
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase, published_topology

# backend-specific: AITER's preshuffled index-K, paged-MQA logits and NoPE prologue.
register_amd_ci(est_time=50, suite="stage-b-test-1-gpu-small-amd-mi35x")

PAGE, HEADS, HEAD_DIM = 64, 32, 128
ROWS = 4096
HIDDEN, Q_LORA = 7168, 1536


def _make_pool() -> DSATokenToKVPool:
    return DSATokenToKVPool(
        size=ROWS,
        page_size=PAGE,
        kv_lora_rank=512,
        dtype=torch.float8_e4m3fn,
        qk_rope_head_dim=64,
        layer_num=1,
        device="cuda",
        index_head_dim=HEAD_DIM,
        enable_memory_saver=False,
        kv_cache_dim=656,
    )


class _IndexerStub:
    _store_index_k_cache = Indexer._store_index_k_cache
    block_size, scale_fmt = 128, "ue8m0"
    fp4_index_k = False


def _store(pool, loc, key) -> torch.Tensor:
    backend = SimpleNamespace(token_to_kv_pool=pool)
    with forward_context(ForwardContext(attn_backend=backend)):
        _IndexerStub()._store_index_k_cache(SimpleNamespace(out_cache_loc=loc), 0, key)
    return pool.get_index_k_with_scale_buffer(0)


@unittest.skipUnless(
    dsa_indexer._use_aiter_preshuffle, "needs SGLANG_USE_AITER=1 and preshuffle"
)
class TestDSAIndexerHip(CustomTestCase):
    def test_dcp_store_keeps_this_ranks_tokens(self):
        dcp_size = 8
        gen = torch.Generator(device="cuda").manual_seed(0)
        loc = torch.randperm(ROWS * dcp_size, device="cuda", generator=gen)[:700]
        loc[::50] = -1  # padded tokens write nothing
        key = torch.randn(loc.numel(), HEAD_DIM, device="cuda", generator=gen)
        key = key.bfloat16()
        dcp_layout._dsa_shards_index_k = True
        try:
            for rank in (0, dcp_size - 1):
                with self.subTest(rank=rank):
                    with published_topology(tp_size=dcp_size, dcp_size=dcp_size):
                        with get_parallel().override(attn_dcp_rank=rank, dcp_rank=rank):
                            got = _store(_make_pool(), loc, key)
                    owned = (loc >= 0) & (loc % dcp_size == rank)
                    with published_topology():
                        ref = _store(_make_pool(), loc[owned] // dcp_size, key[owned])
                    self.assertTrue(ref.any(), "no index-K row was stored")
                    self.assertTrue(torch.equal(got, ref))
        finally:
            dcp_layout._dsa_shards_index_k = False

    def test_verify_logits_match_per_row_call(self):
        num_reqs, next_n, req_pages = 3, 4, 8
        q_offset = num_reqs * next_n
        max_seq_len = req_pages * PAGE
        gen = torch.Generator(device="cuda").manual_seed(1)
        with published_topology():
            pool = _make_pool()
        key = torch.randn(ROWS, HEAD_DIM, device="cuda", generator=gen).bfloat16()
        kv = _store(pool, torch.arange(ROWS, device="cuda"), key)
        kv = kv.view(kv.shape[0], PAGE, 1, HEAD_DIM + 4)
        # The same batch object every forward: only the layer order tells them apart.
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        for step in range(2):
            pages = torch.randperm(ROWS // PAGE, device="cuda", generator=gen)
            tables = pages[: num_reqs * req_pages].view(num_reqs, req_pages).int()
            tables = tables.repeat_interleave(next_n, dim=0)
            last = torch.randint(
                next_n, max_seq_len + 1, (num_reqs, 1), device="cuda", generator=gen
            )
            lens = (last - next_n + 1 + torch.arange(next_n, device="cuda")).view(-1)
            lens = lens.int()
            for layer_id in range(2):
                q = torch.randn(q_offset, HEADS, HEAD_DIM, device="cuda", generator=gen)
                q = q.to(torch.float8_e4m3fn)
                w = torch.randn(q_offset, HEADS, device="cuda", generator=gen)
                self.assertTrue(
                    dsa_indexer_hip.verify_shares_index_k(
                        forward_batch, q, tables, PAGE, q_offset, num_reqs, next_n, 1
                    )
                )
                shared = dsa_indexer_hip.verify_paged_mqa_logits(
                    forward_batch,
                    layer_id,
                    q,
                    kv,
                    w,
                    lens.view(-1, 1),
                    tables,
                    max_seq_len,
                    q_offset,
                    num_reqs,
                    next_n,
                    PAGE,
                )
                per_row = aiter_paged_mqa_logits(
                    q,
                    kv,
                    w,
                    lens,
                    tables,
                    max_seq_len,
                    preshuffle=True,
                    kv_block_size=PAGE,
                )
                for row, n in enumerate(lens.tolist()):
                    with self.subTest(step=step, layer=layer_id, row=row):
                        self.assertTrue(torch.equal(shared[row, :n], per_row[row, :n]))


def _exact_linear(x, weight):
    # fp64 accumulation: the bf16 result does not depend on the GEMM's N or blocking
    return (x.double() @ weight.double().t()).bfloat16()


class _Linear(torch.nn.Module):
    """GEMM stand-in, deterministic (AITER's split-K bf16 GEMMs reduce with atomics)
    and N-independent, so a merged [wk | weights_proj] GEMM matches the separate ones."""

    def __init__(self, n: int, k: int, gen: torch.Generator):
        super().__init__()
        self.weight = (
            torch.randn(n, k, device="cuda", generator=gen) * 0.02
        ).bfloat16()
        self.quant_method = SimpleNamespace(
            apply=lambda layer, x: _exact_linear(x, layer.weight)
        )

    def forward(self, x):
        return _exact_linear(x, self.weight), None


def _kimi_k31_indexer() -> Indexer:
    config = SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        index_topk=2048,
        index_rope_dim=0,
        index_apply_hadamard=False,
        index_k_norm_eps=1e-5,
    )
    with published_topology():
        indexer = Indexer(
            hidden_size=HIDDEN,
            index_n_heads=HEADS,
            index_head_dim=HEAD_DIM,
            rope_head_dim=64,
            index_topk=2048,
            q_lora_rank=Q_LORA,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            layer_id=0,
            scale_fmt="ue8m0",
            block_size=128,
            config=config,
        ).cuda()
    gen = torch.Generator(device="cuda").manual_seed(1)
    indexer.wq_b = _Linear(HEADS * HEAD_DIM, Q_LORA, gen)
    indexer.wk = _Linear(HEAD_DIM, HIDDEN, gen)
    indexer.weights_proj = _Linear(HEADS, HIDDEN, gen)
    with torch.no_grad():
        indexer.k_norm.weight.copy_(torch.rand(HEAD_DIM, generator=gen, device="cuda"))
        indexer.k_norm.bias.copy_(torch.randn(HEAD_DIM, generator=gen, device="cuda"))
    return indexer


@unittest.skipUnless(
    dsa_indexer._use_aiter and dsa_indexer._is_gfx95_supported,
    "needs SGLANG_USE_AITER=1 on gfx950",
)
class TestNoPEPrologueHip(CustomTestCase):
    def test_q_quant_head_gate_matches_tilelang_for_every_amax(self):
        """Every positive bf16 amax (ue8m0 power-of-two edges included) quantizes like
        tilelang act_quant and the indexer's bf16 head gate, the generic ROCm path."""
        from sglang.kernels.ops.quantization.act_quant_tilelang import act_quant

        # 0x0080..0x7F7F: every normal bf16, one head row each (32512 = 1016 x 32 heads)
        amax = torch.arange(0x0080, 0x7F80, dtype=torch.int16, device="cuda")
        amax = amax.view(torch.bfloat16).float()
        rows = amax.numel()
        gen = torch.Generator(device="cuda").manual_seed(0)
        query = torch.rand(rows, HEAD_DIM, device="cuda", generator=gen) * 2 - 1
        query = (query * amax[:, None]).bfloat16()
        sign = 1 - 2 * (torch.arange(rows, device="cuda") % 2)
        query[:, 0] = amax.bfloat16() * sign
        query = query.view(rows // HEADS, HEADS, HEAD_DIM)
        gate = torch.randn(rows // HEADS, HEADS + 128, device="cuda", generator=gen)
        gate = gate.bfloat16()[:, 128:]
        for scale_fmt in (None, "ue8m0"):
            q8, q_scale = act_quant(query, 128, scale_fmt)
            gate_bf16 = (gate * HEADS**-0.5).float()  # Indexer._bf16_head_gate
            weights = gate_bf16.unsqueeze(-1) * q_scale * HEAD_DIM**-0.5
            fused_q8, fused_weights = indexer_q_quant_head_gate(
                query, gate, HEAD_DIM**-0.5, scale_fmt
            )
            same_q8 = torch.equal(fused_q8.view(torch.uint8), q8.view(torch.uint8))
            self.assertTrue(same_q8, f"q_fp8 differs, scale_fmt={scale_fmt}")
            same_w = torch.equal(
                fused_weights.view(torch.int32), weights.view(torch.int32)
            )
            self.assertTrue(same_w, f"head weights differ, scale_fmt={scale_fmt}")

    def _forward(self, indexer, pool, mode, x, q_lora, loc, rocm_prologue):
        captured = {}

        def capture_topk(forward_batch, layer_id, q_fp8, weights, metadata):
            captured.update(q_fp8=q_fp8.clone(), weights=weights.clone())
            return torch.zeros(q_fp8.shape[0], 1, dtype=torch.int32, device="cuda")

        num_tokens = loc.numel()
        seq_lens = torch.full((num_tokens,), 3000, dtype=torch.int32, device="cuda")
        forward_batch = SimpleNamespace(
            forward_mode=mode,
            out_cache_loc=loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.cpu(),
            attn_cp_metadata=None,
        )
        backend = SimpleNamespace(
            token_to_kv_pool=pool, get_indexer_metadata=lambda *_: object()
        )
        spy = patch.object(
            prologue_hip, "indexer_q_quant_head_gate", wraps=indexer_q_quant_head_gate
        )
        generic = patch.object(indexer, "rocm_nope_prologue", None)
        with forward_context(ForwardContext(attn_backend=backend)), spy as fused:
            with (
                patch.object(indexer, "_get_topk_paged", capture_topk),
                contextlib.nullcontext() if rocm_prologue else generic,
            ):
                indexer.forward_cuda(
                    x, q_lora, torch.arange(num_tokens, device="cuda"), forward_batch, 0
                )
        self.assertEqual(fused.called, rocm_prologue)
        torch.cuda.synchronize()
        return captured

    def test_nope_prologue_matches_generic_path(self):
        """Decode and verify: q_fp8, head weights and the stored index-K (a DCP rank's
        share too) equal the generic path's, for a tensor or AITER's 3-tuple input,
        with wk / weights_proj separate or merged by prepare_merged_k_gate."""
        gen = torch.Generator(device="cuda").manual_seed(2)
        cases = [
            (ForwardMode.DECODE, 5, 1, False),
            (ForwardMode.TARGET_VERIFY, 24, 1, True),
            (ForwardMode.TARGET_VERIFY, 48, 8, False),
        ]
        for merged in (False, True):
            indexer = _kimi_k31_indexer()
            self.assertIsNotNone(indexer.rocm_nope_prologue)
            if merged:
                indexer.prepare_merged_k_gate()
                self.assertIsNotNone(indexer.rocm_nope_prologue.k_gate_weight)
            for mode, num_tokens, dcp_size, tuple_input in cases:
                # Swap input forms for merged weights; keep all four combinations.
                tuple_input ^= merged
                with self.subTest(
                    merged=merged,
                    mode=mode,
                    tokens=num_tokens,
                    dcp=dcp_size,
                    tuple_input=tuple_input,
                ):
                    x = torch.randn(num_tokens, HIDDEN, device="cuda", generator=gen)
                    x = x.bfloat16()
                    x[1] *= 1e-3
                    q_lora = torch.randn(
                        num_tokens, Q_LORA, device="cuda", generator=gen
                    ).bfloat16()
                    q_lora[0] = 0  # amax floor of every head
                    if tuple_input:
                        # AITER's fused RMSNorm output; the fp8 half is never read
                        x = (x.view(torch.float8_e4m3fn)[:, :HIDDEN], None, x)
                    loc = torch.randperm(ROWS * dcp_size, device="cuda", generator=gen)
                    loc = loc[:num_tokens]
                    self._check_same(indexer, mode, x, q_lora, loc, dcp_size)

    def _check_same(self, indexer, mode, x, q_lora, loc, dcp_size):
        rank = dcp_size - 1
        dcp_layout._dsa_shards_index_k = dcp_size > 1
        try:
            with published_topology(tp_size=dcp_size, dcp_size=dcp_size):
                with get_parallel().override(attn_dcp_rank=rank, dcp_rank=rank):
                    generic_pool, prologue_pool = _make_pool(), _make_pool()
                    generic = self._forward(
                        indexer, generic_pool, mode, x, q_lora, loc, False
                    )
                    prologue = self._forward(
                        indexer, prologue_pool, mode, x, q_lora, loc, True
                    )
        finally:
            dcp_layout._dsa_shards_index_k = False
        q8, generic_q8 = prologue["q_fp8"], generic["q_fp8"]
        self.assertTrue(torch.equal(q8.view(torch.uint8), generic_q8.view(torch.uint8)))
        weights, generic_weights = prologue["weights"], generic["weights"]
        self.assertTrue(
            torch.equal(weights.view(torch.int32), generic_weights.view(torch.int32))
        )
        generic_k = generic_pool.get_index_k_with_scale_buffer(0)
        prologue_k = prologue_pool.get_index_k_with_scale_buffer(0)
        self.assertTrue(generic_k.any(), "no index-K row was stored")
        self.assertTrue(torch.equal(prologue_k, generic_k), "stored index-K differs")


_E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _fp4_values(codes: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 bytes [..., 64] and ue8m0 exponents [..., 4] -> [..., 128] fp32."""
    lut = torch.tensor(_E2M1, device=codes.device)
    nibbles = torch.stack([codes.long() & 15, codes.long() >> 4], -1).flatten(-2)
    return lut[nibbles] * torch.exp2(exps.float() - 127).repeat_interleave(32, -1)


@unittest.skipUnless(
    dsa_indexer._use_aiter and dsa_indexer._is_gfx95_supported,
    "needs SGLANG_USE_AITER=1 on gfx950",
)
class TestFp4IndexKHip(CustomTestCase):
    def test_decode_and_prefill_logits_read_the_stored_fp4_values(self):
        """AITER's decode (row-group) and prefill logits over keys and queries our FP4
        kernels wrote, the prefill keys gathered and staged back into pages, equal torch
        scores of the same FP4 values: a layout drift on either side breaks them."""
        gen = torch.Generator(device="cuda").manual_seed(0)
        lens = torch.tensor([300, 2500, 1, 1730], dtype=torch.int32)
        batch, next_n = lens.numel(), 5
        pages = (lens + PAGE - 1) // PAGE
        perm = torch.randperm(
            int(pages.sum()) + 3, generator=torch.Generator().manual_seed(0)
        )
        table = torch.zeros(batch, int(pages.max()), dtype=torch.int32)
        for b, first in enumerate(torch.cumsum(pages, 0) - pages):
            table[b, : pages[b]] = perm[first : first + pages[b]].int()
        table, lens_d = table.cuda(), lens.cuda()
        payload = torch.zeros(perm.numel(), PAGE * 64, dtype=torch.uint8, device="cuda")
        scale = torch.zeros(perm.numel(), PAGE * 4, dtype=torch.uint8, device="cuda")

        channel = torch.exp(0.5 * torch.randn(HEAD_DIM, device="cuda", generator=gen))
        keys, locs = [], []
        for b in range(batch):
            rows = torch.arange(int(lens[b]), device="cuda")
            keys.append(
                (
                    torch.randn(rows.numel(), HEAD_DIM, device="cuda", generator=gen)
                    * channel
                ).bfloat16()
            )
            locs.append(table[b, rows // PAGE].long() * PAGE + rows % PAGE)
        key, loc = torch.cat(keys), torch.cat(locs)
        # other DCP ranks' tokens (-1) between this rank's write nothing
        store_fp4_index_k(
            torch.stack([key, torch.randn_like(key)], 1).flatten(0, 1),
            payload,
            scale,
            torch.stack([loc, torch.full_like(loc, -1)], 1).flatten(),
        )
        rows = gather_fp4_index_k_rows(
            payload, scale, table, lens_d, int(lens.sum()), int(lens.max())
        )
        key_fp4 = _fp4_values(rows[:, :64], rows[:, 64:])
        group_err = (key_fp4 - key.float()).view(-1, 4, 32).abs().amax(-1)
        group_amax = key.float().view(-1, 4, 32).abs().amax(-1)
        self.assertTrue(
            bool((group_err <= group_amax / 4 + 1e-6).all()), "keys not FP4-rounded"
        )

        query = torch.randn(
            batch * next_n, HEADS, HEAD_DIM, device="cuda", generator=gen
        ).bfloat16()
        gate = torch.randn(
            batch * next_n, HEADS, device="cuda", generator=gen
        ).bfloat16()
        q, weights = indexer_q_fp4_head_gate(query, gate, HEAD_DIM**-0.5)
        heads = torch.arange(HEADS, device="cuda")
        q_exps = q.scale.view(-1, 4, 16, 4)[:, :, heads % 16, heads // 16].transpose(
            1, 2
        )
        query_fp4 = _fp4_values(q.payload, q_exps)
        ends = (
            (lens[:, None] - next_n + 1 + torch.arange(next_n))
            .clamp_min(1)
            .flatten()
            .cuda()
            .int()
        )
        starts = torch.arange(
            0, batch * next_n + 1, next_n, dtype=torch.int32, device="cuda"
        )
        logits = fp4_paged_mqa_logits(
            q,
            weights.squeeze(-1),
            payload,
            scale,
            table,
            starts,
            ends,
            table.shape[1] * PAGE,
        )
        # prefill reads a bf16 gate and takes the softmax scale as its weight scale
        first = (torch.cumsum(lens, 0) - lens).cuda().int()
        ks = first.repeat_interleave(next_n)
        q_prefill, gate_bf16 = indexer_q_fp4_head_gate(query, gate, 1.0, torch.bfloat16)
        prefill = fp4_ragged_mqa_logits(
            q_prefill,
            gate_bf16.squeeze(-1),
            HEAD_DIM**-0.5,
            stage_fp4_index_k_rows(rows),
            ks,
            ks + ends,
        )
        for r in range(batch * next_n):
            start, end = int(ks[r]), int(ends[r])
            ref = (
                torch.relu(query_fp4[r] @ key_fp4[start : start + end].T) * weights[r]
            ).sum(0)
            tol = dict(rtol=1e-5, atol=1e-5 * float(ref.abs().max()))
            torch.testing.assert_close(logits[r, :end], ref, **tol)
            torch.testing.assert_close(prefill[r, start : start + end], ref, **tol)


if __name__ == "__main__":
    sys.exit(unittest.main())
