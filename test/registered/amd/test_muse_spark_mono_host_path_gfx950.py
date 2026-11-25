"""gfx950 Muse Spark host path beside the AMD monokernels, each piece against the path it replaces.

The vocab-head GEMV (head_gemm) against an FP32 reference, the stacked draft KV commit against the per-layer
commit, a head-major KV pool's writes against an NHD pool's, the selector's fused lattice + walk against the torch
lattice and the walk on it, and the selector params staging. (The compact draft rebuild's in-kernel prefix lengths:
test_dflash_overlap_hostsync.)
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kernels.prefix_valid import assert_bytes_equal, make_kv_cache
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="base-b-test-1-gpu-small-amd-mi35x")

DEVICE = "cuda"


def _rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def _fp8_pool(layers, slots, row_dim):
    """An NHD FP8 pool of ``layers`` one-head caches, as the prefix-commit tests build theirs."""
    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.dtype, pool.store_dtype = fp8_dtype, torch.uint8
    pool.start_layer, pool.device = 0, DEVICE
    pool.kv_cache_layout, pool.use_hnd = "nhd", False
    pool.row_dim = pool.v_row_dim = pool.head_dim = pool.v_head_dim = row_dim
    caches = [make_kv_cache(slots, row_dim, fp8_dtype) for _ in range(layers)]
    pool.k_buffer = [k.view(torch.uint8) for k, _ in caches]
    pool.v_buffer = [v.view(torch.uint8) for _, v in caches]
    return pool


def _mha_pool(slots, heads, dim, head_major):
    """A two-head byte pool, NHD or head-major behind the same [slot][head][dim] tensor."""
    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.dtype = pool.store_dtype = torch.uint8
    pool.start_layer, pool.device = 0, DEVICE
    pool.kv_cache_layout, pool.use_hnd = "nhd", False
    pool.head_num, pool.head_dim, pool.v_head_dim = heads, dim, dim
    pool.row_dim = pool.v_row_dim = heads * dim
    pool.full_head_major = head_major
    # what the NHD row store reads: slots = size + the reserved page
    pool.page_size, pool.size = 16, slots - 16
    pool.device_module, pool.alt_stream = torch.get_device_module(DEVICE), None

    def buffer(fill):
        if head_major:
            flat = torch.full(
                (heads, slots, dim), fill, dtype=torch.uint8, device=DEVICE
            )
            return flat.transpose(0, 1)
        return torch.full((slots, heads, dim), fill, dtype=torch.uint8, device=DEVICE)

    pool.k_buffer, pool.v_buffer = [buffer(0x5A)], [buffer(0xA5)]
    return pool


@unittest.skipUnless(is_gfx95_supported(), "gfx950 (MI35x) only")
class TestMuseSparkMonoHostPathGfx950(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.gen = torch.Generator(device=DEVICE).manual_seed(0)

    def randn(self, *shape):
        return torch.randn(*shape, device=DEVICE, generator=self.gen)

    def test_head_gemm_matches_an_fp32_reference(self):
        from sglang.kernels.ops.gemm import bf16_tiled_small_m_gemm as g

        # the TP8 LM-head shard, a small head, and a row count that is not a multiple of the program's 16
        for n, k in ((25256, 8192), (1000, 1024), (33, 256)):
            w = (self.randn(n, k) * 0.02).to(torch.bfloat16)
            for m in (1, 7, 8, 16):
                x = self.randn(m, k).to(torch.bfloat16)
                ref = x.float() @ w.float().T
                for out_dtype, bound in ((torch.float32, 1e-5), (torch.bfloat16, 1e-2)):
                    with self.subTest(n=n, k=k, m=m, out_dtype=out_dtype):
                        self.assertTrue(g.head_gemm_applies(x, w))
                        y = g.head_gemm(x, w, out_dtype)
                        self.assertEqual((y.dtype, tuple(y.shape)), (out_dtype, (m, n)))
                        self.assertLess(_rel_err(y, ref), bound)

    def test_head_gemm_covers_only_its_operands(self):
        from sglang.kernels.ops.gemm import bf16_tiled_small_m_gemm as g

        w = (self.randn(64, 512) * 0.02).to(torch.bfloat16)
        x = self.randn(8, 512).to(torch.bfloat16)
        self.assertTrue(g.head_gemm_applies(x, w))
        self.assertFalse(g.head_gemm_applies(self.randn(17, 512).to(torch.bfloat16), w))
        self.assertFalse(g.head_gemm_applies(x.float(), w))
        self.assertFalse(g.head_gemm_applies(x[:, :384], w[:, :384]))
        self.assertFalse(g.head_gemm_applies(x, w.float()))
        self.assertFalse(g.head_gemm_applies(x[None], w))

    def test_stacked_prefix_commit_matches_the_per_layer_commit(self):
        layers, slots, block = 3, 48, 4
        loc = torch.tensor(
            [[5, 9, 11, 13], [20, 21, 22, 23], [40, 41, 42, 43]],
            dtype=torch.int64,
            device=DEVICE,
        )
        lengths = torch.tensor([3, 0, 2], dtype=torch.int32, device=DEVICE)
        for row_dim in (128, 2048, 4096):  # the three tile sizes of the launcher
            with self.subTest(row_dim=row_dim):
                stacked = _fp8_pool(layers, slots, row_dim)
                single = _fp8_pool(layers, slots, row_dim)
                # values past the FP8 range take the saturating path
                k = (self.randn(layers, loc.numel(), 1, row_dim) * 200).to(
                    torch.bfloat16
                )
                v = (self.randn(layers, loc.numel(), 1, row_dim) * 200).to(
                    torch.bfloat16
                )
                self.assertTrue(
                    stacked.set_kv_buffer_prefix_valid_layers(
                        tuple(range(layers)), loc, lengths, k, v
                    )
                )
                for i in range(layers):
                    layer = SimpleNamespace(
                        layer_id=i,
                        k_scale=None,
                        v_scale=None,
                        k_scale_float=None,
                        v_scale_float=None,
                    )
                    single.set_kv_buffer_prefix_valid(
                        layer, loc, lengths, k[i].clone(), v[i].clone()
                    )
                    assert_bytes_equal(
                        stacked.k_buffer[i], single.k_buffer[i], f"K of layer {i}"
                    )
                    assert_bytes_equal(
                        stacked.v_buffer[i], single.v_buffer[i], f"V of layer {i}"
                    )
                # a second call reuses the layers' address tables
                self.assertTrue(
                    stacked.set_kv_buffer_prefix_valid_layers(
                        tuple(range(layers)), loc, lengths, k, v
                    )
                )
                self.assertEqual(len(stacked._layer_ptr_tables), 1)

    def test_stacked_prefix_commit_declines_what_it_does_not_cover(self):
        pool = _fp8_pool(2, 16, 128)
        loc = torch.zeros((2, 4), dtype=torch.int64, device=DEVICE)
        lengths = torch.ones(2, dtype=torch.int32, device=DEVICE)
        k = self.randn(2, 8, 1, 128).to(torch.bfloat16)
        before = [buf.clone() for buf in pool.k_buffer + pool.v_buffer]
        # one layer short, FP32 rows, and a head-major pool
        self.assertFalse(
            pool.set_kv_buffer_prefix_valid_layers((0,), loc, lengths, k, k)
        )
        self.assertFalse(
            pool.set_kv_buffer_prefix_valid_layers(
                (0, 1), loc, lengths, k.float(), k.float()
            )
        )
        pool.full_head_major = True
        self.assertFalse(
            pool.set_kv_buffer_prefix_valid_layers((0, 1), loc, lengths, k, k)
        )
        for buf, ref in zip(pool.k_buffer + pool.v_buffer, before):
            assert_bytes_equal(buf, ref, "an untouched cache")

    def test_head_major_pool_stores_the_rows_an_nhd_pool_stores(self):
        slots, heads, dim = 64, 2, 64
        major = _mha_pool(slots, heads, dim, head_major=True)
        nhd = _mha_pool(slots, heads, dim, head_major=False)
        k_buf = major.k_buffer[0]
        # [head][slot][dim] storage behind the [slot][head][dim] tensor
        self.assertEqual(tuple(k_buf.shape), (slots, heads, dim))
        self.assertEqual(k_buf.stride(), (dim, slots * dim, 1))
        self.assertTrue(k_buf.transpose(0, 1).is_contiguous())
        loc = torch.tensor([3, 17, 18, 40, 63], dtype=torch.int64, device=DEVICE)
        k = torch.randint(
            0, 256, (loc.numel(), heads * dim), dtype=torch.uint8, device=DEVICE
        )
        v = torch.randint(
            0, 256, (loc.numel(), heads * dim), dtype=torch.uint8, device=DEVICE
        )
        for pool in (major, nhd):
            pool._store_kv_layer(0, loc, k.clone(), v.clone())
        self.assertTrue(torch.equal(major.k_buffer[0], nhd.k_buffer[0]))
        self.assertTrue(torch.equal(major.v_buffer[0], nhd.v_buffer[0]))
        # a head's rows of one slot run are contiguous bytes of the head-major storage
        head1 = major.k_buffer[0].transpose(0, 1)[1, 17:19]
        self.assertTrue(head1.is_contiguous())
        self.assertTrue(torch.equal(head1.reshape(2, dim), k[1:3, dim:]))

    def test_fused_selector_walk_matches_the_torch_lattice(self):
        from sglang.kernels.ops.speculative.dflash import (
            _selector_lattice_kernel,
            selector_score_walk_triton,
            selector_walk_triton,
        )
        from sglang.srt.models.dflash import _score_edges

        batch, slots, top_k, rank, vocab = 3, 7, 16, 64, 4096
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                predecessor = (self.randn(vocab, rank) * 0.5).to(dtype)
                successor = (self.randn(vocab, rank) * 0.5).to(dtype)
                hidden = self.randn(batch, slots, rank).to(dtype)
                unary = self.randn(batch, slots, top_k)
                # distinct ids a slot, as a top-k gives them
                candidates = torch.stack(
                    [
                        torch.randperm(vocab, device=DEVICE, generator=self.gen)[:top_k]
                        for _ in range(batch * slots)
                    ]
                ).view(batch, slots, top_k)
                anchors = torch.randint(
                    0, vocab, (batch,), device=DEVICE, generator=self.gen
                )
                reference = _score_edges(
                    predecessor_table=predecessor,
                    successor_table=successor,
                    candidate_ids=candidates,
                    unary_logits=unary,
                    hidden=hidden,
                    anchor_token_ids=anchors,
                    top_k=top_k,
                ).float()
                scores = torch.empty(
                    (batch, slots, top_k, top_k), dtype=torch.float32, device=DEVICE
                )
                _selector_lattice_kernel[(batch, slots)](
                    candidates.contiguous(),
                    unary.contiguous(),
                    hidden.contiguous(),
                    anchors.contiguous(),
                    predecessor,
                    successor,
                    scores,
                    slots=slots,
                    top_k=top_k,
                    rank=rank,
                    round_bf16=dtype == torch.bfloat16,
                    num_warps=4,
                )
                # the lattice within the BF16 rounding of an edge (FP32 tables: its accumulation order)
                bound = 2e-2 if dtype == torch.bfloat16 else 1e-5
                self.assertLess(_rel_err(scores, reference), bound)

                # the fused walk is the walk on that lattice: same tokens and q rows, greedy and sampled rows
                uniforms = torch.rand((batch, slots), device=DEVICE, generator=self.gen)
                temperatures = torch.tensor([1.0, 0.7, 1.3], device=DEVICE)
                greedy = torch.tensor([True, False, False], device=DEVICE)
                want_tokens, want_q = selector_walk_triton(
                    candidate_ids=candidates,
                    scores=scores,
                    uniforms=uniforms,
                    temperatures=temperatures,
                    greedy_mask=greedy,
                )
                # the tokens land in rows of a wider buffer (a stride of slots + 1)
                block = torch.full(
                    (batch, slots + 1), -1, dtype=torch.int64, device=DEVICE
                )
                q_out = torch.empty(
                    (batch, slots, top_k), dtype=torch.float32, device=DEVICE
                )
                selector_score_walk_triton(
                    candidate_ids=candidates,
                    unary_logits=unary,
                    hidden=hidden,
                    anchor_token_ids=anchors,
                    predecessor_table=predecessor,
                    successor_table=successor,
                    uniforms=uniforms,
                    temperatures=temperatures,
                    greedy_mask=greedy,
                    tokens_out=block[:, 1:],
                    q_out=q_out,
                )
                self.assertTrue(torch.equal(block[:, 1:], want_tokens))
                self.assertTrue(torch.equal(block[:, 0], torch.full_like(anchors, -1)))
                self.assertTrue(torch.equal(q_out, want_q))

    def test_selector_params_staging(self):
        from sglang.kernels.ops.speculative.dflash import stage_selector_params

        temperatures = torch.tensor(
            [0.0, 0.5, 1.0, 1e-7, 2.0], dtype=torch.float32, device=DEVICE
        )
        top_ks = torch.tensor([1, 40, 0, 1, 1 << 30], dtype=torch.int32, device=DEVICE)
        for n in (1, 3, 5):
            with self.subTest(n=n):
                staged = torch.full((8,), -1.0, dtype=torch.float32, device=DEVICE)
                greedy = torch.zeros(8, dtype=torch.bool, device=DEVICE)
                stage_selector_params(
                    temperatures[:n], top_ks[:n], staged[:n], greedy[:n]
                )
                self.assertTrue(
                    torch.equal(staged[:n], torch.clamp(temperatures[:n], min=1e-5))
                )
                self.assertTrue(torch.equal(greedy[:n], top_ks[:n] <= 1))
                # rows past n are not written
                self.assertTrue(bool((staged[n:] == -1.0).all()))
                self.assertFalse(bool(greedy[n:].any()))


if __name__ == "__main__":
    unittest.main()
