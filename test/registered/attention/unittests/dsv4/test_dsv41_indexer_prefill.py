"""DeepSeek-V4.1 ratio-1/2 prefill indexer on the DeepGEMM dense fp4 kernel:
logits vs a torch golden of the reference scoring, the top-k buffer contract, and
agreement with the torch prefill path on a mixed batch of requests.

GPU only (deep_gemm.fp8_fp4_mqa_logits).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

FULL_PAGE_SIZE = 256
INDEX_PAGE_SIZE = 64
HEAD_DIM = 128
N_HEADS = 32
TOPK = 64


def golden_scores(q, k, weights):
    """q [t, H, d], k [n, d], weights [t, H] -> [t, n], fp32 reference scoring."""
    s = torch.einsum("bhd,nd->bhn", q.float(), k.float())
    return (s.relu() * weights.float().unsqueeze(-1)).sum(dim=1)


class _Indexer:
    """Stands in for the layer indexer: queries / head weights are precomputed
    fp4-grid tensors indexed by token; scores is the fp32 golden (torch path)."""

    def __init__(self, q, weights, topk, *, candidate_source, uses_candidates):
        self.q, self.w = q, weights
        self.index_topk = topk
        self.is_candidate_source = candidate_source
        self.uses_candidates = uses_candidates
        self.candidate_topk_blocks = 2
        self.candidate_block_size = 32
        self.owns_k = False
        self.n_local_heads = self.n_heads = q.shape[1]

    def queries(self, q_lora, freqs):
        return self.q[q_lora]

    def head_weights(self, x):
        return self.w[x]

    def scores(self, q, k, weights):
        return golden_scores(q, k, weights)


def _backend(core, pool, req_to_token):
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend

    backend = object.__new__(DeepseekV4AttnBackend)
    backend.forward_metadata = SimpleNamespace(core_metadata=core)
    backend.token_to_kv_pool = pool
    backend.req_to_token = req_to_token
    backend.candidate_masks = None
    return backend


@unittest.skipUnless(torch.cuda.is_available(), "CUDA only")
class TestPrefillIndexerKernelPath(CustomTestCase):
    def _setup(self, ratio, seq_lens, extend_lens):
        """A batch of requests with random FULL page tables; the fp4 index-K pool
        is written through store_fp4_index_k_cache exactly as production does."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            store_fp4_index_k_cache,
        )
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _low_ratio_sparse_buffers,
        )
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4

        torch.manual_seed(ratio)
        dev = "cuda"
        bs = len(seq_lens)
        slots_per_page = FULL_PAGE_SIZE // ratio
        n_full_pages = max((s + FULL_PAGE_SIZE - 1) // FULL_PAGE_SIZE for s in seq_lens)
        n_phys_pages = bs * n_full_pages + 2
        total_slots = n_phys_pages * slots_per_page
        full_page_table = (
            torch.randperm(n_phys_pages, device=dev)[: bs * n_full_pages]
            .view(bs, n_full_pages)
            .to(torch.int32)
        )
        # req_to_token in FULL token ids: position p of request b lives at
        # page_table[b, p // 256] * 256 + p % 256.
        req_to_token = torch.zeros(
            bs, n_full_pages * FULL_PAGE_SIZE, dtype=torch.int32, device=dev
        )
        p = torch.arange(n_full_pages * FULL_PAGE_SIZE, device=dev)
        for b in range(bs):
            req_to_token[b] = (
                full_page_table[b, p // FULL_PAGE_SIZE].to(torch.int64) * FULL_PAGE_SIZE
                + p % FULL_PAGE_SIZE
            ).to(torch.int32)

        k_all = fake_quant_fp4(
            torch.randn(total_slots, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        )
        k_cache = torch.zeros(
            total_slots // INDEX_PAGE_SIZE,
            INDEX_PAGE_SIZE * 68,
            dtype=torch.uint8,
            device=dev,
        )
        store_fp4_index_k_cache(
            input=k_all,
            cache=k_cache,
            loc=torch.arange(total_slots, dtype=torch.int32, device=dev),
            page_size=INDEX_PAGE_SIZE,
            rne=True,
        )

        def get_fp4(layer_id, slots):
            slots = slots.to(torch.int64)
            page, off = (
                (slots // INDEX_PAGE_SIZE).unsqueeze(-1),
                slots % INDEX_PAGE_SIZE,
            )
            payload = k_cache[
                page, (off * 64).unsqueeze(-1) + torch.arange(64, device=dev)
            ]
            sc = k_cache[
                page,
                (INDEX_PAGE_SIZE * 64 + off * 4).unsqueeze(-1)
                + torch.arange(4, device=dev),
            ]
            return payload.view(torch.int8), sc.contiguous().view(torch.int32).squeeze(
                -1
            )

        pool = SimpleNamespace(
            get_low_ratio_index_k_fp4=get_fp4,
            get_low_ratio_index_k_dequant=lambda layer_id, slots: k_all[
                slots.to(torch.int64)
            ],
        )

        # Tokens of the batch: the extend tail of every request, in batch order.
        pos_list, req_list = [], []
        for b, (s, e) in enumerate(zip(seq_lens, extend_lens)):
            pos_list += list(range(s - e, s))
            req_list += [b] * e
        pos = torch.tensor(pos_list, dtype=torch.int64, device=dev)
        req = torch.tensor(req_list, dtype=torch.int64, device=dev)
        T = pos.numel()
        q = fake_quant_fp4(
            torch.randn(T, N_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16)
        )
        weights = torch.rand(T, N_HEADS, device=dev, dtype=torch.float32)
        tok_ids = torch.arange(T, device=dev)

        lens_clamp1 = ((pos + 1) // ratio).clamp_min(1).to(torch.int32)

        def buffers():
            _, page_indices, raw_indices = _low_ratio_sparse_buffers(
                lens_clamp1, TOPK, is_prefill=True
            )
            core = SimpleNamespace(
                sparse_page_indices=lambda r: page_indices,
                sparse_raw_indices=lambda r: raw_indices,
            )
            return core, page_indices, raw_indices

        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: False, is_extend=lambda: True
            ),
            seq_lens_cpu=list(seq_lens),
            extend_seq_lens_cpu=list(extend_lens),
            extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32, device=dev),
            req_pool_indices=torch.arange(bs, dtype=torch.int32, device=dev),
        )
        return SimpleNamespace(
            ratio=ratio,
            dev=dev,
            bs=bs,
            pool=pool,
            req_to_token=req_to_token,
            k_all=k_all,
            pos=pos,
            req=req,
            q=q,
            weights=weights,
            tok_ids=tok_ids,
            buffers=buffers,
            forward_batch=forward_batch,
            seq_lens=seq_lens,
            extend_lens=extend_lens,
        )

    def _run(
        self, st, path, *, candidate_source=False, uses_candidates=False, masks=None
    ):
        core, page_indices, raw_indices = st.buffers()
        backend = _backend(core, st.pool, st.req_to_token)
        backend.candidate_masks = masks
        indexer = _Indexer(
            st.q,
            st.weights,
            TOPK,
            candidate_source=candidate_source,
            uses_candidates=uses_candidates,
        )
        layer = SimpleNamespace(
            layer_id=0,
            compress_ratio=st.ratio,
            indexer=indexer,
            freqs_cis=torch.zeros(4096, device=st.dev),
        )
        if path == "kernel":
            backend._low_ratio_index_topk_extend(
                layer, st.tok_ids, st.tok_ids, st.pos, st.forward_batch
            )
        else:
            backend._low_ratio_index_topk_torch(
                layer, st.tok_ids, st.tok_ids, st.req, st.pos
            )
        return page_indices, raw_indices, backend.candidate_masks

    def _golden_logits(self, st):
        """Per token: golden scores over its request's visible compressed keys."""
        out = []
        tok = 0
        for b, (s, e) in enumerate(zip(st.seq_lens, st.extend_lens)):
            lc = s // st.ratio
            j = torch.arange(lc, device=st.dev)
            slots = st.req_to_token[b, j * st.ratio].to(torch.int64) // st.ratio
            g = golden_scores(
                st.q[tok : tok + e], st.k_all[slots], st.weights[tok : tok + e]
            )
            lens = ((st.pos[tok : tok + e] + 1) // st.ratio)[:, None]
            out.append((g.masked_fill(j[None, :] >= lens, -torch.inf), slots))
            tok += e
        return out

    def test_dense_logits_match_golden(self):
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor,
        )
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            _dense_fp4_mqa_logits,
        )

        for ratio in (1, 2):
            st = self._setup(ratio, seq_lens=[300, 45, 700], extend_lens=[300, 45, 700])
            k_slots, starts, start = [], [], 0
            for b, s in enumerate(st.seq_lens):
                lc = s // ratio
                j = torch.arange(lc, device=st.dev)
                k_slots.append(st.req_to_token[b, j * ratio].to(torch.int64) // ratio)
                starts.append(start)
                start += lc
            k_slots = torch.cat(k_slots)
            k_fp4, k_sf = st.pool.get_low_ratio_index_k_fp4(0, k_slots)
            T = st.pos.numel()
            q_fp4, q_sf = quantize_fp4_indexer_tensor(st.q.flatten(0, 1), rne=True)
            ks = torch.repeat_interleave(
                torch.tensor(starts, device=st.dev),
                torch.tensor(st.extend_lens, device=st.dev),
            )
            ke = ks + (st.pos + 1) // ratio
            logits = _dense_fp4_mqa_logits(
                (q_fp4.view(T, N_HEADS, 64), q_sf.view(T, N_HEADS)),
                (k_fp4, k_sf),
                st.weights,
                ks.to(torch.int32),
                ke.to(torch.int32),
                max(s // ratio for s in st.seq_lens),
            )
            tok = 0
            for b, (g, _) in enumerate(self._golden_logits(st)):
                e = st.extend_lens[b]
                lc = st.seq_lens[b] // ratio
                got = logits[tok : tok + e, :lc]
                visible = g > -torch.inf
                rel = ((got - g).abs() / g.abs().clamp_min(1.0))[visible].max().item()
                self.assertLess(rel, 2e-3, msg=f"{ratio=} request {b}: max rel {rel}")
                tok += e

    def test_kernel_path_contract_and_selection(self):
        for ratio in (1, 2):
            st = self._setup(ratio, seq_lens=[300, 45, 700], extend_lens=[300, 45, 700])
            page_indices, raw_indices, _ = self._run(st, "kernel")
            tok = 0
            for b, (g, slots) in enumerate(self._golden_logits(st)):
                e = st.extend_lens[b]
                lens = (st.pos[tok : tok + e] + 1) // ratio
                for i in range(0, e, max(1, e // 7)):
                    n_valid = min(TOPK, int(lens[i]))
                    row_raw = raw_indices[tok + i]
                    row_slot = page_indices[tok + i]
                    if n_valid == 0:  # ratio 2, position 0: nothing visible yet
                        self.assertTrue(bool((row_raw == -1).all()))
                        continue
                    self.assertTrue(bool((row_raw[:n_valid] >= 0).all()))
                    self.assertTrue(bool((row_raw[:n_valid] < lens[i]).all()))
                    self.assertTrue(bool((row_raw[n_valid:] == -1).all()))
                    self.assertTrue(
                        torch.equal(
                            row_slot[:n_valid].to(torch.int64),
                            slots[row_raw[:n_valid].to(torch.int64)],
                        )
                    )
                    ref_sel = g[i].topk(n_valid).indices
                    overlap = (
                        len(set(ref_sel.tolist()) & set(row_raw[:n_valid].tolist()))
                        / n_valid
                    )
                    self.assertGreaterEqual(
                        overlap,
                        0.9,
                        msg=f"{ratio=} request {b} row {i}: overlap {overlap}",
                    )
                tok += e

    def test_kernel_path_agrees_with_torch_path(self):
        """Same batch, kernel vs torch prefill path (including a prefix-only
        request): identical valid-prefix structure and near-identical selection,
        with the candidate masks flowing from source to consumer on both."""
        for ratio in (1, 2):
            st = self._setup(ratio, seq_lens=[300, 45, 700], extend_lens=[300, 45, 200])
            k_pi, k_ri, k_masks = self._run(st, "kernel", candidate_source=True)
            t_pi, t_ri, t_masks = self._run(st, "torch", candidate_source=True)
            self.assertTrue(torch.equal(k_ri == -1, t_ri == -1))
            self.assertTrue(torch.equal(k_pi == -1, t_pi == -1))
            agree = ((k_ri == t_ri) | (k_ri == -1)).float().mean().item()
            self.assertGreaterEqual(
                agree, 0.95, msg=f"{ratio=}: raw top-k agreement {agree}"
            )
            self.assertEqual(len(k_masks), len(t_masks))
            for a, b in zip(k_masks, t_masks):
                self.assertEqual(a.shape, b.shape)
                self.assertGreaterEqual((a == b).float().mean().item(), 0.95)
            # Consumer layer fed by the kernel-path masks.
            c_pi, c_ri, _ = self._run(st, "kernel", uses_candidates=True, masks=k_masks)
            d_pi, d_ri, _ = self._run(st, "torch", uses_candidates=True, masks=k_masks)
            self.assertTrue(torch.equal(c_ri == -1, d_ri == -1))
            agree = ((c_ri == d_ri) | (c_ri == -1)).float().mean().item()
            self.assertGreaterEqual(
                agree, 0.95, msg=f"{ratio=}: consumer agreement {agree}"
            )


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA only")
class TestPrefillIndexerPieces(CustomTestCase):
    def test_pool_packed_readback_matches_quantizer(self):
        """get_index_k_fp4 on the real fp4 indexer pool returns exactly the bytes
        store_fp4_index_k_cache wrote: payload and packed scales bitwise equal to
        quantize_fp4_indexer_tensor of the stored rows, through random slots."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor,
        )
        from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4IndexerPool

        torch.manual_seed(5)
        n = 1000
        pool = DeepSeekV4IndexerPool(
            size=n,
            page_size=INDEX_PAGE_SIZE,
            dtype=torch.bfloat16,
            index_head_dim=HEAD_DIM,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            use_fp4_indexer=True,
        )
        pool.index_k_rne = True
        k = fake_quant_fp4(
            torch.randn(n, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        )
        loc = torch.randperm(n, device="cuda").to(torch.int32)
        pool.set_index_fp4(0, loc, k)
        slots = torch.randperm(n, device="cuda")[:333]
        payload, scales = pool.get_index_k_fp4(0, slots)
        # Row stored at loc[i] is k[i]; slot s therefore holds k[argsort(loc)[s]].
        inv = torch.empty_like(loc)
        inv[loc.to(torch.int64)] = torch.arange(n, device="cuda", dtype=torch.int32)
        ref_payload, ref_scales = quantize_fp4_indexer_tensor(
            k[inv[slots].to(torch.int64)], rne=True
        )
        self.assertEqual(payload.dtype, torch.int8)
        self.assertEqual(scales.dtype, torch.int32)
        self.assertTrue(torch.equal(payload, ref_payload))
        self.assertTrue(torch.equal(scales, ref_scales))

    def test_dispatch_conditions(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        use = DeepseekV4AttnBackend._use_dense_fp4_prefill_indexer
        extend = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: True, is_decode=lambda: False
            ),
            seq_lens_cpu=[3],
            extend_seq_lens_cpu=[3],
        )
        self.assertTrue(use(extend))
        with envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.override(True):
            self.assertFalse(use(extend))
        no_cpu = SimpleNamespace(
            forward_mode=extend.forward_mode, seq_lens_cpu=None, extend_seq_lens_cpu=[3]
        )
        self.assertFalse(use(no_cpu))
        decode = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: False, is_decode=lambda: True
            ),
            seq_lens_cpu=[3],
            extend_seq_lens_cpu=[1],
        )
        self.assertFalse(use(decode))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA only")
class TestPrefillIndexerKernelPathEdges(TestPrefillIndexerKernelPath):
    def test_row_chunking_is_neutral(self):
        """The score budget only bounds memory on the kernel path too: per-row
        chunks give the same indices and candidate masks as one chunk, and the
        consumer slices the published masks per chunk."""
        from sglang.srt.layers.attention import deepseek_v4_backend as be

        for ratio in (1, 2):
            st = self._setup(ratio, seq_lens=[300, 45, 700], extend_lens=[300, 45, 200])
            full = self._run(st, "kernel", candidate_source=True)
            full_c = self._run(st, "kernel", uses_candidates=True, masks=full[2])
            saved = be._TORCH_INDEXER_SCORE_BUDGET_BYTES
            be._TORCH_INDEXER_SCORE_BUDGET_BYTES = 1  # one row per chunk
            try:
                tiny = self._run(st, "kernel", candidate_source=True)
                tiny_c = self._run(st, "kernel", uses_candidates=True, masks=full[2])
            finally:
                be._TORCH_INDEXER_SCORE_BUDGET_BYTES = saved
            self.assertTrue(
                torch.equal(tiny[0], full[0]) and torch.equal(tiny[1], full[1])
            )
            self.assertEqual(len(tiny[2]), len(full[2]))
            for a, b in zip(tiny[2], full[2]):
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(
                torch.equal(tiny_c[0], full_c[0]) and torch.equal(tiny_c[1], full_c[1])
            )

    def test_request_with_no_visible_positions(self):
        """A 1-token request at ratio 2 sees no compressed position: its rows stay
        -1, the other requests are unaffected, and the candidate-mask list stays
        aligned with the torch path (both skip it)."""
        ratio = 2
        st = self._setup(ratio, seq_lens=[1, 300, 45], extend_lens=[1, 300, 45])
        k_pi, k_ri, k_masks = self._run(st, "kernel", candidate_source=True)
        t_pi, t_ri, t_masks = self._run(st, "torch", candidate_source=True)
        self.assertTrue(bool((k_ri[0] == -1).all()) and bool((k_pi[0] == -1).all()))
        self.assertTrue(torch.equal(k_ri == -1, t_ri == -1))
        self.assertEqual(len(k_masks), len(t_masks))
        agree = ((k_ri == t_ri) | (k_ri == -1)).float().mean().item()
        self.assertGreaterEqual(agree, 0.95)
        # A batch with only such requests publishes an empty mask list and writes nothing.
        st0 = self._setup(ratio, seq_lens=[1, 1], extend_lens=[1, 1])
        pi, ri, masks = self._run(st0, "kernel", candidate_source=True)
        self.assertTrue(bool((ri == -1).all()))
        self.assertEqual(masks, [])
