"""DeepSeek-V4.1 ratio-1/2 prefill indexer on the DeepGEMM dense fp4 kernel:
logits vs a torch golden of the reference scoring, the top-k buffer contract, and
selection on a mixed batch of requests.

GPU only (deep_gemm.fp8_fp4_mqa_logits).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

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
    """Supply precomputed queries and head weights to the real indexer backend."""

    def __init__(self, q, weights, topk):
        self.q, self.w = q, weights
        self.index_topk = topk
        self.is_candidate_source = False
        self.uses_candidates = False
        self.candidate_topk_blocks = 2
        self.candidate_block_size = 32
        self.owns_k = False
        self.n_local_heads = self.n_heads = q.shape[1]

    def queries(self, q_lora, freqs):
        return self.q[q_lora]

    def head_weights(self, x):
        return self.w[x]


def _backend(core, pool, req_to_token):
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend

    backend = object.__new__(DeepseekV4AttnBackend)
    backend.forward_metadata = SimpleNamespace(core_metadata=core, late_layer_tail=None)
    backend.token_to_kv_pool = pool
    backend.req_to_token = req_to_token
    backend.candidate_masks = None
    return backend


@unittest.skipUnless(torch.cuda.is_available(), "CUDA only")
class _PrefillIndexerFixture(CustomTestCase):
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
        )

        # Tokens of the batch: the extend tail of every request, in batch order.
        pos_list = []
        for s, e in zip(seq_lens, extend_lens):
            pos_list += list(range(s - e, s))
        pos = torch.tensor(pos_list, dtype=torch.int64, device=dev)
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
            pool=pool,
            req_to_token=req_to_token,
            k_all=k_all,
            pos=pos,
            q=q,
            weights=weights,
            tok_ids=tok_ids,
            buffers=buffers,
            forward_batch=forward_batch,
            seq_lens=seq_lens,
            extend_lens=extend_lens,
        )

    def _run(self, st):
        core, page_indices, raw_indices = st.buffers()
        backend = _backend(core, st.pool, st.req_to_token)
        indexer = _Indexer(st.q, st.weights, TOPK)
        layer = SimpleNamespace(
            layer_id=0,
            compress_ratio=st.ratio,
            indexer=indexer,
            freqs_cis=torch.zeros(4096, device=st.dev),
        )
        backend._low_ratio_index_topk_extend(
            layer, st.tok_ids, st.tok_ids, st.pos, st.forward_batch
        )
        return page_indices, raw_indices

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


class TestPrefillIndexerKernelPath(_PrefillIndexerFixture):
    def test_head_weights_match_scaled_linear(self):
        from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer

        config = SimpleNamespace(
            index_n_heads=N_HEADS,
            index_head_dim=HEAD_DIM,
            index_topk=TOPK,
            qk_rope_head_dim=64,
            q_lora_rank=16,
            hidden_size=5120,
            kv_source_layer_ids=[],
            candidate_source_layer_id=-1,
            candidate_topk_blocks=2,
            candidate_block_size=32,
        )
        torch.manual_seed(0)
        indexer = DeepseekV41Indexer(config, 0, 512, None, "indexer").cuda()
        with torch.no_grad():
            indexer.weights_proj.weight.normal_(std=0.02)
            for rows in (1, 33):
                x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
                expected = torch.nn.functional.linear(x, indexer.weights_proj.weight)
                expected *= HEAD_DIM**-0.5 * N_HEADS**-0.5
                torch.testing.assert_close(
                    indexer.head_weights(x), expected, atol=2e-4, rtol=1e-2
                )

    def test_kernel_path_contract_and_selection(self):
        for ratio in (1, 2):
            st = self._setup(ratio, seq_lens=[300, 45, 700], extend_lens=[300, 45, 700])
            page_indices, raw_indices = self._run(st)
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


if __name__ == "__main__":
    unittest.main()
