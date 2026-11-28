"""SDSA sparse attention (indexer -> top-k page-table transform -> token-sparse
MQA on FlashInfer) vs a PyTorch reference of the indexer and attention."""

import math
import types
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="extra-b", runner_config="4-gpu-b200")


def _has_sdsa() -> bool:
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10):
        return False
    try:
        import flashinfer.sdsa  # noqa: F401
    except ImportError:
        return False
    return True


HEADS, HEAD_DIM, INDEX_DIM, TOPK, PAGE = 16, 256, 128, 2048, 64


@unittest.skipUnless(_has_sdsa(), "needs SM100 and FlashInfer >= 0.7.4 (sdsa)")
class TestSdsaSparseAttention(CustomTestCase):
    def setUp(self):
        from sglang.srt.layers.attention.sdsa_backend import SdsaAttnBackend, SdsaRows

        torch.manual_seed(0)
        self.rows_cls = SdsaRows
        # Request lengths around index_topk: below, at, just above and well above.
        self.lens = [1, 100, 2048, 2049, 3000]
        num_pages = sum(-(-n // PAGE) for n in self.lens) + 1
        slots = num_pages * PAGE
        self.k = torch.randn(slots, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        self.v = torch.randn(slots, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        self.index_k = torch.randn(
            slots, INDEX_DIM, device="cuda", dtype=torch.bfloat16
        )
        width = -(-max(self.lens) // PAGE) * PAGE
        # Whole pages per request in a shuffled order, slot 0's page unused.
        pages = (torch.randperm(num_pages - 1) + 1).tolist()
        self.req_to_token = torch.zeros(
            len(self.lens), width, dtype=torch.int32, device="cuda"
        )
        for r, n in enumerate(self.lens):
            own = torch.tensor([pages.pop() for _ in range(-(-n // PAGE))])
            toks = (own[:, None] * PAGE + torch.arange(PAGE)).reshape(-1)[:n]
            self.req_to_token[r, :n] = toks.to(torch.int32)

        backend = SdsaAttnBackend.__new__(SdsaAttnBackend)
        backend.data_type = torch.bfloat16
        backend.index_topk = TOPK
        backend.req_to_token = self.req_to_token
        backend.score_width = width
        backend.workspace_buffer = torch.zeros(
            256 << 20, dtype=torch.uint8, device="cuda"
        )
        # kv_layers reads the pool's layer router.
        backend.token_to_kv_pool = types.SimpleNamespace(
            layer_router=types.SimpleNamespace(
                read=lambda layer_id: types.SimpleNamespace(kv=lambda: (self.k, self.v))
            )
        )
        backend.index_pool = types.SimpleNamespace(
            get_index_k_buffer=lambda layer_id: self.index_k
        )
        self.backend = backend
        self.layer = types.SimpleNamespace(
            layer_id=0,
            tp_q_head_num=HEADS,
            head_dim=HEAD_DIM,
            scaling=1 / math.sqrt(HEAD_DIM),
        )

    def _reference(self, q, index_q, request, context_len):
        slots = self.req_to_token[request, :context_len].long()
        scores = index_q.float() @ self.index_k[slots].float().T
        keep = slots[scores.topk(min(TOPK, context_len)).indices]
        if self.backend.data_type == torch.float8_e4m3fn:
            q = q.to(torch.float8_e4m3fn)
        attn = torch.softmax(
            q.float() @ self.k[keep].float().T / math.sqrt(HEAD_DIM), dim=-1
        )
        return attn @ self.v[keep].float()

    def _use_fp8_cache(self):
        self.k = self.k.to(torch.float8_e4m3fn)
        self.v = self.v.to(torch.float8_e4m3fn)
        self.backend.data_type = torch.float8_e4m3fn

    def _check(self, requests, context_lens, tol=2e-2):
        rows = len(requests)
        q = torch.randn(rows, HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        index_q = torch.randn(rows, INDEX_DIM, device="cuda", dtype=torch.bfloat16)
        sdsa_rows = self.rows_cls(
            requests=torch.tensor(requests, dtype=torch.int32, device="cuda"),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device="cuda"),
        )
        batch = types.SimpleNamespace(
            req_pool_indices=torch.arange(len(self.lens), device="cuda")
        )
        out = self.backend._sparse_attention(q, self.layer, batch, index_q, sdsa_rows)
        self.assertEqual(out.dtype, torch.bfloat16)
        for i, (r, n) in enumerate(zip(requests, context_lens)):
            ref = self._reference(q[i].view(HEADS, HEAD_DIM), index_q[i], r, n)
            err = (out[i].float().view(HEADS, HEAD_DIM) - ref).norm() / ref.norm()
            self.assertLess(err.item(), tol, f"row {i} (request {r}, {n} keys)")

    def test_decode_rows(self):
        self._check(list(range(len(self.lens))), self.lens)

    def test_extend_rows_in_score_chunks(self):
        # Several causal rows per request; a tiny score budget forces one row
        # per chunk, so every chunk boundary is crossed.
        requests = [0, 2, 2, 3, 3, 4, 4, 4]
        context_lens = [1, 2047, 2048, 2048, 2049, 1, 2500, 3000]
        with patch("sglang.srt.layers.attention.sdsa_backend._SCORE_CHUNK_BYTES", 1):
            self._check(requests, context_lens)

    def test_fp8_kv_cache(self):
        # The query is cast to the cache's E4M3; the output stays BF16. The
        # kernel also rounds the probabilities to E4M3, hence the wider bound.
        self._use_fp8_cache()
        self._check(list(range(len(self.lens))), self.lens, tol=5e-2)


if __name__ == "__main__":
    unittest.main()
