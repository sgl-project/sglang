"""Unit tests for the fp8 QSA gather-dequant path in ``qsa/sparse_attn.py``.

Against a bf16 reference pool (~50 MB VRAM):
  1. the compact gather from an fp8_e4m3 pool equals the gather from
     ``pool_bf16.to(fp8).to(bf16)`` bit for bit (the kernel bitcasts the uint8
     view back to fp8 and widens to bf16);
  2. the dequantization error of e4m3 on N(0, 3) K/V stays in the expected band
     (relative RMS, well below the 8 % of an int4-g32 pool).
"""

import unittest

import torch

from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

SLOTS, HEADS, DIM, BATCH, TOPK = 4096, 2, 256, 3, 64


@unittest.skipUnless(torch.cuda.is_available(), "Triton gather kernels require CUDA")
class TestFp8GatherDequant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        dev = cls.dev = "cuda"
        cls.k16 = torch.randn(SLOTS, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        cls.v16 = torch.randn(SLOTS, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        cls.k8 = cls.k16.to(torch.float8_e4m3fn)
        cls.v8 = cls.v16.to(torch.float8_e4m3fn)
        cls.seq_lens = torch.tensor([300, 1200, 4000], dtype=torch.int32, device=dev)
        cls.req_to_token = torch.arange(SLOTS, dtype=torch.int32, device=dev).repeat(
            BATCH, 1
        )
        cls.req_indices = torch.arange(BATCH, dtype=torch.int32, device=dev)
        cls.indices = torch.stack(
            [
                torch.randperm(int(n), device=dev)[:TOPK].sort().values
                for n in cls.seq_lens
            ]
        ).to(torch.int32)
        cls.cu_k = torch.empty(BATCH + 1, dtype=torch.int32, device=dev)
        counts = torch.empty(BATCH, dtype=torch.int32, device=dev)
        qwen_sparse_fa2_cu_seqlens_triton(
            cls.seq_lens, cls.indices, counts, cls.cu_k, BATCH, TOPK
        )
        cls.n = int(cls.cu_k[-1])

    def _gather(self, k, v):
        out_k = torch.empty(self.n, HEADS, DIM, device=self.dev, dtype=torch.bfloat16)
        out_v = torch.empty_like(out_k)
        qwen_sparse_kv_extraction_compact_triton(
            k,
            v,
            self.req_to_token,
            self.req_indices,
            self.indices,
            self.seq_lens,
            self.cu_k,
            out_k,
            out_v,
            BATCH,
            TOPK,
        )
        torch.cuda.synchronize()
        return out_k, out_v

    def test_fp8_gather_matches_torch_cast_bit_exact(self):
        self.assertEqual(self.n, BATCH * TOPK)
        out_k, out_v = self._gather(self.k8, self.v8)
        exp_k, exp_v = self._gather(
            self.k8.to(torch.bfloat16), self.v8.to(torch.bfloat16)
        )
        self.assertTrue(torch.equal(out_k, exp_k), "fp8 K gather != torch fp8->bf16")
        self.assertTrue(torch.equal(out_v, exp_v), "fp8 V gather != torch fp8->bf16")

    def test_e4m3_relative_rms_error(self):
        ref_k, _ = self._gather(self.k16, self.v16)
        out_k, _ = self._gather(self.k8, self.v8)
        err = (
            (out_k.float() - ref_k.float()).pow(2).mean() / ref_k.float().pow(2).mean()
        ).sqrt()
        # e4m3 on N(0, 3): ~2.7 % relative RMS (3 mantissa bits)
        self.assertGreater(float(err), 0.01)
        self.assertLess(float(err), 0.04)


if __name__ == "__main__":
    unittest.main()
