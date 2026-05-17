"""K_label gather kernel correctness — Triton vs torch reference."""

import unittest

import torch

from sglang.srt.mem_cache.sparsity.triton_ops.k_label_kernels import (
    ds_compute_k_label_torch_ref,
    ds_compute_k_label_write,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, suite="stage-b-test-1-gpu-small")


def _alloc(N, H_kv, D, S, T, dtype, device):
    torch.manual_seed(0)
    k = torch.randn(N, H_kv, D, dtype=dtype, device=device)
    chan = torch.stack([torch.randperm(D, device=device)[:S] for _ in range(H_kv)]).to(
        torch.int32
    )
    # Random unique destination ids (no two writes collide → byte equality
    # vs the torch reference).
    perm = torch.randperm(T, device=device)[:N]
    out_loc = perm.to(torch.int64)
    kl_triton = torch.zeros(T, H_kv, S, dtype=dtype, device=device)
    kl_torch = torch.zeros_like(kl_triton)
    return k, chan, out_loc, kl_triton, kl_torch


class TestKLabelKernel(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for K_label Triton kernel")
        self.device = torch.device("cuda")

    def _run_case(self, N, H_kv, D, S, T, dtype):
        k, chan, out_loc, kl_t, kl_ref = _alloc(N, H_kv, D, S, T, dtype, self.device)
        ds_compute_k_label_write(k, chan, out_loc, kl_t)
        ds_compute_k_label_torch_ref(k, chan, out_loc, kl_ref)
        if dtype == torch.bfloat16:
            self.assertTrue(
                torch.allclose(kl_t.float(), kl_ref.float(), atol=0, rtol=0),
                f"Triton vs torch ref mismatch for N={N} H={H_kv} D={D} S={S}",
            )
        else:
            self.assertTrue(torch.equal(kl_t, kl_ref))

    def test_extend_shape_bf16(self):
        # Realistic prefill chunk: large N, GQA with H_kv=8, head_dim=128, S=32.
        self._run_case(N=512, H_kv=8, D=128, S=32, T=4096, dtype=torch.bfloat16)

    def test_decode_shape_bf16(self):
        # Realistic decode step: N == bs, single token per request.
        self._run_case(N=4, H_kv=8, D=128, S=32, T=4096, dtype=torch.bfloat16)

    def test_fp32_klabel(self):
        # FP32 K_label path (escape hatch via --double-sparsity-klabel-dtype fp32).
        N, H, D, S, T = 64, 4, 64, 16, 1024
        torch.manual_seed(1)
        k = torch.randn(N, H, D, dtype=torch.bfloat16, device=self.device)
        chan = torch.stack(
            [torch.randperm(D, device=self.device)[:S] for _ in range(H)]
        ).to(torch.int32)
        out = torch.randperm(T, device=self.device)[:N].to(torch.int64)
        kl_t = torch.zeros(T, H, S, dtype=torch.float32, device=self.device)
        kl_ref = torch.zeros_like(kl_t)
        ds_compute_k_label_write(k, chan, out, kl_t)
        ds_compute_k_label_torch_ref(k, chan, out, kl_ref)
        # bf16 -> fp32 cast is lossless within representable range
        self.assertTrue(torch.equal(kl_t, kl_ref))

    def test_empty_input_is_noop(self):
        N, H, D, S, T = 0, 4, 64, 16, 256
        k = torch.empty(0, H, D, dtype=torch.bfloat16, device=self.device)
        chan = torch.zeros(H, S, dtype=torch.int32, device=self.device)
        out = torch.empty(0, dtype=torch.int64, device=self.device)
        kl = torch.zeros(T, H, S, dtype=torch.bfloat16, device=self.device)
        before = kl.clone()
        ds_compute_k_label_write(k, chan, out, kl)
        self.assertTrue(torch.equal(kl, before))

    def test_permuted_destinations(self):
        # Make sure writes follow out_cache_loc, not row order — the radix
        # cache + prefix reuse case.
        N, H, D, S, T = 16, 2, 32, 8, 64
        torch.manual_seed(2)
        k = torch.randn(N, H, D, dtype=torch.bfloat16, device=self.device)
        chan = torch.stack(
            [torch.randperm(D, device=self.device)[:S] for _ in range(H)]
        ).to(torch.int32)
        # Reverse-order destinations
        out = torch.arange(N - 1, -1, -1, dtype=torch.int64, device=self.device)
        kl_t = torch.zeros(T, H, S, dtype=torch.bfloat16, device=self.device)
        kl_ref = torch.zeros_like(kl_t)
        ds_compute_k_label_write(k, chan, out, kl_t)
        ds_compute_k_label_torch_ref(k, chan, out, kl_ref)
        self.assertTrue(torch.equal(kl_t, kl_ref))

    def test_does_not_touch_unwritten_rows(self):
        N, H, D, S, T = 4, 2, 16, 4, 32
        torch.manual_seed(3)
        k = torch.randn(N, H, D, dtype=torch.bfloat16, device=self.device)
        chan = torch.stack(
            [torch.randperm(D, device=self.device)[:S] for _ in range(H)]
        ).to(torch.int32)
        out = torch.tensor([2, 5, 9, 14], dtype=torch.int64, device=self.device)
        kl = torch.full(
            (T, H, S), float("nan"), dtype=torch.bfloat16, device=self.device
        )
        ds_compute_k_label_write(k, chan, out, kl)
        # Untouched rows must still be NaN
        for t in range(T):
            if t in {2, 5, 9, 14}:
                continue
            self.assertTrue(torch.isnan(kl[t]).all())


if __name__ == "__main__":
    unittest.main()
