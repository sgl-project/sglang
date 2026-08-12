"""Unit tests for the MiniMax MSA NHD->HND KV page gather — no server, no model.

`gather_kv_hnd` (python/sglang/kernels/ops/attention/minimax_sparse/common/
gather_kv.py) feeds fmha_sm100's sparse prefill when per-rank Hkv > 1: it
gathers only the pool pages referenced by the batch page table into a compact,
contiguous HND buffer instead of letting fmha_sm100 `.contiguous()` a permuted
whole-pool view (a full per-layer pool copy). These tests pin the kernel's
contract against a pure-torch reference: bitwise-equal bytes, contiguous HND
output, shuffled page tables, bf16 and fp8_e4m3 pools, and Hkv in {1, 2, 4}.

The kernel is plain Triton — any CUDA GPU suffices (no sm100 requirement).
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.gather_kv import gather_kv_hnd
from sglang.test.test_utils import CustomTestCase

P = 128  # MSA sparse block == page size for MiniMax-M3
D = 128  # MSA requires head_dim 128


def _reference(pool: torch.Tensor, page_ids: torch.Tensor, page_size: int):
    """Whole-pool NHD view -> HND gather via plain torch (the semantics the
    kernel replaces)."""
    slots, H, d = pool.shape
    paged = pool.view(slots // page_size, page_size, H, d)
    return paged[page_ids.long()].permute(0, 2, 1, 3).contiguous()


def _make_pool(slots: int, H: int, dtype: torch.dtype, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    k = torch.randn(slots, H, D, device="cuda", generator=g, dtype=torch.float32)
    v = torch.randn(slots, H, D, device="cuda", generator=g, dtype=torch.float32)
    return k.to(dtype), v.to(dtype)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGatherKvHnd(CustomTestCase):
    def _check(self, dtype, H, n_pages, slots=1 << 18, seed=0):
        k_pool, v_pool = _make_pool(slots, H, dtype, seed)
        g = torch.Generator(device="cuda").manual_seed(seed + 1)
        page_ids = (
            torch.randperm(slots // P, device="cuda", generator=g)[:n_pages]
        ).to(torch.int32)

        k_out, v_out = gather_kv_hnd(k_pool, v_pool, page_ids, P)

        self.assertEqual(k_out.shape, (n_pages, H, P, D))
        self.assertEqual(k_out.dtype, dtype)
        self.assertTrue(k_out.is_contiguous())
        # Bitwise equality: the gather is a pure byte relocation.
        self.assertTrue(
            torch.equal(
                k_out.view(torch.uint8),
                _reference(k_pool, page_ids, P).view(torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                v_out.view(torch.uint8),
                _reference(v_pool, page_ids, P).view(torch.uint8),
            )
        )

    def test_bf16_hkv4(self):
        # PP/TP1 shape for MiniMax-M3 (4 KV heads per rank).
        self._check(torch.bfloat16, H=4, n_pages=257)

    def test_bf16_hkv2(self):
        self._check(torch.bfloat16, H=2, n_pages=64)

    def test_bf16_hkv1_matches_zero_copy_view(self):
        # Hkv == 1: callers skip the gather (the permuted pool view is already
        # contiguous), but the kernel must still be correct if invoked.
        self._check(torch.bfloat16, H=1, n_pages=32)

    def test_fp8_e4m3_hkv4(self):
        # fp8 KV cache pool (1-byte elements exercise the int32-word reinterpretation).
        self._check(torch.float8_e4m3fn, H=4, n_pages=257)

    def test_single_page(self):
        self._check(torch.bfloat16, H=4, n_pages=1)

    def test_empty_page_table(self):
        k_pool, v_pool = _make_pool(1 << 12, 4, torch.bfloat16, seed=3)
        ids = torch.empty(0, dtype=torch.int32, device="cuda")
        k_out, v_out = gather_kv_hnd(k_pool, v_pool, ids, P)
        self.assertEqual(k_out.shape, (0, 4, P, D))
        self.assertEqual(v_out.shape, (0, 4, P, D))

    def test_identity_page_table_roundtrip(self):
        # Gathering ALL pages in order must reproduce the transposed pool exactly.
        slots, H = 1 << 14, 4
        k_pool, v_pool = _make_pool(slots, H, torch.bfloat16, seed=5)
        ids = torch.arange(slots // P, dtype=torch.int32, device="cuda")
        k_out, _ = gather_kv_hnd(k_pool, v_pool, ids, P)
        expected = k_pool.view(slots // P, P, H, D).permute(0, 2, 1, 3).contiguous()
        self.assertTrue(torch.equal(k_out, expected))


if __name__ == "__main__":
    unittest.main()
