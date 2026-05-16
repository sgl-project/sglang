"""Unit tests for utils.py — GPU-required tests"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# set_mla_kv_buffer_triton — GPU
# ---------------------------------------------------------------------------


@unittest.skipUnless(CUDA, "CUDA required")
class TestSetMLAKVBufferTriton(CustomTestCase):
    NOPE, ROPE, N_LOC = 128, 64, 4

    def _tensors(self, nope=None, rope=None, n_loc=None):
        nope = nope or self.NOPE
        rope = rope or self.ROPE
        n_loc = n_loc or self.N_LOC
        kv = torch.zeros(n_loc * 2, nope + rope, dtype=torch.float32, device="cuda")
        loc = torch.arange(n_loc, dtype=torch.int64, device="cuda")
        k_nope = torch.rand(n_loc, nope, dtype=torch.float32, device="cuda")
        k_rope = torch.rand(n_loc, rope, dtype=torch.float32, device="cuda")
        return kv, loc, k_nope, k_rope

    def test_nope_region_written_correctly(self):
        from sglang.srt.mem_cache.utils import set_mla_kv_buffer_triton

        kv, loc, nope, rope = self._tensors()
        set_mla_kv_buffer_triton(kv, loc, nope, rope)
        for i in range(self.N_LOC):
            self.assertTrue(torch.allclose(kv[i, : self.NOPE], nope[i]))

    def test_rope_region_written_correctly(self):
        from sglang.srt.mem_cache.utils import set_mla_kv_buffer_triton

        kv, loc, nope, rope = self._tensors()
        set_mla_kv_buffer_triton(kv, loc, nope, rope)
        for i in range(self.N_LOC):
            self.assertTrue(
                torch.allclose(kv[i, self.NOPE : self.NOPE + self.ROPE], rope[i])
            )

    def test_non_contiguous_locations(self):
        from sglang.srt.mem_cache.utils import set_mla_kv_buffer_triton

        total = self.NOPE + self.ROPE
        kv = torch.zeros(16, total, dtype=torch.float32, device="cuda")
        loc = torch.tensor([0, 3, 7, 12], dtype=torch.int64, device="cuda")
        nope = torch.rand(4, self.NOPE, device="cuda")
        rope = torch.rand(4, self.ROPE, device="cuda")
        set_mla_kv_buffer_triton(kv, loc, nope, rope)
        for idx, l in enumerate(loc.tolist()):
            self.assertTrue(torch.allclose(kv[l, : self.NOPE], nope[idx]))

    def test_boundary_nope_dim_not_multiple_of_block(self):
        from sglang.srt.mem_cache.utils import set_mla_kv_buffer_triton

        nope_dim, rope_dim, n = 160, 64, 2
        kv = torch.zeros(n, nope_dim + rope_dim, dtype=torch.float32, device="cuda")
        loc = torch.arange(n, dtype=torch.int64, device="cuda")
        nope = torch.rand(n, nope_dim, device="cuda")
        rope = torch.rand(n, rope_dim, device="cuda")
        set_mla_kv_buffer_triton(kv, loc, nope, rope)
        for i in range(n):
            self.assertTrue(torch.allclose(kv[i, :nope_dim], nope[i]))
            self.assertTrue(
                torch.allclose(kv[i, nope_dim : nope_dim + rope_dim], rope[i])
            )


# ---------------------------------------------------------------------------
# set_mla_kv_scale_buffer_triton — GPU
# ---------------------------------------------------------------------------


@unittest.skipUnless(CUDA, "CUDA required")
class TestSetMLAKVScaleBufferTriton(CustomTestCase):
    NOPE, ROPE, N_LOC = 128, 64, 4

    def test_scale_nope_region(self):
        from sglang.srt.mem_cache.utils import set_mla_kv_scale_buffer_triton

        total = self.NOPE + self.ROPE
        kv = torch.zeros(self.N_LOC, total, dtype=torch.float32, device="cuda")
        loc = torch.arange(self.N_LOC, dtype=torch.int64, device="cuda")
        nope = torch.rand(self.N_LOC, self.NOPE, device="cuda")
        rope = torch.zeros(self.N_LOC, self.ROPE, device="cuda")
        set_mla_kv_scale_buffer_triton(kv, loc, nope, rope)
        for i in range(self.N_LOC):
            self.assertTrue(torch.allclose(kv[i, : self.NOPE], nope[i]))

    def test_scale_rope_region(self):
        from sglang.srt.mem_cache.utils import set_mla_kv_scale_buffer_triton

        total = self.NOPE + self.ROPE
        kv = torch.zeros(self.N_LOC, total, dtype=torch.float32, device="cuda")
        loc = torch.arange(self.N_LOC, dtype=torch.int64, device="cuda")
        nope = torch.zeros(self.N_LOC, self.NOPE, device="cuda")
        rope = torch.rand(self.N_LOC, self.ROPE, device="cuda")
        set_mla_kv_scale_buffer_triton(kv, loc, nope, rope)
        for i in range(self.N_LOC):
            self.assertTrue(torch.allclose(kv[i, self.NOPE :], rope[i]))

    def test_scale_both_regions_combined(self):
        from sglang.srt.mem_cache.utils import set_mla_kv_scale_buffer_triton

        total = self.NOPE + self.ROPE
        kv = torch.zeros(self.N_LOC, total, dtype=torch.float32, device="cuda")
        loc = torch.arange(self.N_LOC, dtype=torch.int64, device="cuda")
        nope = torch.rand(self.N_LOC, self.NOPE, device="cuda")
        rope = torch.rand(self.N_LOC, self.ROPE, device="cuda")
        set_mla_kv_scale_buffer_triton(kv, loc, nope, rope)
        for i in range(self.N_LOC):
            self.assertTrue(torch.allclose(kv[i, : self.NOPE], nope[i]))
            self.assertTrue(torch.allclose(kv[i, self.NOPE :], rope[i]))

    def test_non_contiguous_locations(self):
        from sglang.srt.mem_cache.utils import set_mla_kv_scale_buffer_triton

        total = self.NOPE + self.ROPE
        kv = torch.zeros(16, total, dtype=torch.float32, device="cuda")
        loc = torch.tensor([1, 5, 9, 13], dtype=torch.int64, device="cuda")
        nope = torch.rand(4, self.NOPE, device="cuda")
        rope = torch.rand(4, self.ROPE, device="cuda")
        set_mla_kv_scale_buffer_triton(kv, loc, nope, rope)
        for idx, l in enumerate(loc.tolist()):
            self.assertTrue(torch.allclose(kv[l, : self.NOPE], nope[idx]))


# ---------------------------------------------------------------------------
# get_mla_kv_buffer_triton — GPU
# ---------------------------------------------------------------------------


@unittest.skipUnless(CUDA, "CUDA required")
class TestGetMLAKVBufferTriton(CustomTestCase):
    NOPE, ROPE, N_LOC = 128, 64, 4

    def test_roundtrip_set_then_get(self):
        from sglang.srt.mem_cache.utils import (
            get_mla_kv_buffer_triton,
            set_mla_kv_buffer_triton,
        )

        total = self.NOPE + self.ROPE
        kv = torch.zeros(self.N_LOC, total, dtype=torch.float32, device="cuda")
        loc = torch.arange(self.N_LOC, dtype=torch.int64, device="cuda")
        nope_in = torch.rand(self.N_LOC, self.NOPE, device="cuda")
        rope_in = torch.rand(self.N_LOC, self.ROPE, device="cuda")
        set_mla_kv_buffer_triton(kv, loc, nope_in, rope_in)
        nope_out = torch.empty_like(nope_in)
        rope_out = torch.empty_like(rope_in)
        get_mla_kv_buffer_triton(kv, loc, nope_out, rope_out)
        self.assertTrue(torch.allclose(nope_in, nope_out, atol=1e-5))
        self.assertTrue(torch.allclose(rope_in, rope_out, atol=1e-5))

    def test_get_reads_correct_locations(self):
        from sglang.srt.mem_cache.utils import get_mla_kv_buffer_triton

        total = self.NOPE + self.ROPE
        kv = torch.rand(16, total, dtype=torch.float32, device="cuda")
        loc = torch.tensor([2, 5], dtype=torch.int64, device="cuda")
        nope_out = torch.empty(2, self.NOPE, device="cuda")
        rope_out = torch.empty(2, self.ROPE, device="cuda")
        get_mla_kv_buffer_triton(kv, loc, nope_out, rope_out)
        self.assertTrue(torch.allclose(nope_out[0], kv[2, : self.NOPE]))
        self.assertTrue(torch.allclose(rope_out[1], kv[5, self.NOPE :]))

    def test_get_single_location(self):
        from sglang.srt.mem_cache.utils import get_mla_kv_buffer_triton

        total = self.NOPE + self.ROPE
        kv = torch.rand(8, total, dtype=torch.float32, device="cuda")
        loc = torch.tensor([3], dtype=torch.int64, device="cuda")
        nope_out = torch.empty(1, self.NOPE, device="cuda")
        rope_out = torch.empty(1, self.ROPE, device="cuda")
        get_mla_kv_buffer_triton(kv, loc, nope_out, rope_out)
        self.assertTrue(torch.allclose(nope_out[0], kv[3, : self.NOPE]))
        self.assertTrue(torch.allclose(rope_out[0], kv[3, self.NOPE :]))


if __name__ == "__main__":
    unittest.main()
