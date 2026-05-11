from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="stage-b-test-1-gpu-small")

import unittest

import torch

from sglang.srt.layers.dp_attention import (
    memcpy_triton,
    memcpy_triton_with_zero_fill,
)


def _reference(dst_shape, src, offset_t, sz_t, offset_src, dtype, device):
    """Two-step reference: zero-fill then memcpy_triton, matching the
    pre-fusion sequence in `_dp_gather_via_all_reduce` / `dp_scatter`."""
    dst = torch.empty(dst_shape, dtype=dtype, device=device).fill_(7)  # garbage
    dst.fill_(0)
    memcpy_triton(dst, src, 0, offset_t, sz_t, offset_src)
    return dst


def _fused(dst_shape, src, offset_t, sz_t, offset_src, dtype, device):
    """Single-step fused kernel under test."""
    dst = torch.empty(dst_shape, dtype=dtype, device=device).fill_(7)  # garbage
    memcpy_triton_with_zero_fill(dst, src, 0, offset_t, sz_t, offset_src)
    return dst


class TestMemcpyTritonWithZeroFill(unittest.TestCase):
    """Correctness parity between `dst.fill_(0) + memcpy_triton(...)` and
    the fused `memcpy_triton_with_zero_fill(...)` introduced by #24938."""

    device = "cuda"

    def _run_case(self, dst_rows, src_rows, hidden, offset, sz, offset_src, dtype):
        dst_shape = (dst_rows, hidden)
        src = torch.randn((src_rows, hidden), device=self.device).to(dtype)
        offset_t = torch.tensor([offset], dtype=torch.int32, device=self.device)
        sz_t = torch.tensor([sz], dtype=torch.int32, device=self.device)

        ref = _reference(dst_shape, src, offset_t, sz_t, offset_src, dtype, self.device)
        out = _fused(dst_shape, src, offset_t, sz_t, offset_src, dtype, self.device)

        torch.testing.assert_close(out, ref)

    def test_gather_path_bf16(self):
        # offset_src=False: write src[0:sz] into dst[offset:offset+sz], rest zeros.
        self._run_case(8192, 1024, 7168, offset=2048, sz=1024, offset_src=False, dtype=torch.bfloat16)

    def test_scatter_path_bf16(self):
        # offset_src=True: write src[offset:offset+sz] into dst[0:sz], rest zeros.
        self._run_case(1024, 8192, 7168, offset=2048, sz=1024, offset_src=True, dtype=torch.bfloat16)

    def test_gather_path_int32(self):
        # Matches the int32 input-id branch in `_dp_gather_via_all_reduce`.
        dst_shape = (8192,)
        src = torch.randint(0, 32000, (1024,), dtype=torch.int32, device=self.device)
        offset_t = torch.tensor([2048], dtype=torch.int32, device=self.device)
        sz_t = torch.tensor([1024], dtype=torch.int32, device=self.device)

        ref = _reference(dst_shape, src, offset_t, sz_t, False, torch.int32, self.device)
        out = _fused(dst_shape, src, offset_t, sz_t, False, torch.int32, self.device)
        torch.testing.assert_close(out, ref)

    def test_zero_offset(self):
        self._run_case(4096, 512, 2048, offset=0, sz=512, offset_src=False, dtype=torch.bfloat16)

    def test_full_coverage(self):
        # sz == dst_rows: every dst position copied, no zeros written.
        self._run_case(1024, 1024, 2048, offset=0, sz=1024, offset_src=False, dtype=torch.bfloat16)

    def test_end_offset(self):
        # Copy region at the tail of dst.
        self._run_case(4096, 512, 2048, offset=3584, sz=512, offset_src=False, dtype=torch.bfloat16)

    def test_non_block_aligned(self):
        # Total elements not a multiple of BLOCK_SIZE (8192).
        self._run_case(999, 333, 131, offset=200, sz=333, offset_src=False, dtype=torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
