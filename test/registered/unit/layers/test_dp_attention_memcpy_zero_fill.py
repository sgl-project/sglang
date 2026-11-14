from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import unittest
from typing import Tuple

import torch

from sglang.kernels.ops.memory.memcpy_triton import (
    memcpy_triton,
    memcpy_triton_with_zero_fill,
)


def _reference(
    dst_shape: Tuple[int, ...],
    src: torch.Tensor,
    offset_t: torch.Tensor,
    sz_t: torch.Tensor,
    offset_src: bool,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Two-step reference: zero-fill then memcpy_triton."""
    dst = torch.empty(dst_shape, dtype=dtype, device=device).fill_(7)  # garbage
    dst.fill_(0)
    memcpy_triton(
        dst=dst, src=src, dim=0, offset=offset_t, sz=sz_t, offset_src=offset_src
    )
    return dst


def _fused(
    dst_shape: Tuple[int, ...],
    src: torch.Tensor,
    offset_t: torch.Tensor,
    sz_t: torch.Tensor,
    offset_src: bool,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Single-step fused kernel under test."""
    dst = torch.empty(dst_shape, dtype=dtype, device=device).fill_(7)  # garbage
    memcpy_triton_with_zero_fill(
        dst=dst, src=src, dim=0, offset=offset_t, sz=sz_t, offset_src=offset_src
    )
    return dst


class TestMemcpyTritonWithZeroFill(unittest.TestCase):
    """Parity between `dst.fill_(0) + memcpy_triton` and the fused kernel (#24938)."""

    device = "cuda"

    def _run_case(
        self,
        dst_rows: int,
        src_rows: int,
        hidden: int,
        offset: int,
        sz: int,
        offset_src: bool,
        dtype: torch.dtype,
    ) -> None:
        dst_shape = (dst_rows, hidden)
        src = torch.randn((src_rows, hidden), device=self.device).to(dtype)
        offset_t = torch.tensor([offset], dtype=torch.int32, device=self.device)
        sz_t = torch.tensor([sz], dtype=torch.int32, device=self.device)

        ref = _reference(
            dst_shape=dst_shape,
            src=src,
            offset_t=offset_t,
            sz_t=sz_t,
            offset_src=offset_src,
            dtype=dtype,
            device=self.device,
        )
        out = _fused(
            dst_shape=dst_shape,
            src=src,
            offset_t=offset_t,
            sz_t=sz_t,
            offset_src=offset_src,
            dtype=dtype,
            device=self.device,
        )

        torch.testing.assert_close(out, ref)

    def test_gather_path_bf16(self):
        # offset_src=False: write src[0:sz] into dst[offset:offset+sz], rest zeros.
        self._run_case(
            8192,
            1024,
            7168,
            offset=2048,
            sz=1024,
            offset_src=False,
            dtype=torch.bfloat16,
        )

    def test_scatter_path_bf16(self):
        # offset_src=True: write src[offset:offset+sz] into dst[0:sz], rest zeros.
        self._run_case(
            1024,
            8192,
            7168,
            offset=2048,
            sz=1024,
            offset_src=True,
            dtype=torch.bfloat16,
        )

    def test_gather_path_int32(self):
        # Matches the int32 input-id branch in `_dp_gather_via_all_reduce`.
        dst_shape = (8192,)
        src = torch.randint(0, 32000, (1024,), dtype=torch.int32, device=self.device)
        offset_t = torch.tensor([2048], dtype=torch.int32, device=self.device)
        sz_t = torch.tensor([1024], dtype=torch.int32, device=self.device)

        ref = _reference(
            dst_shape=dst_shape,
            src=src,
            offset_t=offset_t,
            sz_t=sz_t,
            offset_src=False,
            dtype=torch.int32,
            device=self.device,
        )
        out = _fused(
            dst_shape=dst_shape,
            src=src,
            offset_t=offset_t,
            sz_t=sz_t,
            offset_src=False,
            dtype=torch.int32,
            device=self.device,
        )
        torch.testing.assert_close(out, ref)

    def test_zero_offset(self):
        self._run_case(
            4096, 512, 2048, offset=0, sz=512, offset_src=False, dtype=torch.bfloat16
        )

    def test_full_coverage(self):
        # sz == dst_rows: every dst position copied, no zeros written.
        self._run_case(
            1024, 1024, 2048, offset=0, sz=1024, offset_src=False, dtype=torch.bfloat16
        )

    def test_end_offset(self):
        # Copy region at the tail of dst.
        self._run_case(
            4096, 512, 2048, offset=3584, sz=512, offset_src=False, dtype=torch.bfloat16
        )

    def test_non_block_aligned(self):
        # Total elements not a multiple of BLOCK_SIZE (8192).
        self._run_case(
            999, 333, 131, offset=200, sz=333, offset_src=False, dtype=torch.bfloat16
        )

    def test_dtype_matrix(self):
        # Same shape across the float dtypes the gather/scatter paths see in practice.
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._run_case(
                    2048, 512, 4096, offset=1024, sz=512, offset_src=False, dtype=dtype
                )
                self._run_case(
                    512, 2048, 4096, offset=1024, sz=512, offset_src=True, dtype=dtype
                )


if __name__ == "__main__":
    unittest.main()
