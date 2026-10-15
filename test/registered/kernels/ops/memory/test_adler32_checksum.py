"""Tests for Adler-32 GPU checksum against Python zlib.adler32."""

import unittest
import zlib

import torch

from sglang.kernels.ops.memory.adler32 import (
    adler32_checksum,
    adler32_regions_checksum,
    adler32_strided_checksum,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _ref_adler32(tensor: torch.Tensor) -> int:
    return zlib.adler32(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes())


def _ref_strided_adler32(tensors, indices, strides) -> int:
    parts = []
    for tensor, idx, stride in zip(tensors, indices, strides):
        raw = tensor.cpu().contiguous().flatten().view(torch.uint8)
        for i in idx.cpu().tolist():
            parts.append(raw[i * stride : (i + 1) * stride].numpy().tobytes())
    return zlib.adler32(b"".join(parts))


class TestAdler32(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.manual_seed(42)

    def _check_whole(self, tensor):
        self.assertEqual(adler32_checksum(tensor), _ref_adler32(tensor))

    def test_whole_small(self):
        self._check_whole(torch.tensor([1.0, 2, 3, 4], device="cuda"))

    def test_whole_single_element(self):
        self._check_whole(torch.tensor([42.0], device="cuda"))

    def test_whole_dtypes(self):
        for dtype, shape in [
            (torch.bfloat16, (1024, 128)),
            (torch.float16, (512, 64)),
            (torch.float32, (4096, 256)),
        ]:
            self._check_whole(torch.randn(*shape, dtype=dtype, device="cuda"))

    def _check_strided(self, tensors, indices, strides):
        actual = adler32_strided_checksum(
            [t.data_ptr() for t in tensors], strides, indices
        )
        expected = _ref_strided_adler32(tensors, indices, strides)
        self.assertEqual(actual, expected)

    def test_strided_single_tensor(self):
        t = torch.randn(100, 64, device="cuda")
        idx = torch.tensor([0, 5, 10, 50, 99], dtype=torch.int64, device="cuda")
        self._check_strided([t], [idx], [64 * 4])

    def test_strided_multi_tensor_different_strides(self):
        t1 = torch.randn(50, 32, dtype=torch.float16, device="cuda")
        t2 = torch.randn(80, 64, dtype=torch.float16, device="cuda")
        idx1 = torch.tensor([0, 10, 49], dtype=torch.int64, device="cuda")
        idx2 = torch.tensor([30, 79], dtype=torch.int64, device="cuda")
        self._check_strided([t1, t2], [idx1, idx2], [32 * 2, 64 * 2])

    def test_strided_many_items(self):
        t = torch.randn(1000, 128, dtype=torch.bfloat16, device="cuda")
        idx = torch.arange(1000, dtype=torch.int64, device="cuda")
        self._check_strided([t], [idx], [128 * 2])

    def test_regions(self):
        first = torch.randint(
            0, 256, (5 * 1024 * 1024,), dtype=torch.uint8, device="cuda"
        )
        second = torch.randint(0, 256, (12345,), dtype=torch.uint8, device="cuda")
        actual = adler32_regions_checksum(
            [first.data_ptr(), second.data_ptr()],
            [first.numel(), second.numel()],
            first.device,
        )
        expected = zlib.adler32(
            first.cpu().numpy().tobytes() + second.cpu().numpy().tobytes()
        )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
