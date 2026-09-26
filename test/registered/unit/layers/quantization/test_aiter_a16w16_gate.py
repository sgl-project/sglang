import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.unquant import _prefer_triton_a16w16
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _tensor(rows, cols, *, dtype=torch.bfloat16, device="cuda:0", contiguous=True):
    return SimpleNamespace(
        shape=(rows, cols),
        dtype=dtype,
        device=device,
        is_cuda=device.startswith("cuda"),
        dim=lambda: 2,
        is_contiguous=lambda: contiguous,
    )


class TestAiterA16W16Gate(CustomTestCase):
    def test_supported_decode_shapes(self):
        for rows, k in ((1, 2048), (64, 2048), (512, 512)):
            with self.subTest(rows=rows, k=k):
                self.assertTrue(_prefer_triton_a16w16(_tensor(rows, k), _tensor(16, k)))

    def test_unsupported_shapes(self):
        cases = (
            (_tensor(65, 2048), _tensor(16, 2048)),
            (_tensor(513, 512), _tensor(16, 512)),
            (_tensor(1, 2049), _tensor(16, 2049)),
            (_tensor(1, 2048), _tensor(16, 1024)),
            (_tensor(0, 2048), _tensor(16, 2048)),
            (_tensor(1, 2048, device="cpu"), _tensor(16, 2048, device="cpu")),
            (_tensor(1, 2048, contiguous=False), _tensor(16, 2048)),
        )
        for x, weight in cases:
            with self.subTest(x=x.shape, weight=weight.shape):
                self.assertFalse(_prefer_triton_a16w16(x, weight))


if __name__ == "__main__":
    unittest.main()
