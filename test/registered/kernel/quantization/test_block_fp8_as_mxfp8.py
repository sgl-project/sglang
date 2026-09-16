"""WO-A kernel quantization parity and CUDA graph replay checks."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

DEVICE = "cuda"


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "MXFP8 dense kernels unavailable (needs Blackwell + FlashInfer)",
)
class TestLinearNumerics(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def test_fused_wo_a_quantized_input(self):
        from flashinfer import mxfp8_quantize

        from sglang.kernels.ops.attention.dsv4.wo_a_bf16_small_batch import (
            _quantize_partial,
            _wo_a_reduce,
            wo_a_bf16_small_batch,
            wo_a_bf16_small_batch_mxfp8,
        )

        for rows in range(2, 9):
            for magnitude in (0.0, 1e-37, 1e-7, 1.0, 448.0, 1e10):
                partial = torch.randn(8, rows, 2, 1024, device=DEVICE) * magnitude
                bf16 = torch.empty(rows, 2048, dtype=torch.bfloat16, device=DEVICE)
                _wo_a_reduce[(rows * 8,)](partial, bf16, rows * 2048, num_warps=4)
                expected_q, expected_s = mxfp8_quantize(bf16, True, alignment=32)
                actual_q, actual_s = _quantize_partial(partial)
                torch.testing.assert_close(
                    actual_q.view(torch.uint8),
                    expected_q.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)

            x = torch.randn(rows, 64, 512, device=DEVICE, dtype=torch.bfloat16)[
                :, :16
            ].view(rows, 2, 4096)
            wo_a = (
                torch.randn(2, 1024, 4096, device=DEVICE, dtype=torch.bfloat16) * 0.02
            )
            bf16 = wo_a_bf16_small_batch(x, wo_a).flatten(1)
            q, s = wo_a_bf16_small_batch_mxfp8(x, wo_a)
            expected_q, expected_s = mxfp8_quantize(bf16, True, alignment=32)
            torch.testing.assert_close(
                q.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(s, expected_s, rtol=0, atol=0)

        partial = torch.randn(8, 6, 2, 1024, device=DEVICE)
        _quantize_partial(partial)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q, s = _quantize_partial(partial)
        for _ in range(3):
            partial.normal_()
            # The replay must regenerate scale padding as well as live rows.
            s.fill_(255)
            graph.replay()
            bf16 = torch.empty(6, 2048, dtype=torch.bfloat16, device=DEVICE)
            _wo_a_reduce[(48,)](partial, bf16, 6 * 2048, num_warps=4)
            eq, es = mxfp8_quantize(bf16, True, alignment=32)
            torch.testing.assert_close(
                q.view(torch.uint8), eq.view(torch.uint8), rtol=0, atol=0
            )
            torch.testing.assert_close(s, es, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
