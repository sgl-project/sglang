# Adapted from https://github.com/vllm-project/vllm/blob/main/tests/kernels/quantization/test_awq_triton.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
unittest version of the AWQ Triton kernel tests.

Run with:
    python -m unittest test_awq_dequant.py
"""
import unittest

import torch

from sglang.srt.layers.quantization.awq_triton import (
    AWQ_TRITON_SUPPORTED_GROUP_SIZES,
    awq_dequantize_triton,
    awq_gemm_triton,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=2, suite="stage-a-test-1-amd")

device = "cuda"


def reverse_awq_order(t: torch.Tensor) -> torch.Tensor:
    bits = 4
    AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
    idx = torch.arange(t.shape[-1], dtype=torch.int32, device=t.device)
    idx = idx.view(-1, 32 // bits)[:, AWQ_REVERSE_ORDER].view(-1)
    return (t[:, idx] & 0xF).contiguous()


def awq_dequantize_torch(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if group_size == -1:
        group_size = qweight.shape[0]

    bits = 4
    shifts = torch.arange(0, 32, bits, device=qzeros.device)

    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(
        torch.int8
    )
    iweights = reverse_awq_order(iweights.view(iweights.shape[0], -1))

    zeros = torch.bitwise_right_shift(qzeros[:, :, None], shifts[None, None, :]).to(
        torch.int8
    )
    zeros = reverse_awq_order(zeros.view(qzeros.shape[0], -1))

    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)

    scales = scales.repeat_interleave(group_size, dim=0)
    zeros = zeros.repeat_interleave(group_size, dim=0)
    return (iweights - zeros) * scales


class TestAWQTriton(CustomTestCase):
    def test_dequantize(self):
        rows_list = [3584, 18944, 128, 256, 512, 1024]
        cols_list = [448, 576, 4736, 16, 32, 64, 128]

        for qweight_rows in rows_list:
            for qweight_cols in cols_list:
                for group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES:
                    with self.subTest(
                        rows=qweight_rows, cols=qweight_cols, g=group_size
                    ):
                        self._run_dequant_case(
                            qweight_rows=qweight_rows,
                            qweight_cols=qweight_cols,
                            group_size=group_size,
                        )

    def _run_dequant_case(self, qweight_rows, qweight_cols, group_size):
        if group_size == -1:
            group_size = qweight_rows

        torch.manual_seed(0)

        qweight = torch.randint(
            0,
            torch.iinfo(torch.int32).max,
            (qweight_rows, qweight_cols),
            dtype=torch.int32,
            device=device,
        )
        scales = torch.rand(
            qweight_rows // group_size,
            qweight_cols * 8,
            dtype=torch.float16,
            device=device,
        )
        zeros = torch.randint(
            0,
            torch.iinfo(torch.int32).max,
            (qweight_rows // group_size, qweight_cols),
            dtype=torch.int32,
            device=device,
        )

        ref = awq_dequantize_torch(qweight, scales, zeros, group_size)
        tri = awq_dequantize_triton(qweight, scales, zeros)

        # sanity
        self.assertFalse(torch.any(torch.isinf(tri)) or torch.any(torch.isnan(tri)))
        torch.testing.assert_close(ref, tri)

    # GEMM
    def test_gemm(self):
        N_list = [1, 2, 4, 8, 14, 17, 23, 32]
        K_list = [128]
        M_list = [16, 24, 32]
        splitK_list = [1, 8]

        for N in N_list:
            for K in K_list:
                for M in M_list:
                    for group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES:
                        for splitK in splitK_list:
                            with self.subTest(N=N, K=K, M=M, g=group_size, sk=splitK):
                                self._run_gemm_case(
                                    N=N,
                                    K=K,
                                    M=M,
                                    group_size=group_size,
                                    splitK=splitK,
                                )

    def _run_gemm_case(self, N, K, M, group_size, splitK):
        if group_size == -1:
            group_size = K

        torch.manual_seed(0)

        x = torch.rand((N, K), dtype=torch.float32, device=device)
        qweight = torch.randint(
            0,
            torch.iinfo(torch.int32).max,
            (K, M // 8),
            dtype=torch.int32,
            device=device,
        )
        qzeros = torch.randint(
            0,
            torch.iinfo(torch.int32).max,
            (K // group_size, M // 8),
            dtype=torch.int32,
            device=device,
        )
        scales = torch.rand((K // group_size, M), dtype=torch.float32, device=device)

        tri_out = awq_gemm_triton(x, qweight, scales, qzeros, splitK)

        self.assertFalse(
            torch.any(torch.isinf(tri_out)) or torch.any(torch.isnan(tri_out))
        )

        # dequantize & compare
        w_deq = awq_dequantize_triton(qweight, scales, qzeros)
        ref_out = torch.matmul(x, w_deq)

        self.assertFalse(
            torch.any(torch.isinf(ref_out)) or torch.any(torch.isnan(ref_out))
        )

        torch.testing.assert_close(tri_out.cpu(), ref_out.cpu(), atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
