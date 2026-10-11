"""Weight-streaming BF16 GEMM for decode-sized LM heads.

A row's tiling and FP32 accumulation order depend only on (N, K), so a row must give
the same bits alone or in any batch of up to 16 rows, also under CUDA graph replay.
Row offsets are 64-bit, and the launch follows the operands' device. The LM-head
dispatch is tested in test/registered/unit/layers/test_logits_processor_bf16_lm_head.py.
"""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.gemm.bf16_skinny_gemm import (
    MAX_M,
    bf16_skinny_gemm,
    bf16_skinny_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

# A TP4 DeepSeek-V4.1 LM head shard, and a small shape whose N is not a block multiple.
SHAPES = [(32320, 4096), (1000, 256)]


def _inputs(m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=g).bfloat16()
    w = (torch.randn(n, k, device="cuda", generator=g) / k**0.5).bfloat16()
    return x, w


def _bits(t):
    return t.view(torch.int16)


def _error_bound(x, w):
    """FP64 result and the bound for an FP32-accumulated dot product rounded once to
    BF16: gamma_K * (|x| @ |w|.T) for the accumulation plus one BF16 ulp."""
    ref = x.double() @ w.double().T
    k = x.shape[1]
    gamma = k * 2.0**-24 / (1 - k * 2.0**-24)
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp_min(2.0**-126))) - 7)
    return ref, gamma * (x.double().abs() @ w.double().abs().T) + ulp


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestBf16SkinnyGemm(CustomTestCase):
    def test_within_fp32_accumulation_bound(self):
        for n, k in SHAPES:
            for m in (1, 6, MAX_M):
                with self.subTest(n=n, k=k, m=m):
                    x, w = _inputs(m, n, k)
                    ref, bound = _error_bound(x, w)
                    err = (bf16_skinny_gemm(x, w).double() - ref).abs()
                    self.assertTrue(bool((err <= bound).all()))

    def test_rows_are_batch_invariant(self):
        for n, k in SHAPES:
            x, w = _inputs(MAX_M, n, k, seed=1)
            alone = torch.cat([bf16_skinny_gemm(x[i : i + 1], w) for i in range(MAX_M)])
            for m in range(2, MAX_M + 1):
                with self.subTest(n=n, k=k, m=m):
                    self.assertTrue(
                        torch.equal(_bits(bf16_skinny_gemm(x[:m], w)), _bits(alone[:m]))
                    )

    def test_graph_replay_reads_new_activations(self):
        x, w = _inputs(6, 1000, 256, seed=2)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            bf16_skinny_gemm(x, w)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = bf16_skinny_gemm(x, w)
        x.copy_(_inputs(6, 1000, 256, seed=3)[0])
        graph.replay()
        self.assertTrue(torch.equal(_bits(out), _bits(bf16_skinny_gemm(x, w))))

    def test_strided_rows_match_contiguous(self):
        x, w = _inputs(6, 1000, 256, seed=5)
        x_rows = torch.zeros(6, 384, dtype=x.dtype, device="cuda")[:, :256]
        w_rows = torch.zeros(1000, 512, dtype=w.dtype, device="cuda")[:, :256]
        x_rows.copy_(x)
        w_rows.copy_(w)
        self.assertTrue(bf16_skinny_supported(x_rows, w_rows))
        self.assertTrue(
            torch.equal(
                _bits(bf16_skinny_gemm(x_rows, w_rows)), _bits(bf16_skinny_gemm(x, w))
            )
        )

    def test_row_offsets_beyond_int32(self):
        # Three weight rows 2**30 + 256 elements apart: the last row starts past
        # 2**31 elements, so 32-bit row offsets would wrap.
        n, k, row_stride = 3, 256, 2**30 + 256
        size = (n - 1) * row_stride + k
        self.assertGreater((n - 1) * row_stride, 2**31)
        if torch.cuda.mem_get_info()[0] < size * 2 + 2**30:
            self.skipTest("needs about 5 GiB of free GPU memory")
        x, w = _inputs(6, n, k, seed=7)
        storage = torch.empty(size, dtype=torch.bfloat16, device="cuda")
        w_far = storage.as_strided((n, k), (row_stride, 1))
        w_far.copy_(w)
        self.assertTrue(bf16_skinny_supported(x, w_far))
        self.assertTrue(
            torch.equal(
                _bits(bf16_skinny_gemm(x, w_far)), _bits(bf16_skinny_gemm(x, w))
            )
        )

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "needs two GPUs")
    def test_operands_on_a_non_current_device(self):
        x, w = _inputs(6, 1000, 256, seed=6)
        ref, bound = _error_bound(x, w)
        x1, w1 = x.to("cuda:1"), w.to("cuda:1")
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        with torch.cuda.device(0):
            # Keep device 0 busy, so a launch on its stream instead of device 1's
            # would not have run when device 1 is synchronized below.
            torch.cuda._sleep(2**31)
            out = bf16_skinny_gemm(x1, w1)
        torch.cuda.synchronize(1)
        self.assertEqual(out.device, x1.device)
        err = (out.cpu().double() - ref.cpu()).abs()
        self.assertTrue(bool((err <= bound.cpu()).all()))
        torch.cuda.synchronize(0)

    def test_supported_inputs(self):
        x, w = _inputs(4, 1000, 256)
        self.assertTrue(bf16_skinny_supported(x, w))
        rejected = {
            "rows > 16": (_inputs(MAX_M + 1, 1000, 256)[0], w),
            "float16": (x.half(), w.half()),
            "K % 128": (x[:, :192], w[:, :192]),
            "strided x columns": (x.t().contiguous().t(), w),
            "strided weight columns": (x, w.t().contiguous().t()),
            "weight on CPU": (x, w.cpu()),
        }
        for name, (a, b) in rejected.items():
            with self.subTest(name):
                self.assertFalse(bf16_skinny_supported(a, b))
        with self.subTest("ROCm"), mock.patch.object(torch.version, "hip", "6.4"):
            self.assertFalse(bf16_skinny_supported(x, w))


if __name__ == "__main__":
    unittest.main()
