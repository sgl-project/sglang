"""Weight-streaming BF16 GEMM for decode-sized LM heads.

A row's tiling and FP32 accumulation order depend only on (N, K), so a row must give
the same bits alone or in any batch of up to 16 rows, also under CUDA graph replay.
With SGLANG_ENABLE_BF16_SKINNY_LM_HEAD=1 both LM-head callers (target logits and the DSpark
draft projection) must route qualifying batches to it and everything else to cuBLAS.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.gemm import bf16_skinny_gemm as skinny
from sglang.kernels.ops.gemm.bf16_skinny_gemm import (
    MAX_M,
    bf16_skinny_gemm,
    bf16_skinny_supported,
)
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.models.dspark import project_through_lm_head
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=25, stage="base-b", runner_config="1-gpu-small")

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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestLmHeadDispatch(CustomTestCase):
    """Both production LM-head callers use the kernel only when enabled and eligible."""

    def _target_logits(self, hidden, weight):
        processor = SimpleNamespace(use_fp32_lm_head=False, rl_on_policy_target=None)
        return LogitsProcessor._compute_lm_head(
            processor, hidden, SimpleNamespace(weight=weight)
        )

    def _draft_logits(self, hidden, weight):
        return project_through_lm_head(
            hidden, SimpleNamespace(weight=weight, quant_method=None)
        )

    def test_callers_route_to_the_kernel(self):
        x, w = _inputs(6, 1000, 256, seed=4)
        cases = {
            "eligible": (True, x, w, True),
            "disabled": (False, x, w, False),
            "rows > 16": (True, _inputs(MAX_M + 1, 1000, 256, seed=4)[0], w, False),
            "float16": (True, x.half(), w.half(), False),
            "strided columns": (True, x.t().contiguous().t(), w, False),
        }
        for caller in (self._target_logits, self._draft_logits):
            for name, (enabled, hidden, weight, expect_kernel) in cases.items():
                with self.subTest(caller=caller.__name__, case=name):
                    with (
                        envs.SGLANG_ENABLE_BF16_SKINNY_LM_HEAD.override(enabled),
                        mock.patch.object(
                            skinny, "bf16_skinny_gemm", wraps=bf16_skinny_gemm
                        ) as kernel,
                    ):
                        logits = caller(hidden, weight)
                    self.assertEqual(kernel.called, expect_kernel)
                    if expect_kernel:
                        want = bf16_skinny_gemm(hidden, weight)
                    else:
                        want = hidden @ weight.T
                    self.assertTrue(torch.equal(_bits(logits), _bits(want)))


if __name__ == "__main__":
    unittest.main()
