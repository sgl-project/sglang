"""Allowlist, numerical, device and CUDA Graph tests for the opt-in BA GEMM."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

# This runner maps to RTX 5090 (SM120), unlike 1-gpu-large (H100).
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class TestSm120BaCpuFallback(unittest.TestCase):
    def test_cpu_does_not_query_cuda(self):
        from sglang.kernels.ops.gemm.sm120_ba_gemm import sm120_ba_linear

        x, w = torch.randn(4, 5120), torch.randn(48, 5120)
        with patch("torch.cuda.get_device_capability", side_effect=AssertionError):
            torch.testing.assert_close(sm120_ba_linear(x, w), F.linear(x, w))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "Requires SM120",
)
class TestSm120BaGemm(unittest.TestCase):
    def setUp(self):
        from sglang.kernels.ops.gemm import sm120_ba_gemm

        self.op = sm120_ba_gemm
        torch.manual_seed(20260909)

    def tensors(self, m=4, n=48, k=5120, device="cuda", dtype=torch.bfloat16):
        return (
            torch.randn(m, k, device=device, dtype=dtype),
            torch.randn(n, k, device=device, dtype=dtype) * 0.01,
        )

    def test_all_allowed_rows(self):
        # Strict parity is an empirical regression gate for this environment,
        # not a promise about all cuBLAS versions, weights or input values.
        for m in range(2, 9):
            for seed in range(20):
                with self.subTest(m=m, seed=seed):
                    torch.manual_seed(seed)
                    x, w = self.tensors(m)
                    x0, w0 = x.clone(), w.clone()
                    self.assertTrue(self.op.can_use_sm120_ba_gemm(x, w))
                    out = self.op.sm120_ba_linear(x, w)
                    self.assertTrue(torch.isfinite(out).all())
                    torch.testing.assert_close(out, F.linear(x, w), rtol=0, atol=0)
                    torch.testing.assert_close(x, x0, rtol=0, atol=0)
                    torch.testing.assert_close(w, w0, rtol=0, atol=0)
                    self.assertNotEqual(out.data_ptr(), x.data_ptr())

    def assert_fallback(self, x, w, bias=None):
        self.assertFalse(self.op.can_use_sm120_ba_gemm(x, w, bias))
        with patch.object(self.op, "_ba_dot_kernel", side_effect=AssertionError):
            torch.testing.assert_close(
                self.op.sm120_ba_linear(x, w, bias),
                F.linear(x, w, bias),
                rtol=0,
                atol=0,
            )

    def test_shapes_and_layouts_fall_back(self):
        for m in (0, 1, 9, 16):
            with self.subTest(m=m):
                self.assert_fallback(*self.tensors(m))
        for n, k in ((47, 5120), (49, 5120), (48, 5119), (48, 5121)):
            with self.subTest(n=n, k=k):
                self.assert_fallback(*self.tensors(n=n, k=k))
        x, w = self.tensors()
        self.assert_fallback(x.view(2, 2, 5120), w)
        self.assert_fallback(x.t().contiguous().t(), w)
        self.assert_fallback(x, w.t().contiguous().t())
        self.assert_fallback(x, w, torch.zeros(48, device=x.device, dtype=x.dtype))
        self.assert_fallback(*self.tensors(dtype=torch.float32))
        self.assert_fallback(*self.tensors(dtype=torch.float16))

    def test_autograd_falls_back(self):
        x, w = self.tensors()
        x.requires_grad_(True)
        self.assert_fallback(x, w)
        self.op.sm120_ba_linear(x, w).float().sum().backward()
        self.assertIsNotNone(x.grad)
        x = x.detach()
        w.requires_grad_(True)
        self.assert_fallback(x, w)
        self.op.sm120_ba_linear(x, w).float().sum().backward()
        self.assertIsNotNone(w.grad)

    def test_execution_modes_fall_back(self):
        x, w = self.tensors()
        with torch.autocast("cuda", dtype=torch.float16):
            self.assert_fallback(x, w)
        with patch("torch.are_deterministic_algorithms_enabled", return_value=True):
            self.assert_fallback(x, w)
        with patch("torch.compiler.is_compiling", return_value=True):
            self.assert_fallback(x, w)
        with patch.object(self.op, "_is_sm120", return_value=False):
            self.assert_fallback(x, w)

    def test_compile_uses_original_path(self):
        x, w = self.tensors()
        compiled = torch.compile(
            self.op.sm120_ba_linear, backend="eager", fullgraph=True
        )
        with patch.object(self.op, "_ba_dot_kernel", side_effect=AssertionError):
            torch.testing.assert_close(compiled(x, w), F.linear(x, w), rtol=0, atol=0)

    def test_actual_unquantized_dispatch(self):
        from sglang.srt.layers.quantization import unquant

        for m in range(1, 10):
            x, w = self.tensors(m)
            layer = SimpleNamespace(
                prefix="model.layers.0.linear_attn.in_proj_ba",
                weight=torch.nn.Parameter(w, requires_grad=False),
            )
            with (
                patch.object(unquant, "_sm120_ba_linear", self.op.sm120_ba_linear),
                patch.object(unquant, "use_intel_amx_backend", return_value=False),
                patch.object(unquant, "_use_aiter", False),
                patch.object(
                    unquant, "is_batch_invariant_mode_enabled", return_value=False
                ),
            ):
                out = unquant.UnquantizedLinearMethod().apply(layer, x)
            torch.testing.assert_close(out, F.linear(x, w), rtol=0, atol=0)

    def test_cuda_graph_cold_fallback_and_warm_replay(self):
        for m in range(2, 9):
            with self.subTest(m=m):
                x, w = self.tensors(m)
                key = (x.device.index, m)
                self.op._warmed_shapes.discard(key)
                stream = torch.cuda.Stream(device=x.device)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        F.linear(x, w)
                torch.cuda.current_stream().wait_stream(stream)
                cold = torch.cuda.CUDAGraph()
                with patch.object(
                    self.op, "_ba_dot_kernel", side_effect=AssertionError
                ):
                    with torch.cuda.graph(cold, stream=stream):
                        cold_out = self.op.sm120_ba_linear(x, w)
                    cold.replay()
                torch.testing.assert_close(cold_out, F.linear(x, w), rtol=0, atol=0)
                self.assertNotIn(key, self.op._warmed_shapes)
                self.op.sm120_ba_linear(x, w)
                warm = torch.cuda.CUDAGraph()
                with torch.cuda.graph(warm, stream=stream):
                    warm_out = self.op.sm120_ba_linear(x, w)
                for _ in range(3):
                    x.copy_(torch.randn_like(x))
                    w.copy_(torch.randn_like(w) * 0.01)
                    warm.replay()
                    torch.testing.assert_close(warm_out, F.linear(x, w), rtol=0, atol=0)

    def test_noncurrent_device_and_cross_device_rejection(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("Requires two GPUs")
        original = torch.cuda.current_device()
        other = 1 if original == 0 else 0
        if torch.cuda.get_device_capability(other) != (12, 0):
            self.skipTest("Second GPU must be SM120")
        x, w = self.tensors(device=f"cuda:{other}")
        out = self.op.sm120_ba_linear(x, w)
        self.assertEqual(torch.cuda.current_device(), original)
        torch.testing.assert_close(out, F.linear(x, w), rtol=0, atol=0)
        self.assertFalse(self.op.can_use_sm120_ba_gemm(x, w.to(f"cuda:{original}")))


if __name__ == "__main__":
    unittest.main()
