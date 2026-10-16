"""DSV4.1 low-latency compressor projections: ll_bf16 dispatch, which layers take
the low-M GEMMs, and the 1-8-row routing. GPU (SM100 family with CuTe DSL): the
installed projections match the stock GEMMs at the served shapes."""

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=60, stage="base-b", runner_config="4-gpu-b200")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4 import dsv41_sparse
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor
from sglang.srt.models.dsv41_low_latency import indexer as ll_indexer
from sglang.srt.models.dsv41_low_latency import ll_bf16
from sglang.srt.models.dsv41_low_latency.ll_bf16 import CompileKey, dispatch
from sglang.srt.models.dsv41_low_latency.low_m_gemm import cutedsl_low_m_supported
from sglang.test.test_utils import CustomTestCase

HIDDEN = 5120
HEAD_DIM = 512


class TestLlBf16Dispatch(CustomTestCase):
    def test_tuned_compressor_shapes(self):
        # GB300 configs measured at the DSpark verify size.
        self.assertEqual(
            dispatch(M=6, K=5120, N=1024),
            CompileKey(backend="splitk", split_k=4, num_stages=4),
        )
        self.assertEqual(
            dispatch(M=6, K=5120, N=512),
            CompileKey(backend="dotprod", M=6, K=5120, bs=64),
        )
        self.assertEqual(
            dispatch(M=8, K=5120, N=512),
            CompileKey(backend="splitk", split_k=4, num_stages=6),
        )

    def test_untuned_shapes_use_the_default_dispatch(self):
        dotprod = CompileKey(backend="dotprod", M=3, K=5120, bs=128)
        self.assertEqual(dispatch(M=3, K=5120, N=1024), dotprod)
        # Narrow N would leave split-K with too few CTAs.
        self.assertEqual(dispatch(M=7, K=5120, N=32).backend, "dotprod")
        self.assertEqual(
            dispatch(M=7, K=5120, N=512),
            CompileKey(backend="splitk", split_k=6, num_stages=4),
        )


class TestLlBf16Operands(CustomTestCase):
    def test_trainable_weight_reaches_the_kernel_detached(self):
        """An nn.Linear weight requires grad; DLPack export of it fails at graph capture."""
        seen = []
        w = torch.nn.Parameter(torch.zeros(1024, 5120, dtype=torch.bfloat16))
        x = torch.zeros(6, 5120, dtype=torch.bfloat16)
        with (
            patch.object(
                ll_bf16, "compile_kernel", return_value=lambda *a: seen.extend(a)
            ),
            patch.object(ll_bf16, "_stream", return_value=None),
        ):
            ll_bf16.ll_bf16_gemm(x, w)
        tensors = [a for a in seen if isinstance(a, torch.Tensor)]
        self.assertEqual(len(tensors), 3)
        self.assertFalse(any(t.requires_grad for t in tensors))


def _attn(compressor=True, indexer=True):
    return SimpleNamespace(
        compressor=SimpleNamespace(low_m_bf16_gemm=None, low_m_fp32_gemm=None)
        if compressor
        else None,
        indexer=SimpleNamespace(low_m_gemm=None) if indexer else None,
    )


class TestInstallProjections(CustomTestCase):
    def _install(self, attn, *, ll_bf16_ok=True, cutedsl_ok=True):
        with (
            patch.object(ll_indexer, "ll_bf16_supported", return_value=ll_bf16_ok),
            patch.object(
                ll_indexer, "cutedsl_low_m_supported", return_value=cutedsl_ok
            ),
        ):
            ll_indexer.install_projections(attn)
        return attn

    def test_installs_on_the_compressor_and_indexer(self):
        attn = self._install(_attn())
        self.assertIsNotNone(attn.compressor.low_m_bf16_gemm)
        self.assertIsNotNone(attn.compressor.low_m_fp32_gemm)
        self.assertIsNotNone(attn.indexer.low_m_gemm)
        # Layers without a compressor or an indexer take the other half alone.
        self.assertIsNotNone(self._install(_attn(compressor=False)).indexer.low_m_gemm)
        attn = self._install(_attn(indexer=False))
        self.assertIsNotNone(attn.compressor.low_m_fp32_gemm)

    def test_either_kernel_missing_installs_nothing(self):
        for ll_bf16_ok, cutedsl_ok in ((False, True), (True, False)):
            with self.subTest(ll_bf16=ll_bf16_ok, cutedsl=cutedsl_ok):
                attn = self._install(
                    _attn(), ll_bf16_ok=ll_bf16_ok, cutedsl_ok=cutedsl_ok
                )
                self.assertIsNone(attn.compressor.low_m_bf16_gemm)
                self.assertIsNone(attn.compressor.low_m_fp32_gemm)
                self.assertIsNone(attn.indexer.low_m_gemm)


def _compressor(ratio, installed=True):
    c = DeepseekV41Compressor(64, 32, ratio, 1e-6, fused_compress=True)
    if installed:
        attn = SimpleNamespace(compressor=c, indexer=None)
        with (
            patch.object(ll_indexer, "ll_bf16_supported", return_value=True),
            patch.object(ll_indexer, "cutedsl_low_m_supported", return_value=True),
        ):
            ll_indexer.install_projections(attn)
    return c


class TestCompressorRouting(CustomTestCase):
    def test_ratio2_fp32_projection_switches_at_1_to_8_rows(self):
        c = _compressor(2)
        bf16 = torch.bfloat16
        with (
            patch.object(ll_indexer, "_ll_bf16_gemm", return_value="ll") as ll,
            patch.object(
                dsv41_sparse, "linear_bf16_fp32", return_value="cublas"
            ) as cublas,
        ):
            self.assertEqual(c.project_fused(torch.zeros(1, 64, dtype=bf16)), "ll")
            self.assertEqual(c.project_fused(torch.zeros(8, 64, dtype=bf16)), "ll")
            self.assertEqual(c.project_fused(torch.zeros(9, 64, dtype=bf16)), "cublas")
            self.assertEqual(
                c.project_fused(torch.zeros(64, 6, dtype=bf16).T), "cublas"
            )
        self.assertEqual(ll.call_count, 2)
        self.assertEqual(cublas.call_count, 2)

    def test_ratio1_bf16_projection_switches_at_1_to_8_rows(self):
        bf16 = torch.bfloat16
        with patch.object(ll_indexer, "cutedsl_low_m_mm", return_value="ll") as ll:
            self.assertEqual(
                _compressor(1).project(torch.zeros(8, 64, dtype=bf16)), ("ll", None)
            )
            kv, _ = _compressor(1).project(torch.zeros(9, 64, dtype=bf16))
            stock, _ = _compressor(1, installed=False).project(
                torch.zeros(6, 64, dtype=bf16)
            )
        self.assertEqual(ll.call_count, 1)
        for out, rows in ((kv, 9), (stock, 6)):
            self.assertEqual((out.dtype, out.shape), (bf16, (rows, 32)))


def _gpu_ok() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 10
        and ll_bf16.is_available()
        and cutedsl_low_m_supported()
    )


@unittest.skipUnless(_gpu_ok(), "needs an SM100-family GPU with CuTe DSL")
class TestCompressorKernels(CustomTestCase):
    def test_kernels_match_fp32_reference_at_1_to_8_rows(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        for n in (2 * HEAD_DIM, HEAD_DIM):
            w = (torch.randn(n, HIDDEN, device="cuda", generator=g) * 0.02).bfloat16()
            for rows in range(1, 9):
                with self.subTest(n=n, rows=rows):
                    x = torch.randn(rows, HIDDEN, device="cuda", generator=g).bfloat16()
                    want = x.float() @ w.float().T
                    torch.testing.assert_close(
                        ll_bf16.ll_bf16_gemm(x, w), want, rtol=1e-4, atol=1e-4
                    )
                    torch.testing.assert_close(
                        ll_indexer.cutedsl_low_m_mm(x, w).float(),
                        want,
                        rtol=2**-7,
                        atol=1e-2,
                    )

    def test_installed_compressor_matches_the_stock_one(self):
        torch.manual_seed(0)
        for ratio in (1, 2):
            stock = DeepseekV41Compressor(
                HIDDEN, HEAD_DIM, ratio, 1e-6, fused_compress=True
            ).cuda()
            for p in stock.parameters():
                torch.nn.init.normal_(p, std=0.02)
            fast = DeepseekV41Compressor(
                HIDDEN, HEAD_DIM, ratio, 1e-6, fused_compress=True
            ).cuda()
            fast.load_state_dict(stock.state_dict())
            ll_indexer.install_projections(
                SimpleNamespace(compressor=fast, indexer=None)
            )
            self.assertIsNotNone(fast.low_m_fp32_gemm)
            for rows in (1, 6, 8):
                with self.subTest(ratio=ratio, rows=rows):
                    x = torch.randn(rows, HIDDEN, device="cuda").bfloat16()
                    if ratio == 1:
                        # Only the K summation order differs: one BF16 ulp at most.
                        torch.testing.assert_close(
                            fast.project_kv(x),
                            stock.project_kv(x),
                            atol=1e-2,
                            rtol=8e-3,
                        )
                    else:
                        torch.testing.assert_close(
                            fast.project_fused(x),
                            stock.project_fused(x),
                            atol=1e-4,
                            rtol=1e-4,
                        )


if __name__ == "__main__":
    unittest.main()
