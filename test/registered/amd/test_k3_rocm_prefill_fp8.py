"""Kimi-K3 prefill FP8 copies (k3_rocm_prefill_fp8) on gfx950: the FP8 GEMM
against BF16, the row-chunked call, and the write into a caller's buffer."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization import k3_rocm_prefill_fp8 as p
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _gfx950():
    return (
        bool(torch.version.hip)
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
    )


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@unittest.skipUnless(_gfx950(), "gfx950 (MI35x) only")
class TestK3RocmPrefillFP8(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.n, cls.k = p.GROUP_SHAPES["o_proj"]
        weight = (
            torch.randn(cls.n, cls.k, dtype=torch.bfloat16, device="cuda") / cls.k**0.5
        )
        cls.layer = SimpleNamespace(
            weight=torch.nn.Parameter(weight, requires_grad=False),
            quant_method=UnquantizedLinearMethod(),
        )
        cls.fp8 = p.PrefillFP8Weight("o_proj", weight, min_tokens=1024)
        cls.method = p.PrefillFP8LinearMethod(cls.fp8)

    def _x(self, m):
        return torch.randn(m, self.k, dtype=torch.bfloat16, device="cuda")

    def test_admitted_call_matches_bf16(self):
        x = self._x(2048)
        self.assertTrue(self.fp8.admits(self.layer, x, None))
        y = self.method.apply(self.layer, x)
        self.assertEqual(y.shape, (2048, self.n))
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertLess(
            _rel_err(y, torch.nn.functional.linear(x, self.layer.weight)), 0.05
        )

    def test_small_call_is_the_stock_bf16_path(self):
        x = self._x(64)
        torch.testing.assert_close(
            self.method.apply(self.layer, x),
            UnquantizedLinearMethod().apply(self.layer, x),
            rtol=0,
            atol=0,
        )

    def test_chunked_call_matches_bf16(self):
        # Chunks may pick other tuned kernels than one call, so compare to BF16.
        x = self._x(3000)
        out = torch.zeros(3000, self.n, dtype=torch.bfloat16, device="cuda")
        with mock.patch.object(p, "MAX_ROWS_PER_CALL", 1024):
            chunked = self.fp8.run(x, out)
        self.assertIs(chunked, out)
        self.assertLess(
            _rel_err(out, torch.nn.functional.linear(x, self.layer.weight)), 0.05
        )

    def test_call_above_the_row_bound_is_stable_and_equals_its_chunks(self):
        m = p.MAX_ROWS_PER_CALL + 3616
        x = self._x(m)
        first = self.fp8.run(x)
        for _ in range(4):
            self.assertTrue(torch.equal(self.fp8.run(x), first))
        bound = p.MAX_ROWS_PER_CALL
        self.assertTrue(torch.equal(first[:bound], self.fp8.run(x[:bound])))
        self.assertTrue(torch.equal(first[bound:], self.fp8.run(x[bound:])))

    def test_raw_call_writes_the_callers_buffer(self):
        x = self._x(2048)
        out = torch.zeros(2048, self.n, dtype=torch.bfloat16, device="cuda")
        self.assertTrue(self.method.apply_raw_into(self.layer, x, out))
        self.assertLess(
            _rel_err(out, torch.nn.functional.linear(x, self.layer.weight)), 0.05
        )


if __name__ == "__main__":
    unittest.main()
