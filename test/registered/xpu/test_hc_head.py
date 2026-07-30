import statistics
import time
import types
import unittest

import pytest
import torch

from sglang.kernels.ops.layernorm.mhc_head import fused_hc_head as triton_fused_hc_head
from sglang.srt.models.deepseek_v4 import DeepseekV4Model
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")


@pytest.fixture(autouse=True)
def skip_if_no_xpu():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("XPU not available")


def _sync_xpu() -> None:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


def _bench_ms(fn, warmup: int = 10, iters: int = 50):
    for _ in range(warmup):
        fn()
    _sync_xpu()

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync_xpu()
        samples.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


class TestHCHead(CustomTestCase):
    HC_MULT = 4
    HIDDEN_SIZE = 7168
    NUM_TOKENS = [1, 16, 128, 1024]
    DTYPE = torch.bfloat16
    NORM_EPS = 1e-6
    HC_EPS = 1e-6
    SEED = 2026

    @classmethod
    def setUpClass(cls):
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise unittest.SkipTest("XPU is not available")
        try:
            from sgl_kernel.mhc import fused_hc_head as _  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("sgl-kernel fused_hc_head is not available")
        torch.set_default_device("xpu")

    def _make_inputs(self, num_tokens: int):
        torch.manual_seed(self.SEED + num_tokens)
        x = torch.randn(
            num_tokens,
            self.HC_MULT,
            self.HIDDEN_SIZE,
            dtype=self.DTYPE,
            device="xpu",
        ).contiguous()
        hc_fn = torch.randn(
            self.HC_MULT,
            self.HC_MULT * self.HIDDEN_SIZE,
            dtype=torch.float32,
            device="xpu",
        ).contiguous()
        hc_base = torch.randn(
            self.HC_MULT,
            dtype=torch.float32,
            device="xpu",
        ).contiguous()
        hc_scale = torch.randn(1, dtype=torch.float32, device="xpu").contiguous()
        return x, hc_fn, hc_scale, hc_base

    def test_hc_head_triton_vs_sycl(self):
        for num_tokens in self.NUM_TOKENS:
            with self.subTest(num_tokens=num_tokens):
                x, hc_fn, hc_scale, hc_base = self._make_inputs(num_tokens)

                _ctx = types.SimpleNamespace(norm_eps=self.NORM_EPS, hc_eps=self.HC_EPS)

                with torch.inference_mode():
                    triton_out = triton_fused_hc_head(
                        x,
                        hc_fn,
                        hc_scale,
                        hc_base,
                        self.NORM_EPS,
                        self.HC_EPS,
                    )
                    # Go through the actual DeepseekV4Model.hc_head dispatch path
                    # (on XPU this calls sgl_kernel.mhc.fused_hc_head).
                    sycl_out = DeepseekV4Model.hc_head(
                        _ctx, x, hc_fn, hc_scale, hc_base
                    )

                torch.testing.assert_close(
                    triton_out,
                    sycl_out,
                    atol=2e-2,
                    rtol=2e-2,
                )

                triton_stats = _bench_ms(
                    lambda: triton_fused_hc_head(
                        x,
                        hc_fn,
                        hc_scale,
                        hc_base,
                        self.NORM_EPS,
                        self.HC_EPS,
                    )
                )
                sycl_stats = _bench_ms(
                    lambda: DeepseekV4Model.hc_head(_ctx, x, hc_fn, hc_scale, hc_base)
                )

                print(
                    "hc_head num_tokens={num_tokens}: "
                    "triton mean={triton_mean:.3f} ms median={triton_median:.3f} ms; "
                    "sycl mean={sycl_mean:.3f} ms median={sycl_median:.3f} ms".format(
                        num_tokens=num_tokens,
                        triton_mean=triton_stats["mean_ms"],
                        triton_median=triton_stats["median_ms"],
                        sycl_mean=sycl_stats["mean_ms"],
                        sycl_median=sycl_stats["median_ms"],
                    )
                )


if __name__ == "__main__":
    unittest.main()
