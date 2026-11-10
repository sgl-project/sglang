"""SM70 dense MXFP8 GEMV vs fp32/UE8M0 dequant. Not registered for GPU CI."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.sm70_dsv41_mxfp8_linear import (
    sm70_dsv41_mxfp8_linear,
)
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    dequant_mxfp8_ue8m0_to_fp16,
    prepare_mxfp8_layer_for_sm70_marlin,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=30,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)


def _require_sm70():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")
    if torch.cuda.get_device_capability()[0] != 7:
        raise unittest.SkipTest("SM70 required")


def _mxfp8_tensors(n, k, *, scale_byte=127, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w_f = torch.randn(n, k, generator=g, dtype=torch.float32).clamp(-8, 8)
    weight = w_f.to(device=device, dtype=torch.float8_e4m3fn)
    scales = torch.full((n, k // 32), scale_byte, dtype=torch.uint8, device=device)
    return weight, scales


def _fp32_ref(x, weight, scales):
    n, k = weight.shape
    sf = torch.pow(
        torch.tensor(2.0, device=weight.device),
        scales.view(torch.uint8).to(dtype=torch.float32) - 127.0,
    )
    wf = weight.float() * sf.unsqueeze(-1).expand(-1, -1, 32).reshape(n, k)
    return x.float() @ wf.t()


class TestDsv41Sm70Mxfp8Linear(CustomTestCase):
    def setUp(self):
        _require_sm70()
        self.dev = torch.device("cuda")

    def test_gemv_matches_fp32_dequant(self):
        for m, n, k, byte in (
            (1, 64, 64, 127),
            (2, 96, 128, 120),
            (3, 64, 64, 128),
            (4, 128, 96, 113),
            (1, 32, 64, 109),
        ):
            with self.subTest(m=m, n=n, k=k, byte=byte):
                weight, scales = _mxfp8_tensors(
                    n, k, scale_byte=byte, seed=m + n + k + byte, device=self.dev
                )
                x = torch.randn(m, k, dtype=torch.float16, device=self.dev)
                got = sm70_dsv41_mxfp8_linear(x, weight, scales)
                ref = _fp32_ref(x, weight, scales).to(torch.float16)
                torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)

    def test_gemv_matches_fp16_unpack_except_subnormal_scales(self):
        n, k = 64, 128
        weight, scales = _mxfp8_tensors(n, k, scale_byte=127, seed=9, device=self.dev)
        x = torch.randn(1, k, dtype=torch.float16, device=self.dev)
        got = sm70_dsv41_mxfp8_linear(x, weight, scales)
        w16 = dequant_mxfp8_ue8m0_to_fp16(weight, scales, (1, 32))
        ref = F.linear(x, w16)
        torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)

    def test_byte_109_product_survives(self):
        n, k = 32, 64
        weight = torch.full((n, k), 16.0, device=self.dev).to(torch.float8_e4m3fn)
        scales = torch.full((n, k // 32), 109, dtype=torch.uint8, device=self.dev)
        x = torch.ones(1, k, dtype=torch.float16, device=self.dev)
        got = sm70_dsv41_mxfp8_linear(x, weight, scales)
        expected = k * (2.0 ** (109 - 127 + 4))
        self.assertTrue(torch.isfinite(got).all())
        torch.testing.assert_close(
            got, torch.full_like(got, expected), rtol=2e-3, atol=2e-3
        )

    def test_prepare_then_apply_roundtrip(self):
        n, k = 64, 64
        weight, _ = _mxfp8_tensors(n, k, scale_byte=128, seed=3, device="cpu")
        layer = torch.nn.Module()
        layer.weight_block_size = [32, 32]
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(
            torch.full((n // 32, k // 32), 128, dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            ),
            requires_grad=False,
        )
        layer.orig_dtype = torch.float16
        prepare_mxfp8_layer_for_sm70_marlin(layer)
        self.assertTrue(layer._sm70_mxfp8_w8a16)
        x = torch.randn(1, k, dtype=torch.float16, device=self.dev)
        got = sm70_dsv41_mxfp8_linear(
            x, layer.weight.to(self.dev), layer.weight_scale_inv.to(self.dev)
        )
        w16 = dequant_mxfp8_ue8m0_to_fp16(
            layer.weight.to(self.dev), layer.weight_scale_inv.to(self.dev), (1, 32)
        )
        ref = F.linear(x, w16)
        torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)

    def test_prefill_m_gt_4_uses_dequant_linear(self):
        n, k = 64, 64
        weight, scales = _mxfp8_tensors(n, k, scale_byte=127, seed=4, device=self.dev)
        x = torch.randn(8, k, dtype=torch.float16, device=self.dev)
        got = sm70_dsv41_mxfp8_linear(x, weight, scales)
        w16 = dequant_mxfp8_ue8m0_to_fp16(weight, scales, (1, 32))
        x_pad = torch.zeros(32, k, dtype=x.dtype, device=x.device)
        x_pad[:8] = x
        ref = F.linear(x_pad, w16)[:8]
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    def test_microbench_m1_vs_fp16_linear(self):
        n, k = 5120, 5120
        weight, scales = _mxfp8_tensors(n, k, scale_byte=127, seed=5, device=self.dev)
        x = torch.randn(1, k, dtype=torch.float16, device=self.dev)
        w16 = dequant_mxfp8_ue8m0_to_fp16(weight, scales, (1, 32))
        for _ in range(5):
            sm70_dsv41_mxfp8_linear(x, weight, scales)
            F.linear(x, w16)
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(20):
            sm70_dsv41_mxfp8_linear(x, weight, scales)
        ender.record()
        torch.cuda.synchronize()
        gemv_ms = starter.elapsed_time(ender) / 20
        starter.record()
        for _ in range(20):
            F.linear(x, w16)
        ender.record()
        torch.cuda.synchronize()
        lin_ms = starter.elapsed_time(ender) / 20
        print(
            f"MXFP8 GEMV M=1 {n}x{k}: {gemv_ms:.3f} ms vs FP16 F.linear "
            f"{lin_ms:.3f} ms ({lin_ms / gemv_ms:.2f}x)"
        )
        self.assertTrue(torch.isfinite(sm70_dsv41_mxfp8_linear(x, weight, scales)).all())


if __name__ == "__main__":
    unittest.main()
