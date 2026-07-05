"""Numeric unit tests for the XPU int4 *dense* linear kernels (GPTQ / AWQ).

Fast, deterministic, network-free: builds synthetic int4 weights in-memory and
checks that the XPU int4pack kernels dequantize + matmul correctly against a
pure-torch ``W = (q - zp) * scale`` reference. Gates every PR via
``stage-a-test-1-gpu-xpu``.
"""

import unittest

import torch

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, suite="stage-a-test-1-gpu-xpu")

DEV = "xpu"

REL_TOL = {torch.float16: 1.5e-3, torch.bfloat16: 1e-2}

# (K, N, group_size); K % 8 == 0, N % 8 == 0, K % group_size == 0.
SHAPES = [
    (128, 64, 16),
    (128, 64, 32),
    (256, 128, 64),
    (256, 128, 128),
    (512, 256, 256),
]

# AutoAWQ forward pack order (inverse of reverse [0, 4, 1, 5, 2, 6, 3, 7]).
AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _awq_pack(codes: torch.Tensor) -> torch.Tensor:
    """``[R, C]`` codes (0..15) -> ``[R, C // 8]`` int32 in AutoAWQ order."""
    r, c = codes.shape
    codes = codes.reshape(r, c // 8, 8)[:, :, AWQ_PACK_ORDER]
    packed = torch.zeros(r, c // 8, dtype=torch.int32, device=codes.device)
    for i in range(8):
        packed |= codes[:, :, i].to(torch.int32) << (4 * i)
    return packed


def _gptq_pack_qweight(codes: torch.Tensor) -> torch.Tensor:
    """``[K, N]`` codes -> ``[K // 8, N]`` int32 (packed sequentially along K)."""
    k, n = codes.shape
    codes = codes.reshape(k // 8, 8, n)
    packed = torch.zeros(k // 8, n, dtype=torch.int32, device=codes.device)
    for i in range(8):
        packed |= codes[:, i, :].to(torch.int32) << (4 * i)
    return packed


def _gptq_pack_qzeros(zc: torch.Tensor) -> torch.Tensor:
    """``[ng, N]`` codes -> ``[ng, N // 8]`` int32 (packed sequentially along N)."""
    ng, n = zc.shape
    zc = zc.reshape(ng, n // 8, 8)
    packed = torch.zeros(ng, n // 8, dtype=torch.int32, device=zc.device)
    for j in range(8):
        packed |= zc[:, :, j].to(torch.int32) << (4 * j)
    return packed


def _make_layer():
    """A bare ``LinearBase`` with only ``nn.Module`` machinery initialised."""
    from sglang.srt.layers.linear import LinearBase

    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    return layer


def _awq_config(group_size: int):
    from sglang.srt.layers.quantization.awq import AWQXPUConfig

    cfg = AWQXPUConfig.__new__(AWQXPUConfig)
    cfg.group_size = group_size
    cfg.weight_bits = 4
    cfg.pack_factor = 8
    cfg.zero_point = True
    cfg.lm_head_quantized = False
    cfg.modules_to_not_convert = []
    return cfg


def _gptq_config(group_size: int, desc_act: bool, fmt: str):
    from sglang.srt.layers.quantization.gptq import GPTQXPUConfig

    cfg = GPTQXPUConfig.__new__(GPTQXPUConfig)
    cfg.group_size = group_size
    cfg.desc_act = desc_act
    cfg.checkpoint_format = fmt
    cfg.weight_bits = 4
    cfg.lm_head_quantized = False
    cfg.dynamic = {}
    return cfg


@unittest.skipIf(not is_xpu(), "XPU int4 dense UT requires an Intel XPU")
class TestXPUInt4DenseKernel(CustomTestCase):
    """AWQ / GPTQ int4pack kernel numerics vs a pure-torch dequant reference."""

    def test_awq_numeric(self):
        for dtype in (torch.float16, torch.bfloat16):
            for k, n, gs in SHAPES:
                with self.subTest(dtype=dtype, K=k, N=n, gs=gs):
                    self._run_awq(k, n, gs, dtype)

    def _run_awq(self, k, n, gs, dtype):
        from sglang.srt.hardware_backend.xpu.quantization.awq_kernels import (
            AWQXPULinearKernel,
        )

        torch.manual_seed(0)
        ng = k // gs
        wcodes = torch.randint(0, 16, (k, n), device=DEV)
        zcodes = torch.randint(0, 16, (ng, n), device=DEV)
        scales = torch.rand(ng, n, device=DEV, dtype=dtype) * 0.05 + 0.005

        gidx = torch.arange(k, device=DEV) // gs
        w_ref = (wcodes.to(dtype) - zcodes[gidx].to(dtype)) * scales[gidx]
        x = torch.randn(7, k, device=DEV, dtype=dtype)
        ref = x @ w_ref

        layer = _make_layer()
        layer.qweight = torch.nn.Parameter(_awq_pack(wcodes), requires_grad=False)
        layer.qzeros = torch.nn.Parameter(_awq_pack(zcodes), requires_grad=False)
        layer.scales = torch.nn.Parameter(scales, requires_grad=False)

        kernel = AWQXPULinearKernel(_awq_config(gs))
        kernel.process_weights_after_loading(layer)
        out = kernel.apply(layer, x)

        self.assertEqual(tuple(out.shape), (7, n))
        self.assertTrue(torch.isfinite(out).all())
        rel = (out - ref).abs().max().item() / ref.abs().max().item()
        self.assertLess(rel, REL_TOL[dtype], f"rel={rel:.2e}")

    def test_gptq_numeric(self):
        for dtype in (torch.float16, torch.bfloat16):
            for fmt in ("", "gptq_v2"):
                for desc_act in (False, True):
                    for k, n, gs in SHAPES:
                        with self.subTest(
                            dtype=dtype,
                            fmt=fmt or "gptq_v1",
                            desc_act=desc_act,
                            K=k,
                            N=n,
                            gs=gs,
                        ):
                            self._run_gptq(k, n, gs, dtype, desc_act, fmt)

    def _run_gptq(self, k, n, gs, dtype, desc_act, fmt):
        from sglang.srt.hardware_backend.xpu.quantization.gptq_kernels import (
            GPTQXPULinearKernel,
        )

        torch.manual_seed(0)
        ng = k // gs
        qnat = torch.randint(0, 16, (k, n), device=DEV)
        zc = torch.randint(0, 14, (ng, n), device=DEV)  # room for v1 +1
        scales = torch.rand(ng, n, device=DEV, dtype=dtype) * 0.05 + 0.005

        if desc_act:
            base = torch.arange(k, device=DEV) // gs
            g_idx = base[torch.randperm(k, device=DEV)].to(torch.int32)
        else:
            g_idx = (torch.arange(k, device=DEV) // gs).to(torch.int32)

        zp_eff = zc + (0 if fmt == "gptq_v2" else 1)
        w_true = (qnat.to(dtype) - zp_eff[g_idx].to(dtype)) * scales[g_idx]
        x = torch.randn(5, k, device=DEV, dtype=dtype)
        ref = x @ w_true

        layer = _make_layer()
        layer.qweight = torch.nn.Parameter(
            _gptq_pack_qweight(qnat), requires_grad=False
        )
        layer.qzeros = torch.nn.Parameter(_gptq_pack_qzeros(zc), requires_grad=False)
        layer.scales = torch.nn.Parameter(scales, requires_grad=False)
        layer.g_idx = torch.nn.Parameter(g_idx, requires_grad=False)

        kernel = GPTQXPULinearKernel(_gptq_config(gs, desc_act, fmt))
        kernel.process_weights_after_loading(layer)
        out = kernel.apply(layer, x)

        self.assertEqual(tuple(out.shape), (5, n))
        self.assertTrue(torch.isfinite(out).all())
        rel = (out - ref).abs().max().item() / ref.abs().max().item()
        self.assertLess(rel, REL_TOL[dtype], f"rel={rel:.2e}")


if __name__ == "__main__":
    unittest.main()
