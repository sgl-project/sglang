# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""RMSNorm architecture gate: the JIT rmsnorm family serves only SM100 / SM103. On any
other CUDA device (other SM10x devices included) RMSNorm keeps the sgl_kernel kernels,
so its bf16 output must be bitwise equal to sgl_kernel rmsnorm / fused_add_rmsnorm."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

# Hidden sizes of the DeepSeek-style norms: hidden, q_lora, kv_lora, head dim.
HIDDEN_SIZES = (7168, 1536, 512, 128)
MAX_ROWS = 4096
JIT_ARCHS = ((10, 0), (10, 3))


class TestRMSNormArchGate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            raise unittest.SkipTest("needs a CUDA device")
        if torch.cuda.get_device_capability() in JIT_ARCHS:
            raise unittest.SkipTest("the JIT rmsnorm family is meant for this device")
        try:
            from sgl_kernel import fused_add_rmsnorm, rmsnorm
        except ImportError:
            raise unittest.SkipTest("sgl_kernel is not available")
        cls.ref_rmsnorm = staticmethod(rmsnorm)
        cls.ref_fused_add_rmsnorm = staticmethod(fused_add_rmsnorm)

    def _layers(self):
        from sglang.srt.layers.layernorm import RMSNorm

        gen = torch.Generator(device="cuda").manual_seed(0)
        for hidden in HIDDEN_SIZES:
            layer = RMSNorm(hidden, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
            layer.weight.data.copy_(
                1.0
                + 0.1
                * torch.randn(hidden, device="cuda", generator=gen).to(torch.bfloat16)
            )
            x = torch.randn(MAX_ROWS, hidden, device="cuda", generator=gen)
            r = torch.randn(MAX_ROWS, hidden, device="cuda", generator=gen)
            yield hidden, layer, (4 * x).to(torch.bfloat16), r.to(torch.bfloat16)

    def test_jit_family_off(self):
        for hidden, layer, _, _ in self._layers():
            with self.subTest(hidden=hidden):
                self.assertFalse(layer.use_jit_rmsnorm)

    def test_plain_bitwise_equal_to_sgl_kernel(self):
        with torch.inference_mode():
            for hidden, layer, x, _ in self._layers():
                bad = []
                for rows in range(1, MAX_ROWS + 1):
                    xi = x[:rows].contiguous()
                    got = layer.forward_cuda(xi.clone())
                    want = self.ref_rmsnorm(
                        xi.clone(), layer.weight.data, layer.variance_epsilon
                    )
                    if not torch.equal(got, want):
                        bad.append(rows)
                self.assertFalse(
                    bad, f"hidden={hidden}: {len(bad)} row counts differ: {bad[:8]}"
                )

    def test_fused_add_bitwise_equal_to_sgl_kernel(self):
        with torch.inference_mode():
            for hidden, layer, x, r in self._layers():
                bad = []
                for rows in range(1, MAX_ROWS + 1):
                    xi, ri = x[:rows].contiguous(), r[:rows].contiguous()
                    got_x, got_r = layer.forward_cuda(xi.clone(), ri.clone())
                    want_x, want_r = xi.clone(), ri.clone()
                    self.ref_fused_add_rmsnorm(
                        want_x, want_r, layer.weight.data, layer.variance_epsilon
                    )
                    same_x = torch.equal(got_x, want_x)
                    if not (same_x and torch.equal(got_r, want_r)):
                        bad.append(rows)
                self.assertFalse(
                    bad, f"hidden={hidden}: {len(bad)} row counts differ: {bad[:8]}"
                )


if __name__ == "__main__":
    unittest.main()
