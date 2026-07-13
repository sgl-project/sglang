"""Generic every-backend-vs-native parity harness for BaseFusedOp operators.

Part of RFC #29630, Phase 2. For each reworked fused op, enumerate its
available backends, run each one that is eligible on this platform, and
assert the output matches the pure-torch ``forward_native`` reference within
dtype tolerance. New backends added to an op are picked up automatically —
no per-kernel test boilerplate.
"""

import unittest

import torch

from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")

_DEVICE = "cuda"
# torch_compile is native under the hood; exclude it from the sweep to keep
# CI time down (compilation dominates) — it is exercised in the CPU lane.
_SKIP_BACKENDS = {KernelBackend.TORCH, KernelBackend.TORCH_COMPILE}
_TOLERANCE = {
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=2e-2, rtol=2e-2),
}


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFusedOpGpuParity(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _eligible_backends(self, op):
        return [
            b
            for b in op.available_backends()
            if b not in _SKIP_BACKENDS and op.backend_eligible(b)
        ]

    def _assert_close(self, got, ref, dtype, msg):
        torch.testing.assert_close(got, ref, **_TOLERANCE[dtype], msg=msg)

    def test_rmsnorm_backends_match_native(self):
        from sglang.kernels.ops.layernorm import _RMSNORM

        for dtype in (torch.float16, torch.bfloat16):
            for shape in ((1, 4096), (128, 4096), (7, 2048)):
                x = torch.randn(shape, dtype=dtype, device=_DEVICE)
                w = torch.randn(shape[-1], dtype=dtype, device=_DEVICE)
                ref = _RMSNORM.forward_native(x, w, 1e-6)
                for backend in self._eligible_backends(_RMSNORM):
                    got = _RMSNORM.forward(x, w, 1e-6, backend=backend)
                    self._assert_close(
                        got, ref, dtype, f"rmsnorm {backend.value} {dtype} {shape}"
                    )

    def test_fused_add_rmsnorm_backends_match_native(self):
        from sglang.kernels.ops.layernorm import _FUSED_ADD_RMSNORM

        for dtype in (torch.float16, torch.bfloat16):
            for shape in ((1, 4096), (128, 4096)):
                x0 = torch.randn(shape, dtype=dtype, device=_DEVICE)
                r0 = torch.randn(shape, dtype=dtype, device=_DEVICE)
                w = torch.randn(shape[-1], dtype=dtype, device=_DEVICE)
                x_ref, r_ref = x0.clone(), r0.clone()
                _FUSED_ADD_RMSNORM.forward_native(x_ref, r_ref, w, 1e-6)
                for backend in self._eligible_backends(_FUSED_ADD_RMSNORM):
                    x, r = x0.clone(), r0.clone()
                    _FUSED_ADD_RMSNORM.forward(x, r, w, 1e-6, backend=backend)
                    label = f"fused_add_rmsnorm {backend.value} {dtype} {shape}"
                    self._assert_close(x, x_ref, dtype, label + " (normed)")
                    self._assert_close(r, r_ref, dtype, label + " (residual)")

    def test_gemma_rmsnorm_backends_match_native(self):
        from sglang.kernels.ops.layernorm import _GEMMA_RMSNORM

        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(64, 2048, dtype=dtype, device=_DEVICE)
            w = torch.randn(2048, dtype=dtype, device=_DEVICE)
            ref = _GEMMA_RMSNORM.forward_native(x, w, 1e-6)
            for backend in self._eligible_backends(_GEMMA_RMSNORM):
                got = _GEMMA_RMSNORM.forward(x, w, 1e-6, backend=backend)
                self._assert_close(
                    got, ref, dtype, f"gemma_rmsnorm {backend.value} {dtype}"
                )

    def test_gemma_fused_add_rmsnorm_backends_match_native(self):
        from sglang.kernels.ops.layernorm import _GEMMA_FUSED_ADD_RMSNORM

        for dtype in (torch.float16, torch.bfloat16):
            x0 = torch.randn(64, 2048, dtype=dtype, device=_DEVICE)
            r0 = torch.randn(64, 2048, dtype=dtype, device=_DEVICE)
            w = torch.randn(2048, dtype=dtype, device=_DEVICE)
            x_ref, r_ref = x0.clone(), r0.clone()
            _GEMMA_FUSED_ADD_RMSNORM.forward_native(x_ref, r_ref, w, 1e-6)
            for backend in self._eligible_backends(_GEMMA_FUSED_ADD_RMSNORM):
                x, r = x0.clone(), r0.clone()
                _GEMMA_FUSED_ADD_RMSNORM.forward(x, r, w, 1e-6, backend=backend)
                label = f"gemma_fused_add_rmsnorm {backend.value} {dtype}"
                self._assert_close(x, x_ref, dtype, label + " (normed)")
                self._assert_close(r, r_ref, dtype, label + " (residual)")

    def test_gated_activation_backends_match_native(self):
        from sglang.kernels.ops.activation import (
            _GELU_AND_MUL,
            _GELU_TANH_AND_MUL,
            _SILU_AND_MUL,
        )

        for op in (_SILU_AND_MUL, _GELU_AND_MUL, _GELU_TANH_AND_MUL):
            for dtype in (torch.float16, torch.bfloat16):
                for shape in ((1, 8192), (128, 8192)):
                    x = torch.randn(shape, dtype=dtype, device=_DEVICE)
                    ref = op.forward_native(x)
                    for backend in self._eligible_backends(op):
                        got = op.forward(x, backend=backend)
                        self._assert_close(
                            got, ref, dtype, f"{op.op} {backend.value} {dtype} {shape}"
                        )


if __name__ == "__main__":
    unittest.main()
