"""XPU coverage for the FLUX.2 LayerNorm+modulate and gated residual norm fast paths."""

import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

import sglang.multimodal_gen.runtime.models.dits.flux_2 as flux2
from sglang.kernels.ops.diffusion import (
    can_use_xpu_gated_resnorm,
    can_use_xpu_layernorm_modulate,
    residual_gate_add,
    xpu_gated_resnorm,
    xpu_layernorm_modulate,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

_HIDDEN = 6144


def _has_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _inputs(rows: int, magnitude: float, seed: int):
    torch.manual_seed(seed)
    shape = (1, rows, _HIDDEN)
    x = (
        torch.randn(shape, device="xpu") * magnitude
        + torch.randn(1, rows, 1, device="xpu") * magnitude
    ).bfloat16()
    update = (torch.randn(shape, device="xpu") * magnitude).bfloat16()
    params = (torch.randn(1, 1, 3 * _HIDDEN, device="xpu") * 0.5).bfloat16()
    gate, scale, shift = params.chunk(3, dim=-1)
    return x, update, gate.contiguous(), scale.contiguous(), shift.contiguous()


@unittest.skipUnless(_has_xpu(), "requires an Intel XPU")
class TestFlux2XpuNormFusions(CustomTestCase):
    def setUp(self):
        self.norm = nn.LayerNorm(
            _HIDDEN, elementwise_affine=False, eps=1e-6, device="xpu"
        )

    def test_layernorm_modulate_is_bit_exact(self):
        for rows, magnitude, seed in ((512, 1.0, 0), (17, 30.0, 1), (64, 0.05, 2)):
            x, _, _, scale, shift = _inputs(rows, magnitude, seed)
            self.assertTrue(can_use_xpu_layernorm_modulate(x, scale, shift))
            expected = self.norm(x) * (1 + scale) + shift
            actual = xpu_layernorm_modulate(x, scale, shift, self.norm.eps)
            self.assertTrue(torch.equal(actual, expected), (rows, magnitude))

    def test_gated_resnorm_is_bit_exact(self):
        for rows, magnitude, seed in ((512, 1.0, 0), (17, 30.0, 1), (64, 0.05, 2)):
            residual, update, gate, scale, shift = _inputs(rows, magnitude, seed)
            self.assertTrue(
                can_use_xpu_gated_resnorm(residual, update, gate, scale, shift)
            )
            expected_residual = residual_gate_add(residual, update, gate)
            expected = self.norm(expected_residual) * (1 + scale) + shift
            actual, actual_residual = xpu_gated_resnorm(
                residual, update, gate, scale, shift, self.norm.eps
            )
            self.assertTrue(torch.equal(actual_residual, expected_residual))
            self.assertTrue(torch.equal(actual, expected), (rows, magnitude))

    def test_flux2_dispatch_uses_xpu_kernels(self):
        residual, update, gate, scale, shift = _inputs(32, 1.0, 3)
        expected_residual = residual_gate_add(residual, update, gate)
        expected = self.norm(expected_residual) * (1 + scale) + shift

        pending = flux2._defer_gated_residual(residual, update, gate)
        self.assertIsInstance(pending, tuple)
        actual, actual_residual = flux2._flux2_gated_resnorm(
            self.norm, residual, update, gate, scale, shift
        )
        self.assertTrue(torch.equal(actual_residual, expected_residual))
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(flux2._FLUX2_XPU_GATED_RESNORM.disabled)

        with patch.object(
            flux2, "xpu_layernorm_modulate", wraps=flux2.xpu_layernorm_modulate
        ) as spy:
            out = flux2._flux2_norm_modulate(self.norm, residual, scale, shift)
        spy.assert_called_once()
        self.assertTrue(torch.equal(out, self.norm(residual) * (1 + scale) + shift))

    def test_unsupported_inputs_stay_eager(self):
        for batch, dtype in ((1, torch.float16), (2, torch.bfloat16)):
            residual = torch.randn(batch, 17, _HIDDEN, device="xpu", dtype=dtype)
            update = torch.randn_like(residual)
            gate = torch.randn(batch, 1, _HIDDEN, device="xpu", dtype=dtype)
            actual = flux2._defer_gated_residual(residual, update, gate)
            self.assertIsInstance(actual, torch.Tensor)
            self.assertTrue(
                torch.equal(actual, residual_gate_add(residual, update, gate))
            )

        # Rows narrower than torch-xpu's 1024-wide work-group take another dispatch.
        x = torch.randn(1, 8, 3072, device="xpu", dtype=torch.bfloat16)
        row = torch.randn(3072, device="xpu", dtype=torch.bfloat16)
        self.assertFalse(can_use_xpu_layernorm_modulate(x, row, row))

    def test_flux2_imports_without_intel_triton(self):
        # Regression: the kernel module imported `triton.language.extra.intel`
        # at module scope. The package facade resolves `_EXPORTS` eagerly on
        # attribute access, so flux_2.py's top-level import of the XPU entry
        # points made Intel Triton an import-time requirement on *every*
        # platform -- `import flux_2` then raised ModuleNotFoundError on CUDA,
        # taking out FLUX.2 there entirely. Run in a subprocess so blocking the
        # module cannot leak into this interpreter's import state.
        script = textwrap.dedent(
            """
            import importlib.abc
            import sys

            import triton
            import triton.language  # import triton fully first, as CUDA would

            class Blocker(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name.startswith("triton.language.extra.intel"):
                        raise ImportError(name)
                    return None

            for mod in [m for m in sys.modules if "extra.intel" in m]:
                del sys.modules[mod]
            if hasattr(triton.language.extra, "intel"):
                del triton.language.extra.intel
            sys.meta_path.insert(0, Blocker())

            import sglang.multimodal_gen.runtime.models.dits.flux_2  # noqa: F401
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
