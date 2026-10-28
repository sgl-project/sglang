"""FLUX.2 XPU LayerNorm+modulate and gated residual norm fast paths.

The fusion cases need an Intel XPU; the Intel-Triton import regression runs
in the diffusion unit suite, which installs the model's optional dependencies.
"""

import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

import sglang.multimodal_gen.runtime.models.dits.flux_2 as flux2
from sglang.kernels.ops.diffusion import (
    can_defer_xpu_gated_residual,
    can_use_xpu_layernorm_modulate,
    residual_gate_add,
    xpu_gated_resnorm,
    xpu_layernorm_modulate,
)
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")
register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")

_HIDDEN = 6144


def _has_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _inputs(rows: int, magnitude: float, seed: int, batch: int = 1):
    torch.manual_seed(seed)
    shape = (batch, rows, _HIDDEN)
    x = (
        torch.randn(shape, device="xpu") * magnitude
        + torch.randn(batch, rows, 1, device="xpu") * magnitude
    ).bfloat16()
    update = (torch.randn(shape, device="xpu") * magnitude).bfloat16()
    # Per-batch modulation rows as strided chunks of one adaLN projection.
    params = (torch.randn(batch, 1, 3 * _HIDDEN, device="xpu") * 0.5).bfloat16()
    gate, scale, shift = params.chunk(3, dim=-1)
    return x, update, gate, scale, shift


_CASES = ((512, 1.0, 0, 1), (17, 30.0, 1, 1), (64, 0.05, 2, 1), (64, 1.0, 3, 2))


class TestFlux2ImportWithoutIntelTriton(CustomTestCase):
    def test_flux2_imports_without_intel_triton(self):
        # Importing flux_2 must not need Intel Triton; a subprocess keeps the
        # blocked import from leaking into this interpreter.
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


@unittest.skipUnless(_has_xpu(), "requires an Intel XPU")
class TestFlux2XpuNormFusions(CustomTestCase):
    def setUp(self):
        self.norm = nn.LayerNorm(
            _HIDDEN, elementwise_affine=False, eps=1e-6, device="xpu"
        )

    def test_layernorm_modulate_is_bit_exact(self):
        for rows, magnitude, seed, batch in _CASES:
            x, _, _, scale, shift = _inputs(rows, magnitude, seed, batch)
            self.assertTrue(can_use_xpu_layernorm_modulate(x.dtype, x.shape[-1]))
            expected = self.norm(x) * (1 + scale) + shift
            actual = xpu_layernorm_modulate(x, scale, shift, self.norm.eps)
            self.assertTrue(torch.equal(actual, expected), (rows, magnitude, batch))

    def test_gated_resnorm_is_bit_exact(self):
        for rows, magnitude, seed, batch in _CASES:
            residual, update, gate, scale, shift = _inputs(rows, magnitude, seed, batch)
            self.assertTrue(can_defer_xpu_gated_residual(residual, update, gate))
            expected_residual = residual_gate_add(residual, update, gate)
            expected = self.norm(expected_residual) * (1 + scale) + shift
            actual, actual_residual = xpu_gated_resnorm(
                residual, update, gate, scale, shift, self.norm.eps
            )
            self.assertTrue(torch.equal(actual_residual, expected_residual))
            self.assertTrue(torch.equal(actual, expected), (rows, magnitude, batch))

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
        fp16 = torch.randn(1, 17, _HIDDEN, device="xpu", dtype=torch.float16)
        strided = torch.randn(1, _HIDDEN, 17, device="xpu").bfloat16().transpose(1, 2)
        for residual in (fp16, strided):
            update = torch.randn(residual.shape, device="xpu", dtype=residual.dtype)
            gate = torch.randn(1, 1, _HIDDEN, device="xpu", dtype=residual.dtype)
            actual = flux2._defer_gated_residual(residual, update, gate)
            self.assertIsInstance(actual, torch.Tensor)
            self.assertTrue(
                torch.equal(actual, residual_gate_add(residual, update, gate))
            )

        # Rows narrower than torch-xpu's 1024-wide work-group take another dispatch.
        self.assertFalse(can_use_xpu_layernorm_modulate(torch.bfloat16, 3072))

    def test_swiglu_into_out_projection_input_matches_cat(self):
        # The single-stream block writes SwiGLU straight into the to_out input
        # buffer (row stride 1536 + 4608); it must equal the old cat input.
        torch.manual_seed(4)
        attn = torch.randn(1, 64, 1536, device="xpu", dtype=torch.bfloat16)
        mlp_proj = torch.randn(1, 64, 9216, device="xpu", dtype=torch.bfloat16)
        expected = torch.cat([attn, flux2._flux2_swiglu(mlp_proj)], dim=-1)

        gemm_input = torch.full_like(expected, float("nan"))
        returned = flux2._flux2_swiglu(mlp_proj, out=gemm_input[..., 1536:])
        gemm_input[..., :1536].copy_(attn)

        self.assertEqual(returned.data_ptr(), gemm_input[..., 1536:].data_ptr())
        self.assertTrue(torch.equal(gemm_input, expected))

    def test_attention_reads_fused_projection_slices_in_place(self):
        # Single-block q/k/v are row-strided slices of the fused projection
        # output; the XPU backend must feed them to flash attention without a
        # copy and get the same result as contiguous inputs.
        from sglang.multimodal_gen.runtime.layers.attention.backends import (
            xpu_backend,
        )

        torch.manual_seed(5)
        tokens, heads, head_dim = 256, 12, 128
        proj = torch.randn(1, tokens, 13824, device="xpu", dtype=torch.bfloat16)
        q, k, v = (
            proj[..., i * heads * head_dim : (i + 1) * heads * head_dim].view(
                1, tokens, heads, head_dim
            )
            for i in range(3)
        )
        self.assertEqual(
            xpu_backend._as_fa_input(q, tokens, heads, head_dim).data_ptr(),
            q.data_ptr(),
        )
        impl = xpu_backend.XPUAttentionImpl(
            num_heads=heads,
            head_size=head_dim,
            causal=False,
            softmax_scale=head_dim**-0.5,
        )
        expected = impl.forward(q.contiguous(), k.contiguous(), v.contiguous())
        self.assertTrue(torch.equal(impl.forward(q, k, v), expected))

    def test_launchers_reject_layouts_they_do_not_replicate(self):
        x, update, gate, scale, shift = _inputs(16, 1.0, 4)
        strided = x.transpose(1, 2).contiguous().transpose(1, 2)
        with self.assertRaises(RuntimeError):
            xpu_layernorm_modulate(strided, scale, shift, self.norm.eps)
        with self.assertRaises(RuntimeError):
            xpu_layernorm_modulate(x, scale.expand(2, 1, _HIDDEN), shift, 1e-6)
        with self.assertRaises(RuntimeError):
            xpu_gated_resnorm(x, update[:, :8], gate, scale, shift, self.norm.eps)


if __name__ == "__main__":
    unittest.main()
