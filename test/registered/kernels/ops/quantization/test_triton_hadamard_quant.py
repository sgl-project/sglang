"""Correctness test for the fused Walsh-Hadamard + block FP8 quant Triton kernel
(`sglang.srt.layers.attention.dsa.triton_hadamard_quant.fused_hadamard_act_quant`).

The kernel fuses the DSA-indexer query's Hadamard rotation and block (group=128)
FP8 quant into one pass, optionally folding `weights * q_scale * softmax_scale`.
This checks it against a pure-torch reference built from the module's own +/-1
Sylvester matrix, so the test is backend-portable (e4m3fn on CUDA, e4m3fnuz on
ROCm/gfx950 -- both driven by the module's own _FP8_DTYPE/_FP8_MAX).
"""

import unittest

import torch

from sglang.srt.layers.attention.dsa.triton_hadamard_quant import (
    _FP8_DTYPE,
    _FP8_MAX,
    _hadamard_pm1,
    fused_hadamard_act_quant,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# gfx950 is the deployment target and exercises the e4m3fnuz path.
register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")

_BLOCK = 128
# (rows..., heads, head_dim==block): covers single-token decode and large prefill.
_SHAPES = [(8, 32, _BLOCK), (4096, 32, _BLOCK), (1, 8, _BLOCK), (2, 1, _BLOCK)]


def _reference(x: torch.Tensor, scale_fmt):
    """Pure-torch mirror of the kernel math (docstring of triton_hadamard_quant)."""
    n = x.size(-1)
    # Kernel casts x to bf16, matmuls against the bf16 +/-1 matrix (fp32 accum),
    # then scales by 1/sqrt(N).
    h = _hadamard_pm1(n, x.device, torch.bfloat16).float()
    had = (x.reshape(-1, n).to(torch.bfloat16).float() @ h) * (float(n) ** -0.5)
    amax = had.abs().amax(dim=-1).clamp_min(1e-4)
    if scale_fmt is not None:
        s = torch.exp2(torch.ceil(torch.log2(amax / _FP8_MAX)))  # round_scale
    else:
        s = amax / _FP8_MAX
    y = (had / s[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return y, s


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU for the Triton kernel")
class TestFusedHadamardActQuant(CustomTestCase):
    def _check(self, shape, scale_fmt):
        torch.manual_seed(0)
        x = (
            torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.3
        ).contiguous()

        y_f, s_f = fused_hadamard_act_quant(x.clone(), _BLOCK, scale_fmt)
        y_ref, s_ref = _reference(x, scale_fmt)

        self.assertEqual(y_f.dtype, _FP8_DTYPE)
        self.assertEqual(y_f.shape, x.shape)

        y_f2 = y_f.reshape(-1, shape[-1])
        s_f1 = s_f.reshape(-1)

        # scale within tolerance
        s_rel = ((s_ref - s_f1).abs() / (s_ref.abs() + 1e-9)).max().item()
        self.assertLess(
            s_rel, 1e-2, f"scale rel-err {s_rel} (shape={shape}, fmt={scale_fmt})"
        )

        # dequantized cosine ~= 1
        dq_ref = (y_ref.float() * s_ref[:, None]).flatten()
        dq_f = (y_f2.float() * s_f1[:, None]).flatten()
        cos = torch.nn.functional.cosine_similarity(dq_ref, dq_f, dim=0).item()
        self.assertGreater(
            cos, 0.999, f"dequant cos {cos} (shape={shape}, fmt={scale_fmt})"
        )

        # the vast majority of fp8 bytes match exactly (allow rare 1-ULP boundary diffs)
        exact = (
            (y_ref.view(torch.uint8) == y_f2.view(torch.uint8)).float().mean().item()
        )
        self.assertGreater(
            exact, 0.95, f"fp8 exact frac {exact} (shape={shape}, fmt={scale_fmt})"
        )

    def test_fp8_matches_reference(self):
        for shape in _SHAPES:
            for scale_fmt in (None, "e4m3"):
                with self.subTest(shape=shape, scale_fmt=scale_fmt):
                    self._check(shape, scale_fmt)

    def test_weights_fold(self):
        # The optional 3rd output folds weights * q_scale(s) * softmax_scale.
        softmax_scale = 0.1337
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                torch.manual_seed(1)
                x = (
                    torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.3
                ).contiguous()
                w = torch.randn(
                    *shape[:-1], device="cuda", dtype=torch.bfloat16
                ).contiguous()

                _, s_f, w_fused = fused_hadamard_act_quant(
                    x.clone(), _BLOCK, None, weights=w, softmax_scale=softmax_scale
                )
                self.assertEqual(w_fused.shape, (*shape[:-1], 1))
                w_ref = w.to(torch.float32).unsqueeze(-1) * s_f * softmax_scale
                max_abs = (w_ref.float() - w_fused.float()).abs().max().item()
                self.assertLess(
                    max_abs, 1e-3, f"weights-fold max-abs {max_abs} (shape={shape})"
                )


if __name__ == "__main__":
    unittest.main(verbosity=3)
