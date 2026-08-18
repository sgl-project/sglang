"""FLUX.2 eager fusions must be bit-exact for real packed/view layouts."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.multimodal_gen.runtime.models.dits.flux_2 as flux2
from sglang.multimodal_gen.runtime.models.dits.flux_2 import (
    _flux2_norm_modulate,
    _flux2_swiglu,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFlux2EagerFusions(CustomTestCase):
    def setUp(self):
        flux2._FLUX2_LN_MOD.disabled = False
        flux2._FLUX2_LN_MOD.verified = False
        flux2._FLUX2_LN_MOD_SIGS.clear()
        flux2._FLUX2_SWIGLU.disabled = False
        flux2._FLUX2_SWIGLU.verified = False
        flux2._FLUX2_SWIGLU_SIGS.clear()

    def test_norm_modulate_is_bit_exact_across_sequence_lengths(self):
        torch.manual_seed(0)
        hidden = 256
        norm = torch.nn.LayerNorm(
            hidden, eps=1e-6, elementwise_affine=False, device="cuda"
        )
        # FLUX.2 modulation values are views of one packed projection.
        params = torch.randn(1, 1, 6 * hidden, device="cuda").bfloat16()
        shift, scale = params.chunk(6, dim=-1)[:2]

        for seq in (17, 65):
            x = torch.randn(1, seq, hidden, device="cuda").bfloat16()
            expected = norm(x) * (1 + scale) + shift
            actual = _flux2_norm_modulate(norm, x, scale, shift)
            self.assertTrue(torch.equal(actual, expected))

        self.assertFalse(flux2._FLUX2_LN_MOD.disabled)
        self.assertEqual(len(flux2._FLUX2_LN_MOD_SIGS), 1)

    def test_packed_swiglu_is_bit_exact_for_contiguous_and_strided_views(self):
        torch.manual_seed(1)
        hidden = 384
        inputs = [
            torch.randn(1, 19, 2 * hidden, device="cuda").bfloat16(),
            torch.randn(1, 19, 3 * hidden, device="cuda").bfloat16()[..., : 2 * hidden],
        ]
        for x in inputs:
            expected = F.silu(x[..., :hidden]) * x[..., hidden:]
            actual = _flux2_swiglu(x)
            self.assertTrue(torch.equal(actual, expected))

        self.assertFalse(flux2._FLUX2_SWIGLU.disabled)
        self.assertEqual(len(flux2._FLUX2_SWIGLU_SIGS), 2)

    def test_fp16_preserves_reference_path(self):
        x = torch.randn(1, 17, 512, device="cuda", dtype=torch.float16)
        expected = F.silu(x[..., :256]) * x[..., 256:]
        actual = _flux2_swiglu(x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(flux2._FLUX2_SWIGLU.disabled)

    def test_packed_swiglu_rejects_non_dense_outer_stride(self):
        base = torch.randn(2, 23, 512, device="cuda", dtype=torch.bfloat16)
        x = base[:, :19]
        self.assertNotEqual(x.stride(0), x.shape[1] * x.stride(1))

        expected = F.silu(x[..., :256]) * x[..., 256:]
        actual = _flux2_swiglu(x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(flux2._FLUX2_SWIGLU_SIGS), 0)

    def test_new_swiglu_signature_is_not_verified_during_graph_capture(self):
        first = torch.randn(1, 17, 512, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(
            torch.equal(
                _flux2_swiglu(first),
                F.silu(first[..., :256]) * first[..., 256:],
            )
        )
        self.assertEqual(len(flux2._FLUX2_SWIGLU_SIGS), 1)

        second = torch.randn(1, 19, 768, device="cuda", dtype=torch.bfloat16)
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            actual = _flux2_swiglu(second)

        expected = F.silu(second[..., :384]) * second[..., 384:]
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(flux2._FLUX2_SWIGLU_SIGS), 1)


if __name__ == "__main__":
    unittest.main()
