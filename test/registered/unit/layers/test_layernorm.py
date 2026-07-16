"""Unit tests for srt/layers/layernorm.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

import sglang.srt.layers.layernorm as layernorm_module
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.test_utils import CustomTestCase


class TestRMSNormHipVllmFallback(CustomTestCase):
    """Pin the vllm fused_add_rms_norm call shape used by RMSNorm.forward_hip.

    forward_hip binds vllm's fused_add_rms_norm, whose signature is
    (input, residual, weight, epsilon): four positional args, in-place on
    input and residual — the same shape forward_cuda binds from sgl_kernel.
    It used to be called with AITER's six-arg out-of-place shape
    (out, x, residual_out, residual, weight, eps), so any AMD device without
    AITER raised

        TypeError: fused_add_rms_norm() takes 4 positional arguments but 6 were given

    on every residual RMSNorm, i.e. in every transformer layer. CDNA never hit
    it because SGLANG_USE_AITER routes to forward_aiter instead; RDNA has no
    AITER and is forced down this path. The signature is an external vendor
    API, so it is pinned here rather than left to be re-copied from the AITER
    branch.
    """

    @staticmethod
    def _norm():
        norm = RMSNorm(8, eps=1e-6)
        with torch.no_grad():
            norm.weight.fill_(1.0)
        return norm

    def test_residual_path_uses_vllm_in_place_signature(self):
        norm = self._norm()
        x = torch.randn(2, 8)
        residual = torch.randn(2, 8)
        with (
            patch.object(layernorm_module, "_has_vllm_rms_norm", True),
            patch.object(layernorm_module, "fused_add_rms_norm", create=True) as fused,
        ):
            out, residual_out = norm.forward_hip(x, residual)

        args, kwargs = fused.call_args
        self.assertEqual(kwargs, {})
        self.assertEqual(
            len(args), 4, "vllm fused_add_rms_norm takes 4 positional args"
        )
        self.assertIs(args[0], x)
        self.assertIs(args[1], residual)
        # .data returns a fresh view object each access, so compare storage
        self.assertEqual(args[2].data_ptr(), norm.weight.data_ptr())
        self.assertEqual(args[3], norm.variance_epsilon)
        # in-place contract: the mutated inputs are what is returned, matching
        # forward_cuda, which binds a fused_add_rmsnorm of the same shape.
        self.assertIs(out, x)
        self.assertIs(residual_out, residual)

    def test_no_vllm_falls_back_to_native(self):
        # The guard that keeps a box with neither AITER nor vllm working at all:
        # without it forward_hip would call an unbound name.
        norm = self._norm()
        x = torch.randn(2, 8)
        with patch.object(layernorm_module, "_has_vllm_rms_norm", False):
            out = norm.forward_hip(x)
        torch.testing.assert_close(out, norm.forward_native(x))


if __name__ == "__main__":
    unittest.main()
