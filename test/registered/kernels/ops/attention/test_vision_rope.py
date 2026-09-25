"""Representative parity coverage for the lightweight Kimi-K3 prerequisites."""

import unittest

import torch

from sglang.kernels.ops.attention.vision_rope import (
    apply_fused_qk_complex_rope,
    apply_fused_qk_complex_rope_inplace,
    prepare_fused_qk_complex_rope_inplace,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 896

TOPK = 16

NOPE_DIM = 512

ROPE_DIM = 64

MLA_DIM = NOPE_DIM + ROPE_DIM

MLA_PAGES = 256


class TestKimiK3PrerequisiteOps(CustomTestCase):
    def test_vision_rope(self):
        torch.manual_seed(4)
        qkv = torch.randn(480, 3, 12, 128, device="cuda", dtype=torch.bfloat16)
        q, k, _ = qkv.unbind(1)
        angles = torch.randn(480, 64, device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        freqs_expanded = freqs.unsqueeze(-2)

        def reference(x):
            value = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
            return torch.view_as_real(value * freqs_expanded).flatten(-2).type_as(x)

        actual_q, actual_k = apply_fused_qk_complex_rope(q, k, freqs)
        atol = 2 * torch.finfo(torch.bfloat16).eps
        torch.testing.assert_close(actual_q, reference(q), rtol=0, atol=atol)
        torch.testing.assert_close(actual_k, reference(k), rtol=0, atol=atol)

    def test_vision_rope_inplace(self):
        # VisionAttention hands the applier contiguous q/k, which is what the
        # in-place kernel requires; mirror that rather than qkv.unbind views.
        for dtype in (torch.bfloat16, torch.float16):
            torch.manual_seed(4)
            q = torch.randn(480, 12, 128, device="cuda", dtype=dtype)
            k = torch.randn(480, 12, 128, device="cuda", dtype=dtype)
            angles = torch.randn(480, 64, device="cuda")
            freqs = torch.polar(torch.ones_like(angles), angles)
            freqs_expanded = freqs.unsqueeze(-2)

            def reference(x):
                value = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
                return torch.view_as_real(value * freqs_expanded).flatten(-2).type_as(x)

            expected_q, expected_k = reference(q), reference(k)
            prepared = prepare_fused_qk_complex_rope_inplace(freqs)
            actual_q, actual_k = apply_fused_qk_complex_rope_inplace(q, k, prepared)

            atol = 2 * torch.finfo(dtype).eps
            torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=atol)
            torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=atol)

    def test_vision_rope_inplace_rejects_non_complex_frequencies(self):
        with self.assertRaises(ValueError):
            prepare_fused_qk_complex_rope_inplace(torch.randn(8, 64, device="cuda"))


if __name__ == "__main__":
    unittest.main()
