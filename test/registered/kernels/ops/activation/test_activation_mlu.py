"""MLU SiLU gating semantics, wide tensors and input preservation."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
import torch_mlu  # noqa: F401
import torch_mlu_ops

from sglang.srt.layers.activation import (
    GeluAndMul,
    NewGELU,
    QuickGELU,
    SiluAndMul,
    SituAndMul,
)
from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=30, suite="pr-test-1-mlu")


class TestActivationMLU(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)

    def _check_silu_and_mul(self, x):
        act = SiluAndMul()
        original = x.clone()
        d = x.shape[-1] // 2
        # Match the explicit formula used by community activation tests. Compute
        # it before execution so accidental vendor input writes cannot taint it.
        expected = F.silu(original[..., :d]) * original[..., d:]
        actual = act(x)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        torch.testing.assert_close(x, original, rtol=0, atol=0)

    def test_gate_up_order_and_silu_regimes(self):
        """Preserve leading dimensions, gate/up order and SiLU across regimes."""
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                # Include the negative tail, near-zero region and positive tail.
                gate = torch.tensor(
                    [-20, -8, -2, -0.125, 0, 0.125, 2, 8],
                    device="mlu",
                    dtype=dtype,
                ).repeat(8)
                up = torch.tensor(
                    [3, -2, 0.5, 4, -3, 0, -1, 2], device="mlu", dtype=dtype
                ).repeat(8)
                # Distinct rows also expose accidental broadcasting across tokens.
                x = torch.stack(
                    [torch.cat((gate.roll(i), up.roll(-i))) for i in range(6)]
                ).reshape(2, 3, 128)
                self._check_silu_and_mul(x)

    def test_row_strided_inputs_fall_back(self):
        """A contiguous last dimension does not imply a contiguous vendor input."""
        x = torch.randn(4, 128, device="mlu", dtype=torch.bfloat16)[::2]
        for act in (SiluAndMul(), SituAndMul(), GeluAndMul(), NewGELU(), QuickGELU()):
            with self.subTest(activation=type(act).__name__):
                original = x.clone()
                expected = act.forward_native(x)
                with patch.object(
                    torch_mlu_ops, "active", wraps=torch_mlu_ops.active
                ) as op:
                    actual = act(x)
                op.assert_not_called()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(x, original, rtol=0, atol=0)

    def test_situ_and_mul(self):
        for dtype, beta, linear_beta in (
            (torch.bfloat16, 1.0, None),
            (torch.float16, 1.7, 2.3),
        ):
            with self.subTest(dtype=dtype, beta=beta, linear_beta=linear_beta):
                act = SituAndMul(beta=beta, linear_beta=linear_beta)
                x = torch.randn(2, 3, 128, device="mlu", dtype=dtype)
                original = x.clone()
                expected = act.forward_native(x)
                with patch.object(
                    torch_mlu_ops, "active", wraps=torch_mlu_ops.active
                ) as op:
                    actual = act(x)
                if dtype == torch.bfloat16:
                    op.assert_called()
                torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
                torch.testing.assert_close(x, original, rtol=0, atol=0)
        act = SituAndMul()
        for x in (
            torch.randn(2, 128, device="mlu", dtype=torch.float32),
            torch.empty(0, 128, device="mlu", dtype=torch.bfloat16),
            torch.randn(128, 2, device="mlu", dtype=torch.bfloat16).T,
        ):
            with patch.object(
                torch_mlu_ops, "active", wraps=torch_mlu_ops.active
            ) as op:
                actual = act(x)
            op.assert_not_called()
            torch.testing.assert_close(actual, act.forward_native(x), rtol=0, atol=0)

    def test_gelu_variants(self):
        def check(act, x, fused):
            original = x.clone()
            expected = act.forward_native(x)
            with patch.object(
                torch_mlu_ops, "active", wraps=torch_mlu_ops.active
            ) as op:
                actual = act(x)
            if not fused:
                op.assert_not_called()
            elif x.dtype == torch.bfloat16:
                op.assert_called()
            tolerance = (1e-5 if x.dtype == torch.float32 else 0.03) if fused else 0
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(x, original, rtol=0, atol=0)

        # FP32 support is part of the vendor kernel contract.
        for act, fused_fp32 in (
            (GeluAndMul("none"), False),
            (GeluAndMul("tanh"), True),
            (QuickGELU(), False),
            (NewGELU(), True),
        ):
            with self.subTest(
                activation=type(act).__name__,
                approximate=getattr(act, "approximate", None),
            ):
                for dtype in (torch.float16, torch.bfloat16, torch.float32):
                    x = torch.randn(2, 3, 128, device="mlu", dtype=dtype)
                    fused = dtype != torch.float32 or fused_fp32
                    check(act, x, fused)
                for x in (
                    torch.empty(0, 128, device="mlu", dtype=torch.bfloat16),
                    torch.randn(128, 2, device="mlu", dtype=torch.bfloat16).T,
                ):
                    check(act, x, False)


if __name__ == "__main__":
    unittest.main()
