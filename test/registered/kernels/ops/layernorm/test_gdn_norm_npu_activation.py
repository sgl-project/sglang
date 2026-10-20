"""NPU gated-norm dispatch and math on CPU; no accelerator extension imported."""

import unittest

import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.qwen4_exp_cpu_test_utils import forbidden, load
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PATH = "kernels/ops/attention/fla/layernorm_gated.py"


def module(kernel=forbidden):
    return load(
        PATH,
        {
            "rms_norm_gated": None,
            "LayerNormFn": None,
            "layernorm_fn": None,
            "RMSNorm": None,
        },
        {"_is_npu": True, "_use_cpu": False, "_layer_norm_fwd": kernel},
    )


class TestNPUGatedNormActivation(CustomTestCase):
    def test_sigmoid_matches_reference_for_both_norm_orders(self):
        torch.manual_seed(8)
        code = module()
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for before in (False, True):
                for rms in (False, True):
                    for group in (None, 32):
                        with self.subTest(
                            dtype=dtype, before=before, rms=rms, group=group
                        ):
                            x = torch.randn(3, 12, 128).to(dtype)
                            z = torch.randn(3, 12, 256).to(dtype)[..., ::2]
                            w = torch.randn(128).to(dtype)
                            bias = torch.randn(128).to(dtype)
                            original = x.clone()
                            actual = code.rms_norm_gated(
                                x=x,
                                weight=w,
                                bias=bias,
                                z=z,
                                group_size=group,
                                norm_before_gate=before,
                                is_rms_norm=rms,
                                activation="sigmoid",
                            )
                            a, gate = x.float(), z.float().sigmoid()
                            if not before:
                                a = a * gate
                            groups = a.reshape(
                                3, 12, 128 // (group or 128), group or 128
                            )
                            if rms:
                                ref = F.rms_norm(groups, (group or 128,), eps=1e-6)
                            else:
                                ref = F.layer_norm(groups, (group or 128,), eps=1e-6)
                            ref = ref.reshape_as(a) * w.float() + bias.float()
                            if before:
                                ref *= gate
                            torch.testing.assert_close(actual, ref.to(dtype))
                            torch.testing.assert_close(x, original)
                            self.assertEqual(actual.dtype, dtype)

    def test_actual_rmsnorm_forward_preserves_sigmoid_gate(self):
        norm = module().RMSNorm(
            128, eps=1e-6, activation="sigmoid", dtype=torch.bfloat16
        )
        x = torch.ones(12, 128, dtype=torch.bfloat16)
        # At z=0 sigmoid is 0.5 while swish is zero, exposing accidental substitution.
        actual = norm(x, torch.zeros_like(x))
        torch.testing.assert_close(actual, torch.full_like(x, 0.5))

    def test_swish_and_silu_keep_existing_kernel(self):
        x = torch.randn(4, 128)
        z = torch.randn(2, 2, 256)[..., ::2]
        w = torch.ones(128)

        def kernel(x, weight, bias, eps, *, z, activation, **kwargs):
            self.assertEqual(activation, "swish")
            self.assertEqual(z.shape, x.shape)
            self.assertTrue(z.is_contiguous())
            return F.rms_norm(x, (128,), eps=eps) * F.silu(z), None, None

        code = module(kernel)
        expected = F.rms_norm(x, (128,), eps=1e-6) * F.silu(z.reshape_as(x))
        for activation in ("swish", "silu"):
            actual = code.rms_norm_gated(
                x=x, weight=w, bias=None, z=z, is_rms_norm=True, activation=activation
            )
            torch.testing.assert_close(actual, expected)

    def test_sigmoid_supports_absent_and_strided_three_dimensional_gate(self):
        code = module()
        x = torch.randn(24, 128)
        w = torch.ones(128)
        for gate in (None, torch.randn(2, 12, 256)[..., ::2]):
            actual = code.rms_norm_gated(
                x=x, weight=w, bias=None, z=gate, is_rms_norm=True, activation="sigmoid"
            )
            expected = F.rms_norm(x, (128,), eps=1e-6)
            if gate is not None:
                expected *= gate.reshape_as(x).sigmoid()
            torch.testing.assert_close(actual, expected)

    def test_unknown_activation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "activation"):
            module().rms_norm_gated(
                x=torch.ones(1, 128),
                weight=torch.ones(128),
                bias=None,
                activation="not-an-activation",
            )


if __name__ == "__main__":
    unittest.main()
